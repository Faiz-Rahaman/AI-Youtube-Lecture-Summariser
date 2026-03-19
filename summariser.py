"""
summariser.py — AI module for the Lecture Summariser.
Supports both Groq (default, free, fast) and Google Gemini as AI providers.
Provides: summary, flashcards, quiz, chat, and analytics functions.
"""

import json
import os
import re
import time
from dotenv import load_dotenv

load_dotenv()

# ── Provider Setup ─────────────────────────────
# Priority: Groq first (more generous free tier), then Gemini
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

if GROQ_API_KEY:
    from groq import Groq
    groq_client = Groq(api_key=GROQ_API_KEY)
    PROVIDER = "groq"
    print("[AI Provider] Using Groq (fast, generous free tier)")
elif GEMINI_API_KEY:
    from google import genai
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    PROVIDER = "gemini"
    print("[AI Provider] Using Google Gemini")
else:
    raise ValueError("No API key found! Set GROQ_API_KEY or GEMINI_API_KEY in your .env file.")


def _call_ai(prompt: str) -> str:
    """Send a prompt to the configured AI provider with retry logic."""
    if PROVIDER == "groq":
        return _call_groq(prompt)
    else:
        return _call_gemini(prompt)


def _call_groq(prompt: str) -> str:
    """Call Groq API — very fast, generous free tier."""
    for attempt in range(3):
        try:
            response = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=4096,
            )
            return response.choices[0].message.content
        except Exception as e:
            error_str = str(e)
            if "rate_limit" in error_str.lower() or "429" in error_str:
                wait = 3 * (2 ** attempt)
                print(f"[Groq rate limit] Retrying in {wait}s (attempt {attempt+1}/3)...")
                time.sleep(wait)
            else:
                raise
    raise Exception("Groq rate limit exceeded. Please wait a moment and try again.")


def _call_gemini(prompt: str) -> str:
    """Call Gemini API with retry and fallback."""
    models = ["gemini-2.0-flash", "gemini-2.0-flash-lite"]
    last_error = None

    for model in models:
        for attempt in range(2):
            try:
                response = gemini_client.models.generate_content(
                    model=model,
                    contents=prompt,
                )
                return response.text
            except Exception as e:
                last_error = e
                error_str = str(e)
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                    wait = 3 * (2 ** attempt)
                    print(f"[Rate limit] Retrying in {wait}s (attempt {attempt+1}/2, model={model})...")
                    time.sleep(wait)
                else:
                    raise
        print(f"[Rate limit] Model {model} exhausted, trying next...")

    raise Exception(
        "Your Gemini API key has hit its daily limit. "
        "Try using Groq instead — add GROQ_API_KEY to your .env file. "
        "Get a free key at https://console.groq.com"
    )


def _extract_json(text: str):
    """Extract JSON from a response that might contain markdown fences or conversational filler."""
    text = text.strip()
    
    # 1. Try finding standard markdown fences
    match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except:
            pass
            
    # 2. Try slicing the exact JSON bracket structures
    try:
        first_bracket = text.find('[')
        first_brace = text.find('{')
        
        # Determine if payload is array or object
        if first_bracket != -1 and (first_brace == -1 or first_bracket < first_brace):
            start = first_bracket
            end = text.rfind(']') + 1
        elif first_brace != -1:
            start = first_brace
            end = text.rfind('}') + 1
        else:
            raise ValueError("No JSON structures found")
            
        return json.loads(text[start:end])
    except Exception as e:
        # 3. Final fallback, attempt direct parse
        return json.loads(text)


# ──────────────────────────────────────────────
# 1. STRUCTURED SUMMARY
# ──────────────────────────────────────────────

def summarise_lecture(transcript_text: str) -> str:
    prompt = f"""You are an expert educational content analyst. Analyse the following lecture transcript and produce a well-structured summary in **Markdown** format.

Your summary MUST include these sections in order:

## 📌 Title
A concise, descriptive title for the lecture.

## 📖 Overview
A 3-5 sentence overview of what the lecture covers.

## 🔑 Key Topics
For each major topic discussed:
### Topic Name
- A clear explanation (2-4 sentences)
- Approximate timestamp range if possible (e.g., ~2:00 - 8:30)

## 💡 Key Takeaways
- Bullet list of the 5-8 most important points a student should remember.

## 📝 Study Notes
- Detailed notes suitable for exam preparation, including definitions, formulas, and examples mentioned in the lecture.

---

TRANSCRIPT:
{transcript_text}
"""
    return _call_ai(prompt)


# ──────────────────────────────────────────────
# 2. FLASHCARDS
# ──────────────────────────────────────────────

def generate_flashcards(transcript_text: str) -> list:
    prompt = f"""You are an expert educator. Based on the following lecture transcript, generate 10-15 high-quality flashcards for student revision.

Each flashcard should test understanding of a key concept from the lecture.

Return your response as a pure JSON array (no markdown fences, no extra text) with this format:
[
  {{ "question": "What is ...?", "answer": "..." }},
  ...
]

TRANSCRIPT:
{transcript_text}
"""
    result = _call_ai(prompt)
    return _extract_json(result)


# ──────────────────────────────────────────────
# 3. QUIZ
# ──────────────────────────────────────────────

def generate_quiz(transcript_text: str) -> list:
    prompt = f"""You are an expert educator. Based on the following lecture transcript, generate 10 multiple-choice quiz questions to test a student's understanding.

Each question should have exactly 4 options. Only ONE option is correct.

Return your response as a pure JSON array (no markdown fences, no extra text) with this format:
[
  {{
    "question": "What is ...?",
    "options": ["A) ...", "B) ...", "C) ...", "D) ..."],
    "correctIndex": 0,
    "explanation": "The correct answer is A because ..."
  }},
  ...
]

TRANSCRIPT:
{transcript_text}
"""
    result = _call_ai(prompt)
    return _extract_json(result)


# ──────────────────────────────────────────────
# 4. CHAT WITH LECTURE
# ──────────────────────────────────────────────

def chat_with_lecture(transcript_text: str, user_question: str, chat_history: list | None = None) -> str:
    history_text = ""
    if chat_history:
        for msg in chat_history:
            role = "Student" if msg["role"] == "user" else "Tutor"
            history_text += f"{role}: {msg['text']}\n"

    prompt = f"""You are an AI tutor who has deep knowledge of the following lecture. Answer the student's question accurately, based ONLY on the lecture content. If the question is not covered in the lecture, say so politely.

Keep your answers concise but thorough. Use bullet points or numbered lists where appropriate.

LECTURE TRANSCRIPT:
{transcript_text}

CONVERSATION SO FAR:
{history_text}

Student: {user_question}

Tutor:"""
    return _call_ai(prompt)


# ──────────────────────────────────────────────
# 5. DIFFICULTY & SENTIMENT ANALYSIS
# ──────────────────────────────────────────────

def analyse_sentiment_difficulty(transcript_text: str) -> dict:
    prompt = f"""You are an expert educational analyst. Analyse the following lecture transcript and provide:

1. A list of main topics with estimated difficulty level (1-10 scale).
2. Overall lecture analysis: tone, pace, clarity, engagement level (each on a 1-10 scale).

Return your response as a pure JSON object (no markdown fences, no extra text) with this format:
{{
  "topics": [
    {{ "name": "Topic Name", "difficulty": 7, "description": "Brief description" }},
    ...
  ],
  "overall": {{
    "tone": "informative / casual / formal / etc.",
    "pace": 6,
    "clarity": 8,
    "engagement": 7,
    "difficulty": 6,
    "summary": "Brief 1-2 sentence analysis of the lecture style"
  }}
}}

TRANSCRIPT:
{transcript_text}
"""
    result = _call_ai(prompt)
    return _extract_json(result)
