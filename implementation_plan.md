# AI YouTube Lecture Summariser — Implementation Plan

A full-stack web application that takes a YouTube video URL, extracts the transcript, and uses **Google Gemini** to generate a structured lecture summary with key topics, takeaways, and notes — plus **unique AI-powered features** that set it apart from any generic summariser.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Backend** | Python 3 + Flask |
| **AI/LLM** | Google Gemini API (`google-genai`) |
| **Transcript** | `youtube-transcript-api` |
| **Frontend** | HTML + Vanilla CSS + JavaScript |
| **Charting** | Chart.js (CDN) |
| **Export** | jsPDF (CDN) |

---

## User Review Required

> [!IMPORTANT]
> **API Key**: You will need a **Google Gemini API key** (free tier available at [aistudio.google.com](https://aistudio.google.com)). The app will read it from a `.env` file.

> [!NOTE]
> The app uses `youtube-transcript-api` which extracts auto-generated or manually-added captions. **Videos without any captions/subtitles will not work.** This covers the vast majority of educational content.

---

## Proposed Changes

### Backend — Core Logic

#### [NEW] `requirements.txt`
Dependencies: `flask`, `google-genai`, `youtube-transcript-api`, `python-dotenv`, `flask-cors`

---

#### [NEW] `transcript.py`
- Function `extract_transcript(youtube_url)` that:
  - Parses the video ID from various YouTube URL formats
  - Fetches transcript via `youtube-transcript-api`
  - Returns the full transcript text and video metadata

---

#### [NEW] `summariser.py`
- Function `summarise_lecture(transcript_text)` that:
  - Sends transcript to Gemini with a structured prompt
  - Prompt asks for: **Title**, **Overview**, **Key Topics** (with explanations), **Key Takeaways**, and **Study Notes**
  - Returns the structured summary as markdown
- Function `generate_flashcards(transcript_text)` that:
  - Sends transcript to Gemini with a prompt to extract Q&A flashcards
  - Returns a JSON array of `{ question, answer }` pairs
- Function `generate_quiz(transcript_text)` that:
  - Sends transcript to Gemini with a prompt to generate MCQ quiz questions
  - Returns a JSON array of `{ question, options[], correctIndex, explanation }` objects
- Function `chat_with_lecture(transcript_text, user_question, chat_history)` that:
  - Accepts a user question + the transcript as context
  - Maintains conversational context via `chat_history`
  - Returns an AI answer grounded in the lecture content
- Function `analyse_sentiment_difficulty(transcript_text)` that:
  - Asks Gemini to estimate topic-wise difficulty level and overall lecture sentiment/tone
  - Returns structured JSON for chart rendering

---

#### [NEW] `app.py`
- Flask server with routes:
  - `GET /` — serves the frontend
  - `POST /api/summarise` — accepts `{ "url": "..." }`, returns the summary JSON
  - `POST /api/flashcards` — generates flashcards from transcript
  - `POST /api/quiz` — generates quiz questions from transcript
  - `POST /api/chat` — accepts `{ "url", "question", "history" }` for lecture Q&A
  - `POST /api/analyse` — returns difficulty/sentiment analysis JSON
- CORS enabled, reads API key from `.env`
- In-memory transcript caching (avoids re-fetching for same video)

---

#### [NEW] `.env`
- `GEMINI_API_KEY=your_api_key_here`

---

### Frontend — Modern UI

#### [NEW] `templates/index.html`
- Single-page app with **tabbed interface**:
  1. **Summary Tab** — structured summary with expandable sections
  2. **Flashcards Tab** — interactive flip-card UI for revision
  3. **Quiz Tab** — timed MCQ quiz with scoring and explanations
  4. **Chat Tab** — ask follow-up questions about the lecture
  5. **Analytics Tab** — visual charts for topic difficulty & lecture sentiment
- **Hero section** with animated gradient background
- **URL input** with paste/submit functionality
- **Loading state** with skeleton loader animation
- **Copy-to-clipboard** and **Download as PDF** buttons
- **Dark/Light mode toggle** with local storage persistence
- **Timestamp navigation** — clicking a topic scrolls to its timestamp in an embedded YouTube player
- Responsive design (mobile-friendly)

#### [NEW] `static/style.css`
- Dark-mode glassmorphism design with light-mode alternative
- Animated gradients, hover effects, micro-animations
- Custom scrollbar, smooth transitions
- Google Fonts (Inter)
- Flashcard flip animation (3D CSS transform)
- Tab navigation styles, quiz progress bar

#### [NEW] `static/script.js`
- Handles form submission, API calls, loading states
- Renders markdown summary into styled HTML
- **Tab system** — switches between Summary, Flashcards, Quiz, Chat, Analytics
- **Flashcard engine** — flip animation, next/previous navigation, shuffle
- **Quiz engine** — timer, option selection, scoring, review mode with explanations
- **Chat interface** — message bubbles, auto-scroll, conversation history
- **Analytics renderer** — uses Chart.js for radar/bar charts
- **PDF export** — generates a formatted PDF of the summary using jsPDF
- Copy and download functionality
- Dark/Light mode toggle logic
- **YouTube embed** — embeds the video with timestamp-linked topics

---

## 🌟 Unique Features That Set This Apart

| # | Feature | What Makes It Unique |
|---|---------|---------------------|
| 1 | **Interactive Flashcards** | AI generates Q&A cards from the lecture — users can flip, shuffle, and study |
| 2 | **AI Quiz Generator** | Timed MCQs auto-generated from the lecture with explanations on submit |
| 3 | **Chat With Your Lecture** | Ask follow-up questions — the AI answers using the lecture as context |
| 4 | **Difficulty & Sentiment Analysis** | Visual radar/bar charts showing topic complexity and lecture tone |
| 5 | **PDF Export** | Download a beautifully formatted PDF of the full summary |
| 6 | **Dark/Light Mode** | Toggle between themes with smooth transitions and persistence |
| 7 | **Timestamp Navigation** | Topics link to their timestamps in an embedded YouTube player |
| 8 | **Tabbed Multi-Tool UI** | Not just a summariser — it's a full lecture study companion |

---

## Project Structure

```
AI-Youtube-Lecture-Summariser/
├── app.py                  # Flask server with all API routes
├── transcript.py           # YouTube transcript extraction
├── summariser.py           # Gemini — summary, flashcards, quiz, chat, analytics
├── requirements.txt        # Python dependencies
├── .env                    # API key (gitignored)
├── templates/
│   └── index.html          # Frontend page (tabbed SPA)
└── static/
    ├── style.css           # Styles (dark/light, animations, flashcards, quiz)
    └── script.js           # Client logic (tabs, quiz engine, chat, charts, PDF)
```

---

## Verification Plan

### Automated Tests
1. **Run the Flask server**:
   ```bash
   cd C:\Users\Faiz\Downloads\AI_Youtube_Lecture_Summariser
   pip install -r requirements.txt
   python app.py
   ```
2. **Browser test**: Use the browser tool to navigate to `http://127.0.0.1:5000`, paste a YouTube lecture URL, and verify:
   - Summary tab renders with structured sections
   - Flashcards tab shows flip-able cards
   - Quiz tab presents timed MCQs with scoring
   - Chat tab allows follow-up questions
   - Analytics tab displays charts
   - Dark/Light mode toggle works
   - PDF download generates a file

### Manual Verification
1. Open `http://127.0.0.1:5000` in your browser
2. Paste a YouTube lecture URL (e.g., a Khan Academy or MIT OCW video)
3. Click **Summarise** and verify:
   - A skeleton loading animation appears
   - A structured summary is displayed with sections
   - All 5 tabs are functional
   - Copy and Download PDF buttons work
   - Dark/Light mode toggles correctly
   - Embedded YouTube player shows correct video
