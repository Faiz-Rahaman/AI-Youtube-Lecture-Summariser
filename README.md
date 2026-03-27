<div align="center">

# 🎓 LectureLens
**AI YouTube Lecture Summariser & Study Companion**

[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/Flask-3.x-lightgrey.svg)](https://flask.palletsprojects.com/)
[![Groq](https://img.shields.io/badge/AI-Groq%20LLaMA%203-orange.svg)](https://groq.com/)
[![Gemini](https://img.shields.io/badge/AI-Google%20Gemini-blue.svg)](https://aistudio.google.com/)

*Turn any YouTube lecture into an interactive study session in seconds.*

</div>

---

## 🌟 Overview

LectureLens is a full-stack web application that takes a YouTube video URL, instantly extracts its transcript, and uses advanced AI to generate a complete study guide. 

Unlike basic summarisers, LectureLens creates a **highly interactive, multi-tool study environment** featuring automated flashcards, timed quizzes, and a dedicated AI tutor you can chat with about the video.

### ✨ Features

- **📝 Smart Summary**: Structured markdown notes with key topics, timestamps, and study bullet points.
- **🃏 Interactive Flashcards**: AI-generated Q&A flip-cards for rapid revision, complete with shuffle and navigation.
- **🧠 AI Quiz Engine**: Timed multiple-choice questions with automated scoring and detailed explanations.
- **💬 Chat with Lecture**: Ask follow-up questions — the AI tutor answers using *only* the lecture as context.
- **📊 Difficulty Analytics**: Visual radar and bar charts (via Chart.js) analyzing topic complexity and lecture tone.
- **📄 Export & Preserve**: Download your summary as a formatted PDF (via jsPDF) or copy it to your clipboard.
- **🎨 Premium UI**: Dark/Light mode, custom cursor tracking, glassmorphism design, 3D card tilts, and confetti celebrations.
- **⚡ Dual AI Engines**: Powered by **Groq** (insanely fast, generous free limits) with fallback to **Google Gemini**.

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.9 or higher
- An API Key from [Groq Console](https://console.groq.com/keys) (Recommended) OR [Google AI Studio](https://aistudio.google.com/).

### 2. Installation Setup
Clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/AI-Youtube-Lecture-Summariser.git
cd AI-Youtube-Lecture-Summariser

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Variables
Create a file named `.env` in the root of the project and add your API keys. The app prioritizes Groq for its speed and higher free limits.

```dotenv
GROQ_API_KEY=your_groq_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
```

### 4. Run the Server
Start the Flask development server:

```bash
python app.py
```

Open your browser and navigate to `http://127.0.0.1:5000`.

---

## 🛠️ Tech Stack

- **Backend**: Python, Flask, `youtube-transcript-api`
- **AI Libraries**: `groq`, `google-genai`
- **Frontend**: HTML5, Vanilla CSS3, Vanilla JavaScript
- **UI Components**: `Chart.js` (Analytics), `jsPDF` (Export), `Vanilla-Tilt.js` (3D Interactions), `Canvas-Confetti` (Effects)

---

## 📝 How It Works (Architecture)

1. **Input**: User pastes a YouTube URL into the frontend SPA.
2. **Extraction**: `transcript.py` parses the video ID and uses the YouTube caption API to fetch the spoken text.
3. **AI Generation**: `summariser.py` sends the transcript to the AI provider (Groq/Gemini) with strict JSON and Markdown prompts to generate the 5 different study modules.
4. **Rendering**: The backend sends the structured data back to `script.js` which dynamically builds the tabs, injects the charts, and initializes the interactivity engines.

---

<div align="center">
<i>Built using Python, Flask, and Next-Gen AI models.</i>
</div>
