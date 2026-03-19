"""
app.py — Flask server for AI YouTube Lecture Summariser.
Provides API routes for summarisation, flashcards, quiz, chat, and analytics.
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from transcript import extract_transcript
from summariser import (
    summarise_lecture,
    generate_flashcards,
    generate_quiz,
    chat_with_lecture,
    analyse_sentiment_difficulty,
)

app = Flask(__name__)
CORS(app)

# ── In-memory transcript cache ──────────────────
# Avoids re-fetching transcript for the same video
transcript_cache = {}


def _get_transcript(url: str) -> dict:
    """Fetch transcript from cache or YouTube."""
    if url not in transcript_cache:
        transcript_cache[url] = extract_transcript(url)
    return transcript_cache[url]


# ── Routes ──────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/summarise", methods=["POST"])
def api_summarise():
    """Generate a structured summary of the lecture."""
    data = request.get_json()
    url = data.get("url", "").strip()
    if not url:
        return jsonify({"error": "No URL provided"}), 400

    try:
        transcript_data = _get_transcript(url)
        summary = summarise_lecture(transcript_data["transcript_text"])
        return jsonify({
            "summary": summary,
            "video_id": transcript_data["video_id"],
            "segments": transcript_data["segments"],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/flashcards", methods=["POST"])
def api_flashcards():
    """Generate flashcards from the lecture."""
    data = request.get_json()
    url = data.get("url", "").strip()
    if not url:
        return jsonify({"error": "No URL provided"}), 400

    try:
        transcript_data = _get_transcript(url)
        flashcards = generate_flashcards(transcript_data["transcript_text"])
        return jsonify({"flashcards": flashcards})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/quiz", methods=["POST"])
def api_quiz():
    """Generate quiz questions from the lecture."""
    data = request.get_json()
    url = data.get("url", "").strip()
    if not url:
        return jsonify({"error": "No URL provided"}), 400

    try:
        transcript_data = _get_transcript(url)
        quiz = generate_quiz(transcript_data["transcript_text"])
        return jsonify({"quiz": quiz})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/chat", methods=["POST"])
def api_chat():
    """Chat with the lecture — answer follow-up questions."""
    data = request.get_json()
    url = data.get("url", "").strip()
    question = data.get("question", "").strip()
    history = data.get("history", [])
    if not url or not question:
        return jsonify({"error": "URL and question are required"}), 400

    try:
        transcript_data = _get_transcript(url)
        answer = chat_with_lecture(
            transcript_data["transcript_text"],
            question,
            history,
        )
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/analyse", methods=["POST"])
def api_analyse():
    """Analyse difficulty and sentiment of the lecture."""
    data = request.get_json()
    url = data.get("url", "").strip()
    if not url:
        return jsonify({"error": "No URL provided"}), 400

    try:
        transcript_data = _get_transcript(url)
        analysis = analyse_sentiment_difficulty(transcript_data["transcript_text"])
        return jsonify({"analysis": analysis})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
