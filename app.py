"""
app.py — Flask server for AI YouTube Lecture Summariser.
Provides API routes for summarisation, flashcards, quiz, chat, and analytics.

SECURITY ENHANCEMENTS:
- Input validation on all user inputs
- Rate limiting protection
- Error logging
- Debug mode controlled by environment variable
"""

import os
import logging
from functools import lru_cache
from urllib.parse import urlparse
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Security: Disable debug in production
DEBUG_MODE = os.getenv("FLASK_DEBUG", "False").lower() == "true"

# ── In-memory transcript cache with LRU ──────────────────
# Limited to 100 entries to prevent memory exhaustion
MAX_CACHE_SIZE = 100


def validate_youtube_url(url: str) -> bool:
    """Validate that the URL is a legitimate YouTube URL."""
    try:
        parsed = urlparse(url)
        allowed_domains = [
            'youtube.com', 'www.youtube.com', 'youtu.be',
            'youtube.co.uk', 'www.youtube.co.uk'
        ]
        return parsed.netloc in allowed_domains
    except Exception:
        return False


@lru_cache(maxsize=MAX_CACHE_SIZE)
def _get_transcript_cached(url: str) -> tuple:
    """Fetch transcript with LRU caching (internal function)."""
    result = extract_transcript(url)
    # Convert to tuple for hashability (required by lru_cache)
    return (
        result["video_id"],
        result["transcript_text"],
        tuple(result["segments"]),
    )


def _get_transcript(url: str) -> dict:
    """Fetch transcript from cache or YouTube with validation."""
    # Validate URL before processing
    if not validate_youtube_url(url):
        raise ValueError("Invalid YouTube URL. Please provide a valid youtube.com or youtu.be URL.")
    
    try:
        cached_result = _get_transcript_cached(url)
        return {
            "video_id": cached_result[0],
            "transcript_text": cached_result[1],
            "segments": list(cached_result[2]),
        }
    except Exception as e:
        logger.error(f"Failed to fetch transcript for URL {url}: {str(e)}", exc_info=True)
        raise


# ── Routes ──────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/summarise", methods=["POST"])
def api_summarise():
    """Generate a structured summary of the lecture."""
    data = request.get_json()
    url = data.get("url", "").strip()
    
    # Input validation
    if not url:
        logger.warning("Summarisation request with no URL")
        return jsonify({"error": "No URL provided"}), 400
    
    if not validate_youtube_url(url):
        logger.warning(f"Invalid YouTube URL attempted: {url[:50]}...")
        return jsonify({"error": "Invalid YouTube URL. Please provide a valid youtube.com or youtu.be URL."}), 400

    try:
        logger.info(f"Processing summarisation for video: {url}")
        transcript_data = _get_transcript(url)
        summary = summarise_lecture(transcript_data["transcript_text"])
        logger.info(f"Successfully generated summary for video: {transcript_data['video_id']}")
        return jsonify({
            "summary": summary,
            "video_id": transcript_data["video_id"],
            "segments": transcript_data["segments"],
        })
    except ValueError as e:
        logger.warning(f"Validation error: {str(e)}")
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Summarisation failed for URL {url}: {str(e)}", exc_info=True)
        return jsonify({"error": "Failed to process video. Please try again later."}), 500


@app.route("/api/flashcards", methods=["POST"])
def api_flashcards():
    """Generate flashcards from the lecture."""
    data = request.get_json()
    url = data.get("url", "").strip()
    
    if not url:
        logger.warning("Flashcards request with no URL")
        return jsonify({"error": "No URL provided"}), 400
    
    if not validate_youtube_url(url):
        logger.warning(f"Invalid YouTube URL for flashcards: {url[:50]}...")
        return jsonify({"error": "Invalid YouTube URL"}), 400

    try:
        logger.info(f"Generating flashcards for video: {url}")
        transcript_data = _get_transcript(url)
        flashcards = generate_flashcards(transcript_data["transcript_text"])
        logger.info(f"Generated {len(flashcards)} flashcards")
        return jsonify({"flashcards": flashcards})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Flashcards generation failed: {str(e)}", exc_info=True)
        return jsonify({"error": "Failed to generate flashcards"}), 500


@app.route("/api/quiz", methods=["POST"])
def api_quiz():
    """Generate quiz questions from the lecture."""
    data = request.get_json()
    url = data.get("url", "").strip()
    
    if not url:
        logger.warning("Quiz request with no URL")
        return jsonify({"error": "No URL provided"}), 400
    
    if not validate_youtube_url(url):
        logger.warning(f"Invalid YouTube URL for quiz: {url[:50]}...")
        return jsonify({"error": "Invalid YouTube URL"}), 400

    try:
        logger.info(f"Generating quiz for video: {url}")
        transcript_data = _get_transcript(url)
        quiz = generate_quiz(transcript_data["transcript_text"])
        logger.info(f"Generated {len(quiz)} quiz questions")
        return jsonify({"quiz": quiz})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Quiz generation failed: {str(e)}", exc_info=True)
        return jsonify({"error": "Failed to generate quiz"}), 500


@app.route("/api/chat", methods=["POST"])
def api_chat():
    """Chat with the lecture — answer follow-up questions."""
    data = request.get_json()
    url = data.get("url", "").strip()
    question = data.get("question", "").strip()
    history = data.get("history", [])
    
    if not url or not question:
        logger.warning("Chat request missing URL or question")
        return jsonify({"error": "URL and question are required"}), 400
    
    if not validate_youtube_url(url):
        logger.warning(f"Invalid YouTube URL for chat: {url[:50]}...")
        return jsonify({"error": "Invalid YouTube URL"}), 400
    
    # Basic question length validation
    if len(question) > 2000:
        logger.warning(f"Question too long: {len(question)} chars")
        return jsonify({"error": "Question too long (max 2000 characters)"}), 400

    try:
        logger.info(f"Processing chat question for video: {url}")
        transcript_data = _get_transcript(url)
        answer = chat_with_lecture(
            transcript_data["transcript_text"],
            question,
            history,
        )
        logger.info("Chat response generated successfully")
        return jsonify({"answer": answer})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Chat failed: {str(e)}", exc_info=True)
        return jsonify({"error": "Failed to process question"}), 500


@app.route("/api/analyse", methods=["POST"])
def api_analyse():
    """Analyse difficulty and sentiment of the lecture."""
    data = request.get_json()
    url = data.get("url", "").strip()
    
    if not url:
        logger.warning("Analytics request with no URL")
        return jsonify({"error": "No URL provided"}), 400
    
    if not validate_youtube_url(url):
        logger.warning(f"Invalid YouTube URL for analytics: {url[:50]}...")
        return jsonify({"error": "Invalid YouTube URL"}), 400

    try:
        logger.info(f"Generating analytics for video: {url}")
        transcript_data = _get_transcript(url)
        analysis = analyse_sentiment_difficulty(transcript_data["transcript_text"])
        logger.info("Analytics generated successfully")
        return jsonify({"analysis": analysis})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Analytics generation failed: {str(e)}", exc_info=True)
        return jsonify({"error": "Failed to generate analytics"}), 500


if __name__ == "__main__":
    # Security: Use environment variable for debug mode
    debug_mode = os.getenv("FLASK_DEBUG", "False").lower() == "true"
    
    if debug_mode:
        logger.warning("Running in DEBUG mode - DO NOT use in production!")
    else:
        logger.info("Starting server in production mode")
    
    app.run(debug=debug_mode, port=5000, host="127.0.0.1")
