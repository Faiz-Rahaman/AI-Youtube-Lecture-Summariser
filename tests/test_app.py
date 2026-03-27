"""
Test suite for LectureLens AI.
Run with: pytest tests/ -v
"""
import pytest
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import validate_youtube_url, app
from summariser import _extract_json as extract_json
from database import init_db, save_summary, get_summary


class TestYouTubeURLValidation:
    """Test URL validation security feature."""
    
    def test_valid_youtube_urls(self):
        """Test that valid YouTube URLs are accepted."""
        valid_urls = [
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "https://youtube.com/watch?v=abc123",
            "https://youtu.be/dQw4w9WgXcQ",
            "http://www.youtube.com/watch?v=test",
        ]
        for url in valid_urls:
            assert validate_youtube_url(url) is True, f"Valid URL rejected: {url}"
    
    def test_invalid_urls(self):
        """Test that invalid/malicious URLs are rejected."""
        invalid_urls = [
            "https://evil.com/video.mp4",
            "https://google.com",
            "javascript:alert('xss')",
            "",
            "not-a-url",
        ]
        for url in invalid_urls:
            assert validate_youtube_url(url) is False, f"Invalid URL accepted: {url}"


class TestJSONExtraction:
    """Test JSON extraction from AI responses."""
    
    def test_extract_json_with_markdown_fences(self):
        """Test extraction from markdown code blocks."""
        text = '''Here is the quiz:
```json
[{"question": "What is AI?", "answer": "Artificial Intelligence"}]
```
End of response.'''
        result = extract_json(text)
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["question"] == "What is AI?"
    
    def test_extract_json_raw(self):
        """Test extraction from raw JSON."""
        text = '[{"question": "Test?", "answer": "Yes"}]'
        result = extract_json(text)
        assert isinstance(result, list)
        assert len(result) == 1
    
    def test_extract_json_conversational(self):
        """Test extraction from conversational text."""
        text = '''Sure! Here's the JSON you requested: [{"q": "Q1", "a": "A1"}]. Hope it helps!'''
        result = extract_json(text)
        assert isinstance(result, list)
        assert len(result) == 1


class TestDatabase:
    """Test database functionality."""
    
    def test_database_initialization(self):
        """Test that database initializes correctly."""
        # init_db() is called on module load, just verify it works
        assert os.path.exists("lectures.db")
    
    def test_save_and_retrieve_summary(self):
        """Test saving and retrieving summaries."""
        video_id = "test_video_123"
        title = "Test Lecture"
        summary = "This is a test summary."
        quiz = [{"question": "Test?", "options": ["A", "B"], "answer": "A"}]
        
        # Save
        success = save_summary(video_id, title, summary, quiz)
        assert success is True
        
        # Retrieve
        retrieved = get_summary(video_id)
        assert retrieved is not None
        assert retrieved["video_id"] == video_id
        assert retrieved["title"] == title
        assert retrieved["summary"] == summary
        assert retrieved["quiz"] == quiz


class TestAppRoutes:
    """Test Flask application routes."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        app.config['TESTING'] = True
        with app.test_client() as client:
            yield client
    
    def test_index_route(self, client):
        """Test index page loads."""
        response = client.get('/')
        assert response.status_code == 200
    
    def test_summarise_no_url(self, client):
        """Test summarise endpoint with no URL."""
        response = client.post('/api/summarise', json={})
        assert response.status_code == 400
        assert 'error' in response.get_json()
    
    def test_summarise_invalid_url(self, client):
        """Test summarise endpoint with invalid URL."""
        response = client.post('/api/summarise', json={"url": "https://evil.com/video"})
        assert response.status_code == 400
        assert 'error' in response.get_json()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
