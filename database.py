"""
Database module for LectureLens AI.
Handles SQLite integration for persistent storage of summaries.
"""
import sqlite3
import json
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DB_NAME = "lectures.db"

def get_db_connection():
    """Create a database connection."""
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initialize the database with required tables."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Create summaries table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS summaries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id TEXT UNIQUE NOT NULL,
            title TEXT NOT NULL,
            summary_text TEXT NOT NULL,
            quiz_data TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            views INTEGER DEFAULT 0
        )
    ''')
    
    # Create index for faster lookups
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_video_id ON summaries(video_id)')
    
    conn.commit()
    conn.close()
    logger.info("Database initialized successfully.")

def save_summary(video_id: str, title: str, summary: str, quiz: List[Dict]) -> bool:
    """Save a summary to the database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute('''
            INSERT OR REPLACE INTO summaries (video_id, title, summary_text, quiz_data, created_at)
            VALUES (?, ?, ?, ?, ?)
        ''', (video_id, title, summary, json.dumps(quiz), datetime.now()))
        conn.commit()
        logger.info(f"Summary saved for video: {video_id}")
        return True
    except Exception as e:
        logger.error(f"Error saving summary: {e}")
        return False
    finally:
        conn.close()

def get_summary(video_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve a summary from the database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute('SELECT * FROM summaries WHERE video_id = ?', (video_id,))
        row = cursor.fetchone()
        
        if row:
            # Increment view count
            cursor.execute('UPDATE summaries SET views = views + 1 WHERE video_id = ?', (video_id,))
            conn.commit()
            
            return {
                "video_id": row["video_id"],
                "title": row["title"],
                "summary": row["summary_text"],
                "quiz": json.loads(row["quiz_data"]),
                "created_at": row["created_at"],
                "views": row["views"] + 1 # Return updated view count
            }
        return None
    except Exception as e:
        logger.error(f"Error retrieving summary: {e}")
        return None
    finally:
        conn.close()

def get_recent_summaries(limit: int = 5) -> List[Dict[str, Any]]:
    """Get the most recent summaries."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute('''
            SELECT video_id, title, created_at, views 
            FROM summaries 
            ORDER BY created_at DESC 
            LIMIT ?
        ''', (limit,))
        
        rows = cursor.fetchall()
        return [
            {
                "video_id": row["video_id"],
                "title": row["title"],
                "created_at": row["created_at"],
                "views": row["views"]
            }
            for row in rows
        ]
    except Exception as e:
        logger.error(f"Error retrieving recent summaries: {e}")
        return []
    finally:
        conn.close()

# Initialize DB on module load
init_db()
