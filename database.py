"""SQLite storage for MeetMind.

The database file is created automatically the first time the app starts.
Every query uses `?` placeholders (parameterized SQL), never string formatting.
"""

import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime

from config import DB_PATH


class DatabaseError(Exception):
    """Raised when something goes wrong with the database (shown as a friendly message)."""


@contextmanager
def get_connection():
    """Open a connection, commit on success, roll back on error, always close."""
    conn = None
    try:
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        yield conn
        conn.commit()
    except sqlite3.Error as exc:
        if conn:
            conn.rollback()
        raise DatabaseError("The session database could not be accessed.") from exc
    finally:
        if conn:
            conn.close()


def init_db():
    """Create the tables if they do not exist yet."""
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                title            TEXT NOT NULL,
                session_type     TEXT NOT NULL,
                description      TEXT DEFAULT '',
                created_at       TEXT NOT NULL,
                file_name        TEXT DEFAULT '',
                whisper_model    TEXT DEFAULT '',
                status           TEXT DEFAULT 'Completed',
                transcript       TEXT DEFAULT '',
                summary          TEXT DEFAULT '',
                detailed_summary TEXT DEFAULT '',
                key_points       TEXT DEFAULT '',
                topics           TEXT DEFAULT '',
                concepts         TEXT DEFAULT '',
                important_dates  TEXT DEFAULT '',
                announcements    TEXT DEFAULT '',
                decisions        TEXT DEFAULT '',
                mode_extras      TEXT DEFAULT '',
                questions        TEXT DEFAULT '',
                notes            TEXT DEFAULT ''
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS action_items (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id   INTEGER NOT NULL,
                task         TEXT NOT NULL,
                responsible  TEXT DEFAULT 'Not specified',
                deadline     TEXT DEFAULT 'Not specified',
                priority     TEXT DEFAULT 'Medium',
                status       TEXT DEFAULT 'Pending',
                FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
            )
            """
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )


# ---------------------------------------------------------------- sessions

# Columns that may be updated through update_session_fields (guards the SQL).
_UPDATABLE_COLUMNS = {
    "title", "description", "status", "transcript", "summary", "detailed_summary",
    "key_points", "topics", "concepts", "important_dates", "announcements",
    "decisions", "mode_extras", "questions", "notes",
}


def create_session(title, session_type, description, file_name, whisper_model,
                   transcript, insights, action_items, status="Completed"):
    """Save a fully processed session plus its action items. Returns the new session id."""
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    with get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO sessions (
                title, session_type, description, created_at, file_name, whisper_model,
                status, transcript, summary, detailed_summary, key_points, topics,
                concepts, important_dates, announcements, decisions, mode_extras, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                title, session_type, description, created_at, file_name, whisper_model,
                status, transcript,
                insights.get("summary", ""),
                insights.get("detailed_summary", ""),
                insights.get("key_points", ""),
                insights.get("topics", ""),
                insights.get("concepts", ""),
                insights.get("important_dates", ""),
                insights.get("announcements", ""),
                insights.get("decisions", ""),
                insights.get("mode_extras", ""),
                insights.get("notes", ""),
            ),
        )
        session_id = cursor.lastrowid
        for item in action_items:
            conn.execute(
                """
                INSERT INTO action_items (session_id, task, responsible, deadline, priority, status)
                VALUES (?, ?, ?, ?, ?, 'Pending')
                """,
                (session_id, item["task"], item["responsible"], item["deadline"], item["priority"]),
            )
    return session_id


def get_session(session_id):
    """Return one session as a dict, or None."""
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
    return dict(row) if row else None


def list_sessions(search_text=""):
    """List sessions (newest first). Optional search across title, transcript and AI text."""
    with get_connection() as conn:
        if search_text.strip():
            pattern = f"%{search_text.strip()}%"
            rows = conn.execute(
                """
                SELECT * FROM sessions
                WHERE title LIKE ? OR description LIKE ? OR transcript LIKE ?
                   OR summary LIKE ? OR detailed_summary LIKE ? OR key_points LIKE ?
                   OR topics LIKE ? OR concepts LIKE ? OR notes LIKE ? OR questions LIKE ?
                ORDER BY id DESC
                """,
                (pattern,) * 10,
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM sessions ORDER BY id DESC").fetchall()
    return [dict(row) for row in rows]


def update_session_fields(session_id, **fields):
    """Update selected text columns of a session (e.g. questions=...)."""
    bad = set(fields) - _UPDATABLE_COLUMNS
    if bad:
        raise DatabaseError("Invalid field update.")
    if not fields:
        return
    assignments = ", ".join(f"{column} = ?" for column in fields)  # column names are whitelisted
    with get_connection() as conn:
        conn.execute(
            f"UPDATE sessions SET {assignments} WHERE id = ?",
            (*fields.values(), session_id),
        )


def save_insights(session_id, insights, action_items, status="Completed"):
    """Store AI results for an existing session (used when AI analysis is run or retried later)."""
    with get_connection() as conn:
        conn.execute(
            """
            UPDATE sessions SET status = ?, summary = ?, detailed_summary = ?, key_points = ?,
                topics = ?, concepts = ?, important_dates = ?, announcements = ?,
                decisions = ?, mode_extras = ?, notes = ?
            WHERE id = ?
            """,
            (
                status,
                insights.get("summary", ""), insights.get("detailed_summary", ""),
                insights.get("key_points", ""), insights.get("topics", ""),
                insights.get("concepts", ""), insights.get("important_dates", ""),
                insights.get("announcements", ""), insights.get("decisions", ""),
                insights.get("mode_extras", ""), insights.get("notes", ""),
                session_id,
            ),
        )
        conn.execute("DELETE FROM action_items WHERE session_id = ?", (session_id,))
        for item in action_items:
            conn.execute(
                """
                INSERT INTO action_items (session_id, task, responsible, deadline, priority, status)
                VALUES (?, ?, ?, ?, ?, 'Pending')
                """,
                (session_id, item["task"], item["responsible"], item["deadline"], item["priority"]),
            )


def delete_session(session_id):
    """Delete a session and (via CASCADE) its action items."""
    with get_connection() as conn:
        conn.execute("DELETE FROM action_items WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))


def clear_all_sessions():
    """Remove every session and action item."""
    with get_connection() as conn:
        conn.execute("DELETE FROM action_items")
        conn.execute("DELETE FROM sessions")


# ------------------------------------------------------------ action items

def get_action_items(session_id=None):
    """Action items for one session, or for all sessions when session_id is None."""
    with get_connection() as conn:
        if session_id is None:
            rows = conn.execute(
                """
                SELECT a.*, s.title AS session_title FROM action_items a
                JOIN sessions s ON s.id = a.session_id ORDER BY a.id DESC
                """
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT a.*, s.title AS session_title FROM action_items a
                JOIN sessions s ON s.id = a.session_id
                WHERE a.session_id = ? ORDER BY a.id
                """,
                (session_id,),
            ).fetchall()
    return [dict(row) for row in rows]


def set_action_status(item_id, status):
    """Mark an action item 'Pending' or 'Completed'."""
    if status not in ("Pending", "Completed"):
        raise DatabaseError("Invalid status.")
    with get_connection() as conn:
        conn.execute("UPDATE action_items SET status = ? WHERE id = ?", (status, item_id))


# ------------------------------------------------------------------ stats

def get_stats():
    """Numbers for the dashboard and analytics pages."""
    with get_connection() as conn:
        total = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        lectures = conn.execute(
            "SELECT COUNT(*) FROM sessions WHERE session_type = 'Lecture'"
        ).fetchone()[0]
        meetings = conn.execute(
            "SELECT COUNT(*) FROM sessions WHERE session_type = 'Meeting'"
        ).fetchone()[0]
        items = conn.execute("SELECT COUNT(*) FROM action_items").fetchone()[0]
        done = conn.execute(
            "SELECT COUNT(*) FROM action_items WHERE status = 'Completed'"
        ).fetchone()[0]
    return {
        "total_sessions": total,
        "lectures": lectures,
        "meetings": meetings,
        "action_items": items,
        "completed": done,
        "pending": items - done,
    }


# --------------------------------------------------------------- settings

def get_setting(key, default):
    """Read a saved setting, falling back to a default."""
    try:
        with get_connection() as conn:
            row = conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else default
    except DatabaseError:
        return default


def save_setting(key, value):
    """Save (insert or replace) a setting."""
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO settings (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, str(value)),
        )
