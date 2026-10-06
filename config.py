"""Central configuration for MeetMind.

Everything that might change (paths, model names, limits) lives here so the
rest of the code never hard-codes it. No API keys are needed: Whisper and
Ollama both run locally.
"""

import os

# ---------- Paths ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "database", "meetmind.db")
CSS_PATH = os.path.join(BASE_DIR, "assets", "style.css")
TEMP_DIR = os.path.join(BASE_DIR, "temp")  # temporary uploads / extracted audio

# ---------- Upload rules ----------
AUDIO_EXTENSIONS = ["wav", "mp3", "m4a", "ogg"]
VIDEO_EXTENSIONS = ["mp4", "mov", "webm"]
ALLOWED_EXTENSIONS = AUDIO_EXTENSIONS + VIDEO_EXTENSIONS
MAX_UPLOAD_MB = 1024  # must match .streamlit/config.toml
MAX_DOWNLOAD_MB = 1024  # largest media downloaded from a URL
MAX_MEDIA_HOURS = 4  # longest recording accepted from a URL

# ---------- Whisper ----------
WHISPER_MODELS = ["tiny", "base", "small"]
DEFAULT_WHISPER_MODEL = "base"

# ---------- Ollama (local LLM) ----------
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
DEFAULT_OLLAMA_MODEL = "llama3.2:1b"
OLLAMA_TIMEOUT_SECONDS = 600
# Transcripts longer than this are split into chunks so small local models
# do not lose the beginning of a long lecture.
MAX_CHARS_PER_CHUNK = 6000

# ---------- Session defaults ----------
SESSION_TYPES = ["Lecture", "Meeting"]
DEFAULT_SESSION_TYPE = "Lecture"

# ---------- Question generator ----------
QUESTION_COUNTS = [5, 10, 15, 20]
QUESTION_TYPES = ["Multiple Choice", "Short Answer", "True/False"]

# ---------- Pages ----------
PAGES = [
    "🏠 Dashboard",
    "🎙️ New Session",
    "📝 Transcript",
    "🧠 AI Insights",
    "✅ Action Items",
    "❓ Question Generator",
    "💬 Ask MeetMind",
    "📚 Session History",
    "📊 Analytics",
    "⚙️ Settings",
]
