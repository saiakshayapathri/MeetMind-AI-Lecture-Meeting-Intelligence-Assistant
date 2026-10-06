"""Small helper functions shared by the app and the modules."""

import csv
import io
import json
import os
import re

from config import ALLOWED_EXTENSIONS, AUDIO_EXTENSIONS, MAX_UPLOAD_MB, MAX_CHARS_PER_CHUNK


# ------------------------------------------------------------------ files

def get_extension(file_name):
    return os.path.splitext(file_name)[1].lower().lstrip(".")


def is_video(file_name):
    return get_extension(file_name) not in AUDIO_EXTENSIONS


def format_file_size(num_bytes):
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024


def validate_upload(uploaded_file):
    """Return an error message (str) if the upload is not usable, otherwise None."""
    if uploaded_file is None:
        return "Please upload a valid audio or video file."
    if get_extension(uploaded_file.name) not in ALLOWED_EXTENSIONS:
        return ("This file type is not supported. Please upload a valid audio or video file "
                f"({', '.join(e.upper() for e in ALLOWED_EXTENSIONS)}).")
    if uploaded_file.size == 0:
        return "This file is empty. Please upload a valid audio or video file."
    if uploaded_file.size > MAX_UPLOAD_MB * 1024 * 1024:
        return f"This file is too large. The maximum size is {MAX_UPLOAD_MB // 1024} GB."
    return None


# ------------------------------------------------------------------- text

def split_into_chunks(text, max_chars=MAX_CHARS_PER_CHUNK):
    """Split long text into pieces at sentence boundaries."""
    text = text.strip()
    if len(text) <= max_chars:
        return [text]
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks, current = [], ""
    for sentence in sentences:
        # A single very long "sentence" (no punctuation) is cut hard.
        while len(sentence) > max_chars:
            if current:
                chunks.append(current.strip())
                current = ""
            chunks.append(sentence[:max_chars])
            sentence = sentence[max_chars:]
        if len(current) + len(sentence) + 1 > max_chars and current:
            chunks.append(current.strip())
            current = ""
        current += sentence + " "
    if current.strip():
        chunks.append(current.strip())
    return chunks


def extract_json_array(text):
    """Pull a JSON list out of an LLM reply (models sometimes add extra words or code fences)."""
    if not text:
        return []
    cleaned = re.sub(r"```(?:json)?", "", text).strip()
    start, end = cleaned.find("["), cleaned.rfind("]")
    if start == -1 or end <= start:
        return []
    try:
        data = json.loads(cleaned[start:end + 1])
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


def clean_action_items(raw_items):
    """Normalise parsed action items so the database always gets valid values."""
    cleaned, seen = [], set()
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        task = str(item.get("task", "")).strip()
        if not task or task.lower() in seen:
            continue
        seen.add(task.lower())
        priority = str(item.get("priority", "Medium")).strip().capitalize()
        if priority not in ("High", "Medium", "Low"):
            priority = "Medium"
        cleaned.append({
            "task": task,
            "responsible": str(item.get("responsible") or "Not specified").strip() or "Not specified",
            "deadline": str(item.get("deadline") or "Not specified").strip() or "Not specified",
            "priority": priority,
        })
    return cleaned


def safe_filename(text):
    name = re.sub(r"[^A-Za-z0-9_-]+", "_", text).strip("_")
    return name[:60] or "session"


# ---------------------------------------------------------------- reports

def action_items_to_csv(items):
    """Return action items as CSV text."""
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["Task", "Responsible", "Deadline", "Priority", "Status"])
    for item in items:
        writer.writerow([item["task"], item["responsible"], item["deadline"],
                         item["priority"], item["status"]])
    return buffer.getvalue()


def build_report(session, action_items):
    """Build the full plain-text report for a session."""
    def section(title, body):
        return f"{title}\n{'-' * len(title)}\n{body.strip() or 'Not available.'}\n"

    if action_items:
        lines = [
            f"- [{'x' if i['status'] == 'Completed' else ' '}] {i['task']} "
            f"(Responsible: {i['responsible']}; Deadline: {i['deadline']}; Priority: {i['priority']})"
            for i in action_items
        ]
        action_text = "\n".join(lines)
    else:
        action_text = "No action items found."

    header = (
        f"MEETMIND SESSION REPORT\n{'=' * 24}\n"
        f"Title: {session['title']}\nDate: {session['created_at']}\n"
        f"Type: {session['session_type']}\nFile: {session['file_name']}\n"
    )
    parts = [
        header,
        section("SUMMARY", session["summary"]),
        section("DETAILED SUMMARY", session["detailed_summary"]),
        section("KEY POINTS", session["key_points"]),
        section("IMPORTANT TOPICS", session["topics"]),
        section("ACTION ITEMS", action_text),
        section("IMPORTANT DATES", session["important_dates"]),
        section("QUESTIONS", session["questions"] or "No questions generated yet."),
        section("NOTES", session["notes"]),
    ]
    return "\n".join(parts)


# ------------------------------------------------------------------- html

def status_badge(text):
    """Small coloured pill used in tables and lists."""
    css_class = {
        "completed": "badge-success", "pending": "badge-warning", "failed": "badge-warning",
        "high": "badge-warning", "medium": "badge-blue", "low": "badge-muted",
    }.get(str(text).lower(), "badge-blue")
    safe = str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return f'<span class="badge {css_class}">{safe}</span>'


def escape_html(text):
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
