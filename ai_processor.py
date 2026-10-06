"""Local AI processing through Ollama.

`generate_ai_response(prompt)` is the only function that talks to Ollama.
Everything else builds prompts (see utils/prompts.py) and organises results.
"""

import requests

from config import DEFAULT_OLLAMA_MODEL, MAX_CHARS_PER_CHUNK, OLLAMA_TIMEOUT_SECONDS, OLLAMA_URL
from utils import prompts
from utils.helpers import clean_action_items, extract_json_array, split_into_chunks


class OllamaError(Exception):
    """Friendly, user-facing Ollama error."""


# ------------------------------------------------------------ Ollama access

def generate_ai_response(prompt, model=None):
    """Send a prompt to the local Ollama server and return the reply text."""
    model = model or DEFAULT_OLLAMA_MODEL
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False,
                  "options": {"temperature": 0.2, "num_ctx": 4096}},
            timeout=OLLAMA_TIMEOUT_SECONDS,
        )
    except requests.exceptions.ConnectionError as exc:
        raise OllamaError("Ollama is not running. Please start Ollama and try again.") from exc
    except requests.exceptions.Timeout as exc:
        raise OllamaError("The AI model took too long to respond. Try a smaller model.") from exc
    except requests.exceptions.RequestException as exc:
        raise OllamaError("Could not talk to Ollama. Please start Ollama and try again.") from exc

    if response.status_code == 404:
        raise OllamaError(
            f"The AI model '{model}' is not installed. Open a terminal and run: ollama pull {model}"
        )
    if response.status_code != 200:
        raise OllamaError("Ollama returned an error. Please try again.")

    try:
        return (response.json().get("response") or "").strip()
    except ValueError as exc:
        raise OllamaError("Ollama returned an unreadable answer. Please try again.") from exc


def list_installed_models():
    """Names of models installed in Ollama (empty list if Ollama is not reachable)."""
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        response.raise_for_status()
        return [m["name"] for m in response.json().get("models", [])]
    except (requests.exceptions.RequestException, ValueError, KeyError):
        return []


def is_ollama_running():
    try:
        return requests.get(f"{OLLAMA_URL}/api/tags", timeout=3).status_code == 200
    except requests.exceptions.RequestException:
        return False


# ------------------------------------------------------------ long transcripts

def get_working_text(transcript, session_type, model, progress=None):
    """Return text that fits in the model's context.

    Short transcripts are used as they are. Long ones are condensed chunk by
    chunk into detailed digests, so nothing from the start of a long lecture is lost.
    """
    if len(transcript) <= MAX_CHARS_PER_CHUNK:
        return transcript
    chunks = split_into_chunks(transcript)
    digests = []
    for number, chunk in enumerate(chunks, start=1):
        if progress:
            progress(f"Reading part {number} of {len(chunks)}")
        prompt = prompts.chunk_digest_prompt(chunk, session_type, number, len(chunks))
        digests.append(generate_ai_response(prompt, model))
    return "\n\n".join(digests)


def extract_action_items(transcript, session_type, model):
    """Find action items. Long transcripts are scanned chunk by chunk to keep exact details."""
    found = []
    for chunk in split_into_chunks(transcript):
        reply = generate_ai_response(prompts.action_items_prompt(chunk, session_type), model)
        found.extend(extract_json_array(reply))
    return clean_action_items(found)


# ---------------------------------------------------------------- full analysis

def generate_insights(transcript, session_type, title, model, progress=None):
    """Generate fast AI insights using a single Ollama request."""

    if progress:
        progress("Generating AI insights...")

    prompt = f"""
You are MeetMind, an AI lecture and meeting assistant.

Analyze the following {session_type.lower()} transcript.

Session title: {title}

Give a concise and useful response in exactly this format:

SUMMARY:
Write 3-5 simple sentences.

KEY POINTS:
- Point 1
- Point 2
- Point 3
- Point 4
- Point 5

TOPICS:
- Topic 1
- Topic 2
- Topic 3

IMPORTANT DATES:
- Mention important dates or deadlines.
- If none are mentioned, write "None mentioned."

CONCEPTS:
- Explain the most important concepts briefly.

ACTION ITEMS:
- List tasks or things the student/user needs to do.
- If none are mentioned, write "None."

STUDY NOTES:
Write short useful notes for revision.

Transcript:
{transcript}
"""

    reply = generate_ai_response(prompt, model)

    insights = {
        "summary": reply,
        "detailed_summary": reply,
        "key_points": "",
        "topics": "",
        "concepts": "",
        "announcements": "",
        "important_dates": "",
        "decisions": "",
        "mode_extras": "",
        "notes": reply,
    }

    action_items = []

    if progress:
        progress("AI insights completed.")

    return insights, action_items