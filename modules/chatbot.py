"""'Ask MeetMind': answer questions using only the selected session."""

import re

from config import MAX_CHARS_PER_CHUNK
from modules.ai_processor import generate_ai_response
from utils import prompts
from utils.helpers import split_into_chunks

_STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "what", "who", "when", "where", "why", "how",
    "did", "does", "do", "of", "in", "on", "to", "for", "and", "or", "this", "that", "from",
    "about", "me", "give", "tell", "explain", "any", "there", "with", "it", "be", "by",
}


def _keywords(text):
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOP_WORDS and len(w) > 2}


def _relevant_transcript(question, transcript):
    """Whole transcript if short, otherwise the chunks that best match the question."""
    chunks = split_into_chunks(transcript, 1500)
    if len(transcript) <= MAX_CHARS_PER_CHUNK:
        return transcript
    words = _keywords(question)
    ranked = sorted(chunks, key=lambda c: len(words & _keywords(c)), reverse=True)
    picked, size = [], 0
    for chunk in ranked:
        if size + len(chunk) > MAX_CHARS_PER_CHUNK:
            break
        picked.append(chunk)
        size += len(chunk)
    # Keep the original order so the context reads naturally.
    return "\n...\n".join(c for c in chunks if c in picked)


def build_context(question, session, action_items):
    """Collect the saved information the chatbot is allowed to use."""
    task_lines = "\n".join(
        f"- {i['task']} | Responsible: {i['responsible']} | Deadline: {i['deadline']} "
        f"| Priority: {i['priority']} | Status: {i['status']}"
        for i in action_items
    ) or "None"
    return (
        f"Title: {session['title']} ({session['session_type']}, {session['created_at']})\n\n"
        f"Summary:\n{session['summary']}\n\n"
        f"Key points:\n{session['key_points']}\n\n"
        f"Important dates:\n{session['important_dates']}\n\n"
        f"Decisions:\n{session['decisions']}\n\n"
        f"Action items:\n{task_lines}\n\n"
        f"Transcript:\n{_relevant_transcript(question, session['transcript'])}"
    )


def answer_question(question, session, action_items, history, model):
    """Answer one question. `history` is a list of {"role", "content"} dicts."""
    recent = history[-6:]
    history_text = "\n".join(
        f"{'User' if m['role'] == 'user' else 'MeetMind'}: {m['content']}" for m in recent
    )
    context = build_context(question, session, action_items)
    prompt = prompts.chatbot_prompt(question, context, session["session_type"], history_text)
    reply = generate_ai_response(prompt, model)
    return reply or prompts.NOT_FOUND_MESSAGE
