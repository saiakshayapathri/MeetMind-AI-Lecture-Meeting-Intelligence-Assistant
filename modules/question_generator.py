"""Generate practice questions from a transcript."""

from modules.ai_processor import generate_ai_response, get_working_text
from utils import prompts


def generate_questions(transcript, session_type, count, question_type, model):
    """Return question text (Markdown) for the chosen count and type."""
    source = get_working_text(transcript, session_type, model)
    prompt = prompts.question_prompt(source, session_type, count, question_type)
    reply = generate_ai_response(prompt, model)
    if not reply:
        raise ValueError("empty reply")
    return reply
