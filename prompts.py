"""All LLM prompts used by MeetMind, in one place.

Every prompt starts with the same rule so the model stays grounded in the
transcript and does not invent facts.
"""

GROUNDING_RULE = (
    "Use only the provided transcript. Do not invent information. "
    "If something is not mentioned in the transcript, leave it out. "
    "Write in clear, simple English using Markdown."
)

NOT_FOUND_MESSAGE = "I couldn't find this information in the selected session."


def _wrap(task, transcript, session_type):
    """Common layout: rule, task, then the transcript."""
    return (
        f"You are MeetMind, an assistant that analyses a {session_type.lower()} recording.\n"
        f"{GROUNDING_RULE}\n\n"
        f"TASK:\n{task}\n\n"
        f"TRANSCRIPT:\n\"\"\"\n{transcript}\n\"\"\"\n"
    )


# ------------------------------------------------------------ long transcripts

def chunk_digest_prompt(chunk, session_type, part, total):
    return _wrap(
        f"This is part {part} of {total} of a long transcript. Write a detailed digest of this part "
        "as bullet points. Keep every concept, definition, example, decision, name, date, deadline "
        "and task that appears. Do not add anything that is not in the text.",
        chunk, session_type,
    )


# ------------------------------------------------------------------ summaries

def executive_summary_prompt(transcript, session_type):
    return _wrap(
        "Write a short, easy-to-understand executive summary in 3 to 5 sentences.",
        transcript, session_type,
    )


def detailed_summary_prompt(transcript, session_type):
    if session_type == "Meeting":
        task = ("Write a structured meeting summary with these headings: **Overview**, "
                "**Discussion Points**, **Outcomes**. Use short paragraphs and bullets.")
    else:
        task = ("Write a structured lecture summary with these headings: **Overview**, "
                "**Main Explanation**, **Conclusion**. Use short paragraphs and bullets.")
    return _wrap(task, transcript, session_type)


def key_points_prompt(transcript, session_type):
    return _wrap(
        "List the most important points as a bullet list (maximum 12 bullets). One line each.",
        transcript, session_type,
    )


def topics_prompt(transcript, session_type):
    return _wrap(
        "List the major topics that were discussed as a bullet list. "
        "Each bullet: topic name, then a dash and a one-line description.",
        transcript, session_type,
    )


def concepts_prompt(transcript, session_type):
    return _wrap(
        "Identify the important concepts mentioned and explain each in one or two sentences, "
        "using only what the transcript says. Format: **Concept** - explanation. "
        "If no concepts are explained, write: None mentioned.",
        transcript, session_type,
    )


# ---------------------------------------------------------------------- dates

def important_dates_prompt(transcript, session_type):
    return _wrap(
        "Extract every date, deadline, exam, assignment, submission and event mentioned. "
        "Format each as a bullet: **Date or time** - what it is for. "
        "If there are none, write exactly: None mentioned.",
        transcript, session_type,
    )


def announcements_prompt(transcript, session_type):
    return _wrap(
        "List the important announcements (notices, changes, reminders) as bullets. "
        "If there are none, write exactly: None mentioned.",
        transcript, session_type,
    )


def decisions_prompt(transcript, session_type):
    return _wrap(
        "List the decisions that were made as bullets. Only include things clearly agreed or decided. "
        "If there are none, write exactly: None mentioned.",
        transcript, session_type,
    )


# --------------------------------------------------------------- action items

def action_items_prompt(transcript, session_type):
    return _wrap(
        "Find every task, assignment or follow-up that someone must do. "
        "Return ONLY a JSON array (no extra text). Each element must be an object with keys: "
        "\"task\" (string), \"responsible\" (person name, or \"Not specified\"), "
        "\"deadline\" (as said in the transcript, or \"Not specified\"), "
        "\"priority\" (one of \"High\", \"Medium\", \"Low\"; use High when urgent or due soon). "
        "If there are no tasks, return [].",
        transcript, session_type,
    )


# ---------------------------------------------------------- notes / mode extras

def lecture_extras_prompt(transcript):
    return _wrap(
        "Produce these sections in Markdown, each with its own heading:\n"
        "## Key Concepts\n## Important Definitions\n## Examples\n## Exam-Relevant Points\n"
        "## Possible Exam Questions (5 questions, no answers)\n"
        "If a section has nothing in the transcript, write: None mentioned.",
        transcript, "Lecture",
    )


def meeting_extras_prompt(transcript):
    return _wrap(
        "Produce these sections in Markdown, each with its own heading:\n"
        "## Discussion Points\n## Responsible Persons (who owns what)\n## Follow-up Tasks\n"
        "If a section has nothing in the transcript, write: None mentioned.",
        transcript, "Meeting",
    )


def lecture_notes_prompt(transcript, title):
    return _wrap(
        f"Write structured study notes titled \"{title}\" using exactly these numbered sections:\n"
        "1. Introduction\n2. Main Topics\n3. Important Concepts\n4. Examples\n5. Key Points\n"
        "6. Important Definitions\n7. Conclusion\n"
        "Use Markdown headings and bullets. Write 'None mentioned.' for empty sections.",
        transcript, "Lecture",
    )


def meeting_minutes_prompt(transcript, title):
    return _wrap(
        f"Write meeting minutes titled \"{title}\" using exactly these numbered sections:\n"
        "1. Meeting Purpose\n2. Discussion\n3. Decisions\n4. Action Items\n5. Deadlines\n6. Follow-up\n"
        "Use Markdown headings and bullets. Write 'None mentioned.' for empty sections.",
        transcript, "Meeting",
    )


# ------------------------------------------------------------------ questions

_QUESTION_TYPE_RULES = {
    "Multiple Choice": (
        "Multiple-choice questions. Use exactly this format for each:\n"
        "**Q1. question text**\nA) ...\nB) ...\nC) ...\nD) ...\n"
        "**Correct Answer:** letter\n**Explanation:** one or two sentences"
    ),
    "Short Answer": "Short-answer questions. After each question write **Answer:** with 1-2 sentences.",
    "True/False": (
        "True/False statements. Use exactly this format for each:\n"
        "**Q1. statement**\n**Answer:** True or False\n**Explanation:** one sentence"
    ),
}


def question_prompt(transcript, session_type, count, question_type):
    rules = _QUESTION_TYPE_RULES[question_type]
    return _wrap(
        f"Generate exactly {count} questions from the transcript.\n{rules}\n"
        "Number the questions. Every question must be answerable from the transcript.",
        transcript, session_type,
    )


# -------------------------------------------------------------------- chatbot

def chatbot_prompt(question, context, session_type, history_text=""):
    history_block = f"\nPREVIOUS CHAT:\n{history_text}\n" if history_text else ""
    return (
        f"You are MeetMind, answering questions about one {session_type.lower()} recording.\n"
        f"{GROUNDING_RULE}\n"
        "Answer ONLY from the SESSION INFORMATION below. "
        f"If the answer is not there, reply exactly: {NOT_FOUND_MESSAGE}\n"
        "Be concise and direct.\n\n"
        f"SESSION INFORMATION:\n\"\"\"\n{context}\n\"\"\"\n"
        f"{history_block}\n"
        f"QUESTION: {question}\n\nANSWER:"
    )
