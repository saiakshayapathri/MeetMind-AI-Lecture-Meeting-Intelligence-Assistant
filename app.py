"""MeetMind - AI Lecture & Meeting Intelligence Assistant.

Run with:  python -m streamlit run app.py

This file only contains the user interface. The real work happens in:
  modules/media_downloader.py      yt-dlp (URL -> temporary media file)
  modules/audio_processor.py       FFmpeg (media -> audio)
  modules/whisper_transcriber.py   Whisper (audio -> text)
  modules/ai_processor.py          Ollama (text -> insights)
  modules/question_generator.py    practice questions
  modules/chatbot.py               "Ask MeetMind"
  database.py                      SQLite storage
"""

import logging

import pandas as pd
import streamlit as st

import database as db
from config import (
    CSS_PATH, DEFAULT_OLLAMA_MODEL, DEFAULT_SESSION_TYPE, DEFAULT_WHISPER_MODEL,
    ALLOWED_EXTENSIONS, PAGES, QUESTION_COUNTS, QUESTION_TYPES, SESSION_TYPES, WHISPER_MODELS,
)
from database import DatabaseError
from modules import audio_processor, chatbot, media_downloader, question_generator, whisper_transcriber
from modules import ai_processor
from modules.ai_processor import OllamaError
from modules.audio_processor import AudioProcessingError
from modules.media_downloader import MediaDownloadError
from modules.whisper_transcriber import TranscriptionError
from utils.helpers import (
    action_items_to_csv, build_report, escape_html, format_file_size, get_extension,
    is_video, safe_filename, status_badge, validate_upload,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("meetmind")

# Errors whose messages are already written for normal users.
FRIENDLY_ERRORS = (MediaDownloadError, AudioProcessingError, TranscriptionError, OllamaError, DatabaseError)
GENERIC_ERROR = "Something went wrong. Please try again."

st.set_page_config(page_title="MeetMind", page_icon="🎙️", layout="wide")


# =============================================================== setup helpers

def load_css():
    try:
        with open(CSS_PATH, encoding="utf-8") as css_file:
            st.markdown(f"<style>{css_file.read()}</style>", unsafe_allow_html=True)
    except OSError:
        pass  # the app still works without custom styling


def init_state():
    """Load saved settings into session_state once per browser session."""
    if "settings_loaded" in st.session_state:
        return
    st.session_state.whisper_model = db.get_setting("whisper_model", DEFAULT_WHISPER_MODEL)
    st.session_state.default_type = db.get_setting("default_type", DEFAULT_SESSION_TYPE)
    st.session_state.ollama_model = db.get_setting("ollama_model", DEFAULT_OLLAMA_MODEL)
    st.session_state.current_id = None
    st.session_state.chats = {}
    st.session_state.settings_loaded = True


def go_to(page_name):
    """Button callback: switch the sidebar page."""
    st.session_state.page = page_name


def open_session(session_id, page_name):
    """Button callback: select a session and jump to a page."""
    st.session_state.current_id = session_id
    st.session_state.page = page_name


def page_header(title, subtitle=""):
    st.markdown(f"## {title}")
    if subtitle:
        st.markdown(f'<p class="muted">{subtitle}</p>', unsafe_allow_html=True)


def metric_card(label, value):
    return (f'<div class="metric-card"><div class="metric-value">{value}</div>'
            f'<div class="metric-label">{label}</div></div>')


def show_error(error):
    """Show a friendly message; log technical details to the console only."""
    if isinstance(error, FRIENDLY_ERRORS):
        st.error(str(error))
    else:
        logger.exception("Unexpected error", exc_info=error)
        st.error(GENERIC_ERROR)


def session_selector(key):
    """Dropdown to choose which session a page works on. Returns the session dict or None."""
    sessions = db.list_sessions()
    if not sessions:
        st.info("No sessions yet. Process a recording first.")
        st.button("🎙️ Create your first session", on_click=go_to, args=("🎙️ New Session",), key=f"first_{key}")
        return None
    ids = [s["id"] for s in sessions]
    labels = {s["id"]: f"{s['title']}  ·  {s['session_type']}  ·  {s['created_at']}" for s in sessions}
    index = ids.index(st.session_state.current_id) if st.session_state.current_id in ids else 0
    chosen = st.selectbox("Session", ids, index=index, format_func=labels.get, key=key)
    st.session_state.current_id = chosen
    return db.get_session(chosen)


def show_markdown_block(text, empty_text="Nothing to show yet."):
    cleaned = (text or "").strip()
    st.markdown(cleaned if cleaned else f"_{empty_text}_")


# ============================================================ processing pipeline

# Checklist shown while a recording is processed: (done label, active label)
URL_STEPS = [
    ("URL validated", "Validating URL"),
    ("Media downloaded", "Downloading media"),
    ("Audio extracted", "Extracting audio"),
    ("Transcript generated", "Transcribing with Whisper"),
    ("AI analysis completed", "Generating AI insights"),
    ("Session saved", "Saving session"),
]
FILE_STEPS = [
    ("File validated", "Validating file"),
    ("File received", "Reading file"),
] + URL_STEPS[2:]


class ProgressChecklist:
    """Draws '✓ done / … current / ○ waiting' lines plus a thin progress bar."""

    def __init__(self, heading, steps):
        self.heading, self.steps = heading, steps
        self.bar = st.progress(0.0)
        self.box = st.empty()

    def show(self, current, note=""):
        lines = [f"**{self.heading}**"]
        for index, (done_text, active_text) in enumerate(self.steps):
            if index < current:
                lines.append(f"✓ {done_text}")
            elif index == current:
                lines.append(f"⏳ {active_text}…" + (f" ({note})" if note else ""))
            else:
                lines.append(f"<span class='muted'>○ {done_text}</span>")
        self.bar.progress(min(current / len(self.steps), 1.0))
        self.box.markdown("  \n".join(lines), unsafe_allow_html=True)

    def finish(self):
        self.show(len(self.steps))

    def clear(self):
        self.bar.empty()
        self.box.empty()


def run_pipeline(kind, source, title, session_type, description, whisper_model):
    """Validate -> get media -> FFmpeg -> Whisper -> Ollama -> SQLite -> delete temp files.

    kind is "url" (source is the link) or "file" (source is the uploaded file).
    Returns the new session id, or None if something went wrong.
    """
    is_url = kind == "url"
    checklist = ProgressChecklist("Processing URL" if is_url else "Processing file",
                                  URL_STEPS if is_url else FILE_STEPS)
    media_path = audio_path = None
    try:
        # Step 1: validate
        checklist.show(0)
        if is_url:
            source = media_downloader.validate_url(source)
        else:
            problem = validate_upload(source)
            if problem:
                raise AudioProcessingError(problem)
        if not audio_processor.is_ffmpeg_installed():
            raise AudioProcessingError(
                "FFmpeg is not installed. Please install FFmpeg and add it to your PATH (see README).")

        # Step 2: download (URL) or save the upload (file) to a temporary file
        checklist.show(1)
        if is_url:
            media_path, media_title = media_downloader.download_media(source)
            source_name = source
        else:
            media_path = audio_processor.save_upload_to_temp(source)
            media_title, source_name = source.name.rsplit(".", 1)[0], source.name
        title = title or media_title or "Untitled session"

        # Step 3: audio extraction, then the downloaded media is deleted immediately
        checklist.show(2)
        audio_path = audio_processor.extract_audio(media_path)
        audio_processor.remove_file(media_path)
        media_path = None

        # Step 4: Whisper (local)
        checklist.show(3, f"{whisper_model} model; the first run downloads the model")
        transcript = whisper_transcriber.transcribe_audio(audio_path, whisper_model)
        audio_processor.remove_file(audio_path)
        audio_path = None

        # Step 5: AI insights + action items
        checklist.show(4)
        insights, action_items, status = {}, [], "Completed"
        try:
            insights, action_items = ai_processor.generate_insights(
                transcript, session_type, title, st.session_state.ollama_model,
                progress=lambda msg: checklist.show(4, msg),
            )
        except OllamaError as error:
            # Keep the transcript so the long transcription work is not lost.
            status = "AI pending"
            st.warning(f"{error} Your transcript was saved. "
                       "You can generate the AI insights later from the AI Insights page.")

        # Step 6: save
        checklist.show(5)
        session_id = db.create_session(title, session_type, description, source_name,
                                       whisper_model, transcript, insights, action_items, status)
        checklist.finish()
        return session_id
    except Exception as error:  # noqa: BLE001 - always convert to a friendly message
        checklist.clear()
        show_error(error)
        return None
    finally:
        # Temporary media and audio are always deleted, even after an error.
        audio_processor.remove_file(media_path)
        audio_processor.remove_file(audio_path)


def run_ai_analysis(session):
    """(Re)generate insights for a saved session, e.g. after Ollama was not running."""
    placeholder = st.empty()
    try:
        with st.spinner("Generating AI insights. This can take a few minutes..."):
            insights, action_items = ai_processor.generate_insights(
                session["transcript"], session["session_type"], session["title"],
                st.session_state.ollama_model,
                progress=lambda msg: placeholder.info(msg),
            )
            db.save_insights(session["id"], insights, action_items, "Completed")
        placeholder.empty()
        return True
    except Exception as error:  # noqa: BLE001
        placeholder.empty()
        show_error(error)
        return False


# =============================================================== page: dashboard

def page_dashboard():
    st.markdown(
        """
        <div class="app-title">MeetMind</div>
        <div class="app-subtitle">AI Lecture &amp; Meeting Intelligence Assistant</div>
        <p class="muted app-desc">Turn lectures and meetings into searchable transcripts, summaries,
        notes, action items and practice questions.</p>
        """,
        unsafe_allow_html=True,
    )
    st.button("Start New Session", type="primary", on_click=go_to, args=("🎙️ New Session",))

    stats = db.get_stats()
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    columns = st.columns(4)
    for column, (label, key) in zip(columns, [
        ("Total Sessions", "total_sessions"), ("Lectures", "lectures"),
        ("Meetings", "meetings"), ("Action Items", "action_items"),
    ]):
        column.markdown(metric_card(label, stats[key]), unsafe_allow_html=True)

    st.markdown('<div class="section-title">Recent sessions</div>', unsafe_allow_html=True)
    recent = db.list_sessions()[:5]
    if not recent:
        st.info("No sessions yet. Click “Start New Session” to process your first recording.")
        return
    for session in recent:
        with st.container(border=True):
            c_title, c_type, c_status, c_open = st.columns([5, 2, 2, 1.5])
            c_title.markdown(
                f'<div class="row-title">{escape_html(session["title"])}</div>'
                f'<div class="muted">{session["created_at"]}</div>', unsafe_allow_html=True)
            c_type.markdown(status_badge(session["session_type"]), unsafe_allow_html=True)
            c_status.markdown(status_badge(session["status"]), unsafe_allow_html=True)
            c_open.button("Open", key=f"dash_open_{session['id']}",
                          on_click=open_session, args=(session["id"], "🧠 AI Insights"))


# ============================================================= page: new session

def page_new_session():
    page_header("New Session", "Paste a link to a lecture or meeting recording, or upload a file.")

    type_index = SESSION_TYPES.index(st.session_state.default_type) \
        if st.session_state.default_type in SESSION_TYPES else 0
    model_index = WHISPER_MODELS.index(st.session_state.whisper_model) \
        if st.session_state.whisper_model in WHISPER_MODELS else 0

    c_type, c_model = st.columns([2, 1])
    session_type = c_type.radio("Session type", SESSION_TYPES, index=type_index, horizontal=True)
    whisper_model = c_model.selectbox(
        "Whisper model", WHISPER_MODELS, index=model_index,
        help="tiny is fastest. base is a good balance. small is the most accurate but slower.")
    title = st.text_input("Session title", placeholder="DBMS Normalization Lecture",
                          help="Optional. If left empty, the video title or file name is used.")
    description = st.text_area("Description (optional)", height=80)
    st.caption("🔒 Whisper runs locally on your computer. Your recording is never sent to an "
               "external transcription service.")

    url_tab, file_tab = st.tabs(["Paste URL", "Upload File"])  # URL is the first (default) tab
    request = None  # becomes ("url", link) or ("file", uploaded_file)

    with url_tab:
        url = st.text_input("Video / Audio URL",
                            placeholder="Paste your public video or audio URL here, e.g. https://www.youtube.com/watch?v=...")
        st.caption("Paste a public video/audio link. You don't need to download the video first. "
                   "The media is downloaded temporarily and deleted after processing.")
        if st.button("Process URL", type="primary"):
            request = ("url", url)

    with file_tab:
        uploaded = st.file_uploader(
            "Upload audio or video", type=ALLOWED_EXTENSIONS,
            help="MP4, MOV, WEBM, MP3, WAV, M4A, OGG · up to about 1 GB")
        if uploaded is not None:
            c1, c2, c3 = st.columns(3)
            c1.markdown(f'**File name**<br><span class="muted">{escape_html(uploaded.name)}</span>',
                        unsafe_allow_html=True)
            c2.markdown(f'**File type**<br><span class="muted">{get_extension(uploaded.name).upper()} '
                        f'({"Video" if is_video(uploaded.name) else "Audio"})</span>',
                        unsafe_allow_html=True)
            c3.markdown(f'**File size**<br><span class="muted">{format_file_size(uploaded.size)}</span>',
                        unsafe_allow_html=True)
            if validate_upload(uploaded) is None:
                if is_video(uploaded.name):
                    st.video(uploaded)
                else:
                    st.audio(uploaded)
        if st.button("Process File", type="primary"):
            request = ("file", uploaded)

    if request:
        kind, source = request
        # Quick checks first, so simple mistakes get a simple message.
        problem = None
        if kind == "url":
            try:
                media_downloader.validate_url(source)
            except MediaDownloadError as error:
                problem = str(error)
        else:
            problem = validate_upload(source)
        if problem:
            st.error(problem)
        else:
            session_id = run_pipeline(kind, source, title.strip(), session_type,
                                      description.strip(), whisper_model)
            if session_id:
                st.session_state.current_id = session_id
                st.success("Your session is ready.")
                st.button("View results →", on_click=open_session,
                          args=(session_id, "🧠 AI Insights"), type="primary")


# ============================================================== page: transcript

def page_transcript():
    page_header("Transcript", "The full text of your recording.")
    session = session_selector("transcript_session")
    if not session:
        return
    transcript = session["transcript"]
    words = len(transcript.split())
    st.markdown(f'<p class="muted">{words:,} words · {session["file_name"]}</p>', unsafe_allow_html=True)
    # The text area is read-friendly: click inside and press Ctrl+A, Ctrl+C to copy.
    st.text_area("Transcript", transcript, height=420, label_visibility="collapsed",
                 key=f"tx_{session['id']}")
    st.download_button("⬇️ Download Transcript (.txt)", transcript,
                       file_name=f"{safe_filename(session['title'])}_transcript.txt",
                       mime="text/plain")


# ============================================================== page: AI insights

def page_insights():
    page_header("AI Insights", "Summary, key points, topics, dates and notes generated from the transcript.")
    session = session_selector("insights_session")
    if not session:
        return

    if not session["summary"].strip():
        st.warning("AI insights have not been generated for this session yet.")
        st.caption(f"Ollama model in use: {st.session_state.ollama_model} (change it in Settings).")
        if st.button("✨ Generate AI insights now", type="primary"):
            if run_ai_analysis(session):
                st.rerun()
        return

    is_meeting = session["session_type"] == "Meeting"
    tab_names = ["Summary", "Key Points", "Topics & Concepts", "Dates & Announcements"]
    tab_names.append("Decisions" if is_meeting else "Lecture Details")
    if is_meeting:
        tab_names.append("Meeting Details")
    tab_names.append("Notes")
    tabs = st.tabs(tab_names)
    tab = dict(zip(tab_names, tabs))

    with tab["Summary"]:
        st.markdown("#### Executive Summary")
        show_markdown_block(session["summary"])
        st.markdown("#### Detailed Summary")
        show_markdown_block(session["detailed_summary"])
    with tab["Key Points"]:
        show_markdown_block(session["key_points"])
    with tab["Topics & Concepts"]:
        st.markdown("#### Main Topics")
        show_markdown_block(session["topics"])
        st.markdown("#### Important Concepts")
        show_markdown_block(session["concepts"])
    with tab["Dates & Announcements"]:
        st.markdown("#### Important Dates")
        show_markdown_block(session["important_dates"])
        st.markdown("#### Important Announcements")
        show_markdown_block(session["announcements"])
    if is_meeting:
        with tab["Decisions"]:
            show_markdown_block(session["decisions"])
        with tab["Meeting Details"]:
            show_markdown_block(session["mode_extras"])
    else:
        with tab["Lecture Details"]:
            show_markdown_block(session["mode_extras"])
    with tab["Notes"]:
        show_markdown_block(session["notes"])
        st.download_button("⬇️ Download Notes (.txt)", session["notes"],
                           file_name=f"{safe_filename(session['title'])}_notes.txt", mime="text/plain")

    with st.expander("Regenerate insights"):
        st.caption("Run the AI analysis again, for example with a different Ollama model.")
        if st.button("🔄 Regenerate", key="regen"):
            if run_ai_analysis(session):
                st.rerun()


# ============================================================ page: action items

def toggle_action_item(item_id, widget_key):
    """Checkbox callback: save Completed / Pending to the database."""
    try:
        db.set_action_status(item_id, "Completed" if st.session_state[widget_key] else "Pending")
    except DatabaseError as error:
        st.session_state["action_error"] = str(error)


def page_action_items():
    page_header("Action Items", "Tasks found in your recording. Tick a task to mark it completed.")
    show_all = st.toggle("Show action items from all sessions", value=False)
    if show_all:
        session = None
        items = db.get_action_items()
    else:
        session = session_selector("action_session")
        if not session:
            return
        items = db.get_action_items(session["id"])

    if st.session_state.pop("action_error", None):
        st.error("The task could not be updated. Please try again.")

    if not items:
        if session is not None and not session["summary"].strip():
            st.warning("AI insights have not been generated for this session yet. "
                       "Open the AI Insights page to generate them.")
        else:
            st.info("No action items were found.")
        return

    status_filter = st.radio("Show", ["All", "Pending", "Completed"], horizontal=True)
    shown = [i for i in items if status_filter == "All" or i["status"] == status_filter]
    done = sum(1 for i in items if i["status"] == "Completed")
    st.markdown(f'<p class="muted">{done} of {len(items)} completed</p>', unsafe_allow_html=True)

    widths = [0.7, 4, 2, 2, 1.2, 1.4] if not show_all else [0.7, 3.5, 1.8, 1.8, 1.1, 1.3]
    header = st.columns(widths)
    for column, text in zip(header, ["Done", "Task", "Responsible", "Deadline", "Priority", "Status"]):
        column.markdown(f'<span class="muted"><b>{text}</b></span>', unsafe_allow_html=True)
    for item in shown:
        with st.container(border=True):
            cols = st.columns(widths)
            key = f"done_{item['id']}"
            cols[0].checkbox("Done", value=item["status"] == "Completed", key=key,
                             label_visibility="collapsed",
                             on_change=toggle_action_item, args=(item["id"], key))
            task_html = escape_html(item["task"])
            if show_all:
                task_html += f'<div class="muted">{escape_html(item["session_title"])}</div>'
            cols[1].markdown(task_html, unsafe_allow_html=True)
            cols[2].markdown(escape_html(item["responsible"]))
            cols[3].markdown(escape_html(item["deadline"]))
            cols[4].markdown(status_badge(item["priority"]), unsafe_allow_html=True)
            cols[5].markdown(status_badge(item["status"]), unsafe_allow_html=True)

    st.download_button("⬇️ Download action items (CSV)", action_items_to_csv(shown),
                       file_name="action_items.csv", mime="text/csv")


# ======================================================== page: question generator

def page_questions():
    page_header("Question Generator", "Create practice questions from the transcript.")
    session = session_selector("question_session")
    if not session:
        return

    c1, c2 = st.columns(2)
    count = c1.selectbox("Number of questions", QUESTION_COUNTS, index=1)
    question_type = c2.selectbox("Question type", QUESTION_TYPES)

    if st.button("✨ Generate Questions", type="primary"):
        try:
            with st.spinner("Generating questions..."):
                text = question_generator.generate_questions(
                    session["transcript"], session["session_type"], count, question_type,
                    st.session_state.ollama_model)
            saved = f"## {count} {question_type} questions\n\n{text}"
            db.update_session_fields(session["id"], questions=saved)
            session["questions"] = saved
        except Exception as error:  # noqa: BLE001
            show_error(error)

    if session["questions"].strip():
        with st.container(border=True):
            st.markdown(session["questions"])
        st.download_button("⬇️ Download Questions (.txt)", session["questions"],
                           file_name=f"{safe_filename(session['title'])}_questions.txt",
                           mime="text/plain")
    else:
        st.info("Choose the options above and click “Generate Questions”.")


# ================================================================ page: chatbot

def page_chat():
    page_header("Ask MeetMind", "Ask anything about the selected session. Answers use only its content.")
    session = session_selector("chat_session")
    if not session:
        return

    history = st.session_state.chats.setdefault(session["id"], [])

    suggestions = (
        ["What was discussed in this meeting?", "What decisions were made?",
         "Who was assigned which task?", "When is the deadline?"]
        if session["session_type"] == "Meeting" else
        ["What was discussed in this lecture?", "What are the important topics?",
         "What assignment was given?", "When is the deadline?"]
    )
    pending = None
    if not history:
        st.markdown('<p class="muted">Try one of these:</p>', unsafe_allow_html=True)
        for column, text in zip(st.columns(len(suggestions)), suggestions):
            if column.button(text, key=f"sugg_{session['id']}_{text}"):
                pending = text

    for message in history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    typed = st.chat_input("Ask a question about this session...")
    question = typed or pending
    if not question:
        return

    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                answer = chatbot.answer_question(
                    question, session, db.get_action_items(session["id"]),
                    history, st.session_state.ollama_model)
            st.markdown(answer)
            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": answer})
        except Exception as error:  # noqa: BLE001
            show_error(error)


# ============================================================== page: history

def delete_and_refresh(session_id):
    """Callback for the confirmed delete button."""
    try:
        db.delete_session(session_id)
        st.session_state.chats.pop(session_id, None)
        if st.session_state.current_id == session_id:
            st.session_state.current_id = None
        st.session_state.history_open = None
        st.session_state.confirm_delete = None
    except DatabaseError as error:
        st.session_state["history_error"] = str(error)


def set_history_state(key, value):
    st.session_state[key] = value


def page_history():
    page_header("Session History", "Search, open, download or delete your past sessions.")
    if st.session_state.pop("history_error", None):
        st.error("The session could not be deleted. Please try again.")

    search = st.text_input("Search", placeholder="Search sessions, e.g. DBMS, normalization, project meeting, assignment",
                           label_visibility="collapsed")
    sessions = db.list_sessions(search)
    st.markdown(f'<p class="muted">{len(sessions)} session(s) found</p>', unsafe_allow_html=True)

    open_id = st.session_state.get("history_open")
    selected = db.get_session(open_id) if open_id else None
    if open_id and selected is None:
        st.session_state.history_open = None

    if selected:
        render_session_detail(selected)

    if not sessions:
        st.info("No sessions match your search." if search else "No sessions yet.")
        return
    for session in sessions:
        with st.container(border=True):
            c_title, c_type, c_status, c_open = st.columns([5, 1.7, 1.7, 1.4])
            c_title.markdown(
                f'<div class="row-title">{escape_html(session["title"])}</div>'
                f'<div class="muted">{session["created_at"]} · {escape_html(session["file_name"])}</div>',
                unsafe_allow_html=True)
            c_type.markdown(status_badge(session["session_type"]), unsafe_allow_html=True)
            c_status.markdown(status_badge(session["status"]), unsafe_allow_html=True)
            c_open.button("Open", key=f"hist_open_{session['id']}",
                          on_click=set_history_state, args=("history_open", session["id"]))


def render_session_detail(session):
    items = db.get_action_items(session["id"])
    with st.container(border=True):
        head, close = st.columns([6, 1])
        head.markdown(f"### {escape_html(session['title'])}")
        close.button("Close", key="hist_close", on_click=set_history_state, args=("history_open", None))
        st.markdown(f'<p class="muted">{session["created_at"]} · {session["session_type"]} · '
                    f'{escape_html(session["file_name"])}</p>', unsafe_allow_html=True)

        t_tx, t_ins, t_act, t_q = st.tabs(["Transcript", "AI Insights", "Action Items", "Questions"])
        with t_tx:
            st.text_area("Transcript", session["transcript"], height=260,
                         label_visibility="collapsed", key=f"hist_tx_{session['id']}")
        with t_ins:
            if session["summary"].strip():
                st.markdown("**Summary**")
                show_markdown_block(session["summary"])
                st.markdown("**Key points**")
                show_markdown_block(session["key_points"])
                st.markdown("**Topics**")
                show_markdown_block(session["topics"])
                st.markdown("**Important dates**")
                show_markdown_block(session["important_dates"])
                st.markdown("**Notes**")
                show_markdown_block(session["notes"])
            else:
                st.info("AI insights are not available for this session yet.")
        with t_act:
            if items:
                st.dataframe(
                    pd.DataFrame(items)[["task", "responsible", "deadline", "priority", "status"]]
                    .rename(columns=str.title),
                    hide_index=True, width="stretch")
            else:
                st.info("No action items.")
        with t_q:
            show_markdown_block(session["questions"], "No questions generated yet.")

        b1, b2, b3, b4, b5 = st.columns(5)
        b1.download_button("⬇️ Report (TXT)", build_report(session, items),
                           file_name=f"{safe_filename(session['title'])}_report.txt",
                           mime="text/plain", key=f"rep_{session['id']}")
        b2.download_button("⬇️ Action items (CSV)", action_items_to_csv(items),
                           file_name=f"{safe_filename(session['title'])}_actions.csv",
                           mime="text/csv", key=f"csv_{session['id']}", disabled=not items)
        b3.button("💬 Ask MeetMind", key=f"to_chat_{session['id']}",
                  on_click=open_session, args=(session["id"], "💬 Ask MeetMind"))
        if st.session_state.get("confirm_delete") == session["id"]:
            b4.button("Yes, delete", type="primary", key=f"del_yes_{session['id']}",
                      on_click=delete_and_refresh, args=(session["id"],))
            b5.button("Cancel", key=f"del_no_{session['id']}",
                      on_click=set_history_state, args=("confirm_delete", None))
        else:
            b4.button("🗑️ Delete", key=f"del_{session['id']}",
                      on_click=set_history_state, args=("confirm_delete", session["id"]))


# =============================================================== page: analytics

def page_analytics():
    page_header("Analytics", "A quick overview of your sessions and tasks.")
    stats = db.get_stats()
    if stats["total_sessions"] == 0:
        st.info("Analytics will appear after you process your first session.")
        return

    cols = st.columns(3)
    labels = [("Total Sessions", "total_sessions"), ("Lectures", "lectures"), ("Meetings", "meetings"),
              ("Total Action Items", "action_items"), ("Completed Tasks", "completed"),
              ("Pending Tasks", "pending")]
    for i, (label, key) in enumerate(labels):
        cols[i % 3].markdown(metric_card(label, stats[key]), unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-title">Lecture vs Meeting</div>', unsafe_allow_html=True)
        st.bar_chart(pd.DataFrame({"Sessions": [stats["lectures"], stats["meetings"]]},
                                  index=["Lecture", "Meeting"]), color="#3F6EDB")
    with c2:
        st.markdown('<div class="section-title">Completed vs Pending Tasks</div>', unsafe_allow_html=True)
        st.bar_chart(pd.DataFrame({"Tasks": [stats["completed"], stats["pending"]]},
                                  index=["Completed", "Pending"]), color="#3F6EDB")

    st.markdown('<div class="section-title">Sessions over time</div>', unsafe_allow_html=True)
    frame = pd.DataFrame(db.list_sessions())
    per_day = frame["created_at"].str[:10].value_counts().sort_index()
    st.line_chart(pd.DataFrame({"Sessions": per_day}), color="#3F6EDB")


# =============================================================== page: settings

def clear_history_callback():
    try:
        db.clear_all_sessions()
        st.session_state.chats = {}
        st.session_state.current_id = None
        st.session_state.history_open = None
        st.session_state["settings_notice"] = "All session history was cleared."
    except DatabaseError as error:
        st.session_state["settings_notice_error"] = str(error)


def page_settings():
    page_header("Settings", "Choose your defaults. Everything runs locally on your computer.")
    if note := st.session_state.pop("settings_notice", None):
        st.success(note)
    if note := st.session_state.pop("settings_notice_error", None):
        st.error(note)

    with st.container(border=True):
        st.markdown('<div class="section-title">Processing</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        whisper_model = c1.selectbox(
            "Whisper model", WHISPER_MODELS,
            index=WHISPER_MODELS.index(st.session_state.whisper_model)
            if st.session_state.whisper_model in WHISPER_MODELS else 0)
        default_type = c2.selectbox(
            "Session default", SESSION_TYPES,
            index=SESSION_TYPES.index(st.session_state.default_type)
            if st.session_state.default_type in SESSION_TYPES else 0)

        installed = ai_processor.list_installed_models()
        ollama_model = st.text_input("Ollama model", value=st.session_state.ollama_model,
                                     help="Any model you have pulled, e.g. llama3.2, mistral, qwen2.5")
        if installed:
            st.caption("Installed models: " + ", ".join(installed))
        else:
            st.caption("No installed models found. Start Ollama and run: ollama pull llama3.2")

        if st.button("💾 Save settings", type="primary"):
            if not ollama_model.strip():
                st.error("Please enter an Ollama model name.")
            else:
                try:
                    db.save_setting("whisper_model", whisper_model)
                    db.save_setting("default_type", default_type)
                    db.save_setting("ollama_model", ollama_model.strip())
                    st.session_state.whisper_model = whisper_model
                    st.session_state.default_type = default_type
                    st.session_state.ollama_model = ollama_model.strip()
                    st.success("Settings saved.")
                except DatabaseError as error:
                    show_error(error)

    with st.container(border=True):
        st.markdown('<div class="section-title">System check</div>', unsafe_allow_html=True)
        ffmpeg_ok = audio_processor.is_ffmpeg_installed()
        ollama_ok = ai_processor.is_ollama_running()
        try:
            import whisper  # noqa: F401
            whisper_ok = True
        except ImportError:
            whisper_ok = False
        model_ok = ollama_ok and any(
            m == st.session_state.ollama_model or m.split(":")[0] == st.session_state.ollama_model.split(":")[0]
            for m in installed)
        for label, ok, hint in [
            ("FFmpeg", ffmpeg_ok, "Install FFmpeg and add it to PATH (see README)."),
            ("Whisper", whisper_ok, "Run: pip install openai-whisper"),
            ("Ollama", ollama_ok, "Start Ollama, then reload this page."),
            (f"Ollama model “{st.session_state.ollama_model}”", model_ok,
             f"Run: ollama pull {st.session_state.ollama_model}"),
        ]:
            st.markdown(f"{'✅' if ok else '⚠️'} **{label}** — "
                        f"{'ready' if ok else hint}")

    with st.container(border=True):
        st.markdown('<div class="section-title">Danger zone</div>', unsafe_allow_html=True)
        st.caption("This permanently deletes every session, transcript and action item.")
        sure = st.checkbox("I understand this cannot be undone")
        st.button("🗑️ Clear All Session History", disabled=not sure, on_click=clear_history_callback)


# ===================================================================== main

PAGE_FUNCTIONS = {
    "🏠 Dashboard": page_dashboard,
    "🎙️ New Session": page_new_session,
    "📝 Transcript": page_transcript,
    "🧠 AI Insights": page_insights,
    "✅ Action Items": page_action_items,
    "❓ Question Generator": page_questions,
    "💬 Ask MeetMind": page_chat,
    "📚 Session History": page_history,
    "📊 Analytics": page_analytics,
    "⚙️ Settings": page_settings,
}


def main():
    load_css()
    try:
        db.init_db()
    except DatabaseError as error:
        show_error(error)
        return
    init_state()

    with st.sidebar:
        st.markdown('<div class="sidebar-brand">🎙️ MeetMind</div>', unsafe_allow_html=True)
        st.radio("Navigation", PAGES, key="page", label_visibility="collapsed")
        st.markdown('<p class="muted" style="margin-top:2rem">🔒 Private by design: Whisper and '
                    'Ollama run locally. Your recordings are never sent to external AI services.</p>',
                    unsafe_allow_html=True)

    try:
        PAGE_FUNCTIONS[st.session_state.page]()
    except Exception as error:  # noqa: BLE001 - last safety net, no tracebacks for users
        show_error(error)


main()
