# MeetMind – AI Lecture & Meeting Intelligence Assistant

## Project description

MeetMind helps students and professionals understand lectures and meetings. Paste a public video/audio URL (or upload a file) and MeetMind turns it into a transcript, summary, key points, topics, action items,
important dates, study notes / meeting minutes and practice questions. You can then chat with the
recording and search all your past sessions.

Everything runs **locally**: speech-to-text uses Whisper, the AI uses Ollama, and sessions are stored
in SQLite. Your recordings are never uploaded to an external service and no API keys are needed.

## Features

- **URL-first:** paste a public YouTube (or other yt-dlp supported) link, no manual download needed
- Optional file upload: WAV, MP3, M4A, OGG, MP4, MOV, WEBM (up to ~1 GB)
- yt-dlp downloads the media temporarily, FFmpeg extracts audio, then the media is deleted
- Whisper (tiny / base / small, default base) creates the transcript locally
- Local AI insights: executive and detailed summary, key points, topics, concepts, dates,
  announcements, decisions (meetings)
- Lecture mode: definitions, examples, exam-relevant points, study notes, possible exam questions
- Meeting mode: discussion points, decisions, responsible persons, follow-ups, meeting minutes
- Action items with responsible person, deadline, priority and a completed checkbox
- Question generator: 5/10/15/20 questions; multiple choice, short answer, true/false
- "Ask MeetMind" chatbot that answers only from the selected session
- Searchable session history, TXT report and CSV action-item downloads, delete sessions
- Analytics dashboard and settings page
- Clean light UI (white, light gray, blue)

## Technology stack

Python 3.11 · Streamlit · yt-dlp · OpenAI Whisper (+ PyTorch) · FFmpeg · Ollama · SQLite · Pandas · Requests

## Folder structure

```
meetmind/
├── app.py                      # Streamlit user interface (all pages)
├── database.py                 # SQLite tables and queries
├── config.py                   # Paths, limits, defaults
├── requirements.txt
├── README.md
├── modules/
│   ├── media_downloader.py     # yt-dlp: validate URL, temporary download
│   ├── audio_processor.py      # FFmpeg: save upload, extract audio, clean up
│   ├── whisper_transcriber.py  # Whisper speech-to-text
│   ├── ai_processor.py         # Ollama client + insight generation
│   ├── question_generator.py   # Practice questions
│   └── chatbot.py              # Ask MeetMind
├── utils/
│   ├── prompts.py              # Every LLM prompt
│   └── helpers.py              # Validation, parsing, report building
├── database/
│   └── meetmind.db             # Created automatically on first run
├── assets/
│   └── style.css               # Light theme
└── .streamlit/
    └── config.toml             # 1 GB upload limit + light theme
```

## Installation (Windows)

1. Install **Python 3.11** from python.org (tick "Add Python to PATH").
2. Open a terminal in the `meetmind` folder and create a virtual environment:
   ```
   python -m venv venv
   venv\Scripts\activate
   ```
3. Install the packages:
   ```
   pip install -r requirements.txt
   ```
   (PyTorch is large. If installation is slow, see https://pytorch.org for the CPU build.)

## FFmpeg setup

Whisper and MeetMind both need FFmpeg.

1. Easiest on Windows: open a terminal and run `winget install Gyan.FFmpeg`
   (or download a build from https://www.gyan.dev/ffmpeg/builds/ and unzip it, e.g. to `C:\ffmpeg`).
2. If you unzipped manually, add `C:\ffmpeg\bin` to your **PATH** (Start → "Edit environment variables").
3. Open a **new** terminal and check: `ffmpeg -version`

## Whisper setup

Whisper is installed by `pip install -r requirements.txt` (package `openai-whisper`).
yt-dlp is also installed by `requirements.txt`. Keep it up to date (`pip install -U yt-dlp`) because
video sites change often.

The model you choose (tiny, base or small) is downloaded automatically the first time it is used,
so the first transcription needs an internet connection. After that it works offline.

## Ollama setup

1. Install Ollama from https://ollama.com/download
2. Download a model (this only needs to be done once):
   ```
   ollama pull llama3.2
   ```
3. Make sure Ollama is running (it starts automatically after installation; the tray icon should be visible).
4. To use another model, type its name in **Settings → Ollama model**.

## How to run

```
python -m streamlit run app.py
```

Open the address shown in the terminal (usually http://localhost:8501). The database is created
automatically. Use **Settings → System check** to confirm FFmpeg, Whisper and Ollama are ready.

## Troubleshooting

| Problem | Fix |
|---|---|
| "Unable to process this URL" | Check that the link is public and opens in a browser without logging in. Update yt-dlp: `pip install -U yt-dlp`. Private, deleted, live or very long (4 h+) videos are rejected. |
| YouTube asks to "sign in" / bot check | Some videos or networks are blocked by the site. Download the file yourself and use the **Upload File** tab. |
| "FFmpeg is not installed" | Install FFmpeg, add it to PATH, open a **new** terminal, restart the app. |
| "Ollama is not running" | Start Ollama and try again. Your transcript is saved; generate insights later from the AI Insights page. |
| "The AI model … is not installed" | Run `ollama pull <model-name>` or change the model in Settings. |
| Whisper model fails to load | Check your internet connection for the first download, or pick a smaller model. |
| "No speech was found" | The file may be silent or have no audio track. Try another recording. |
| Processing is slow | Use the `tiny` Whisper model and a smaller Ollama model (e.g. `llama3.2`). Long recordings take longer. |
| Poor transcript accuracy | Use `base` or `small`; clearer audio helps. |
| Download or upload over 1 GB | Compress the recording or extract the audio first. |

## Future enhancements

- Speaker identification (who said what) and timestamps in the transcript
- Export reports as PDF or Word
- Multi-language transcription and translation
- Flashcards and spaced-repetition review
- Calendar export for deadlines
- Live recording from the microphone
- Playlist and batch URL processing
