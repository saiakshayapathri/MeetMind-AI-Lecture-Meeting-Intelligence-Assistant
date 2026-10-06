"""Speech-to-text with OpenAI Whisper, running fully on this computer."""

_loaded_models = {}  # keeps models in memory so they load only once per run


class TranscriptionError(Exception):
    """Friendly, user-facing transcription error."""


def _load_model(model_name):
    try:
        import whisper  # imported here so the app still opens if Whisper is missing
    except ImportError as exc:
        raise TranscriptionError(
            "Whisper is not installed. Run: pip install openai-whisper (see README)."
        ) from exc

    if model_name not in _loaded_models:
        try:
            _loaded_models[model_name] = whisper.load_model(model_name)
        except Exception as exc:  # download or memory problems
            raise TranscriptionError(
                f"The Whisper '{model_name}' model could not be loaded. "
                "Check your internet connection for the first download, or try a smaller model."
            ) from exc
    return _loaded_models[model_name]


def transcribe_audio(wav_path, model_name="tiny"):
    """Transcribe a WAV file and return the transcript text."""
    model = _load_model(model_name)
    try:
        # fp16=False avoids a warning/crash on computers without a GPU.
        result = model.transcribe(wav_path, fp16=False)
    except Exception as exc:
        raise TranscriptionError(
            "The speech could not be transcribed. The audio may be corrupted."
        ) from exc

    text = (result.get("text") or "").strip()
    if not text:
        raise TranscriptionError(
            "No speech was found in this recording, so the transcript is empty."
        )
    return text
