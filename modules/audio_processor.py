"""Media handling with FFmpeg.

Every upload (audio or video) is converted to a 16 kHz mono WAV file. This
extracts the audio track from videos and gives Whisper a clean, uniform input.
Temporary files are always deleted after processing.
"""

import os
import shutil
import subprocess
import uuid

from config import TEMP_DIR
from utils.helpers import get_extension


class AudioProcessingError(Exception):
    """Friendly, user-facing media error."""


def is_ffmpeg_installed():
    return shutil.which("ffmpeg") is not None


def save_upload_to_temp(uploaded_file):
    """Write the Streamlit upload to a temporary file and return its path."""
    os.makedirs(TEMP_DIR, exist_ok=True)
    extension = get_extension(uploaded_file.name)
    path = os.path.join(TEMP_DIR, f"upload_{uuid.uuid4().hex}.{extension}")
    try:
        with open(path, "wb") as target:
            # Copy in blocks so a 1 GB file is never duplicated in memory.
            uploaded_file.seek(0)
            shutil.copyfileobj(uploaded_file, target, length=8 * 1024 * 1024)
    except OSError as exc:
        remove_file(path)
        raise AudioProcessingError(
            "The file could not be saved for processing. Please check your free disk space."
        ) from exc
    return path


def extract_audio(input_path):
    """Convert any supported audio/video file to a mono 16 kHz WAV. Returns the WAV path."""
    if not is_ffmpeg_installed():
        raise AudioProcessingError(
            "FFmpeg is not installed. Please install FFmpeg and add it to your PATH (see README)."
        )

    output_path = os.path.join(TEMP_DIR, f"audio_{uuid.uuid4().hex}.wav")
    command = [
        "ffmpeg", "-y", "-i", input_path,
        "-vn",                    # ignore any video stream
        "-ac", "1",               # mono
        "-ar", "16000",           # 16 kHz, what Whisper expects
        "-c:a", "pcm_s16le",
        output_path,
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=3600)
    except subprocess.TimeoutExpired as exc:
        remove_file(output_path)
        raise AudioProcessingError("FFmpeg took too long to process this file.") from exc
    except OSError as exc:
        raise AudioProcessingError("FFmpeg could not be started. Please reinstall FFmpeg.") from exc

    if result.returncode != 0 or not os.path.exists(output_path) or os.path.getsize(output_path) < 1000:
        remove_file(output_path)
        raise AudioProcessingError(
            "FFmpeg could not process this file. It may be corrupted or have no audio track."
        )
    return output_path


def remove_file(path):
    """Delete a temporary file, ignoring errors."""
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except OSError:
        pass
