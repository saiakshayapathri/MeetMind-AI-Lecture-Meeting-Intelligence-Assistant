"""Download media from a public URL with yt-dlp.

The file is saved to a temporary folder only. The caller (app.py) deletes it
right after the audio has been extracted, so downloaded media is never kept.
"""

import glob
import os
import uuid
from urllib.parse import urlparse

from config import MAX_DOWNLOAD_MB, MAX_MEDIA_HOURS, TEMP_DIR
from modules.audio_processor import remove_file


class _SilentLogger:
    """Keeps yt-dlp quiet so no technical text reaches the console or the user."""
    def debug(self, msg): pass
    def info(self, msg): pass
    def warning(self, msg): pass
    def error(self, msg): pass


class MediaDownloadError(Exception):
    """Friendly, user-facing download error."""


GENERIC_URL_ERROR = "Unable to process this URL. Please check that the link is public and valid."


def validate_url(url):
    """Return the cleaned URL, or raise MediaDownloadError with a friendly message."""
    url = (url or "").strip()
    if not url:
        raise MediaDownloadError("Please paste a video or audio URL.")
    parsed = urlparse(url)
    # Only web links are accepted (this blocks file:// and similar schemes).
    if parsed.scheme not in ("http", "https") or not parsed.netloc or " " in url:
        raise MediaDownloadError(
            "This doesn't look like a valid link. It should start with http:// or https://")
    host = parsed.hostname or ""
    if "." not in host and host != "localhost":
        raise MediaDownloadError(
            "This doesn't look like a valid link. It should start with http:// or https://")
    return url


def _friendly_message(technical_text):
    """Translate a yt-dlp error into a short message for normal users."""
    text = technical_text.lower()
    if any(w in text for w in ("private video", "sign in", "log in", "login", "members-only",
                               "members only", "age-restricted", "confirm your age", "cookies")):
        return ("This video is private or needs a login, so it can't be downloaded. "
                "Please use a public link, or upload the file instead.")
    if any(w in text for w in ("video unavailable", "has been removed", "been deleted", "no longer available",
                               "does not exist", "http error 404", "http error 410", "not available")):
        return "This video is unavailable or has been removed."
    if "unsupported url" in text:
        return "This link is not supported. Please paste a direct link to a public video or audio page."
    if any(w in text for w in ("urlopen error", "name resolution", "getaddrinfo", "timed out", "connection",
                               "network is unreachable", "unable to download webpage")):
        return "A network problem stopped the download. Please check your internet connection and try again."
    if "live" in text and "stream" in text:
        return "Live streams can't be processed. Please wait until the recording is available."
    return GENERIC_URL_ERROR


def download_media(url, progress=None):
    """Download the best audio stream of a public URL.

    Returns (file_path, media_title). Raises MediaDownloadError on any problem.
    """
    try:
        import yt_dlp
        from yt_dlp.utils import DownloadError, ExtractorError
    except ImportError as exc:
        raise MediaDownloadError("yt-dlp is not installed. Run: pip install yt-dlp (see README).") from exc

    url = validate_url(url)
    os.makedirs(TEMP_DIR, exist_ok=True)
    prefix = f"dl_{uuid.uuid4().hex}"
    options = {
        "format": "bestaudio/best",          # audio is enough, so videos download faster
        "outtmpl": os.path.join(TEMP_DIR, prefix + ".%(ext)s"),
        "noplaylist": True,                   # one video only, never a whole playlist
        "quiet": True,
        "logger": _SilentLogger(),
        "no_warnings": True,
        "noprogress": True,
        "socket_timeout": 30,
        "retries": 3,
        "max_filesize": MAX_DOWNLOAD_MB * 1024 * 1024,
        "restrictfilenames": True,
    }
    try:
        with yt_dlp.YoutubeDL(options) as ydl:
            info = ydl.extract_info(url, download=False)  # look first, download second
            if info is None:
                raise MediaDownloadError(GENERIC_URL_ERROR)
            if info.get("_type") == "playlist":
                entries = [e for e in (info.get("entries") or []) if e]
                if not entries:
                    raise MediaDownloadError(GENERIC_URL_ERROR)
                info = entries[0]
            if info.get("is_live"):
                raise MediaDownloadError(
                    "Live streams can't be processed. Please wait until the recording is available.")
            duration = info.get("duration") or 0
            if duration and duration > MAX_MEDIA_HOURS * 3600:
                raise MediaDownloadError(
                    f"This media is too long (over {MAX_MEDIA_HOURS} hours). Please use a shorter recording.")
            size_guess = info.get("filesize") or info.get("filesize_approx") or 0
            if size_guess and size_guess > MAX_DOWNLOAD_MB * 1024 * 1024:
                raise MediaDownloadError(
                    "This media is too large to process. Please use a shorter recording.")
            if progress:
                progress("downloading")
            ydl.process_ie_result(info, download=True)
    except MediaDownloadError:
        _cleanup(prefix)
        raise
    except (DownloadError, ExtractorError) as exc:
        _cleanup(prefix)
        raise MediaDownloadError(_friendly_message(str(exc))) from exc
    except Exception as exc:  # anything unexpected must not show a traceback
        _cleanup(prefix)
        raise MediaDownloadError(_friendly_message(str(exc))) from exc

    files = [p for p in glob.glob(os.path.join(TEMP_DIR, prefix + ".*"))
             if not p.endswith((".part", ".ytdl"))]
    if not files or os.path.getsize(files[0]) == 0:
        _cleanup(prefix)
        raise MediaDownloadError(
            "The media could not be downloaded. It may be too large, or the site blocked the download.")
    return files[0], (info.get("title") or "").strip()


def _cleanup(prefix):
    """Remove any partial files left by a failed download."""
    for path in glob.glob(os.path.join(TEMP_DIR, prefix + ".*")):
        remove_file(path)
