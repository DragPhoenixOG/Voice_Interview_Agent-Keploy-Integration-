"""
audio/tts.py
============
Text-to-Speech using pyttsx3 — 100% OFFLINE, no internet, no API key.
pyttsx3 uses your operating system's built-in speech engine:
  - Windows: SAPI5
  - macOS:   NSSpeechSynthesizer
  - Linux:   eSpeak / eSpeak-ng

Install on Linux if voices are missing:
    sudo apt-get install espeak espeak-data libespeak-dev
"""

import os
import io
import wave
import logging
import tempfile
import threading
from functools import lru_cache
from dotenv import load_dotenv
import threading

tts_lock = threading.Lock()

load_dotenv()
logger = logging.getLogger(__name__)

TTS_RATE   = int(os.getenv("TTS_RATE", "400"))      # Words per minute
TTS_VOLUME = float(os.getenv("TTS_VOLUME", "1.0"))  # 0.0 – 1.0
TTS_VOICE_INDEX = int(os.getenv("TTS_VOICE_INDEX", "0"))



# pyttsx3 is NOT thread-safe — use a lock for all engine calls
_tts_lock = threading.Lock()


@lru_cache(maxsize=1)
def _get_engine():
    """
    Create and configure the pyttsx3 engine (cached singleton).
    Called once; reused on every TTS request for efficiency.
    """
    
    try:
        import pyttsx3
    except ImportError:
        raise RuntimeError("pyttsx3 not installed. Run: pip install pyttsx3")

    engine = pyttsx3.init()
    engine.setProperty("rate",   TTS_RATE)
    engine.setProperty("volume", TTS_VOLUME)

    # Select voice by index (0 = first system voice)
    voices = engine.getProperty("voices")

    for v in voices:
        if "Zira" in v.name:
            engine.setProperty("voice", v.id)
            logger.info(f"TTS voice: {voices[TTS_VOICE_INDEX].name}")
        else:
            logger.warning(f"Voice index {TTS_VOICE_INDEX} not found — using default")

    return engine
    


def text_to_speech(text: str, output_path: str = None, play_audio: bool = True) -> str:
    """
    Convert text to speech using pyttsx3 (offline, no internet).
    Speaks aloud and/or saves to a WAV file.

    Args:
        text:        Text to speak
        output_path: Optional WAV file path (temp file if None)
        play_audio:  Speak aloud immediately via system speaker

    Returns:
        Path to saved WAV file (or None if only spoken)
    """
    if os.getenv("DISABLE_TTS", "false").lower() == "true":
        return b""
    if not text.strip():
        logger.warning("Empty text passed to TTS — skipping")
        return None
    
    with _tts_lock:
        try:
            engine = _get_engine()

            engine.stop()

            sentences = text.split(". ")

            for sentence in sentences:
                sentence = sentence.strip()

                if sentence:
                    engine.say(sentence)
                    engine.runAndWait()

            # Force wait so playback fully settles
            import time

            estimated_duration = max(3, len(text) / 12)

            logger.info(f"Sleeping {estimated_duration}s for TTS completion")

            time.sleep(estimated_duration)

            # Save to WAV file
            if output_path is None:
                tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                output_path = tmp.name
                tmp.close()

            engine.save_to_file(text, output_path)
            engine.runAndWait()
            logger.info(f"TTS audio saved: {output_path}")

            if play_audio:
                _play_wav(output_path)

            return output_path

        except Exception as e:
            logger.error(f"pyttsx3 TTS failed: {e}")
            raise RuntimeError(f"Text-to-speech failed: {e}")


def get_tts_audio_bytes(text: str) -> bytes:
    """
    Generate TTS audio and return WAV bytes.
    Used by the FastAPI /synthesize-speech endpoint to stream audio
    to the Streamlit frontend for in-browser playback.

    Returns:
        WAV audio as bytes, or empty bytes on failure
    """
    if not text.strip():
        return b""

    tmp_path = None
    try:
        # Save to temp WAV, read bytes, clean up
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_path = tmp.name
        tmp.close()

        with _tts_lock:
            engine = _get_engine()
            engine.save_to_file(text, tmp_path)
            engine.runAndWait()

        with open(tmp_path, "rb") as f:
            return f.read()

    except Exception as e:
        logger.error(f"pyttsx3 byte generation failed: {e}")
        return b""
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


def _play_wav(file_path: str):
    """
    Play a WAV file using the sounddevice library.
    Falls back gracefully if sounddevice is unavailable.
    """
    try:
        import sounddevice as sd
        import scipy.io.wavfile as wav_io

        rate, data = wav_io.read(file_path)
        sd.play(data, rate)
        sd.wait()
    except Exception as e:
        logger.warning(f"WAV playback via sounddevice failed: {e}. Audio was saved but not played.")


def list_available_voices() -> list:
    """
    Return a list of available system voices.
    Useful for selecting the right TTS_VOICE_INDEX in .env.
    """
    try:
        engine = _get_engine()
        voices = engine.getProperty("voices")
        return [(i, v.name, v.id) for i, v in enumerate(voices)]
    except Exception:
        return []


def speak_directly(text: str):
    """
    Convenience function: speak text immediately without saving.
    Used internally by agents during local playback.
    """
    text_to_speech(text, output_path=None, play_audio=True)
