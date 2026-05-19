"""
audio/stt.py
============
Speech-to-Text using Faster-Whisper — 100% local, completely free.
No API key, no internet needed at runtime. Model is downloaded once
and cached in ~/.cache/huggingface/hub/

Faster-Whisper: https://github.com/SYSTRAN/faster-whisper
"""

import os
import logging
from functools import lru_cache
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")


@lru_cache(maxsize=1)
def _get_model():
    """
    Load and cache the Faster-Whisper model (singleton pattern).
    The model is downloaded once on first call; subsequent calls reuse it.
    Model sizes and disk usage:
      tiny     ~75 MB   — fastest, lower accuracy
      base     ~150 MB  — great balance  ← recommended
      small    ~480 MB  — better accuracy
      medium   ~1.5 GB  — high accuracy
      large-v2 ~3 GB    — best accuracy, slow on CPU
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise RuntimeError("faster-whisper not installed. Run: pip install faster-whisper")

    logger.info(f"Loading Whisper model '{WHISPER_MODEL}' (downloads once if not cached)...")

    model = WhisperModel(
        WHISPER_MODEL,
        device="cpu",         # Use "cuda" if you have an NVIDIA GPU
        compute_type="int8",  # int8 quantization — fast + memory-efficient on CPU
    )
    logger.info("Whisper model ready")
    return model


def transcribe_audio(audio_file_path: str, language: str = "en") -> str:
    """
    Transcribe a WAV/MP3 audio file to text using local Faster-Whisper.

    Args:
        audio_file_path: Path to the audio file
        language:        Language code, e.g. "en", "hi", "fr"

    Returns:
        Transcribed text string (empty string if no speech detected)
    """
    if not os.path.exists(audio_file_path):
        raise FileNotFoundError(f"Audio file not found: {audio_file_path}")

    try:
        model = _get_model()
        logger.info(f"Transcribing: {audio_file_path}")

        segments, info = model.transcribe(
            audio_file_path,
            language=language,
            beam_size=5,
            best_of=5,
            temperature=0.0,    # Deterministic — more consistent results
            vad_filter=True,    # Skip silence / non-speech regions
            vad_parameters=dict(
                min_silence_duration_ms=500,
                # threshold=0.5,
            ),
        )

        parts = [seg.text.strip() for seg in segments if seg.text.strip()]
        result = " ".join(parts).strip()

        if result:
            logger.info(f"Transcription: '{result[:80]}'")
        else:
            logger.warning("No speech detected in recording")

        return result

    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        raise RuntimeError(f"Speech-to-text failed: {e}")


def is_meaningful_response(text: str, min_words: int = 3) -> bool:
    """
    Return True if the transcription contains enough words to process.
    Filters out noise triggers or ultra-short utterances like "um" or "okay".
    """
    return bool(text) and len(text.split()) >= min_words
