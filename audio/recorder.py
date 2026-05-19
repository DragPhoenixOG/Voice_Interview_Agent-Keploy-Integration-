"""
audio/recorder.py
=================
Microphone recording with automatic silence detection.
Uses sounddevice for cross-platform audio capture.
Recording stops automatically after N seconds of silence.
"""

import os
import logging
import tempfile
import numpy as np
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

SAMPLE_RATE = int(os.getenv("AUDIO_SAMPLE_RATE", "16000"))
CHANNELS    = int(os.getenv("AUDIO_CHANNELS", "1"))


def record_audio(
    max_duration: float   = 45.0,
    silence_threshold: float = 0.01,
    silence_duration: float  = 2.0,
    output_path: str = None,
) -> str:
    """
    Record from the default microphone until silence is detected or max_duration reached.
    Auto-stops after `silence_duration` seconds of quiet (after speech has started).

    Args:
        max_duration:       Maximum recording length in seconds
        silence_threshold:  RMS amplitude below which audio is considered silence
        silence_duration:   Seconds of continuous silence before stopping
        output_path:        Where to save the WAV file (temp file if None)

    Returns:
        Path to saved WAV file
    """
    try:
        import sounddevice as sd
        from scipy.io.wavfile import write as wav_write
    except ImportError:
        raise RuntimeError(
            "sounddevice and scipy are required.\n"
            "Run: pip install sounddevice scipy"
        )

    chunk_size          = int(SAMPLE_RATE * 0.1)   # 100 ms chunks
    total_chunks        = int(max_duration / 0.1)
    silence_chunks_needed = int(silence_duration / 0.1)

    all_chunks      = []
    silence_count   = 0
    speech_started  = False

    logger.info(f"Recording: max={max_duration}s, silence_stop={silence_duration}s")

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
        ) as stream:

            for _ in range(total_chunks):
                chunk, _ = stream.read(chunk_size)
                all_chunks.append(chunk.copy())

                rms = float(np.sqrt(np.mean(chunk ** 2)))

                if rms > silence_threshold:
                    speech_started = True
                    silence_count  = 0
                elif speech_started:
                    silence_count += 1

                # Stop after sustained silence following speech
                if speech_started and silence_count >= silence_chunks_needed:
                    logger.info(
                        f"Silence detected — stopped at "
                        f"{len(all_chunks) * 0.1:.1f}s"
                    )
                    break

    except Exception as e:
        raise RuntimeError(f"Microphone recording failed: {e}")

    if not all_chunks:
        raise RuntimeError("No audio captured")

    # Concatenate float32 → convert to int16 for WAV
    audio = np.concatenate(all_chunks, axis=0)
    audio_int16 = (audio * 32767).astype(np.int16)

    # Save to file
    if output_path is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        output_path = tmp.name
        tmp.close()

    from scipy.io.wavfile import write as wav_write
    wav_write(output_path, SAMPLE_RATE, audio_int16)
    logger.info(f"Saved {len(audio_int16) / SAMPLE_RATE:.1f}s of audio → {output_path}")
    return output_path


def cleanup_audio_file(path: str):
    """Delete a temporary audio file after processing."""
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except Exception as e:
        logger.warning(f"Could not remove audio file {path}: {e}")
