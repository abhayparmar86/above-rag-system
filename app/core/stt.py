"""
Speech-to-Text module using faster-whisper.
Loads the Whisper model once at import time (same pattern as embedder in database.py).
Exposes a single transcribe() function used by the /stt endpoint in main.py.
"""

import os
import io
import tempfile
import torch
from faster_whisper import WhisperModel
from core.logger import get_logger
from core.translation import get_whisper_code, SUPPORTED_LANGUAGES

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Model Configuration
# ---------------------------------------------------------------------------
# "small" is the sweet spot: fast on GPU, good multilingual accuracy.
# Change to "medium" if you want better accuracy at the cost of ~2x latency.
WHISPER_MODEL_SIZE = os.environ.get("WHISPER_MODEL_SIZE", "large-v3-turbo")

# Reuse the same GPU your embedder and vLLM are on.
# Falls back to CPU automatically if CUDA is not available.
_device = "cuda" if torch.cuda.is_available() else "cpu"
_compute_type = "int8_float16" if _device == "cuda" else "int8"

print(f"[STT] Loading faster-whisper model '{WHISPER_MODEL_SIZE}' on {_device.upper()} ({_compute_type})...")
logger.info("Loading faster-whisper model | model=%s device=%s compute_type=%s", WHISPER_MODEL_SIZE, _device, _compute_type)

# Model is downloaded to HF cache on first run, then cached locally.
# In Docker, this resolves to /models/hf_cache (mounted volume) if HF_HOME is set.
whisper_model = WhisperModel(
    WHISPER_MODEL_SIZE,
    device=_device,
    compute_type=_compute_type,
    download_root=os.environ.get("HF_HOME", None),  # respects your existing HF cache mount
)

print(f"[STT] ✅ faster-whisper '{WHISPER_MODEL_SIZE}' ready.")
logger.info("faster-whisper model ready | model=%s", WHISPER_MODEL_SIZE)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def transcribe(audio_bytes: bytes, hint_language: str | None = None) -> str:
    """
    Transcribes raw audio bytes (WebM, WAV, MP3, etc.) to text.

    Args:
        audio_bytes: Raw audio file content sent from the browser via MediaRecorder.
        hint_language: Language NAME from our supported set (e.g. "Spanish", "English"),
                       or None/"auto" to let Whisper auto-detect. Only the 5 languages in
                       SUPPORTED_LANGUAGES are ever passed through as a hint — anything
                       else is treated as "auto" rather than silently forcing English,
                       which used to be the default and is a likely source of the
                       "even English has mistakes" reports (forcing language='en' on
                       non-English audio makes Whisper confidently mistranscribe instead
                       of detecting correctly).

    Returns:
        Transcribed text string. Empty string if audio is silent, unintelligible, or
        every segment was low-confidence enough that we chose not to trust it.
    """
    whisper_code = None
    if hint_language and hint_language.lower() != "auto":
        whisper_code = get_whisper_code(hint_language)  # None if not in our supported set

    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        segments, info = whisper_model.transcribe(
            tmp_path,
            language=whisper_code,        # None = auto-detect (correct default, not 'en')
            beam_size=5,
            vad_filter=True,              # Silences/noise suppression — very useful for mic input
            vad_parameters=dict(
                min_silence_duration_ms=500   # Don't cut off slow speakers
            ),
            # --- Hallucination-suppression parameters ---
            # Whisper is autoregressive: left unchecked it can confidently generate
            # plausible-sounding text that was never said, especially in noise/silence,
            # and can cascade a bad guess into the rest of the transcript.
            condition_on_previous_text=False,  # don't let one bad segment contaminate the next
            no_speech_threshold=0.6,           # segments Whisper itself flags as "probably silence"
            log_prob_threshold=-1.0,           # drop segments with low average confidence
            compression_ratio_threshold=2.4,   # catches the classic repetitive-hallucination signature
        )

        # segments is a lazy generator — consume it here, dropping low-confidence segments
        # instead of trusting everything Whisper emits.
        kept, dropped = [], 0
        for seg in segments:
            if seg.no_speech_prob is not None and seg.no_speech_prob > 0.6:
                dropped += 1
                continue
            text = seg.text.strip()
            if text:
                kept.append(text)

        transcript = " ".join(kept).strip()

        print(f"[STT] Transcribed ({info.language}, {info.duration:.1f}s, hint={whisper_code or 'auto'}, dropped_segments={dropped}): '{transcript}'")
        logger.debug(
            "Transcription complete | detected_language=%s hint=%s duration_s=%.1f transcript_length=%d dropped_segments=%d",
            info.language, whisper_code or "auto", info.duration, len(transcript), dropped
        )
        return transcript
    finally:
        # Always clean up the temp file
        os.unlink(tmp_path)