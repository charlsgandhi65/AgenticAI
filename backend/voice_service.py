import io
import logging
import os
from pathlib import Path
import re
import sysconfig
import time
import wave
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Preserve NVIDIA CUDA DLL initialization required by faster-whisper on Windows
if os.name == "nt":
    nvidia_dir = Path(sysconfig.get_paths()["purelib"]) / "nvidia"
    dll_directories = []
    for package in ("cublas", "cudnn", "cuda_runtime", "cuda_nvrtc"):
        dll_dir = nvidia_dir / package / "bin"
        if dll_dir.is_dir():
            dll_directories.append(str(dll_dir))
            try:
                os.add_dll_directory(str(dll_dir))
            except Exception:
                pass
    if dll_directories:
        os.environ["PATH"] = os.pathsep.join(dll_directories + [os.environ.get("PATH", "")])

from faster_whisper import WhisperModel
import piper

from ollama_client import chat_with_ollama
from system_prompt import build_system_prompt

_whisper_model: WhisperModel | None = None
_piper_voice: piper.PiperVoice | None = None


def get_whisper_model() -> WhisperModel:
    """Load and return the faster-whisper model singleton (CUDA float16, with CPU fallback)."""
    global _whisper_model
    if _whisper_model is None:
        try:
            logger.info("Initializing faster-whisper (base.en, cuda, float16)...")
            _whisper_model = WhisperModel("base.en", device="cuda", compute_type="float16")
            logger.info("faster-whisper loaded successfully on CUDA.")
        except Exception as exc:
            logger.warning("Failed to load faster-whisper on CUDA (%s). Falling back to CPU.", exc)
            _whisper_model = WhisperModel("base.en", device="cpu", compute_type="int8")
            logger.info("faster-whisper loaded on CPU.")
    return _whisper_model


def get_piper_voice() -> piper.PiperVoice:
    """Load and return the Piper voice model singleton."""
    global _piper_voice
    if _piper_voice is None:
        base_dir = Path(__file__).resolve().parent
        model_path = base_dir / "en_US-lessac-medium.onnx"
        config_path = base_dir / "en_US-lessac-medium.onnx.json"
        if not model_path.exists():
            raise FileNotFoundError(f"Piper ONNX model not found at {model_path}")
        logger.info("Initializing Piper voice model from %s...", model_path)
        _piper_voice = piper.PiperVoice.load(
            str(model_path),
            config_path=str(config_path) if config_path.exists() else None,
        )
        logger.info("Piper voice model loaded successfully.")
    return _piper_voice


def preload_voice_models() -> None:
    """Pre-load Whisper and Piper models into memory on server startup."""
    try:
        get_whisper_model()
    except Exception as exc:
        logger.error("Failed to preload Whisper model: %s", exc)
    try:
        get_piper_voice()
    except Exception as exc:
        logger.error("Failed to preload Piper voice model: %s", exc)


import av

def transcribe_audio_bytes(audio_bytes: bytes) -> str:
    """Transcribes audio bytes (WAV, WebM, M4A, or raw 16kHz PCM) using faster-whisper."""
    if not audio_bytes or len(audio_bytes) < 800:
        return ""

    model = get_whisper_model()

    # Determine input type: container (WAV, WebM, M4A, MP3, etc.) or raw 16-bit PCM
    try:
        container_buf = io.BytesIO(audio_bytes)
        container = av.open(container_buf)
        container.close()
        container_buf.seek(0)
        audio_input: Any = container_buf
    except Exception:
        # Fall back to raw 16-bit signed PCM mono audio (assumed 16kHz)
        trimmed_len = len(audio_bytes) - (len(audio_bytes) % 2)
        pcm_bytes = audio_bytes[:trimmed_len]
        audio_np = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        audio_input = audio_np

    try:
        segments, _ = model.transcribe(
            audio_input,
            beam_size=1,
            language="en",
            condition_on_previous_text=False,
            vad_filter=True,
        )
        texts = []
        for segment in segments:
            text = segment.text.strip()
            # Discard non-speech artifacts like [BLANK_AUDIO] or (music)
            if text and not (text.startswith("[") and text.endswith("]")) and not (text.startswith("(") and text.endswith(")")):
                texts.append(text)
        return " ".join(texts).strip()
    except Exception as exc:
        logger.exception("Whisper transcription error: %s", exc)
        return ""


def clean_text_for_speech(text: str) -> str:
    """Strips markdown, URLs, symbols, and formatting for clean Piper TTS output."""
    if not text:
        return ""
    # Remove URLs
    text = re.sub(r"https?://\S+", "", text)
    # Remove markdown symbols (*, _, #, `, ~, >, - at line starts)
    text = re.sub(r"[\*\_#`~]", "", text)
    text = re.sub(r"^\s*[-•]\s+", "", text, flags=re.MULTILINE)
    # Remove angle brackets or XML-like tags
    text = re.sub(r"<[^>]+>", "", text)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def synthesize_text_to_wav(text: str) -> bytes:
    """Synthesizes text into 22050Hz mono WAV bytes using Piper."""
    cleaned = clean_text_for_speech(text)
    if not cleaned:
        return b""
    voice = get_piper_voice()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav_file:
        voice.synthesize_wav(cleaned, wav_file)
    return buf.getvalue()


def generate_voice_response(
    history: list[dict[str, str]],
    user_text: str,
    system_prompt: str | None = None,
) -> str:
    """Generates a conversational voice reply from Qwen/Ollama with multi-turn memory and system prompt."""
    sys_prompt = system_prompt or build_system_prompt()
    messages = [{"role": "system", "content": sys_prompt}]
    # Append recent conversation history (keep last 10 turns to avoid exceeding context window)
    for item in history[-10:]:
        role = item.get("role", "user")
        content = item.get("content", "")
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": user_text})

    try:
        raw_reply = chat_with_ollama(
            messages=messages,
            think=False,
            options={"temperature": 0.7, "num_predict": 100},
        )
        return clean_text_for_speech(raw_reply)
    except Exception as exc:
        logger.exception("Error generating voice response from Qwen/Ollama: %s", exc)
        return "I am sorry, I had trouble processing that. Could you please repeat?"
