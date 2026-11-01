import asyncio
import io
import sys
import unittest
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

import numpy as np

from voice_service import (
    clean_text_for_speech,
    generate_voice_response,
    get_piper_voice,
    get_whisper_model,
    synthesize_text_to_wav,
    transcribe_audio_bytes,
)
from voice_session_manager import VoiceSession, VoiceSessionManager


class TestVoiceServiceUnit(unittest.TestCase):
    def test_clean_text_for_speech(self):
        dirty = "Hello **world**! Check out https://example.com/api #1 *item* - bullet item `code`"
        cleaned = clean_text_for_speech(dirty)
        self.assertNotIn("**", cleaned)
        self.assertNotIn("https://", cleaned)
        self.assertNotIn("`", cleaned)
        self.assertIn("Hello world", cleaned)

    def test_clean_text_empty_or_whitespace(self):
        self.assertEqual(clean_text_for_speech(""), "")
        self.assertEqual(clean_text_for_speech("   \n\t  "), "")

    def test_piper_voice_singleton_and_synthesis(self):
        voice = get_piper_voice()
        self.assertIsNotNone(voice)
        voice2 = get_piper_voice()
        self.assertIs(voice, voice2)

        wav_bytes = synthesize_text_to_wav("Hello, this is a test.")
        self.assertGreater(len(wav_bytes), 1000)
        self.assertTrue(wav_bytes.startswith(b"RIFF"))

        # Verify it can be read as standard WAV
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            self.assertEqual(wf.getnchannels(), 1)
            self.assertEqual(wf.getframerate(), 22050)
            self.assertGreater(wf.getnframes(), 0)

    def test_whisper_model_singleton(self):
        model1 = get_whisper_model()
        self.assertIsNotNone(model1)
        model2 = get_whisper_model()
        self.assertIs(model1, model2)

    def test_transcribe_synthetic_silence(self):
        # 1 second of silence at 16kHz
        silence = np.zeros(16000, dtype=np.int16).tobytes()
        transcript = transcribe_audio_bytes(silence)
        # Silence with VAD should return empty string or non-blank text
        self.assertIsInstance(transcript, str)

    def test_transcribe_empty_bytes(self):
        self.assertEqual(transcribe_audio_bytes(b""), "")
        self.assertEqual(transcribe_audio_bytes(b"short"), "")

    @patch("voice_service.chat_with_ollama")
    def test_generate_voice_response(self, mock_chat):
        mock_chat.return_value = "I can certainly help you with scheduling your appointment tomorrow."
        history = [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hello, how can I help?"},
        ]
        reply = generate_voice_response(history, "Can you help me schedule?")
        self.assertIn("scheduling your appointment", reply)
        mock_chat.assert_called_once()


class TestVoiceSessionManager(unittest.IsolatedAsyncioTestCase):
    async def test_session_creation_and_touch(self):
        mgr = VoiceSessionManager()
        s1 = await mgr.get_or_create(None, "user-123")
        self.assertIsNotNone(s1.session_id)
        self.assertEqual(s1.user_id, "user-123")
        self.assertEqual(len(s1.history), 0)

        # Retrieve existing
        s2 = await mgr.get_or_create(s1.session_id, "user-123")
        self.assertIs(s1, s2)

    async def test_multi_turn_history(self):
        mgr = VoiceSessionManager()
        s = await mgr.get_or_create("session-multi", "user-abc")
        s.add_turn("Hi", "Hello!")
        s.add_turn("What's my name?", "You didn't mention it yet.")
        self.assertEqual(len(s.history), 4)
        self.assertEqual(s.history[0]["content"], "Hi")
        self.assertEqual(s.history[3]["content"], "You didn't mention it yet.")

    async def test_cancel_active_task_on_interrupt(self):
        mgr = VoiceSessionManager()
        s = await mgr.get_or_create("session-cancel", "user-cancel")
        s.is_ai_speaking = True

        async def dummy_task():
            await asyncio.sleep(5)

        task = asyncio.create_task(dummy_task())
        s.current_task = task

        s.cancel_active_task()
        self.assertFalse(s.is_ai_speaking)
        # Yield control to allow cancellation to propagate
        try:
            await task
        except asyncio.CancelledError:
            pass
        self.assertTrue(task.cancelled())


if __name__ == "__main__":
    unittest.main()
