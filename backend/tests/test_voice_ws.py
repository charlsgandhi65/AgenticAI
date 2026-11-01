import json
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from fastapi.testclient import TestClient

from main import app
from voice_session_manager import session_manager


class TestVoiceWebSocket(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_ws_connection_and_session_ready(self):
        with self.client.websocket_connect("/ws/voice-conversation") as websocket:
            data = websocket.receive_json()
            self.assertEqual(data.get("type"), "session_ready")
            self.assertIn("session_id", data)
            self.assertEqual(data.get("state"), "listening")

    def test_ws_barge_in_interrupt(self):
        with self.client.websocket_connect("/ws/voice-conversation") as websocket:
            ready = websocket.receive_json()
            self.assertEqual(ready["type"], "session_ready")

            # Send interrupt message
            websocket.send_json({"type": "interrupt"})

            interrupted = websocket.receive_json()
            self.assertEqual(interrupted.get("type"), "interrupted")
            self.assertIn("turn_id", interrupted)

            state = websocket.receive_json()
            self.assertEqual(state.get("type"), "state")
            self.assertEqual(state.get("state"), "listening")

    @patch("voice_ws.synthesize_text_to_wav")
    @patch("voice_ws.generate_voice_response")
    def test_ws_text_turn_pipeline(self, mock_llm, mock_tts):
        mock_llm.return_value = "The capital of France is Paris."
        mock_tts.return_value = b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00dummy-wav-data"

        with self.client.websocket_connect("/ws/voice-conversation") as websocket:
            ready = websocket.receive_json()
            self.assertEqual(ready["type"], "session_ready")

            # Send text input
            websocket.send_json({"type": "text_input", "text": "What is the capital of France?"})

            # 1. state: processing
            msg1 = websocket.receive_json()
            self.assertEqual(msg1.get("type"), "state")
            self.assertEqual(msg1.get("state"), "processing")

            # 2. transcript user
            msg2 = websocket.receive_json()
            self.assertEqual(msg2.get("type"), "transcript")
            self.assertEqual(msg2.get("role"), "user")
            self.assertEqual(msg2.get("text"), "What is the capital of France?")

            # 3. transcript assistant
            msg3 = websocket.receive_json()
            self.assertEqual(msg3.get("type"), "transcript")
            self.assertEqual(msg3.get("role"), "assistant")
            self.assertEqual(msg3.get("text"), "The capital of France is Paris.")

            # 4. ai_audio_start
            msg4 = websocket.receive_json()
            self.assertEqual(msg4.get("type"), "ai_audio_start")
            self.assertEqual(msg4.get("format"), "wav")

            # 5. binary WAV audio
            binary_audio = websocket.receive_bytes()
            self.assertTrue(binary_audio.startswith(b"RIFF"))

            # 6. state: speaking
            msg5 = websocket.receive_json()
            self.assertEqual(msg5.get("type"), "state")
            self.assertEqual(msg5.get("state"), "speaking")

            # 7. Frontend informs audio ended
            websocket.send_json({"type": "audio_playback_ended"})
            msg6 = websocket.receive_json()
            self.assertEqual(msg6.get("type"), "state")
            self.assertEqual(msg6.get("state"), "listening")

    def test_ws_clear_history(self):
        with self.client.websocket_connect("/ws/voice-conversation") as websocket:
            ready = websocket.receive_json()
            self.assertEqual(ready["type"], "session_ready")

            websocket.send_json({"type": "clear_history"})
            cleared = websocket.receive_json()
            self.assertEqual(cleared.get("type"), "history_cleared")


if __name__ == "__main__":
    unittest.main()
