import asyncio
import json
import logging
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from fastapi.testclient import TestClient
from main import app
from voice_service import generate_voice_response
from voice_session_manager import session_manager
from system_prompt import build_system_prompt

logger = logging.getLogger(__name__)


class TestVoiceConversationPipeline(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_ws_profile_initialization_via_query_params(self):
        """Verify frontend query parameters populate user identity and assistant name."""
        unique_sid = f"test-session-{int(time.time()*1000)}"
        url = f"/ws/voice-conversation?session_id={unique_sid}&first_name=Charles&last_name=Smith&assistant_name=Alaana"
        with self.client.websocket_connect(url) as ws:
            ready = ws.receive_json()
            self.assertEqual(ready["type"], "session_ready")
            self.assertEqual(ready["first_name"], "Charles")
            self.assertEqual(ready["last_name"], "Smith")
            self.assertEqual(ready["assistant_name"], "Alaana")

    def test_ws_update_profile_runtime_event(self):
        """Verify frontend can update user and assistant identity during an active session."""
        with self.client.websocket_connect("/ws/voice-conversation") as ws:
            ready = ws.receive_json()
            self.assertEqual(ready["type"], "session_ready")

            ws.send_json({
                "type": "update_profile",
                "first_name": "Rudra",
                "last_name": "Gandhi",
                "assistant_name": "Siri",
            })
            resp = ws.receive_json()
            self.assertEqual(resp["type"], "profile_updated")
            self.assertEqual(resp["first_name"], "Rudra")
            self.assertEqual(resp["last_name"], "Gandhi")
            self.assertEqual(resp["assistant_name"], "Siri")

    @patch("voice_ws.synthesize_text_to_wav")
    @patch("voice_ws.generate_voice_response")
    def test_scenarios_multi_turn_pipeline(self, mock_llm, mock_tts):
        """Test multi-turn voice pipeline covering scenarios 1, 2, 3, 4, 5 without session refresh."""
        mock_tts.return_value = b"RIFF....dummy-wav-data"

        # Sequential responses matching the 5 turns
        mock_llm.side_effect = [
            "Hello! I am Alaana, Charles Smith's personal telephone assistant.",
            "I understand. For important financial transactions, Charles handles those directly. May I take a message?",
            "Certainly, I will make sure Charles is reminded about the badge today.",
            "His office is located on the second floor.",
            "You are very welcome! Have a wonderful day. Goodbye!",
        ]

        unique_sid = f"test-session-{int(time.time()*1000)}"
        url = f"/ws/voice-conversation?session_id={unique_sid}&first_name=Charles&last_name=Smith&assistant_name=Alaana"
        with self.client.websocket_connect(url) as ws:
            ready = ws.receive_json()
            self.assertEqual(ready["type"], "session_ready")
            self.assertEqual(ready["assistant_name"], "Alaana")

            turns = [
                ("What is your name?", "Alaana"),
                ("I wanted to discuss an important financial transaction with Charles.", "financial transactions"),
                ("Can you remind him about the badge today?", "reminded about the badge"),
                ("Where is his office located?", "second floor"),
                ("Thank you for your help, goodbye!", "Goodbye"),
            ]

            for i, (user_input, expected_keyword) in enumerate(turns, 1):
                ws.send_json({"type": "text_input", "text": user_input})

                # State -> processing
                msg_state = ws.receive_json()
                self.assertEqual(msg_state["type"], "state")
                self.assertEqual(msg_state["state"], "processing")

                # User transcript
                msg_user = ws.receive_json()
                self.assertEqual(msg_user["type"], "transcript")
                self.assertEqual(msg_user["role"], "user")
                self.assertEqual(msg_user["text"], user_input)

                # Assistant transcript
                msg_ai = ws.receive_json()
                self.assertEqual(msg_ai["type"], "transcript")
                self.assertEqual(msg_ai["role"], "assistant")
                self.assertIn(expected_keyword, msg_ai["text"])

                # Audio delivery
                msg_audio = ws.receive_json()
                self.assertEqual(msg_audio["type"], "ai_audio_start")
                raw_bytes = ws.receive_bytes()
                self.assertTrue(len(raw_bytes) > 0)

                # State -> speaking
                msg_speaking = ws.receive_json()
                self.assertEqual(msg_speaking["type"], "state")
                self.assertEqual(msg_speaking["state"], "speaking")

                # Inform playback complete
                ws.send_json({"type": "audio_playback_ended"})
                msg_listening = ws.receive_json()
                self.assertEqual(msg_listening["type"], "state")
                self.assertEqual(msg_listening["state"], "listening")

            # Verify mock_llm received accumulating history across all 5 turns
            self.assertEqual(mock_llm.call_count, 5)
            # The 5th call should have 4 previous back-and-forth turns in history (8 messages)
            fifth_call_history = mock_llm.call_args_list[4][0][0]
            self.assertEqual(len(fifth_call_history), 8)
            self.assertEqual(fifth_call_history[0]["content"], "What is your name?")


class TestLiveOllamaVoiceScenarios(unittest.TestCase):
    """Executes live Ollama Qwen multi-turn conversation with actual prompt."""

    def test_live_5_turn_conversation_with_qwen(self):
        """Verifies actual 5-turn conversation against live local Qwen model."""
        profile = {
            "first_name": "Charles",
            "last_name": "Smith",
            "assistant_name": "Alaana",
        }
        prompt = build_system_prompt(profile)

        history = []
        user_utterances = [
            "What is your name?",
            "I wanted to discuss an important financial transaction with Charles.",
            "Can you remind him about the badge today?",
            "When will he be free to talk?",
            "Thank you so much for your assistance.",
        ]

        print("\n=== STARTING LIVE 5-TURN CONVERSATION WITH QWEN ===")
        for turn_idx, user_text in enumerate(user_utterances, 1):
            t0 = time.time()
            reply = generate_voice_response(history, user_text, system_prompt=prompt)
            elapsed = time.time() - t0

            print(f"[Turn {turn_idx}] Caller: {user_text}")
            print(f"[Turn {turn_idx}] AI ({profile['assistant_name']}): {reply} ({elapsed:.2f}s)")

            self.assertTrue(len(reply) > 5, f"Turn {turn_idx} response too short")
            # Verify no markdown symbols or emojis
            self.assertNotIn("**", reply)
            self.assertNotIn("##", reply)

            # Append to history for next turn
            history.append({"role": "user", "content": user_text})
            history.append({"role": "assistant", "content": reply})

        print("=== LIVE 5-TURN CONVERSATION COMPLETED SUCCESSFULLY ===\n")
        self.assertEqual(len(history), 10)


if __name__ == "__main__":
    unittest.main()
