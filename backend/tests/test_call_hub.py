import asyncio
import json
import struct
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from fastapi.testclient import TestClient
from main import app
from prescreening_service import PreScreeningContext
from call_decision_service import CallDecision, CallDecisionResponse, CallerType
from voice_session_manager import CallChannel, CallMode, call_channel_manager
from call_hub import pcm16_to_wav, execute_caller_ai_turn


class TestCallHub(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_pcm16_to_wav_header(self):
        """Verify that pcm16_to_wav wraps raw PCM in a valid 44-byte WAV header."""
        dummy_pcm = b"\x00\x00" * 160  # 160 samples (10ms at 16kHz)
        wav_data = pcm16_to_wav(dummy_pcm, sample_rate=16000)

        self.assertEqual(len(wav_data), 44 + len(dummy_pcm))
        self.assertEqual(wav_data[:4], b"RIFF")
        self.assertEqual(wav_data[8:12], b"WAVE")
        self.assertEqual(wav_data[12:16], b"fmt ")
        self.assertEqual(wav_data[36:40], b"data")

        # Check sample rate at offset 24
        sample_rate = struct.unpack_from("<I", wav_data, 24)[0]
        self.assertEqual(sample_rate, 16000)

        # Check data size at offset 40
        data_size = struct.unpack_from("<I", wav_data, 40)[0]
        self.assertEqual(data_size, len(dummy_pcm))

    def test_owner_registration_and_ready_event(self):
        """Owner connects to call hub and receives owner_ready event."""
        with self.client.websocket_connect("/ws/call-hub?role=owner&owner_user_id=owner-123") as ws:
            ready = ws.receive_json()
            self.assertEqual(ready.get("type"), "owner_ready")
            self.assertEqual(ready.get("owner_user_id"), "owner-123")
            self.assertIn("call_id", ready)
            self.assertIn("assistant_name", ready)

    @patch("call_hub.synthesize_text_to_wav")
    @patch("call_hub.decide_call")
    @patch("call_hub.get_prescreening_context")
    def test_caller_regular_handle_flow(self, mock_prescreen, mock_decide, mock_tts):
        """Regular caller connects, pre-screened as known contact, decision HANDLE, AI greets."""
        mock_ctx = PreScreeningContext(
            caller_phone_number="9824378001",
            normalized_caller_phone_number="9824378001",
            contacts=[{"contact_id": "c1", "contact_name": "Alice Smith", "phone_number": "9824378001"}],
            priority_contacts=[],
            spam_cache=[],
            call_history=[],
            user_preferences=[],
        )
        mock_prescreen.return_value = mock_ctx
        mock_decide.return_value = CallDecisionResponse(
            decision=CallDecision.HANDLE,
            caller_type=CallerType.KNOWN_CONTACT,
            reason="Known contact. Let AI take the call.",
            confidence=0.9,
        )
        mock_tts.return_value = b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00dummy-wav"

        with self.client.websocket_connect("/ws/call-hub?role=caller&caller_phone_number=9824378001") as ws:
            # 1. call_connected event
            connected_msg = ws.receive_json()
            self.assertEqual(connected_msg.get("type"), "call_connected")
            self.assertEqual(connected_msg.get("mode"), "AI_SCREENING")
            self.assertIn("Alaana", connected_msg.get("assistant_name"))

            # 2. Greeting transcript
            transcript_msg = ws.receive_json()
            self.assertEqual(transcript_msg.get("type"), "transcript")
            self.assertEqual(transcript_msg.get("role"), "assistant")

            # 3. ai_audio_start event
            audio_start = ws.receive_json()
            self.assertEqual(audio_start.get("type"), "ai_audio_start")
            self.assertEqual(audio_start.get("format"), "wav")

            # 4. Binary WAV audio payload
            wav_bytes = ws.receive_bytes()
            self.assertTrue(len(wav_bytes) > 0)

            # 5. state: speaking
            state_msg = ws.receive_json()
            self.assertEqual(state_msg.get("type"), "state")
            self.assertEqual(state_msg.get("state"), "speaking")

    @patch("call_hub.synthesize_text_to_wav")
    @patch("call_hub.get_prescreening_context")
    def test_priority_contact_direct_routing_without_ai(self, mock_prescreen, mock_tts):
        """Priority caller connects -> direct connection to owner without AI intervention."""
        mock_ctx = PreScreeningContext(
            caller_phone_number="6356414812",
            normalized_caller_phone_number="6356414812",
            contacts=[],
            priority_contacts=[
                {
                    "priority_contact_id": "p1",
                    "contact_name": "VIP Client",
                    "phone_number": "6356414812",
                    "category": "CRITICAL_BUSINESS",
                    "is_active": True,
                }
            ],
            spam_cache=[],
            call_history=[],
            user_preferences=[],
        )
        mock_prescreen.return_value = mock_ctx
        mock_tts.return_value = b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00priority-wav"

        with self.client.websocket_connect("/ws/call-hub?role=caller&caller_phone_number=6356414812") as ws:
            # 1. Caller receives call_ringing with priority direct notice
            ring_msg = ws.receive_json()
            self.assertEqual(ring_msg.get("type"), "call_ringing")
            self.assertEqual(ring_msg.get("mode"), "HOLD_AND_WHISPER")
            self.assertTrue(ring_msg.get("is_priority"))
            self.assertEqual(ring_msg.get("caller_name"), "VIP Client")

            # 2. Audio announcement
            audio_start = ws.receive_json()
            self.assertEqual(audio_start.get("type"), "ai_audio_start")
            wav_bytes = ws.receive_bytes()
            self.assertTrue(len(wav_bytes) > 0)

    @patch("call_hub.synthesize_text_to_wav")
    @patch("call_hub.decide_call")
    @patch("call_hub.get_prescreening_context")
    def test_caller_decline_flow(self, mock_prescreen, mock_decide, mock_tts):
        """Spam caller connects -> decision DECLINE -> call_declined message & goodbye audio."""
        mock_ctx = PreScreeningContext(
            caller_phone_number="1800000000",
            normalized_caller_phone_number="1800000000",
            contacts=[],
            priority_contacts=[],
            spam_cache=[],
            call_history=[],
            user_preferences=[],
        )
        mock_prescreen.return_value = mock_ctx
        mock_decide.return_value = CallDecisionResponse(
            decision=CallDecision.DECLINE,
            caller_type=CallerType.SPAM,
            reason="Confirmed spam caller.",
            confidence=0.95,
        )
        mock_tts.return_value = b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00spam-wav"

        with self.client.websocket_connect("/ws/call-hub?role=caller&caller_phone_number=1800000000") as ws:
            decline_msg = ws.receive_json()
            self.assertEqual(decline_msg.get("type"), "call_declined")
            self.assertIn("not accepting calls", decline_msg.get("message", ""))

    @patch("call_hub.synthesize_text_to_wav")
    @patch("call_hub.generate_voice_response")
    @patch("call_hub.transcribe_audio_bytes")
    def test_execute_caller_ai_turn_and_whisper_escalation(self, mock_stt, mock_qwen, mock_tts):
        """When caller requests escalation in-turn, owner receives private whisper alert."""
        mock_stt.return_value = "I need to talk to Charles about an urgent financial transaction."
        mock_qwen.return_value = "Let me put you on hold while I connect you with Charles."
        mock_tts.return_value = b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00hold-wav"

        caller_ws = AsyncMock()
        owner_ws = AsyncMock()

        channel = CallChannel(
            call_id="test-escalation-call",
            owner_user_id="dev-local-user",
            caller_phone_number="9824378001",
            caller_name="Alice Smith",
            caller_ws=caller_ws,
            owner_ws=owner_ws,
            mode=CallMode.AI_SCREENING,
        )

        from voice_session_manager import VoiceSession
        session = VoiceSession(session_id="voice-test-escalation", user_id="dev-local-user")
        session.user_profile = {"first_name": "Charles", "last_name": "Smith", "assistant_name": "Alaana"}
        session.system_prompt = "You are Alaana."
        channel.voice_session = session

        # Execute turn
        asyncio.run(execute_caller_ai_turn(channel, b"\x00" * 32000, 1))

        # Check channel switched to HOLD_AND_WHISPER
        self.assertEqual(channel.mode, CallMode.HOLD_AND_WHISPER)

        # Check owner_ws received whisper_escalation
        owner_sent_messages = [call.args[0] for call in owner_ws.send_json.call_args_list if call.args]
        whisper_events = [m for m in owner_sent_messages if m.get("type") == "whisper_escalation"]
        self.assertTrue(len(whisper_events) > 0)
        self.assertIn("Alice Smith", whisper_events[0].get("whisper_text", ""))
        self.assertTrue(whisper_events[0].get("requires_takeover"))

    @patch("call_hub.synthesize_text_to_wav")
    @patch("call_hub.get_prescreening_context")
    def test_two_device_audio_bridge_and_takeover(self, mock_prescreen, mock_tts):
        """Test full two-device coordination: Owner connects, Caller calls, Owner accepts transfer, and audio bridge transfers frames."""
        mock_ctx = PreScreeningContext(
            caller_phone_number="6356414812",
            normalized_caller_phone_number="6356414812",
            contacts=[],
            priority_contacts=[
                {
                    "priority_contact_id": "p1",
                    "contact_name": "VIP Investor",
                    "phone_number": "6356414812",
                    "category": "INVESTOR",
                    "is_active": True,
                }
            ],
            spam_cache=[],
            call_history=[],
            user_preferences=[],
        )
        mock_prescreen.return_value = mock_ctx
        mock_tts.return_value = b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00dummy-wav"

        # 1. Owner connects
        with self.client.websocket_connect("/ws/call-hub?role=owner&owner_user_id=dev-local-user") as owner_ws:
            owner_ready = owner_ws.receive_json()
            self.assertEqual(owner_ready.get("type"), "owner_ready")

            # 2. Caller connects from mobile
            with self.client.websocket_connect("/ws/call-hub?role=caller&caller_phone_number=6356414812&owner_user_id=dev-local-user") as caller_ws:
                # Caller receives ringing/priority notice
                caller_msg = caller_ws.receive_json()
                self.assertEqual(caller_msg.get("type"), "call_ringing")
                self.assertTrue(caller_msg.get("is_priority"))
                # Discard audio start & bytes for caller
                caller_ws.receive_json() # ai_audio_start
                caller_ws.receive_bytes() # wav bytes

                # Owner receives incoming_call notification!
                owner_call_alert = owner_ws.receive_json()
                self.assertEqual(owner_call_alert.get("type"), "incoming_call")
                self.assertEqual(owner_call_alert.get("caller_name"), "VIP Investor")
                self.assertTrue(owner_call_alert.get("is_priority"))
                self.assertTrue(owner_call_alert.get("requires_takeover"))

                # 3. Owner taps Accept Transfer / Takeover
                owner_ws.send_json({"type": "accept_transfer"})

                owner_bridge = owner_ws.receive_json()
                self.assertEqual(owner_bridge.get("type"), "bridge_connected")
                self.assertEqual(owner_bridge.get("mode"), "AUDIO_BRIDGE")

                caller_bridge = caller_ws.receive_json()
                self.assertEqual(caller_bridge.get("type"), "call_transferred")
                self.assertEqual(caller_bridge.get("mode"), "AUDIO_BRIDGE")

                # 4. Audio Bridging: Caller mic sends PCM chunk -> Owner laptop speaker receives WAV frame
                caller_pcm = b"\x12\x34" * 160
                caller_ws.send_bytes(caller_pcm)

                forwarded_to_owner = owner_ws.receive_bytes()
                self.assertEqual(len(forwarded_to_owner), 44 + len(caller_pcm))
                self.assertEqual(forwarded_to_owner[:4], b"RIFF")
                self.assertEqual(forwarded_to_owner[44:], caller_pcm)

                # 5. Audio Bridging: Owner mic sends PCM chunk -> Caller phone speaker receives WAV frame
                owner_pcm = b"\x56\x78" * 160
                owner_ws.send_bytes(owner_pcm)

                forwarded_to_caller = caller_ws.receive_bytes()
                self.assertEqual(len(forwarded_to_caller), 44 + len(owner_pcm))
                self.assertEqual(forwarded_to_caller[:4], b"RIFF")
                self.assertEqual(forwarded_to_caller[44:], owner_pcm)

                # 6. Owner terminates call
                owner_ws.send_json({"type": "end_call"})

                caller_ended = caller_ws.receive_json()
                self.assertEqual(caller_ended.get("type"), "call_ended")

                owner_ended = owner_ws.receive_json()
                self.assertEqual(owner_ended.get("type"), "call_ended")

    @patch("call_hub.synthesize_text_to_wav")
    @patch("call_hub.get_prescreening_context")
    def test_owner_reject_transfer(self, mock_prescreen, mock_tts):
        """Owner rejects transfer: caller receives AI fallback message offering to take a note."""
        mock_ctx = PreScreeningContext(
            caller_phone_number="6356414812",
            normalized_caller_phone_number="6356414812",
            contacts=[],
            priority_contacts=[
                {
                    "priority_contact_id": "p1",
                    "contact_name": "VIP Contact",
                    "phone_number": "6356414812",
                    "category": "WORK",
                    "is_active": True,
                }
            ],
            spam_cache=[],
            call_history=[],
            user_preferences=[],
        )
        mock_prescreen.return_value = mock_ctx
        mock_tts.return_value = b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00fallback-wav"

        with self.client.websocket_connect("/ws/call-hub?role=owner&owner_user_id=dev-local-user") as owner_ws:
            owner_ws.receive_json() # owner_ready

            with self.client.websocket_connect("/ws/call-hub?role=caller&caller_phone_number=6356414812&owner_user_id=dev-local-user") as caller_ws:
                caller_ws.receive_json() # call_ringing
                caller_ws.receive_json() # ai_audio_start
                caller_ws.receive_bytes() # wav bytes
                owner_ws.receive_json() # incoming_call

                # Owner rejects transfer
                owner_ws.send_json({"type": "reject_transfer"})

                owner_reply = owner_ws.receive_json()
                self.assertEqual(owner_reply.get("type"), "transfer_rejected")

                caller_transcript = caller_ws.receive_json()
                self.assertEqual(caller_transcript.get("type"), "transcript")
                self.assertIn("leave a message", caller_transcript.get("text", ""))


if __name__ == "__main__":
    unittest.main()
