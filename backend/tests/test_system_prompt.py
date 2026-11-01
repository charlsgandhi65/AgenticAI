import asyncio
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from system_prompt import (
    DEFAULT_SYSTEM_PROMPT_TEMPLATE,
    build_system_prompt,
    fetch_dynamic_context,
    fetch_user_profile,
    load_raw_system_prompt_template,
)
from voice_service import generate_voice_response
from voice_session_manager import VoiceSession, VoiceSessionManager


class TestSystemPrompt(unittest.TestCase):
    def test_load_raw_system_prompt_template(self):
        template = load_raw_system_prompt_template()
        self.assertIsInstance(template, str)
        self.assertGreater(len(template), 500)
        self.assertIn("{{assistant_name}}", template)
        self.assertIn("{{user_first_name}}", template)
        self.assertIn("{{user_last_name}}", template)

    def test_build_system_prompt_placeholders_populated(self):
        profile = {
            "first_name": "Charls",
            "last_name": "Gandhi",
            "assistant_name": "Alaana",
        }
        prompt = build_system_prompt(profile)
        self.assertNotIn("{{assistant_name}}", prompt)
        self.assertNotIn("{{user_first_name}}", prompt)
        self.assertNotIn("{{user_last_name}}", prompt)
        self.assertIn("Alaana", prompt)
        self.assertIn("Charls Gandhi", prompt)
        self.assertIn("Charls", prompt)

    def test_user_isolation_different_assistants(self):
        # User 1: Charls Gandhi / Alaana
        profile1 = {
            "first_name": "Charls",
            "last_name": "Gandhi",
            "assistant_name": "Alaana",
        }
        prompt1 = build_system_prompt(profile1)

        # User 2: Sarah Connor / Jarvis
        profile2 = {
            "first_name": "Sarah",
            "last_name": "Connor",
            "assistant_name": "Jarvis",
        }
        prompt2 = build_system_prompt(profile2)

        self.assertIn("Alaana", prompt1)
        self.assertIn("Charls", prompt1)
        self.assertNotIn("Jarvis", prompt1)

        self.assertIn("Jarvis", prompt2)
        self.assertIn("Sarah", prompt2)
        self.assertNotIn("Alaana", prompt2)

    def test_missing_profile_handled_gracefully(self):
        # Missing all values
        with self.assertLogs("system_prompt", level="WARNING") as cm:
            prompt = build_system_prompt(None)
            self.assertIn("Assistant", prompt)
            self.assertIn("the user", prompt)
            self.assertTrue(any("missing" in log for log in cm.output))

    def test_partial_profile_missing_last_name(self):
        with self.assertLogs("system_prompt", level="WARNING") as cm:
            profile = {
                "first_name": "John",
                "last_name": "",
                "assistant_name": "Friday",
            }
            prompt = build_system_prompt(profile)
            self.assertIn("Friday", prompt)
            self.assertIn("John", prompt)
            self.assertNotIn("{{", prompt)

    def test_dynamic_context_appended(self):
        profile = {
            "first_name": "Alice",
            "last_name": "Smith",
            "assistant_name": "Echo",
        }
        dynamic_ctx = (
            "\n\n## 15. Dynamic Call Context & Internal Memories\n"
            "- Relevant User Preferences: Always take messages for sales calls."
        )
        prompt = build_system_prompt(profile, dynamic_context=dynamic_ctx)
        self.assertIn("Echo", prompt)
        self.assertIn("Alice Smith", prompt)
        self.assertIn("Dynamic Call Context", prompt)
        self.assertIn("Always take messages for sales calls", prompt)
        self.assertIn("CONFIDENTIALITY", prompt.upper())

    @patch("system_prompt.get_supabase_client")
    def test_fetch_user_profile_from_supabase(self, mock_get_client):
        mock_client = MagicMock()
        mock_query = MagicMock()
        mock_client.table.return_value = mock_query
        mock_query.select.return_value = mock_query
        mock_query.eq.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.execute.return_value = MagicMock(
            data=[
                {
                    "first_name": "Rudra",
                    "last_name": "Gandhi",
                    "assistant_name": "Siri",
                }
            ]
        )
        mock_get_client.return_value = mock_client

        valid_uuid = "00000000-0000-0000-0000-000000000001"
        profile = fetch_user_profile(valid_uuid)
        self.assertIsNotNone(profile)
        self.assertEqual(profile["first_name"], "Rudra")
        self.assertEqual(profile["last_name"], "Gandhi")
        self.assertEqual(profile["assistant_name"], "Siri")

        # Non-UUID test ID should be intercepted gracefully without querying
        self.assertIsNone(fetch_user_profile("not-a-valid-uuid"))

    @patch("system_prompt.get_supabase_client")
    def test_fetch_dynamic_context_with_caller_and_preferences(self, mock_get_client):
        mock_client = MagicMock()

        def table_side_effect(table_name):
            tbl = MagicMock()
            tbl.select.return_value = tbl
            tbl.eq.return_value = tbl
            tbl.limit.return_value = tbl
            tbl.order.return_value = tbl
            if table_name == "user_preferences":
                tbl.execute.return_value = MagicMock(
                    data=[{"preference_text": "Do not schedule meetings before 10 AM."}]
                )
            elif table_name == "priority_contacts":
                tbl.execute.return_value = MagicMock(
                    data=[{"contact_name": "Executive Boss", "category": "WORK"}]
                )
            elif table_name == "historical_memory":
                tbl.execute.return_value = MagicMock(
                    data=[{"memory_content": "Caller discussed project milestone on Tuesday."}]
                )
            elif table_name == "learned_memory":
                tbl.execute.return_value = MagicMock(
                    data=[{"memory_content": "User prefers concise 1-sentence answers."}]
                )
            else:
                tbl.execute.return_value = MagicMock(data=[])
            return tbl

        mock_client.table.side_effect = table_side_effect
        mock_get_client.return_value = mock_client

        valid_uuid = "00000000-0000-0000-0000-000000000099"
        ctx = fetch_dynamic_context(valid_uuid, caller_phone_number="9824378001")
        self.assertIn("PRIORITY CONTACT: Executive Boss", ctx)
        self.assertIn("Do not schedule meetings before 10 AM", ctx)
        self.assertIn("project milestone on Tuesday", ctx)
        self.assertIn("concise 1-sentence answers", ctx)
        self.assertIn("CONFIDENTIALITY MANDATE", ctx)

        # Non-UUID test ID should be intercepted gracefully without querying
        self.assertEqual(fetch_dynamic_context("invalid-uuid-string"), "")

    @patch("voice_service.chat_with_ollama")
    def test_generate_voice_response_uses_system_prompt(self, mock_chat):
        mock_chat.return_value = "Hello, this is Alaana speaking."
        custom_system = "You are Alaana, personal assistant to Charls."
        reply = generate_voice_response(
            history=[],
            user_text="Hi",
            system_prompt=custom_system,
        )
        self.assertEqual(reply, "Hello, this is Alaana speaking.")
        args, kwargs = mock_chat.call_args
        messages = kwargs.get("messages", [])
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], custom_system)


if __name__ == "__main__":
    unittest.main()
