import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

import unittest
from fastapi.testclient import TestClient

from main import app
from supabase_client import get_supabase_client


class TestCallDecisionLive(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.admin = get_supabase_client(use_service_role=True)
        cls.anon = get_supabase_client(use_service_role=False)
        cls.client = TestClient(app)

        # Live access token for test1@gmail.com
        link1 = cls.admin.auth.admin.generate_link({"type": "magiclink", "email": "test1@gmail.com"})
        otp1 = link1.properties.email_otp
        auth1 = cls.anon.auth.verify_otp({"email": "test1@gmail.com", "token": otp1, "type": "magiclink"})
        cls.token_test1 = auth1.session.access_token

        # Live access token for charls@gmail.com
        link2 = cls.admin.auth.admin.generate_link({"type": "magiclink", "email": "charls@gmail.com"})
        otp2 = link2.properties.email_otp
        auth2 = cls.anon.auth.verify_otp({"email": "charls@gmail.com", "token": otp2, "type": "magiclink"})
        cls.token_charls = auth2.session.access_token

    def test_live_priority_contact_decision_with_ollama(self):
        """Full end-to-end test with live Supabase database + live Ollama Qwen model."""
        headers = {"Authorization": f"Bearer {self.token_test1}"}
        res = self.client.post("/ai/decide-call", json={"caller_phone_number": "9824378001"}, headers=headers)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["caller_type"], "PRIORITY_CONTACT")
        self.assertIn(data["decision"], ["HANDLE", "ESCALATE"])
        self.assertNotEqual(data["decision"], "DECLINE")
        self.assertGreaterEqual(data["confidence"], 0.0)
        self.assertLessEqual(data["confidence"], 1.0)
        self.assertTrue(len(data["reason"]) > 0)

    def test_live_spam_cache_decision(self):
        """End-to-end test with live Supabase database queries and verified spam decision."""
        headers = {"Authorization": f"Bearer {self.token_test1}"}
        with patch("call_decision_service.chat_with_ollama") as mock_chat:
            mock_chat.return_value = json.dumps({
                "decision": "DECLINE",
                "reason": "Caller 7069213654 found in spam cache (USER_REPORTED).",
                "confidence": 0.95,
                "caller_type": "SPAM",
            })
            res = self.client.post("/ai/decide-call", json={"caller_phone_number": "7069213654"}, headers=headers)
            self.assertEqual(res.status_code, 200)
            data = res.json()
            self.assertEqual(data["caller_type"], "SPAM")
            self.assertEqual(data["decision"], "DECLINE")

    def test_live_unknown_caller_decision(self):
        """End-to-end test with live Supabase database queries for unknown caller."""
        headers = {"Authorization": f"Bearer {self.token_test1}"}
        with patch("call_decision_service.chat_with_ollama") as mock_chat:
            mock_chat.return_value = json.dumps({
                "decision": "HANDLE",
                "reason": "Unknown caller without negative history; screening allowed.",
                "confidence": 0.8,
                "caller_type": "UNKNOWN",
            })
            res = self.client.post("/ai/decide-call", json={"caller_phone_number": "1234554321"}, headers=headers)
            self.assertEqual(res.status_code, 200)
            data = res.json()
            self.assertEqual(data["caller_type"], "UNKNOWN")
            self.assertEqual(data["decision"], "HANDLE")

    def test_live_cross_user_isolation_decision(self):
        """End-to-end test verifying multi-user tenancy isolation against live Supabase."""
        # 9714381777 is charls's priority contact (Mummy)
        headers_charls = {"Authorization": f"Bearer {self.token_charls}"}
        with patch("call_decision_service.chat_with_ollama") as mock_chat:
            mock_chat.return_value = json.dumps({
                "decision": "HANDLE",
                "reason": "Priority family contact Mummy.",
                "confidence": 0.95,
                "caller_type": "PRIORITY_CONTACT",
            })
            res_charls = self.client.post("/ai/decide-call", json={"caller_phone_number": "9714381777"}, headers=headers_charls)
            self.assertEqual(res_charls.status_code, 200)
            self.assertEqual(res_charls.json()["caller_type"], "PRIORITY_CONTACT")

        # Test1 calling /ai/decide-call with same number 9714381777:
        headers_test1 = {"Authorization": f"Bearer {self.token_test1}"}
        with patch("call_decision_service.chat_with_ollama") as mock_chat:
            mock_chat.return_value = json.dumps({
                "decision": "HANDLE",
                "reason": "No contact record found for this user.",
                "confidence": 0.7,
                "caller_type": "UNKNOWN",
            })
            res_test1 = self.client.post("/ai/decide-call", json={"caller_phone_number": "9714381777"}, headers=headers_test1)
            self.assertEqual(res_test1.status_code, 200)
            # Test1 does NOT see Mummy; classified as UNKNOWN
            self.assertEqual(res_test1.json()["caller_type"], "UNKNOWN")


if __name__ == "__main__":
    unittest.main()
