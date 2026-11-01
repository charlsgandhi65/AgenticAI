import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

import unittest
from fastapi.testclient import TestClient

from auth import AuthenticatedUser, get_current_user
from call_decision_service import (
    CallDecision,
    CallDecisionResponse,
    CallerType,
    apply_deterministic_safety_rules,
    classify_caller_type,
    decide_call,
    parse_and_validate_decision_json,
)
from main import app
from prescreening_service import DatabaseQueryError, PreScreeningContext


class TestCallDecisionUnit(unittest.IsolatedAsyncioTestCase):
    def test_classify_caller_type_precedence(self):
        # 1. Priority contact only
        ctx1 = PreScreeningContext(
            caller_phone_number="9824378001",
            normalized_caller_phone_number="9824378001",
            priority_contacts=[{"contact_name": "Boss", "category": "WORK"}],
        )
        self.assertEqual(classify_caller_type(ctx1), CallerType.PRIORITY_CONTACT)

        # 2. Priority and Regular contact overlap (Priority retains precedence)
        ctx2 = PreScreeningContext(
            caller_phone_number="9824378001",
            normalized_caller_phone_number="9824378001",
            priority_contacts=[{"contact_name": "Boss", "category": "WORK"}],
            contacts=[{"contact_name": "Boss"}],
        )
        self.assertEqual(classify_caller_type(ctx2), CallerType.PRIORITY_CONTACT)

        # 3. Regular contact only
        ctx3 = PreScreeningContext(
            caller_phone_number="9824378001",
            normalized_caller_phone_number="9824378001",
            contacts=[{"contact_name": "Friend"}],
        )
        self.assertEqual(classify_caller_type(ctx3), CallerType.KNOWN_CONTACT)

        # 4. Spam cache only
        ctx4 = PreScreeningContext(
            caller_phone_number="9824378001",
            normalized_caller_phone_number="9824378001",
            spam_cache=[{"source": "TRUECALLER"}],
        )
        self.assertEqual(classify_caller_type(ctx4), CallerType.SPAM)

        # 5. Unknown
        ctx5 = PreScreeningContext(
            caller_phone_number="9824378001",
            normalized_caller_phone_number="9824378001",
        )
        self.assertEqual(classify_caller_type(ctx5), CallerType.UNKNOWN)

    async def test_1_priority_contact_no_suspicious_signals(self):
        ctx = PreScreeningContext(
            caller_phone_number="9824378001",
            normalized_caller_phone_number="9824378001",
            priority_contacts=[{"contact_name": "Executive", "category": "WORK"}],
        )

        def mock_chat(messages, **kwargs):
            return json.dumps({
                "decision": "ESCALATE",
                "reason": "Priority contact from work category requires direct user attention.",
                "confidence": 0.95,
                "caller_type": "PRIORITY_CONTACT",
            })

        decision = await decide_call(ctx, chat_fn=mock_chat)
        self.assertEqual(decision.decision, CallDecision.ESCALATE)
        self.assertEqual(decision.caller_type, CallerType.PRIORITY_CONTACT)
        self.assertGreaterEqual(decision.confidence, 0.9)

    async def test_1b_priority_contact_model_says_handle_overridden_to_escalate(self):
        ctx = PreScreeningContext(
            caller_phone_number="6356414812",
            normalized_caller_phone_number="6356414812",
            priority_contacts=[{"contact_name": "charls", "category": "Family"}],
        )

        def mock_chat(messages, **kwargs):
            return json.dumps({
                "decision": "HANDLE",
                "reason": "Priority contact with no negative indicators.",
                "confidence": 0.95,
                "caller_type": "PRIORITY_CONTACT",
            })

        decision = await decide_call(ctx, chat_fn=mock_chat)
        self.assertEqual(decision.decision, CallDecision.ESCALATE)
        self.assertEqual(decision.caller_type, CallerType.PRIORITY_CONTACT)

    async def test_1c_priority_contact_with_explicit_ai_handling_preference(self):
        ctx = PreScreeningContext(
            caller_phone_number="9824378001",
            normalized_caller_phone_number="9824378001",
            priority_contacts=[{"contact_name": "Executive", "category": "WORK"}],
            user_preferences=[{"preference_text": "Allow AI to handle priority contacts", "source": "MANUAL"}],
        )

        def mock_chat(messages, **kwargs):
            return json.dumps({
                "decision": "HANDLE",
                "reason": "User preferences explicitly permit AI handling for priority contacts.",
                "confidence": 0.95,
                "caller_type": "PRIORITY_CONTACT",
            })

        decision = await decide_call(ctx, chat_fn=mock_chat)
        self.assertEqual(decision.decision, CallDecision.HANDLE)
        self.assertEqual(decision.caller_type, CallerType.PRIORITY_CONTACT)

    async def test_2_known_non_priority_contact(self):
        ctx = PreScreeningContext(
            caller_phone_number="6357373812",
            normalized_caller_phone_number="6357373812",
            contacts=[{"contact_name": "Princy"}],
        )

        def mock_chat(messages, **kwargs):
            return json.dumps({
                "decision": "HANDLE",
                "reason": "Known regular contact Princy with no negative signals.",
                "confidence": 0.9,
                "caller_type": "KNOWN_CONTACT",
            })

        decision = await decide_call(ctx, chat_fn=mock_chat)
        self.assertEqual(decision.decision, CallDecision.HANDLE)
        self.assertEqual(decision.caller_type, CallerType.KNOWN_CONTACT)

    async def test_3_spam_cache_number(self):
        ctx = PreScreeningContext(
            caller_phone_number="7069213654",
            normalized_caller_phone_number="7069213654",
            spam_cache=[{"source": "USER_REPORTED"}],
        )

        def mock_chat(messages, **kwargs):
            return json.dumps({
                "decision": "DECLINE",
                "reason": "Caller listed in spam cache with source USER_REPORTED.",
                "confidence": 0.92,
                "caller_type": "SPAM",
            })

        decision = await decide_call(ctx, chat_fn=mock_chat)
        self.assertEqual(decision.decision, CallDecision.DECLINE)
        self.assertEqual(decision.caller_type, CallerType.SPAM)

    async def test_4_unknown_number_not_automatically_declined(self):
        ctx = PreScreeningContext(
            caller_phone_number="1234567890",
            normalized_caller_phone_number="1234567890",
        )

        def mock_chat(messages, **kwargs):
            return json.dumps({
                "decision": "HANDLE",
                "reason": "Unknown caller without negative indicators; screening permitted.",
                "confidence": 0.75,
                "caller_type": "UNKNOWN",
            })

        decision = await decide_call(ctx, chat_fn=mock_chat)
        self.assertIn(decision.decision, [CallDecision.HANDLE, CallDecision.ESCALATE])
        self.assertEqual(decision.caller_type, CallerType.UNKNOWN)

    async def test_5_conflicting_signals_priority_in_spam_escalates(self):
        ctx = PreScreeningContext(
            caller_phone_number="9824378001",
            normalized_caller_phone_number="9824378001",
            priority_contacts=[{"contact_name": "Mom", "category": "FAMILY"}],
            spam_cache=[{"source": "AI_DETECTED"}],
        )

        # Even if the LLM wrongly said DECLINE, safety rules must override to ESCALATE
        def mock_chat(messages, **kwargs):
            return json.dumps({
                "decision": "DECLINE",
                "reason": "Found in spam cache",
                "confidence": 0.8,
                "caller_type": "SPAM",
            })

        decision = await decide_call(ctx, chat_fn=mock_chat)
        self.assertEqual(decision.decision, CallDecision.ESCALATE)
        self.assertEqual(decision.caller_type, CallerType.PRIORITY_CONTACT)
        self.assertIn("Conflicting signals", decision.reason)

    async def test_6_explicit_user_preference_applied(self):
        ctx = PreScreeningContext(
            caller_phone_number="5551234567",
            normalized_caller_phone_number="5551234567",
            user_preferences=[{
                "preference_text": "Always decline calls from unknown callers during focus hours.",
                "source": "MANUAL",
            }],
        )

        def mock_chat(messages, **kwargs):
            return json.dumps({
                "decision": "DECLINE",
                "reason": "User preference explicitly instructs declining unknown callers.",
                "confidence": 0.95,
                "caller_type": "UNKNOWN",
            })

        decision = await decide_call(ctx, chat_fn=mock_chat)
        self.assertEqual(decision.decision, CallDecision.DECLINE)
        self.assertEqual(decision.caller_type, CallerType.UNKNOWN)

    async def test_7_missing_pre_screening_data_handled(self):
        ctx = PreScreeningContext(
            caller_phone_number="9998887777",
            normalized_caller_phone_number="9998887777",
            contacts=[],
            priority_contacts=[],
            spam_cache=[],
            call_history=[],
            user_preferences=[],
        )

        def mock_chat(messages, **kwargs):
            return json.dumps({
                "decision": "HANDLE",
                "reason": "No matching records found; call should be handled with standard greeting.",
                "confidence": 0.7,
                "caller_type": "UNKNOWN",
            })

        decision = await decide_call(ctx, chat_fn=mock_chat)
        self.assertEqual(decision.decision, CallDecision.HANDLE)
        self.assertEqual(decision.caller_type, CallerType.UNKNOWN)

    async def test_8_invalid_json_fallback_to_escalate(self):
        ctx = PreScreeningContext(
            caller_phone_number="9824378001",
            normalized_caller_phone_number="9824378001",
            priority_contacts=[{"contact_name": "Colleague", "category": "WORK"}],
        )

        def mock_chat(messages, **kwargs):
            return "This is not valid JSON at all: error 404 text"

        decision = await decide_call(ctx, chat_fn=mock_chat)
        self.assertEqual(decision.decision, CallDecision.ESCALATE)
        self.assertEqual(decision.caller_type, CallerType.PRIORITY_CONTACT)
        self.assertEqual(decision.confidence, 0.3)
        self.assertIn("fallback", decision.reason.lower())

    async def test_9_ollama_unavailable_or_timeout_fallback_to_escalate(self):
        ctx = PreScreeningContext(
            caller_phone_number="9824378001",
            normalized_caller_phone_number="9824378001",
        )

        def mock_chat(messages, **kwargs):
            raise TimeoutError("Ollama inference timed out after 60s")

        decision = await decide_call(ctx, chat_fn=mock_chat)
        self.assertEqual(decision.decision, CallDecision.ESCALATE)
        self.assertEqual(decision.confidence, 0.3)
        self.assertEqual(decision.caller_type, CallerType.UNKNOWN)
        self.assertIn("TimeoutError", decision.reason)


class TestCallDecisionEndpoint(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_10_missing_auth_token_returns_401(self):
        res = self.client.post("/ai/decide-call", json={"caller_phone_number": "9824378001"})
        self.assertEqual(res.status_code, 401)
        self.assertIn("Bearer authentication is required", res.json()["detail"])

    def test_10_invalid_auth_token_returns_401(self):
        with patch("auth.get_supabase_client") as mock_sb:
            mock_auth = MagicMock()
            mock_auth.auth.get_user.side_effect = Exception("Expired token")
            mock_sb.return_value = mock_auth

            res = self.client.post(
                "/ai/decide-call",
                json={"caller_phone_number": "9824378001"},
                headers={"Authorization": "Bearer bad-expired-token"},
            )
            self.assertEqual(res.status_code, 401)
            self.assertIn("Invalid or expired Supabase access token", res.json()["detail"])

    def test_11_user_isolation_cannot_access_other_user_data(self):
        # User A authenticates
        user_a = "user-uuid-1111"
        app.dependency_overrides[get_current_user] = lambda: AuthenticatedUser(user_id=user_a)

        try:
            # Prescreening for user A should return empty contacts for User B's number
            with patch("main.get_prescreening_context") as mock_prescreen:
                mock_prescreen.return_value = PreScreeningContext(
                    caller_phone_number="9824378001",
                    normalized_caller_phone_number="9824378001",
                    contacts=[],
                    priority_contacts=[],
                    spam_cache=[],
                    call_history=[],
                    user_preferences=[],
                )
                with patch("main.decide_call") as mock_decide:
                    mock_decide.return_value = CallDecisionResponse(
                        decision=CallDecision.HANDLE,
                        reason="No contacts for this user.",
                        confidence=0.8,
                        caller_type=CallerType.UNKNOWN,
                    )
                    res = self.client.post(
                        "/ai/decide-call",
                        json={"caller_phone_number": "9824378001"},
                        headers={"Authorization": "Bearer dummy-token"},
                    )
                    self.assertEqual(res.status_code, 200)
                    data = res.json()
                    self.assertEqual(data["caller_type"], "UNKNOWN")
                    # Verified get_prescreening_context called with User A's user_id
                    mock_prescreen.assert_called_once_with(
                        user_id=user_a,
                        caller_phone_number="9824378001",
                    )
        finally:
            app.dependency_overrides.clear()

    def test_12_existing_endpoints_still_work(self):
        # 1. Root and Health
        res_root = self.client.get("/")
        self.assertEqual(res_root.status_code, 200)
        self.assertIn("AI Calling Agent", res_root.json()["message"])

        res_health = self.client.get("/health")
        self.assertEqual(res_health.status_code, 200)
        self.assertEqual(res_health.json()["status"], "ok")

        # 2. Existing AI Chat
        with patch("main.generate_chat_response") as mock_chat:
            mock_chat.return_value = "Hello from mock Qwen!"
            res_chat = self.client.post("/ai/chat", json={"message": "Hi"})
            self.assertEqual(res_chat.status_code, 200)
            self.assertEqual(res_chat.json()["response"], "Hello from mock Qwen!")

        # 3. Existing Prescreening Test
        app.dependency_overrides[get_current_user] = lambda: AuthenticatedUser(user_id="test-user")
        try:
            with patch("main.get_prescreening_context") as mock_ctx:
                mock_ctx.return_value = PreScreeningContext(
                    caller_phone_number="9824378001",
                    normalized_caller_phone_number="9824378001",
                )
                res_pre = self.client.post(
                    "/prescreening/test",
                    json={"caller_phone_number": "9824378001"},
                    headers={"Authorization": "Bearer dummy-token"},
                )
                self.assertEqual(res_pre.status_code, 200)
                self.assertEqual(res_pre.json()["caller_phone_number"], "9824378001")
        finally:
            app.dependency_overrides.clear()

    def test_13_invalid_phone_number_returns_400(self):
        app.dependency_overrides[get_current_user] = lambda: AuthenticatedUser(user_id="test-user")
        try:
            res = self.client.post(
                "/ai/decide-call",
                json={"caller_phone_number": "no-digits"},
                headers={"Authorization": "Bearer dummy-token"},
            )
            self.assertEqual(res.status_code, 400)
            self.assertIn("Caller phone number must contain digits", res.json()["detail"])
        finally:
            app.dependency_overrides.clear()

    def test_14_database_failure_returns_502_sanitized(self):
        app.dependency_overrides[get_current_user] = lambda: AuthenticatedUser(user_id="test-user")
        try:
            with patch("main.get_prescreening_context") as mock_ctx:
                mock_ctx.side_effect = DatabaseQueryError("Internal connection failure")
                res = self.client.post(
                    "/ai/decide-call",
                    json={"caller_phone_number": "9824378001"},
                    headers={"Authorization": "Bearer dummy-token"},
                )
                self.assertEqual(res.status_code, 502)
                self.assertEqual(
                    res.json()["detail"],
                    "Database service error during pre-screening retrieval.",
                )
        finally:
            app.dependency_overrides.clear()


if __name__ == "__main__":
    unittest.main()
