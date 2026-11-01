import os
import sys
from pathlib import Path

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

import unittest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from auth import AuthenticatedUser, get_current_user
from main import app
from prescreening_service import (
    DatabaseQueryError,
    _phone_matches,
    get_prescreening_context,
    normalize_phone_number,
)


class TestPhoneNormalization(unittest.TestCase):
    def test_normalize_clean_digits(self):
        self.assertEqual(normalize_phone_number("9824378001"), "9824378001")

    def test_normalize_with_formatting(self):
        self.assertEqual(normalize_phone_number("+91 (982) 437-8001"), "919824378001")
        self.assertEqual(normalize_phone_number("  982-437-8001  "), "9824378001")

    def test_normalize_invalid_raises_value_error(self):
        with self.assertRaises(ValueError):
            normalize_phone_number("abc")
        with self.assertRaises(ValueError):
            normalize_phone_number("")
        with self.assertRaises(ValueError):
            normalize_phone_number("---")

    def test_phone_matches_exact(self):
        self.assertTrue(_phone_matches("9824378001", "9824378001"))

    def test_phone_matches_with_country_code(self):
        # Stored has 10 digits, caller has +91 (12 digits)
        self.assertTrue(_phone_matches("9824378001", "919824378001"))
        # Stored has +91 (12 digits), caller has 10 digits
        self.assertTrue(_phone_matches("+919824378001", "9824378001"))
        # Stored has 10 digits, caller has leading 0 (11 digits)
        self.assertTrue(_phone_matches("9824378001", "09824378001"))

    def test_phone_matches_non_matching(self):
        self.assertFalse(_phone_matches("9824378001", "9824378002"))
        self.assertFalse(_phone_matches(None, "9824378001"))
        self.assertFalse(_phone_matches(1234567890, "1234567890"))

    def test_phone_matches_short_numbers_no_fuzzy_suffix(self):
        # Short numbers should not loosely suffix-match
        self.assertFalse(_phone_matches("78001", "9824378001"))


class MockSupabaseQueryBuilder:
    def __init__(self, data=None, error=None):
        self._data = data if data is not None else []
        self._error = error

    def select(self, *args, **kwargs):
        return self

    def eq(self, *args, **kwargs):
        return self

    def limit(self, *args, **kwargs):
        return self

    def execute(self):
        if self._error:
            raise self._error
        mock_res = MagicMock()
        mock_res.data = self._data
        return mock_res


class TestPrescreeningServiceLogic(unittest.IsolatedAsyncioTestCase):
    async def test_matching_priority_contact_authenticated_user(self):
        user_a = "user-uuid-1111"
        caller = "9824378001"

        def table_router(table_name):
            if table_name == "priority_contacts":
                return MockSupabaseQueryBuilder([
                    {
                        "priority_contact_id": "p-1",
                        "user_id": user_a,
                        "contact_name": "Executive Boss",
                        "phone_number": "9824378001",
                        "category": "WORK",
                        "is_active": True,
                    }
                ])
            elif table_name == "users":
                return MockSupabaseQueryBuilder([{"user_id": user_a}])
            return MockSupabaseQueryBuilder([])

        mock_client = MagicMock()
        mock_client.table.side_effect = table_router

        ctx = await get_prescreening_context(user_a, caller, client=mock_client)
        self.assertEqual(len(ctx.priority_contacts), 1)
        self.assertEqual(ctx.priority_contacts[0]["contact_name"], "Executive Boss")
        self.assertEqual(len(ctx.contacts), 0)
        self.assertEqual(len(ctx.spam_cache), 0)

    async def test_priority_contact_belongs_to_another_user_isolated(self):
        user_a = "user-uuid-1111"
        user_b = "user-uuid-2222"
        caller = "9824378001"

        def table_router(table_name):
            if table_name == "priority_contacts":
                # Contact belongs to user_b, so query filtered by user_a returns empty
                return MockSupabaseQueryBuilder([])
            return MockSupabaseQueryBuilder([{"user_id": user_a}])

        mock_client = MagicMock()
        mock_client.table.side_effect = table_router

        ctx = await get_prescreening_context(user_a, caller, client=mock_client)
        self.assertEqual(len(ctx.priority_contacts), 0)

    async def test_matching_contact_and_spam_cache(self):
        user_a = "user-uuid-1111"
        caller = "+91 98243 78001"

        def table_router(table_name):
            if table_name == "contacts":
                return MockSupabaseQueryBuilder([
                    {
                        "contact_id": "c-1",
                        "user_id": user_a,
                        "contact_name": "Colleague",
                        "phone_number": "9824378001",
                        "is_active": True,
                    }
                ])
            elif table_name == "spam_cache":
                return MockSupabaseQueryBuilder([
                    {
                        "spam_id": "s-1",
                        "phone_number": "9824378001",
                        "source": "AI_DETECTED",
                        "is_active": True,
                    }
                ])
            elif table_name == "user_preferences":
                return MockSupabaseQueryBuilder([
                    {
                        "preference_id": "pref-1",
                        "user_id": user_a,
                        "preference_text": "Block spam immediately",
                        "is_active": True,
                    }
                ])
            return MockSupabaseQueryBuilder([{"user_id": user_a}])

        mock_client = MagicMock()
        mock_client.table.side_effect = table_router

        ctx = await get_prescreening_context(user_a, caller, client=mock_client)
        self.assertEqual(len(ctx.contacts), 1)
        self.assertEqual(ctx.contacts[0]["contact_name"], "Colleague")
        self.assertEqual(len(ctx.spam_cache), 1)
        self.assertEqual(ctx.spam_cache[0]["source"], "AI_DETECTED")
        self.assertEqual(len(ctx.user_preferences), 1)

    async def test_formatting_differences_matching(self):
        user_a = "user-uuid-1111"
        caller = "+91-982-437-8001"

        def table_router(table_name):
            if table_name == "contacts":
                return MockSupabaseQueryBuilder([
                    {
                        "contact_id": "c-1",
                        "user_id": user_a,
                        "contact_name": "Mom",
                        "phone_number": "9824378001",
                        "is_active": True,
                    }
                ])
            return MockSupabaseQueryBuilder([{"user_id": user_a}])

        mock_client = MagicMock()
        mock_client.table.side_effect = table_router

        ctx = await get_prescreening_context(user_a, caller, client=mock_client)
        self.assertEqual(len(ctx.contacts), 1)
        self.assertEqual(ctx.contacts[0]["contact_name"], "Mom")

    async def test_no_matching_records_returns_empty_arrays(self):
        user_a = "user-uuid-1111"
        caller = "1111111111"

        def table_router(table_name):
            return MockSupabaseQueryBuilder([])

        mock_client = MagicMock()
        mock_client.table.side_effect = table_router

        ctx = await get_prescreening_context(user_a, caller, client=mock_client)
        self.assertEqual(ctx.contacts, [])
        self.assertEqual(ctx.priority_contacts, [])
        self.assertEqual(ctx.spam_cache, [])
        self.assertEqual(ctx.call_history, [])
        self.assertEqual(ctx.user_preferences, [])

    async def test_database_error_raises_database_query_error(self):
        user_a = "user-uuid-1111"
        caller = "9824378001"

        def table_router(table_name):
            if table_name == "contacts":
                return MockSupabaseQueryBuilder(error=RuntimeError("Connection refused by database"))
            return MockSupabaseQueryBuilder([])

        mock_client = MagicMock()
        mock_client.table.side_effect = table_router

        with self.assertRaises(DatabaseQueryError):
            await get_prescreening_context(user_a, caller, client=mock_client)


class TestPrescreeningEndpoint(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_missing_auth_header_returns_401(self):
        response = self.client.post(
            "/prescreening/test",
            json={"caller_phone_number": "9824378001"},
        )
        self.assertEqual(response.status_code, 401)
        self.assertIn("Bearer authentication is required", response.json()["detail"])

    def test_invalid_bearer_token_returns_401(self):
        with patch("auth.get_supabase_client") as mock_sb:
            mock_auth = MagicMock()
            mock_auth.auth.get_user.side_effect = Exception("Invalid token")
            mock_sb.return_value = mock_auth

            response = self.client.post(
                "/prescreening/test",
                json={"caller_phone_number": "9824378001"},
                headers={"Authorization": "Bearer fake-invalid-token"},
            )
            self.assertEqual(response.status_code, 401)
            self.assertIn("Invalid or expired Supabase access token", response.json()["detail"])

    def test_invalid_caller_phone_number_returns_400(self):
        app.dependency_overrides[get_current_user] = lambda: AuthenticatedUser(user_id="test-user-id")
        try:
            response = self.client.post(
                "/prescreening/test",
                json={"caller_phone_number": "invalid-no-digits"},
                headers={"Authorization": "Bearer dummy-token"},
            )
            self.assertEqual(response.status_code, 400)
            self.assertIn("Caller phone number must contain digits", response.json()["detail"])
        finally:
            app.dependency_overrides.clear()

    def test_database_error_returns_502_sanitized(self):
        app.dependency_overrides[get_current_user] = lambda: AuthenticatedUser(user_id="test-user-id")
        try:
            with patch("main.get_prescreening_context") as mock_service:
                mock_service.side_effect = DatabaseQueryError("Internal table failure")
                response = self.client.post(
                    "/prescreening/test",
                    json={"caller_phone_number": "9824378001"},
                    headers={"Authorization": "Bearer dummy-token"},
                )
                self.assertEqual(response.status_code, 502)
                self.assertEqual(
                    response.json()["detail"],
                    "Database service error during pre-screening retrieval.",
                )
        finally:
            app.dependency_overrides.clear()


if __name__ == "__main__":
    unittest.main()
