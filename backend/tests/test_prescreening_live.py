import os
import sys
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

import unittest
from fastapi.testclient import TestClient

from main import app
from supabase_client import get_supabase_client


class TestPrescreeningLiveSupabase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.admin = get_supabase_client(use_service_role=True)
        cls.anon = get_supabase_client(use_service_role=False)
        cls.client = TestClient(app)

        # Generate live access token for test1@gmail.com
        link1 = cls.admin.auth.admin.generate_link({"type": "magiclink", "email": "test1@gmail.com"})
        otp1 = link1.properties.email_otp
        auth1 = cls.anon.auth.verify_otp({"email": "test1@gmail.com", "token": otp1, "type": "magiclink"})
        cls.token_test1 = auth1.session.access_token

        # Generate live access token for charls@gmail.com
        link2 = cls.admin.auth.admin.generate_link({"type": "magiclink", "email": "charls@gmail.com"})
        otp2 = link2.properties.email_otp
        auth2 = cls.anon.auth.verify_otp({"email": "charls@gmail.com", "token": otp2, "type": "magiclink"})
        cls.token_charls = auth2.session.access_token

    def test_live_test1_matching_contact(self):
        headers = {"Authorization": f"Bearer {self.token_test1}"}
        res = self.client.post("/prescreening/test", json={"caller_phone_number": "6357373812"}, headers=headers)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(len(data["contacts"]), 1)
        self.assertEqual(data["contacts"][0]["contact_name"], "Princy")
        self.assertEqual(data["contacts"][0]["phone_number"], "6357373812")
        self.assertGreaterEqual(len(data["user_preferences"]), 1)

    def test_live_test1_matching_priority_contact_and_user_preferences(self):
        headers = {"Authorization": f"Bearer {self.token_test1}"}
        res = self.client.post("/prescreening/test", json={"caller_phone_number": "9824378001"}, headers=headers)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(len(data["priority_contacts"]), 1)
        self.assertEqual(data["priority_contacts"][0]["contact_name"], "Test Priority Contact")
        self.assertEqual(data["priority_contacts"][0]["phone_number"], "9824378001")
        self.assertEqual(data["priority_contacts"][0]["category"], "WORK")
        self.assertGreaterEqual(len(data["user_preferences"]), 1)

    def test_live_test1_caller_formatting_differences(self):
        headers = {"Authorization": f"Bearer {self.token_test1}"}
        res = self.client.post("/prescreening/test", json={"caller_phone_number": "+91 (982) 437-8001"}, headers=headers)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(len(data["priority_contacts"]), 1)
        self.assertEqual(data["priority_contacts"][0]["contact_name"], "Test Priority Contact")

    def test_live_test1_matching_spam_cache(self):
        headers = {"Authorization": f"Bearer {self.token_test1}"}
        res = self.client.post("/prescreening/test", json={"caller_phone_number": "7069213654"}, headers=headers)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(len(data["spam_cache"]), 1)
        self.assertEqual(data["spam_cache"][0]["source"], "USER_REPORTED")

    def test_live_user_isolation(self):
        # test1 must NOT see charls's priority contact (Mummy: 9714381777)
        headers1 = {"Authorization": f"Bearer {self.token_test1}"}
        res1 = self.client.post("/prescreening/test", json={"caller_phone_number": "9714381777"}, headers=headers1)
        self.assertEqual(res1.status_code, 200)
        self.assertEqual(len(res1.json()["priority_contacts"]), 0)

        # charls must see their own priority contact (Mummy: 9714381777)
        headers2 = {"Authorization": f"Bearer {self.token_charls}"}
        res2 = self.client.post("/prescreening/test", json={"caller_phone_number": "9714381777"}, headers=headers2)
        self.assertEqual(res2.status_code, 200)
        self.assertEqual(len(res2.json()["priority_contacts"]), 1)
        self.assertEqual(res2.json()["priority_contacts"][0]["contact_name"], "Mummy")

        # charls must NOT see test1's contact (Princy: 6357373812) or priority contact (9824378001)
        res3 = self.client.post("/prescreening/test", json={"caller_phone_number": "6357373812"}, headers=headers2)
        self.assertEqual(res3.status_code, 200)
        self.assertEqual(len(res3.json()["contacts"]), 0)

        res4 = self.client.post("/prescreening/test", json={"caller_phone_number": "9824378001"}, headers=headers2)
        self.assertEqual(res4.status_code, 200)
        self.assertEqual(len(res4.json()["priority_contacts"]), 0)

    def test_live_no_match(self):
        headers = {"Authorization": f"Bearer {self.token_test1}"}
        res = self.client.post("/prescreening/test", json={"caller_phone_number": "1234554321"}, headers=headers)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(len(data["contacts"]), 0)
        self.assertEqual(len(data["priority_contacts"]), 0)
        self.assertEqual(len(data["spam_cache"]), 0)
        self.assertEqual(len(data["call_history"]), 0)
        # user preferences are still returned for the authenticated user
        self.assertGreaterEqual(len(data["user_preferences"]), 1)


if __name__ == "__main__":
    unittest.main()
