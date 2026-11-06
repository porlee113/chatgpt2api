from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("CHATGPT2API_AUTH_KEY", "test-auth")

from services.account_service import AccountService
from services.config import config
from services.storage.json_storage import JSONStorageBackend


class BatchAccountStatusTests(unittest.TestCase):
    def setUp(self) -> None:
        self._prev = config.data.get("max_normal_accounts")
        config.data["max_normal_accounts"] = 5

    def tearDown(self) -> None:
        if self._prev is None:
            config.data.pop("max_normal_accounts", None)
        else:
            config.data["max_normal_accounts"] = self._prev

    def _svc(self, tmp: str) -> AccountService:
        return AccountService(JSONStorageBackend(Path(tmp) / "accounts.json"))

    def test_batch_disable_all_selected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            s = self._svc(tmp)
            s.add_account_items([
                {"access_token": "a1", "status": "正常", "quota": 2},
                {"access_token": "a2", "status": "限流", "quota": 0},
                {"access_token": "a3", "status": "异常", "quota": 0},
                {"access_token": "a4", "status": "禁用", "quota": 1},
            ])
            result = s.batch_update_status(["a1", "a2", "a3", "a4"], "禁用")
            self.assertEqual(result["disabled"], 3)
            self.assertEqual(result["skipped"], 1)  # already disabled
            self.assertEqual(result["failed"], 0)
            self.assertTrue(all(a["status"] == "禁用" for a in s.list_accounts()))

    def test_batch_enable_only_disabled_and_cap_when_zero_normal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            s = self._svc(tmp)
            s.add_account_items([{"access_token": f"d{i}", "status": "禁用", "quota": 3} for i in range(8)])
            result = s.batch_update_status([f"d{i}" for i in range(8)], "正常")
            self.assertEqual(result["enabled"], 5)
            self.assertEqual(result["capped"], 3)
            self.assertEqual(result["skipped"], 0)
            normals = [a for a in s.list_accounts() if a["status"] == "正常"]
            self.assertEqual(len(normals), 5)

    def test_batch_enable_when_three_normal_already(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            s = self._svc(tmp)
            s.add_account_items([{"access_token": f"n{i}", "status": "正常", "quota": 2} for i in range(3)])
            s.add_account_items([{"access_token": f"d{i}", "status": "禁用", "quota": 2} for i in range(5)])
            result = s.batch_update_status([f"d{i}" for i in range(5)], "正常")
            self.assertEqual(result["enabled"], 2)
            self.assertEqual(result["capped"], 3)
            normals = [a for a in s.list_accounts() if a["status"] == "正常"]
            self.assertEqual(len(normals), 5)

    def test_batch_enable_at_full_cap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            s = self._svc(tmp)
            s.add_account_items([{"access_token": f"n{i}", "status": "正常", "quota": 2} for i in range(5)])
            s.add_account_items([{"access_token": "d0", "status": "禁用", "quota": 2}])
            result = s.batch_update_status(["d0"], "正常")
            self.assertEqual(result["enabled"], 0)
            self.assertEqual(result["capped"], 1)
            self.assertEqual(s.get_account("d0")["status"], "禁用")

    def test_batch_enable_skips_rate_limited_and_abnormal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            s = self._svc(tmp)
            s.add_account_items([
                {"access_token": "lim", "status": "限流", "quota": 0},
                {"access_token": "bad", "status": "异常", "quota": 0},
                {"access_token": "dis", "status": "禁用", "quota": 4},
            ])
            result = s.batch_update_status(["lim", "bad", "dis"], "正常")
            self.assertEqual(result["enabled"], 1)
            self.assertEqual(result["skipped"], 2)
            self.assertEqual(s.get_account("lim")["status"], "限流")
            self.assertEqual(s.get_account("bad")["status"], "异常")
            self.assertEqual(s.get_account("dis")["status"], "正常")

    def test_batch_errors_do_not_include_raw_token(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            s = self._svc(tmp)
            secret = "sk-very-secret-token-value-123456"
            result = s.batch_update_status([secret], "禁用")
            self.assertEqual(result["failed"], 1)
            blob = str(result["errors"])
            self.assertNotIn(secret, blob)

    def test_batch_enable_empty_selection_of_disabled_only_mix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            s = self._svc(tmp)
            s.add_account_items([
                {"access_token": "n0", "status": "正常", "quota": 1},
                {"access_token": "lim", "status": "限流", "quota": 0},
            ])
            result = s.batch_update_status(["n0", "lim"], "正常")
            self.assertEqual(result["enabled"], 0)
            self.assertEqual(result["skipped"], 2)
            self.assertEqual(result["capped"], 0)


if __name__ == "__main__":
    unittest.main()
