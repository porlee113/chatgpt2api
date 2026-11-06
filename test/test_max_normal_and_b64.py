from __future__ import annotations

import base64
import os
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("CHATGPT2API_AUTH_KEY", "test-auth")

from services.account_service import AccountService
from services.config import config
from services.protocol import conversation as conversation_mod
from services.storage.json_storage import JSONStorageBackend


class MaxNormalAccountTests(unittest.TestCase):
    def setUp(self) -> None:
        self._prev = config.data.get("max_normal_accounts")
        config.data["max_normal_accounts"] = 5

    def tearDown(self) -> None:
        if self._prev is None:
            config.data.pop("max_normal_accounts", None)
        else:
            config.data["max_normal_accounts"] = self._prev

    def _service(self, tmp: str) -> AccountService:
        return AccountService(JSONStorageBackend(Path(tmp) / "accounts.json"))

    def test_import_fills_only_five_normal_slots(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = self._service(tmp)
            tokens = [f"tok-{i}" for i in range(49)]
            result = service.add_accounts(tokens)
            self.assertEqual(result["added"], 49)
            items = service.list_accounts()
            normals = [a for a in items if a.get("status") == "正常"]
            disabled = [a for a in items if a.get("status") == "禁用"]
            self.assertEqual(len(normals), 5)
            self.assertEqual(len(disabled), 44)

    def test_import_does_not_replace_existing_normals(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = self._service(tmp)
            service.add_account_items(
                [{"access_token": f"keep-{i}", "status": "正常", "quota": 3} for i in range(5)]
            )
            keep = {a["access_token"] for a in service.list_accounts() if a.get("status") == "正常"}
            self.assertEqual(len(keep), 5)
            service.add_accounts([f"new-{i}" for i in range(10)])
            normals = [a for a in service.list_accounts() if a.get("status") == "正常"]
            self.assertEqual(len(normals), 5)
            self.assertEqual({a["access_token"] for a in normals}, keep)
            self.assertTrue(all(a.get("status") == "禁用" for a in service.list_accounts() if a["access_token"].startswith("new-")))

    def test_concurrent_import_never_exceeds_cap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = self._service(tmp)

            def add_batch(start: int) -> None:
                service.add_accounts([f"c-{start}-{i}" for i in range(20)])

            with ThreadPoolExecutor(max_workers=4) as pool:
                list(pool.map(add_batch, range(4)))
            normals = [a for a in service.list_accounts() if a.get("status") == "正常"]
            self.assertLessEqual(len(normals), 5)
            self.assertEqual(len(service.list_accounts()), 80)

    def test_refresh_skips_disabled_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = self._service(tmp)
            service.add_account_items(
                [
                    {"access_token": "n1", "status": "正常", "quota": 2},
                    {"access_token": "d1", "status": "禁用", "quota": 2},
                ]
            )
            called: list[str] = []

            def fake_fetch(token, event="fetch_remote_info", defer_invalid_removal=True):
                called.append(token)
                return service.get_account(token)

            service.fetch_remote_info = fake_fetch  # type: ignore[method-assign]
            service.refresh_accounts(["n1", "d1"])
            self.assertEqual(called, ["n1"])

    def test_explicit_refresh_preserves_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = self._service(tmp)
            service.add_account_items([{"access_token": "d1", "status": "禁用", "quota": 0}])

            with patch(
                "services.openai_backend_api.OpenAIBackendAPI.get_user_info",
                return_value={"quota": 10, "status": "正常", "email": "a@b.c", "type": "plus"},
            ):
                # force include_disabled path
                service.refresh_access_token = lambda token, **kwargs: token  # type: ignore[method-assign]
                result = service.refresh_accounts(["d1"], include_disabled=True)
            account = service.get_account("d1")
            self.assertIsNotNone(account)
            self.assertEqual(account["status"], "禁用")
            self.assertEqual(account["quota"], 10)
            self.assertEqual(result["refreshed"], 1)

    def test_list_expiring_skips_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = self._service(tmp)
            service.add_account_items(
                [
                    {
                        "access_token": "n1",
                        "status": "正常",
                        "refresh_token": "rt-n",
                    },
                    {
                        "access_token": "d1",
                        "status": "禁用",
                        "refresh_token": "rt-d",
                    },
                ]
            )
            service._token_needs_refresh = lambda token, force=False: True  # type: ignore[method-assign]
            tokens = service.list_expiring_access_tokens()
            self.assertEqual(tokens, ["n1"])

    def test_rate_limit_restore_respects_cap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = self._service(tmp)
            service.add_account_items(
                [{"access_token": f"n{i}", "status": "正常", "quota": 3} for i in range(5)]
            )
            service.add_account_items([{"access_token": "lim", "status": "限流", "quota": 0}])
            # remote says restored
            updated = service.update_account("lim", {"status": "正常", "quota": 20})
            self.assertEqual(updated["status"], "限流")
            self.assertEqual(updated["quota"], 20)
            normals = [a for a in service.list_accounts() if a.get("status") == "正常"]
            self.assertEqual(len(normals), 5)

    def test_manual_enable_blocked_at_cap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = self._service(tmp)
            service.add_account_items(
                [{"access_token": f"n{i}", "status": "正常", "quota": 1} for i in range(5)]
            )
            service.add_account_items([{"access_token": "d1", "status": "禁用", "quota": 5}])
            updated = service.update_account("d1", {"status": "正常"})
            self.assertEqual(updated["status"], "禁用")

    def test_mark_image_result_restore_respects_cap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = self._service(tmp)
            service.add_account_items(
                [{"access_token": f"n{i}", "status": "正常", "quota": 2} for i in range(5)]
            )
            service.add_account_items([{"access_token": "lim", "status": "限流", "quota": 1}])
            updated = service.mark_image_result("lim", success=True)
            # success with quota 1→0 becomes 限流; use quota 2 path
            service.update_account("lim", {"status": "限流", "quota": 2})
            # bypass disabled preserve: set lim quota path via mark from 限流 with remaining
            # mark_image_result: success decrements; if was 限流 and quota>0 after → tries 正常
            service._accounts["lim"]["status"] = "限流"
            service._accounts["lim"]["quota"] = 2
            updated = service.mark_image_result("lim", success=True)
            self.assertIsNotNone(updated)
            # quota 1, would want 正常 but cap full → stay 限流
            self.assertEqual(updated["quota"], 1)
            self.assertEqual(updated["status"], "限流")


class B64NoSaveTests(unittest.TestCase):
    def test_b64_json_does_not_call_save(self) -> None:
        payload = base64.b64encode(b"fake-image-bytes").decode("ascii")
        with patch.object(conversation_mod, "save_image_bytes", side_effect=AssertionError("save should not run")) as mocked:
            result = conversation_mod.format_image_result(
                [{"b64_json": payload}],
                prompt="p",
                response_format="b64_json",
                base_url="https://example.test",
            )
            mocked.assert_not_called()
        self.assertEqual(len(result["data"]), 1)
        self.assertEqual(result["data"][0]["b64_json"], payload)
        self.assertNotIn("url", result["data"][0])

    def test_url_format_still_saves(self) -> None:
        payload = base64.b64encode(b"fake-image-bytes").decode("ascii")
        with patch.object(conversation_mod, "save_image_bytes", return_value="https://example.test/images/x.png") as mocked:
            result = conversation_mod.format_image_result(
                [{"b64_json": payload}],
                prompt="p",
                response_format="url",
                base_url="https://example.test",
            )
            mocked.assert_called_once()
        self.assertEqual(result["data"][0]["url"], "https://example.test/images/x.png")
        self.assertNotIn("b64_json", result["data"][0])


if __name__ == "__main__":
    unittest.main()
