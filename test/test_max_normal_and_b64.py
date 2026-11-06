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


class ManualNormalAndImportTests(unittest.TestCase):
    """Default: max_normal_accounts=0 (unlimited). Cap tests set a positive value."""

    def setUp(self) -> None:
        self._prev = config.data.get("max_normal_accounts")
        config.data["max_normal_accounts"] = 0

    def tearDown(self) -> None:
        if self._prev is None:
            config.data.pop("max_normal_accounts", None)
        else:
            config.data["max_normal_accounts"] = self._prev

    def _service(self, tmp: str) -> AccountService:
        return AccountService(JSONStorageBackend(Path(tmp) / "accounts.json"))

    def test_import_defaults_to_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = self._service(tmp)
            result = service.add_accounts([f"tok-{i}" for i in range(49)])
            self.assertEqual(result["added"], 49)
            items = service.list_accounts()
            self.assertEqual(sum(1 for a in items if a.get("status") == "正常"), 0)
            self.assertEqual(sum(1 for a in items if a.get("status") == "禁用"), 49)

    def test_reimport_preserves_existing_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = self._service(tmp)
            service.add_account_items(
                [{"access_token": f"keep-{i}", "status": "正常", "quota": 3} for i in range(5)]
            )
            keep = {a["access_token"] for a in service.list_accounts() if a.get("status") == "正常"}
            self.assertEqual(len(keep), 5)
            # re-import same tokens + new ones
            service.add_accounts([f"keep-{i}" for i in range(5)] + [f"new-{i}" for i in range(10)])
            normals = [a for a in service.list_accounts() if a.get("status") == "正常"]
            self.assertEqual(len(normals), 5)
            self.assertEqual({a["access_token"] for a in normals}, keep)
            self.assertTrue(
                all(a.get("status") == "禁用" for a in service.list_accounts() if a["access_token"].startswith("new-"))
            )

    def test_manual_enable_beyond_five_when_unlimited(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = self._service(tmp)
            service.add_account_items(
                [{"access_token": f"n{i}", "status": "正常", "quota": 1} for i in range(5)]
            )
            service.add_account_items([{"access_token": "d1", "status": "禁用", "quota": 5}])
            updated = service.update_account("d1", {"status": "正常"})
            self.assertEqual(updated["status"], "正常")
            normals = [a for a in service.list_accounts() if a.get("status") == "正常"]
            self.assertEqual(len(normals), 6)

    def test_positive_cap_still_blocks_manual_enable(self) -> None:
        config.data["max_normal_accounts"] = 5
        with tempfile.TemporaryDirectory() as tmp:
            service = self._service(tmp)
            service.add_account_items(
                [{"access_token": f"n{i}", "status": "正常", "quota": 1} for i in range(5)]
            )
            service.add_account_items([{"access_token": "d1", "status": "禁用", "quota": 5}])
            updated = service.update_account("d1", {"status": "正常"})
            self.assertEqual(updated["status"], "禁用")

    def test_concurrent_import_all_disabled_when_unlimited(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = self._service(tmp)

            def add_batch(start: int) -> None:
                service.add_accounts([f"c-{start}-{i}" for i in range(20)])

            with ThreadPoolExecutor(max_workers=4) as pool:
                list(pool.map(add_batch, range(4)))
            items = service.list_accounts()
            self.assertEqual(len(items), 80)
            self.assertEqual(sum(1 for a in items if a.get("status") == "禁用"), 80)

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
                    {"access_token": "n1", "status": "正常", "refresh_token": "rt-n"},
                    {"access_token": "d1", "status": "禁用", "refresh_token": "rt-d"},
                ]
            )
            service._token_needs_refresh = lambda token, force=False: True  # type: ignore[method-assign]
            tokens = service.list_expiring_access_tokens()
            self.assertEqual(tokens, ["n1"])

    def test_rate_limit_restore_allowed_when_unlimited(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = self._service(tmp)
            service.add_account_items(
                [{"access_token": f"n{i}", "status": "正常", "quota": 3} for i in range(5)]
            )
            service.add_account_items([{"access_token": "lim", "status": "限流", "quota": 0}])
            updated = service.update_account("lim", {"status": "正常", "quota": 20})
            self.assertEqual(updated["status"], "正常")
            self.assertEqual(updated["quota"], 20)

    def test_rate_limit_restore_respects_positive_cap(self) -> None:
        config.data["max_normal_accounts"] = 5
        with tempfile.TemporaryDirectory() as tmp:
            service = self._service(tmp)
            service.add_account_items(
                [{"access_token": f"n{i}", "status": "正常", "quota": 3} for i in range(5)]
            )
            service.add_account_items([{"access_token": "lim", "status": "限流", "quota": 0}])
            updated = service.update_account("lim", {"status": "正常", "quota": 20})
            self.assertEqual(updated["status"], "限流")
            self.assertEqual(updated["quota"], 20)


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
