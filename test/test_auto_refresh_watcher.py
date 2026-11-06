from __future__ import annotations

import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("CHATGPT2API_AUTH_KEY", "test-auth")

from api.support import start_limited_account_watcher
from services.account_service import AccountService
from services.config import config
from services.storage.json_storage import JSONStorageBackend


class AutoRefreshWatcherTests(unittest.TestCase):
    def setUp(self) -> None:
        self._prev_auto = config.data.get("auto_refresh_accounts_enabled")
        self._prev_interval = config.data.get("refresh_account_interval_minute")
        config.data["refresh_account_interval_minute"] = 60  # long wait; we stop quickly

    def tearDown(self) -> None:
        if self._prev_auto is None:
            config.data.pop("auto_refresh_accounts_enabled", None)
        else:
            config.data["auto_refresh_accounts_enabled"] = self._prev_auto
        if self._prev_interval is None:
            config.data.pop("refresh_account_interval_minute", None)
        else:
            config.data["refresh_account_interval_minute"] = self._prev_interval

    def _run_one_cycle(self, *, auto_enabled: bool) -> tuple[list, list]:
        config.data["auto_refresh_accounts_enabled"] = auto_enabled
        stop = threading.Event()
        refresh_calls: list = []
        keepalive_calls: list = []

        def fake_refresh(tokens, *args, **kwargs):
            refresh_calls.append(list(tokens))
            return {"refreshed": len(tokens), "errors": [], "items": []}

        def fake_keepalive(tokens):
            keepalive_calls.append(list(tokens))
            return {"refreshed": 0, "errors": [], "items": []}

        with patch("api.support.account_service") as svc:
            svc.list_limited_tokens.return_value = ["lim"]
            svc.list_normal_tokens.return_value = ["n1", "n2"]
            svc.list_expiring_access_tokens.return_value = ["exp"]
            svc.list_refresh_token_keepalive_tokens.return_value = ["ka1"]
            svc.refresh_accounts.side_effect = fake_refresh
            svc.keepalive_refresh_tokens.side_effect = fake_keepalive

            # Make wait return immediately once so worker exits after first cycle+wait
            original_wait = stop.wait

            def wait_once(timeout=None):
                stop.set()
                return True

            stop.wait = wait_once  # type: ignore[method-assign]
            thread = start_limited_account_watcher(stop)
            thread.join(timeout=5)
            self.assertFalse(thread.is_alive())
        return refresh_calls, keepalive_calls

    def test_disabled_skips_info_refresh_but_runs_keepalive(self) -> None:
        refresh_calls, keepalive_calls = self._run_one_cycle(auto_enabled=False)
        self.assertEqual(refresh_calls, [])
        self.assertEqual(keepalive_calls, [["ka1"]])

    def test_enabled_runs_info_refresh_and_keepalive(self) -> None:
        refresh_calls, keepalive_calls = self._run_one_cycle(auto_enabled=True)
        self.assertEqual(len(refresh_calls), 1)
        self.assertEqual(refresh_calls[0], ["lim", "n1", "n2", "exp"])
        # ka1 is not in expiring set, so keepalive still runs for ka1
        self.assertEqual(keepalive_calls, [["ka1"]])


class ImageConcurrencyIndependentTests(unittest.TestCase):
    """Closing auto info refresh must not change image_account_concurrency semantics."""

    def setUp(self) -> None:
        self._prev_c = config.data.get("image_account_concurrency")
        self._prev_auto = config.data.get("auto_refresh_accounts_enabled")
        config.data["image_account_concurrency"] = 3
        config.data["auto_refresh_accounts_enabled"] = False

    def tearDown(self) -> None:
        if self._prev_c is None:
            config.data.pop("image_account_concurrency", None)
        else:
            config.data["image_account_concurrency"] = self._prev_c
        if self._prev_auto is None:
            config.data.pop("auto_refresh_accounts_enabled", None)
        else:
            config.data["auto_refresh_accounts_enabled"] = self._prev_auto

    def test_inflight_cap_three_then_other_accounts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            svc = AccountService(JSONStorageBackend(Path(tmp) / "a.json"))
            svc.add_account_items([
                {"access_token": "a", "status": "正常", "quota": 10},
                {"access_token": "b", "status": "正常", "quota": 10},
            ])
            got = [svc._acquire_next_candidate_token() for _ in range(3)]
            self.assertEqual(svc._image_inflight.get("a", 0) + svc._image_inflight.get("b", 0), 3)
            # each account may hold up to 3 concurrent; round-robin fills both
            self.assertLessEqual(svc._image_inflight.get("a", 0), 3)
            self.assertLessEqual(svc._image_inflight.get("b", 0), 3)
            # fourth acquire still ok if under per-account cap across pool
            fourth = svc._acquire_next_candidate_token()
            self.assertIn(fourth, {"a", "b"})
            self.assertEqual(sum(svc._image_inflight.values()), 4)
            # fill to 3 each = 6
            while sum(svc._image_inflight.values()) < 6:
                svc._acquire_next_candidate_token()
            self.assertEqual(svc._image_inflight.get("a"), 3)
            self.assertEqual(svc._image_inflight.get("b"), 3)
            # restart semantics: new service instance clears memory counters
            svc2 = AccountService(JSONStorageBackend(Path(tmp) / "a.json"))
            self.assertEqual(svc2._image_inflight, {})
            self.assertEqual(svc2._index, 0)


class KeepaliveDueTests(unittest.TestCase):
    def setUp(self) -> None:
        self._prev = config.data.get("auto_refresh_accounts_enabled")
        config.data["auto_refresh_accounts_enabled"] = False

    def tearDown(self) -> None:
        if self._prev is None:
            config.data.pop("auto_refresh_accounts_enabled", None)
        else:
            config.data["auto_refresh_accounts_enabled"] = self._prev

    def test_disabled_accounts_excluded_from_keepalive_list(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            svc = AccountService(JSONStorageBackend(Path(tmp) / "a.json"))
            svc.add_account_items([
                {"access_token": "n1", "status": "正常", "refresh_token": "rt-n", "quota": 1,
                 "last_token_refresh_at": "2000-01-01T00:00:00+00:00"},
                {"access_token": "d1", "status": "禁用", "refresh_token": "rt-d", "quota": 1,
                 "last_token_refresh_at": "2000-01-01T00:00:00+00:00"},
            ])
            due = svc.list_refresh_token_keepalive_tokens()
            self.assertIn("n1", due)
            self.assertNotIn("d1", due)



class KeepaliveOAuthCallTests(unittest.TestCase):
    def setUp(self) -> None:
        self._prev = config.data.get("auto_refresh_accounts_enabled")
        config.data["auto_refresh_accounts_enabled"] = False

    def tearDown(self) -> None:
        if self._prev is None:
            config.data.pop("auto_refresh_accounts_enabled", None)
        else:
            config.data["auto_refresh_accounts_enabled"] = self._prev

    def test_keepalive_calls_oauth_only_when_due(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            svc = AccountService(JSONStorageBackend(Path(tmp) / "a.json"))
            svc.add_account_items([
                {"access_token": "n1", "status": "正常", "refresh_token": "rt-n", "quota": 1},
            ])
            called = []

            def fake_refresh(token, force=False, event=""):
                called.append({"token": token, "force": force, "event": event})
                return token

            svc.refresh_access_token = fake_refresh  # type: ignore[method-assign]
            # not due -> empty list -> no call
            with patch.object(svc, "list_refresh_token_keepalive_tokens", return_value=[]):
                result = svc.keepalive_refresh_tokens([])
                self.assertEqual(result["refreshed"], 0)
                self.assertEqual(called, [])
            # due token -> OAuth refresh path via refresh_access_token(force=True)
            result = svc.keepalive_refresh_tokens(["n1"])
            self.assertEqual(len(called), 1)
            self.assertTrue(called[0]["force"])
            self.assertEqual(called[0]["event"], "refresh_token_keepalive")

    def test_manual_refresh_still_works_when_auto_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            svc = AccountService(JSONStorageBackend(Path(tmp) / "a.json"))
            svc.add_account_items([{"access_token": "n1", "status": "正常", "quota": 2}])
            hits = []

            def fake_fetch(token, event="fetch_remote_info", defer_invalid_removal=True):
                hits.append(event)
                return svc.get_account(token)

            svc.fetch_remote_info = fake_fetch  # type: ignore[method-assign]
            result = svc.refresh_accounts(["n1"])
            self.assertEqual(result["refreshed"], 1)
            self.assertEqual(hits, ["refresh_accounts"])



if __name__ == "__main__":
    unittest.main()
