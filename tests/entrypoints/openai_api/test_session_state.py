# Tests for the file-backed session store that powers cross-request
# voice continuity for TTS.
import importlib.util
import os
import sys
import time
import types

import pytest


def _load_session_state():
    existing = sys.modules.get("vllm_omni.entrypoints.openai.session_state")
    if existing is not None:
        parent = sys.modules.get("vllm_omni.entrypoints.openai")
        if parent is not None:
            parent.session_state = existing
            entrypoints = sys.modules.get("vllm_omni.entrypoints")
            if entrypoints is not None:
                entrypoints.openai = parent
        return existing

    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, "..", "..", ".."))
    ss_path = os.path.join(root, "vllm_omni", "entrypoints", "openai", "session_state.py")
    for pkg in (
        "vllm_omni",
        "vllm_omni.entrypoints",
        "vllm_omni.entrypoints.openai",
    ):
        if pkg not in sys.modules:
            m = types.ModuleType(pkg)
            m.__path__ = []
            sys.modules[pkg] = m
        if "." in pkg:
            parent_name, child_name = pkg.rsplit(".", 1)
            setattr(sys.modules[parent_name], child_name, sys.modules[pkg])
    spec = importlib.util.spec_from_file_location("vllm_omni.entrypoints.openai.session_state", ss_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    sys.modules["vllm_omni.entrypoints.openai"].session_state = mod
    return mod


session_state = _load_session_state()

FileSessionStore = session_state.FileSessionStore
compute_signature = session_state.compute_signature
drop_request_binding = session_state.drop_request_binding
get_session_store = session_state.get_session_store
on_request_complete = session_state.on_request_complete
register_request_session = session_state.register_request_session

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def store(tmp_path):
    """Fresh store backed by a temp directory, no LRU/TTL kick-in."""
    return FileSessionStore(
        root=str(tmp_path / "session_state"),
        max_sessions=1024,
        idle_ttl_s=60.0,
        max_prefix_tokens=1000,
    )


# ---------- compute_signature ------------------------------------------------


class TestComputeSignature:
    def test_returns_stable_16_hex(self):
        sig = compute_signature(voice="maya_conv", instruct="friendly")
        assert isinstance(sig, str)
        assert len(sig) == 16
        assert all(c in "0123456789abcdef" for c in sig)

    def test_is_deterministic(self):
        a = compute_signature(voice="maya_conv", instruct="friendly")
        b = compute_signature(voice="maya_conv", instruct="friendly")
        assert a == b

    def test_changes_with_voice(self):
        a = compute_signature(voice="maya_conv", instruct="friendly")
        b = compute_signature(voice="maya_warm", instruct="friendly")
        assert a != b

    def test_changes_with_instructions(self):
        a = compute_signature(voice="maya_conv", instruct="friendly")
        b = compute_signature(voice="maya_conv", instruct="serious")
        assert a != b

    def test_handles_none_inputs(self):
        # Both Nones should still produce a valid signature (not raise).
        sig = compute_signature(voice=None, instruct=None)
        assert len(sig) == 16


# ---------- FileSessionStore: basic lookup / create --------------------------


class TestSessionStoreLookup:
    def test_empty_session_id_returns_empty(self, store):
        assert store.get_prior_codec_for_request("", "sig") == []
        assert store.get_prior_texts("") == []

    def test_new_session_creates_empty_entry(self, store):
        frames = store.get_prior_codec_for_request("s1", "sigA")
        assert frames == []
        assert store.get_prior_texts("s1") == []
        # Signature persists
        state = store.debug_state("s1")
        assert state is not None
        assert state["signature"] == "sigA"
        assert state["codec_frames"] == 0

    def test_existing_session_hit_returns_accumulated_frames(self, store):
        store.get_prior_codec_for_request("s1", "sigA")
        store.append_codec_frames("s1", [[1, 2, 3], [4, 5, 6]])
        frames = store.get_prior_codec_for_request("s1", "sigA")
        assert frames == [[1, 2, 3], [4, 5, 6]]

    def test_prior_texts_round_trip(self, store):
        store.get_prior_codec_for_request("s1", "sigA")
        store.append_prior_text("s1", "hello")
        store.append_prior_text("s1", "world")
        assert store.get_prior_texts("s1") == ["hello", "world"]


# ---------- FileSessionStore: signature gate ---------------------------------


class TestSignatureGate:
    def test_signature_change_resets_state(self, store):
        store.get_prior_codec_for_request("s1", "sigA")
        store.append_codec_frames("s1", [[1, 2]])
        store.append_prior_text("s1", "hello")

        # Different signature -> reset
        frames = store.get_prior_codec_for_request("s1", "sigB")
        assert frames == []
        assert store.get_prior_texts("s1") == []

        # After reset, new signature is recorded
        state = store.debug_state("s1")
        assert state is not None
        assert state["signature"] == "sigB"
        assert state["codec_frames"] == 0

    def test_same_signature_preserves_state(self, store):
        store.get_prior_codec_for_request("s1", "sigA")
        store.append_codec_frames("s1", [[1, 2]])

        frames = store.get_prior_codec_for_request("s1", "sigA")
        assert frames == [[1, 2]]

    def test_cross_signature_context_can_be_preserved(self, tmp_path):
        store = FileSessionStore(
            root=str(tmp_path / "s"),
            allow_cross_signature_context=True,
        )
        store.get_prior_codec_for_request("s1", "sigA")
        store.append_prior_text("s1", "warm first")
        store.append_codec_frames("s1", [[1], [2]])

        context = store.get_prior_context_for_request("s1", "sigB")

        assert context.codec_tokens == [[1], [2]]
        assert context.prior_texts == ["warm first"]
        assert context.selected_turns == 1
        state = store.debug_state("s1")
        assert state is not None
        assert state["signature"] == "sigB"


# ---------- FileSessionStore: cap reset --------------------------------------


class TestCapReset:
    def test_cap_reset_when_projected_exceeds_max(self, tmp_path):
        # Tight cap so we can trigger the reset without thousands of frames.
        store = FileSessionStore(
            root=str(tmp_path / "s"),
            max_prefix_tokens=50,
        )
        store.get_prior_codec_for_request("s1", "sigA")
        store.append_codec_frames("s1", [[0]] * 40)  # 40 frames

        # new_prompt_est pushes projected past the 50-frame cap -> reset
        frames = store.get_prior_codec_for_request("s1", "sigA", new_prompt_est=20)
        assert frames == []

        # The entry was reset in-place: subsequent read for same signature
        # returns empty and does not recreate the frames.
        state = store.debug_state("s1")
        assert state is not None
        assert state["signature"] == "sigA"
        assert state["codec_frames"] == 0

    def test_cap_overflow_trims_oldest_turns_before_resetting(self, tmp_path):
        store = FileSessionStore(
            root=str(tmp_path / "s"),
            max_prefix_tokens=65,
            max_context_turns=0,
        )
        store.get_prior_codec_for_request("s1", "sigA")
        store.append_prior_text("s1", "first")
        store.append_codec_frames("s1", [[1]] * 20)
        store.append_prior_text("s1", "second")
        store.append_codec_frames("s1", [[2]] * 20)
        store.append_prior_text("s1", "third")
        store.append_codec_frames("s1", [[3]] * 20)

        frames = store.get_prior_codec_for_request("s1", "sigA", new_prompt_est=20)

        assert frames == [[2]] * 20 + [[3]] * 20
        assert store.get_prior_texts("s1") == ["second", "third"]


class TestContextWindow:
    def test_default_context_window_returns_last_complete_turn(self, tmp_path):
        store = FileSessionStore(root=str(tmp_path / "s"))
        store.get_prior_codec_for_request("s1", "sigA")
        store.append_prior_text("s1", "first")
        store.append_codec_frames("s1", [[1], [1]])
        store.append_prior_text("s1", "second")
        store.append_codec_frames("s1", [[2], [2], [2]])

        context = store.get_prior_context_for_request("s1", "sigA")

        assert context.codec_tokens == [[2], [2], [2]]
        assert context.prior_texts == ["second"]
        assert context.selected_turns == 1
        assert context.total_complete_turns == 2

    def test_zero_context_window_preserves_full_rolling_context(self, tmp_path):
        store = FileSessionStore(
            root=str(tmp_path / "s"),
            max_context_turns=0,
        )
        store.get_prior_codec_for_request("s1", "sigA")
        store.append_prior_text("s1", "first")
        store.append_codec_frames("s1", [[1], [1]])
        store.append_prior_text("s1", "second")
        store.append_codec_frames("s1", [[2]])

        context = store.get_prior_context_for_request("s1", "sigA")

        assert context.codec_tokens == [[1], [1], [2]]
        assert context.prior_texts == ["first", "second"]
        assert context.selected_turns == 2
        assert context.total_complete_turns == 2

    def test_context_window_can_select_more_than_one_recent_turn(self, tmp_path):
        store = FileSessionStore(
            root=str(tmp_path / "s"),
            max_context_turns=2,
        )
        store.get_prior_codec_for_request("s1", "sigA")
        for text, frame in (("first", 1), ("second", 2), ("third", 3)):
            store.append_prior_text("s1", text)
            store.append_codec_frames("s1", [[frame]])

        context = store.get_prior_context_for_request("s1", "sigA")

        assert context.codec_tokens == [[2], [3]]
        assert context.prior_texts == ["second", "third"]
        assert context.selected_turns == 2
        assert context.total_complete_turns == 3

    def test_dangling_text_is_not_selected_without_matching_codec(self, tmp_path):
        store = FileSessionStore(
            root=str(tmp_path / "s"),
            max_context_turns=2,
        )
        store.get_prior_codec_for_request("s1", "sigA")
        store.append_prior_text("s1", "complete")
        store.append_codec_frames("s1", [[1]])
        store.append_prior_text("s1", "in flight")

        context = store.get_prior_context_for_request("s1", "sigA")

        assert context.codec_tokens == [[1]]
        assert context.prior_texts == ["complete"]
        assert context.selected_turns == 1
        assert context.total_complete_turns == 1


class TestContextRefresh:
    def test_context_refresh_forces_cold_turn_after_configured_hits(self, tmp_path):
        store = FileSessionStore(
            root=str(tmp_path / "s"),
            max_context_turns=1,
            max_context_uses=2,
        )
        store.get_prior_codec_for_request("s1", "sigA")
        store.append_prior_text("s1", "one")
        store.append_codec_frames("s1", [[1]])

        first = store.get_prior_context_for_request("s1", "sigA")
        second = store.get_prior_context_for_request("s1", "sigA")
        refreshed = store.get_prior_context_for_request("s1", "sigA")

        assert first.codec_tokens == [[1]]
        assert second.codec_tokens == [[1]]
        assert refreshed.codec_tokens == []
        state = store.debug_state("s1")
        assert state is not None
        assert state["codec_frames"] == 0
        assert state["prior_texts_count"] == 0
        assert state["context_uses_since_refresh"] == 0

    def test_context_refresh_disabled_by_default(self, tmp_path):
        store = FileSessionStore(
            root=str(tmp_path / "s"),
            max_context_turns=1,
            max_context_uses=0,
        )
        store.get_prior_codec_for_request("s1", "sigA")
        store.append_prior_text("s1", "one")
        store.append_codec_frames("s1", [[1]])

        for _ in range(5):
            context = store.get_prior_context_for_request("s1", "sigA")
            assert context.codec_tokens == [[1]]


class TestTurnReset:
    def test_anomalously_long_turn_resets_session(self, tmp_path):
        store = FileSessionStore(
            root=str(tmp_path / "s"),
            max_turn_frames=4,
        )
        store.get_prior_codec_for_request("s1", "sigA")
        store.append_prior_text("s1", "hello")

        store.append_codec_frames("s1", [[0]] * 5)

        state = store.debug_state("s1")
        assert state is not None
        assert state["signature"] == "sigA"
        assert state["codec_frames"] == 0
        assert state["prior_texts_count"] == 0

    def test_normal_turn_below_turn_cap_is_preserved(self, tmp_path):
        store = FileSessionStore(
            root=str(tmp_path / "s"),
            max_turn_frames=4,
        )
        store.get_prior_codec_for_request("s1", "sigA")

        store.append_codec_frames("s1", [[1], [2], [3], [4]])

        assert store.get_prior_codec_for_request("s1", "sigA") == [
            [1],
            [2],
            [3],
            [4],
        ]

    def test_reset_to_fresh_preserves_signature_and_clears_state(self, store):
        sig = compute_signature(voice="maya_conv", instruct="")
        store.get_prior_codec_for_request("s1", sig)
        store.append_prior_text("s1", "hello")
        store.append_codec_frames("s1", [[1, 2], [3, 4]])

        store.reset_to_fresh("s1", reason="EMPTY-RESET")

        state = store.debug_state("s1")
        assert state is not None
        assert state["signature"] == sig
        assert state["codec_frames"] == 0
        assert state["prior_texts_count"] == 0


# ---------- FileSessionStore: TTL + LRU eviction -----------------------------


class TestEviction:
    def test_idle_session_evicted_on_next_access(self, tmp_path):
        store = FileSessionStore(
            root=str(tmp_path / "s"),
            idle_ttl_s=0.1,
            eviction_check_interval_s=0.0,
        )
        store.get_prior_codec_for_request("old", "sigA")
        store.append_codec_frames("old", [[9, 9]])

        # Wait past TTL, then trigger scan via any request
        time.sleep(0.2)
        store.get_prior_codec_for_request("new", "sigB")

        # 'old' should have been evicted during the lazy scan
        assert store.debug_state("old") is None

    def test_lru_cap_evicts_oldest(self, tmp_path):
        store = FileSessionStore(
            root=str(tmp_path / "s"),
            max_sessions=2,
            idle_ttl_s=1000.0,  # avoid TTL in this test
            eviction_check_interval_s=0.0,
        )
        store.get_prior_codec_for_request("a", "s")
        store.get_prior_codec_for_request("b", "s")
        # Touch 'b' so it is newer than 'a'
        store.append_codec_frames("b", [[0]])
        time.sleep(0.01)

        # Eviction is lazy: created entries count toward the cap on the
        # NEXT access. Two accesses after creating 'c' are therefore
        # needed to observe the LRU drop.
        store.get_prior_codec_for_request("c", "s")
        store.get_prior_codec_for_request("c", "s")

        surviving = {name for name in ("a", "b", "c") if store.debug_state(name) is not None}
        # 'a' is the oldest and should go; 'b' and 'c' remain
        assert "a" not in surviving
        assert "b" in surviving
        assert "c" in surviving

    def test_eviction_scan_is_throttled_by_interval(self, tmp_path, monkeypatch):
        store = FileSessionStore(
            root=str(tmp_path / "s"),
            max_sessions=10,
            idle_ttl_s=60.0,
            eviction_check_interval_s=10.0,
        )
        scan_calls = 0
        real_listdir = os.listdir

        def listdir_spy(path: str) -> list[str]:
            nonlocal scan_calls
            scan_calls += 1
            return real_listdir(path)

        monkeypatch.setattr(
            "vllm_omni.entrypoints.openai.session_state.os.listdir",
            listdir_spy,
        )

        store.get_prior_codec_for_request("s1", "sigA")
        store.get_prior_codec_for_request("s1", "sigA")
        # One eviction scan happens (which itself lists twice: once for TTL,
        # and once again for LRU trimming). The second request is skipped by
        # the interval gate.
        assert scan_calls == 2


# ---------- FileSessionStore: atomic write + cross-process semantics ---------


class TestAtomicWrite:
    def test_save_is_atomic_via_tmp_rename(self, store, monkeypatch):
        calls: list[tuple[str, str]] = []
        real_replace = os.replace

        def record(src, dst):
            calls.append((os.path.basename(src), os.path.basename(dst)))
            return real_replace(src, dst)

        monkeypatch.setattr("vllm_omni.entrypoints.openai.session_state.os.replace", record)
        store.get_prior_codec_for_request("s1", "sigA")

        # Every save goes via ``*.tmp -> final`` atomic rename.
        assert calls, "expected at least one atomic rename"
        for tmp_name, dst_name in calls:
            assert tmp_name.endswith(".tmp")
            assert not dst_name.endswith(".tmp")

    def test_corrupted_pickle_is_tolerated(self, store):
        # A ``get`` against a corrupted pickle should not raise; it should
        # recover by creating a fresh session.
        path = store._session_path("s1")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as fp:
            fp.write(b"garbage-not-a-pickle")
        frames = store.get_prior_codec_for_request("s1", "sigA")
        assert frames == []


# ---------- FileSessionStore: reset + debug_state ----------------------------


class TestResetAndDebug:
    def test_reset_removes_session_file(self, store):
        store.get_prior_codec_for_request("s1", "sigA")
        store.append_codec_frames("s1", [[1, 2]])
        assert store.debug_state("s1") is not None

        store.reset("s1")
        assert store.debug_state("s1") is None

    def test_debug_state_fields(self, store):
        store.get_prior_codec_for_request("s1", "sigA")
        store.append_prior_text("s1", "hi")
        store.append_codec_frames("s1", [[1], [2], [3]])
        state = store.debug_state("s1")
        assert state is not None
        assert state["signature"] == "sigA"
        assert state["codec_frames"] == 3
        assert state["prior_texts_count"] == 1
        assert state["prompt_lens_total_chars"] == 2
        assert state["last_activity"] is not None


# ---------- request <-> session binding + on_request_complete ----------------


class TestRequestBinding:
    def test_register_and_pop(self, store):
        store.register_request("req-1", "sess-A")
        assert store.pop_request_session("req-1") == "sess-A"
        # Second pop returns None (consumed)
        assert store.pop_request_session("req-1") is None

    def test_pop_unknown_returns_none(self, store):
        assert store.pop_request_session("never-registered") is None


class TestConvenienceWrappers:
    """Verify the module-level wrappers that other subsystems import."""

    def test_register_and_on_complete_flow(self, tmp_path, monkeypatch):
        store = FileSessionStore(root=str(tmp_path / "s"))
        monkeypatch.setattr("vllm_omni.entrypoints.openai.session_state._SINGLETON", store)
        store.get_prior_codec_for_request("sess-A", "sigA")

        register_request_session("req-1", "sess-A")
        on_request_complete("req-1", [[10, 11], [12, 13]])

        assert store.get_prior_codec_for_request("sess-A", "sigA") == [
            [10, 11],
            [12, 13],
        ]
        # Binding has been consumed
        assert store.pop_request_session("req-1") is None

    def test_on_complete_empty_frames_clears_bound_session(self, tmp_path, monkeypatch):
        store = FileSessionStore(root=str(tmp_path / "s"))
        monkeypatch.setattr("vllm_omni.entrypoints.openai.session_state._SINGLETON", store)
        store.get_prior_codec_for_request("sess-A", "sigA")
        store.append_prior_text("sess-A", "hello")
        store.append_codec_frames("sess-A", [[1, 2], [3, 4]])
        register_request_session("req-empty", "sess-A")

        on_request_complete("req-empty", [])

        state = store.debug_state("sess-A")
        assert state is not None
        assert state["signature"] == "sigA"
        assert state["codec_frames"] == 0
        assert state["prior_texts_count"] == 0
        assert store.pop_request_session("req-empty") is None

    def test_on_complete_drops_frames_when_no_binding(self, tmp_path, monkeypatch):
        store = FileSessionStore(root=str(tmp_path / "s"))
        monkeypatch.setattr("vllm_omni.entrypoints.openai.session_state._SINGLETON", store)
        # No register_request_session, so on_request_complete is a no-op
        on_request_complete("orphan-req", [[1, 2]])
        # No session exists, so no state created
        assert store.debug_state("sess-A") is None

    def test_drop_request_binding_removes_file(self, tmp_path, monkeypatch):
        store = FileSessionStore(root=str(tmp_path / "s"))
        monkeypatch.setattr("vllm_omni.entrypoints.openai.session_state._SINGLETON", store)
        register_request_session("req-1", "sess-A")
        drop_request_binding("req-1")
        assert store.pop_request_session("req-1") is None


class TestGetSessionStoreSingleton:
    def test_returns_singleton(self, monkeypatch):
        monkeypatch.setattr("vllm_omni.entrypoints.openai.session_state._SINGLETON", None)
        a = get_session_store()
        b = get_session_store()
        assert a is b

    def test_respects_env_store_dir(self, tmp_path, monkeypatch):
        # DEFAULT_STORE_DIR is bound at import time, but a FileSessionStore
        # constructed with an explicit root should honour it.
        root = str(tmp_path / "custom_root")
        store = FileSessionStore(root=root)
        store.get_prior_codec_for_request("s1", "sigA")
        assert os.path.isdir(os.path.join(root, "sessions"))
        assert os.path.isdir(os.path.join(root, "req_bindings"))


# ---------- FileSessionStore: prompt_lens / text-aware cap projection -------


class TestPromptLensTracking:
    """``prompt_lens`` must track prior-text char counts so the cap-reset
    projection can include the text dimension in addition to codec frames.

    Before the fix, ``prompt_lens`` was described in the design docs but
    never actually populated — a session with very long prior text could
    blow past ``max_prefix_tokens`` without a CAP-RESET ever firing.
    """

    def test_append_prior_text_tracks_lengths(self, store):
        store.get_prior_codec_for_request("s1", "sigA")
        store.append_prior_text("s1", "hello")
        store.append_prior_text("s1", "world!")
        state = store.debug_state("s1")
        assert state is not None
        assert state["prompt_lens_total_chars"] == len("hello") + len("world!")

    def test_cap_reset_counts_prior_text_length(self, tmp_path):
        # Tight cap. With char-to-token ratio 3 (see _CHARS_PER_TOKEN),
        # 600 chars ≈ 200 prior-text tokens — exceeds a 150-token cap even
        # with zero codec frames.
        store = FileSessionStore(
            root=str(tmp_path / "s"),
            max_prefix_tokens=150,
        )
        store.get_prior_codec_for_request("s1", "sigA")
        store.append_prior_text("s1", "x" * 600)

        # No codec frames, small new-prompt estimate, but prior text alone
        # projects over the cap -> reset.
        frames = store.get_prior_codec_for_request("s1", "sigA", new_prompt_est=10)
        assert frames == []
        state = store.debug_state("s1")
        assert state is not None
        assert state["prior_texts_count"] == 0
        assert state["prompt_lens_total_chars"] == 0

    def test_short_text_below_cap_is_preserved(self, tmp_path):
        # Same tight cap as above, but prior text is far below the cap.
        store = FileSessionStore(
            root=str(tmp_path / "s"),
            max_prefix_tokens=150,
        )
        store.get_prior_codec_for_request("s1", "sigA")
        store.append_prior_text("s1", "short")
        frames = store.get_prior_codec_for_request("s1", "sigA", new_prompt_est=10)
        assert frames == []  # no codec captured yet
        state = store.debug_state("s1")
        assert state is not None
        assert state["prior_texts_count"] == 1


# ---------- FileSessionStore: eviction lock ---------------------------------


class TestEvictionLock:
    """Concurrent eviction scans must not race each other.

    Before the fix, ``_maybe_evict_stale`` scanned the sessions directory
    without any cross-process serialisation; two workers could both
    observe and unlink the same file, or split a partial LRU sort.
    """

    def test_concurrent_evictions_do_not_raise(self, tmp_path):
        import threading

        store = FileSessionStore(
            root=str(tmp_path / "s"),
            max_sessions=5,
            idle_ttl_s=0.05,
            eviction_check_interval_s=0.0,
        )
        # Seed several sessions.
        for sid in ("a", "b", "c", "d", "e", "f", "g"):
            store.get_prior_codec_for_request(sid, "sig")
        time.sleep(0.1)  # make them all past TTL

        errors: list[BaseException] = []

        def scan():
            try:
                # Trigger a fresh access so eviction runs.
                store.get_prior_codec_for_request("scan", "sig")
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=scan) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        # All seeded sessions are gone (TTL), scanner-created "scan"
        # remains at worst, bounded by max_sessions=5.
        alive = sum(1 for sid in ("a", "b", "c", "d", "e", "f", "g") if store.debug_state(sid) is not None)
        assert alive == 0

    def test_eviction_lock_file_is_created_under_sessions_dir(self, tmp_path):
        store = FileSessionStore(
            root=str(tmp_path / "s"),
            idle_ttl_s=0.05,
            eviction_check_interval_s=0.0,
        )
        store.get_prior_codec_for_request("s1", "sig")
        time.sleep(0.1)
        # Force the eviction path once.
        store.get_prior_codec_for_request("s2", "sig")
        lock_path = os.path.join(str(tmp_path / "s"), "sessions", ".eviction.lock")
        assert os.path.isfile(lock_path)
