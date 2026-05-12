"""Server-side session state for cross-request voice continuity.

This module provides :class:`FileSessionStore`, a small file-backed
per-session cache of prior-turn codec frames and text. It is used by the
``/v1/audio/speech`` endpoint to implement cross-turn voice continuity for
autoregressive TTS models (Qwen3-TTS today): when a client tags requests
with an ``X-Session-Id`` header, the server can inject prior-turn codec
frames into the talker's in-context-learning path so that successive turns
sound like continuations of the same voice, without re-vocalising prior
text.

Design notes
------------
vLLM-Omni runs the HTTP API server and the engine core in separate OS
processes. The API server builds per-request parameters and needs the
prior codec frames to inject via ``voice_clone_prompt``. The engine core's
chunk transfer adapter emits codec frames and, at request completion, needs
to append them to the session state. For the next request to see the prior
turn's codec, the state must therefore be shared across processes.

Rather than introducing a ``multiprocessing.Manager`` or a separate
daemon, this module uses a trivial filesystem-backed keyed index under
``${VLLM_OMNI_SESSION_STATE_DIR}`` (default ``/tmp/vllm_session_state``):

* One pickle file per session, written atomically via tmp-file + rename.
* An :mod:`fcntl` advisory lock (``flock``) per session, so concurrent
  writers serialise cleanly across processes.
* A tiny per-request binding file that maps vLLM request ids back to
  their originating session id, so the engine side can look up the
  session without passing it through the vLLM request plumbing.

Eviction policies (applied lazily on access):

* LRU: when ``len(store) > max_sessions``, oldest pickle files are removed.
* TTL: sessions untouched for more than ``idle_ttl_s`` are evicted.
* Context window: by default, only the most recent complete turn is
  selected for ICL; set ``VLLM_OMNI_SESSION_MAX_CONTEXT_TURNS=0`` to use
  the full rolling context.
* Refresh cadence: set ``VLLM_OMNI_SESSION_MAX_CONTEXT_USES`` to a positive
  integer to force one stateless refresh after that many consecutive
  continued turns. This bounds self-conditioning drift without quality gates.
* Voice/instruction changes: by default, the session resets when either
  changes. Set ``VLLM_OMNI_SESSION_ALLOW_CROSS_SIGNATURE_CONTEXT=true`` to
  keep the selected recent-turn context across those changes for cross-tone
  experiments.
* Overflow: when the selected codec/text prefix would exceed
  ``max_prefix_tokens`` on the next turn, the oldest complete turns are
  trimmed; if the prefix still cannot fit, the session is reset.
"""

from __future__ import annotations

import fcntl
import hashlib
import logging
import os
import pickle
import threading
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean (true/false), got {raw!r}")


DEFAULT_STORE_DIR = os.environ.get("VLLM_OMNI_SESSION_STATE_DIR", "/tmp/vllm_session_state")
DEFAULT_MAX_SESSIONS = int(os.environ.get("VLLM_OMNI_SESSION_MAX", "1024"))
DEFAULT_IDLE_TTL_S = float(os.environ.get("VLLM_OMNI_SESSION_TTL_S", "600"))
DEFAULT_MAX_PREFIX_TOKENS = int(os.environ.get("VLLM_OMNI_SESSION_MAX_PREFIX_TOKENS", "500"))
DEFAULT_MAX_TURN_FRAMES = int(os.environ.get("VLLM_OMNI_SESSION_MAX_TURN_FRAMES", "256"))
DEFAULT_MAX_CONTEXT_TURNS = int(os.environ.get("VLLM_OMNI_SESSION_MAX_CONTEXT_TURNS", "1"))
DEFAULT_MAX_CONTEXT_USES = int(os.environ.get("VLLM_OMNI_SESSION_MAX_CONTEXT_USES", "0"))
DEFAULT_ALLOW_CROSS_SIGNATURE_CONTEXT = _env_bool("VLLM_OMNI_SESSION_ALLOW_CROSS_SIGNATURE_CONTEXT", False)


def compute_signature(voice: str | None, instruct: str | None) -> str:
    """Stable short signature of the (voice, instruct) tuple.

    Sessions are reset whenever this signature changes between requests,
    so a client switching voice or instructions mid-session does not get
    cross-contamination from the previous voice's codec context.
    """
    payload = f"{voice or ''}||{instruct or ''}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _safe_session_key(session_id: str) -> str:
    """Map an opaque session id to a filesystem-safe fixed-length key."""
    return hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:32]


@dataclass(frozen=True)
class PriorContext:
    """Aligned text+codec context selected for an ICL request."""

    codec_tokens: list[list[int]]
    prior_texts: list[str]
    selected_turns: int
    total_complete_turns: int


class _FileLock:
    """Advisory exclusive lock via :func:`fcntl.flock`.

    Cross-process safe on Linux/macOS. The lock is held for the lifetime of
    the ``with`` block.
    """

    def __init__(self, path: str) -> None:
        self._path = path
        self._fp: Any = None

    def __enter__(self) -> _FileLock:
        os.makedirs(os.path.dirname(self._path), exist_ok=True)
        self._fp = open(self._path, "a+b")
        fcntl.flock(self._fp.fileno(), fcntl.LOCK_EX)
        return self

    def __exit__(self, *exc: Any) -> None:
        try:
            if self._fp is not None:
                fcntl.flock(self._fp.fileno(), fcntl.LOCK_UN)
                self._fp.close()
        finally:
            self._fp = None


class FileSessionStore:
    """Cross-process file-backed session store.

    Layout under ``root``:

    .. code-block:: text

        sessions/{safe_session_id}.pkl    # per-session state
        sessions/{safe_session_id}.lock   # per-session advisory lock
        req_bindings/{safe_request_id}    # request id -> session id binding

    Each per-session pickle holds::

        {
            "signature":   <sha256(voice, instruct)[:16]>,
            "codec_tokens": [[int, ...], ...],  # prior-turn codec frames
            "prior_texts": [str, ...],          # prior-turn plain text
            "codec_lens": [int, ...],           # per-turn frame counts
            "last_activity": <unix timestamp>,
        }
    """

    def __init__(
        self,
        *,
        root: str = DEFAULT_STORE_DIR,
        max_sessions: int = DEFAULT_MAX_SESSIONS,
        idle_ttl_s: float = DEFAULT_IDLE_TTL_S,
        max_prefix_tokens: int = DEFAULT_MAX_PREFIX_TOKENS,
        max_turn_frames: int = DEFAULT_MAX_TURN_FRAMES,
        max_context_turns: int = DEFAULT_MAX_CONTEXT_TURNS,
        eviction_check_interval_s: float = 2.5,
        max_context_uses: int = DEFAULT_MAX_CONTEXT_USES,
        allow_cross_signature_context: bool = DEFAULT_ALLOW_CROSS_SIGNATURE_CONTEXT,
    ) -> None:
        self.root = root
        self.max_sessions = int(max_sessions)
        self.idle_ttl_s = float(idle_ttl_s)
        self.max_prefix_tokens = int(max_prefix_tokens)
        self.max_turn_frames = int(max_turn_frames)
        self.eviction_check_interval_s = float(eviction_check_interval_s)
        self.max_context_turns = int(max_context_turns)
        self.max_context_uses = int(max_context_uses)
        self.allow_cross_signature_context = bool(allow_cross_signature_context)
        os.makedirs(os.path.join(self.root, "sessions"), exist_ok=True)
        os.makedirs(os.path.join(self.root, "req_bindings"), exist_ok=True)

    # ---- session entry paths -----------------------------------------

    def _session_path(self, session_id: str) -> str:
        return os.path.join(self.root, "sessions", _safe_session_key(session_id) + ".pkl")

    def _lock_path(self, session_id: str) -> str:
        return os.path.join(self.root, "sessions", _safe_session_key(session_id) + ".lock")

    # ---- load/save helpers -------------------------------------------

    def _load_entry(self, session_id: str) -> dict[str, Any] | None:
        path = self._session_path(session_id)
        try:
            with open(path, "rb") as fp:
                return pickle.load(fp)
        except (FileNotFoundError, EOFError, pickle.UnpicklingError):
            return None

    def _save_entry_atomic(self, session_id: str, entry: dict[str, Any]) -> None:
        path = self._session_path(session_id)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "wb") as fp:
            pickle.dump(entry, fp, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)

    def _fresh_entry(self, signature: str) -> dict[str, Any]:
        return {
            "signature": signature,
            "codec_tokens": [],
            "prior_texts": [],
            "codec_lens": [],
            # Char-count proxy for tokenised prior-text length, summed into
            # the cap projection so multi-turn sessions with long text
            # trigger CAP-RESET before exhausting ``max_prefix_tokens``.
            "prompt_lens": [],
            "context_uses_since_refresh": 0,
            "last_activity": time.time(),
        }

    # Chars per token for the cap projection's text contribution.
    # Conservative under-estimate so the cap is slightly eager; the
    # prior-codec dimension is the dominant contributor in practice.
    _CHARS_PER_TOKEN = 3

    def _projected_prefix_tokens(
        self,
        entry: dict[str, Any],
        *,
        new_prompt_est: int,
        max_context_turns: int | None = None,
    ) -> int:
        mct = self.max_context_turns if max_context_turns is None else int(max_context_turns)
        start, end, selected_turns = self._context_bounds(
            entry,
            max_context_turns=mct,
        )

        codec_lens = entry.get("codec_lens", [])
        prompt_lens = entry.get("prompt_lens", [])
        selected_codec_frames = sum(int(n) for n in codec_lens[start:end])
        selected_prompt_chars = sum(int(n) for n in prompt_lens[start:end])
        if selected_turns <= 0:
            # If there are no complete turns yet, still bound on all emitted
            # codec frames and any pending unmatched prior text.
            selected_codec_frames = sum(int(n) for n in codec_lens)
            selected_prompt_chars = sum(int(n) for n in prompt_lens)

        pending_prompt_tokens = selected_prompt_chars // self._CHARS_PER_TOKEN
        return selected_codec_frames + pending_prompt_tokens + int(new_prompt_est)

    def _context_bounds(
        self,
        entry: dict[str, Any],
        *,
        max_context_turns: int,
    ) -> tuple[int, int, int]:
        codec_lens = entry.get("codec_lens", [])
        prior_texts = entry.get("prior_texts", [])
        complete_turns = min(len(codec_lens), len(prior_texts))
        if complete_turns <= 0:
            return 0, 0, 0
        selected_turns = complete_turns if max_context_turns <= 0 else min(max_context_turns, complete_turns)
        start = complete_turns - selected_turns
        return start, complete_turns, selected_turns

    def _select_prior_context(
        self,
        entry: dict[str, Any],
        *,
        max_context_turns: int,
    ) -> PriorContext:
        codec_lens = entry.get("codec_lens", [])
        codec_tokens = entry.get("codec_tokens", [])
        prior_texts = entry.get("prior_texts", [])
        complete_turns = min(len(codec_lens), len(prior_texts))
        start, end, selected_turns = self._context_bounds(
            entry,
            max_context_turns=max_context_turns,
        )
        if selected_turns <= 0:
            if entry.get("codec_lens"):
                return PriorContext(
                    codec_tokens=codec_tokens,
                    prior_texts=[],
                    selected_turns=0,
                    total_complete_turns=complete_turns,
                )
            return PriorContext(
                codec_tokens=[],
                prior_texts=prior_texts,
                selected_turns=0,
                total_complete_turns=complete_turns,
            )
        frame_start = sum(int(n) for n in codec_lens[:start])
        frame_end = frame_start + sum(int(n) for n in codec_lens[start:end])
        return PriorContext(
            codec_tokens=codec_tokens[frame_start:frame_end],
            prior_texts=prior_texts[start:end],
            selected_turns=selected_turns,
            total_complete_turns=complete_turns,
        )

    def _trim_oldest_complete_turn(self, entry: dict[str, Any]) -> bool:
        """Drop one oldest text+codec turn from ``entry``.

        Returns False if this entry predates per-turn codec lengths or is
        internally inconsistent. In that case callers should reset rather than
        guessing a text/frame alignment.
        """
        codec_lens = entry.get("codec_lens", [])
        codec_tokens = entry.get("codec_tokens", [])
        prior_texts = entry.get("prior_texts", [])
        prompt_lens = entry.get("prompt_lens", [])
        if not codec_lens:
            return False
        if not isinstance(codec_lens[0], int):
            return False
        drop_frames = int(codec_lens.pop(0))
        if drop_frames < 0 or drop_frames > len(codec_tokens):
            return False
        del codec_tokens[:drop_frames]
        if prior_texts:
            del prior_texts[0]
        if prompt_lens:
            del prompt_lens[0]
        return True

    # ---- eviction ----------------------------------------------------

    def _eviction_lock_path(self) -> str:
        return os.path.join(self.root, "sessions", ".eviction.lock")

    def _maybe_evict_stale(self) -> None:
        """Drop stale session files.

        Called lazily from :meth:`get_prior_codec_for_request`. Serialised
        across processes via a dedicated ``.eviction.lock`` file so that two
        concurrent requests cannot race each other's directory scans.
        Per-session entries are still protected by the TTL invariant (only
        files older than ``idle_ttl_s`` are unlinked), so an active writer
        cannot lose its own file to eviction.
        """
        sessions_dir = os.path.join(self.root, "sessions")
        lock_path = self._eviction_lock_path()
        lock_exists = os.path.exists(lock_path)
        with _FileLock(lock_path):
            now = time.time()
            if self.eviction_check_interval_s > 0 and lock_exists:
                try:
                    if now - os.path.getmtime(lock_path) < self.eviction_check_interval_s:
                        return
                except FileNotFoundError:
                    pass

            # Mark the lock path as the last scan time before doing work so
            # concurrent requests can skip redundant scans.
            try:
                os.utime(self._eviction_lock_path(), (now, now))
            except FileNotFoundError:
                pass

            try:
                entries = os.listdir(sessions_dir)
            except FileNotFoundError:
                return
            now = time.time()
            pkl_files = [e for e in entries if e.endswith(".pkl")]

            # TTL
            for name in pkl_files:
                path = os.path.join(sessions_dir, name)
                try:
                    if now - os.path.getmtime(path) > self.idle_ttl_s:
                        os.unlink(path)
                except FileNotFoundError:
                    continue

            # LRU cap
            pkl_files = [e for e in os.listdir(sessions_dir) if e.endswith(".pkl")]
            if len(pkl_files) > self.max_sessions:
                with_mtime: list[tuple[float, str]] = []
                for name in pkl_files:
                    path = os.path.join(sessions_dir, name)
                    try:
                        mtime = os.path.getmtime(path)
                        last_activity = mtime
                        try:
                            with open(path, "rb") as fp:
                                entry = pickle.load(fp)
                            last_activity = float(entry.get("last_activity", mtime))
                        except (EOFError, OSError, TypeError, ValueError, pickle.UnpicklingError):
                            pass
                        with_mtime.append((max(last_activity, mtime), path))
                    except FileNotFoundError:
                        continue
                with_mtime.sort()
                excess = len(with_mtime) - self.max_sessions
                for _, path in with_mtime[:excess]:
                    try:
                        os.unlink(path)
                    except FileNotFoundError:
                        pass

    # ---- public API --------------------------------------------------

    def get_prior_codec_for_request(
        self,
        session_id: str,
        signature: str,
        *,
        new_prompt_est: int = 200,
    ) -> list[list[int]]:
        """Return the accumulated prior codec frames for ``session_id``.

        Applies the signature gate (reset on voice/instruct change) and
        the cap-reset (overflow if appending ``new_prompt_est`` tokens
        would push the session past ``max_prefix_tokens``). Returns an
        empty list on NEW session, signature-reset, or cap-reset.
        """
        return self.get_prior_context_for_request(
            session_id,
            signature,
            new_prompt_est=new_prompt_est,
        ).codec_tokens

    def get_prior_context_for_request(
        self,
        session_id: str,
        signature: str,
        *,
        new_prompt_est: int = 200,
        max_context_turns: int | None = None,
    ) -> PriorContext:
        """Return aligned prior text+codec context for ``session_id``.

        ``max_context_turns=0`` preserves the full rolling context
        behavior. Positive values select only the most recent complete
        turns while the hard prefix-token cap still applies.
        """
        empty = PriorContext(
            codec_tokens=[],
            prior_texts=[],
            selected_turns=0,
            total_complete_turns=0,
        )
        if not session_id:
            return empty
        context_turns = self.max_context_turns if max_context_turns is None else int(max_context_turns)
        with _FileLock(self._lock_path(session_id)):
            self._maybe_evict_stale()
            entry = self._load_entry(session_id)
            now = time.time()

            if entry is None:
                logger.info(
                    "[session_state] session=%s NEW (signature=%s)",
                    session_id,
                    signature,
                )
                self._save_entry_atomic(session_id, self._fresh_entry(signature))
                return empty

            if entry.get("signature") != signature:
                if self.allow_cross_signature_context:
                    logger.info(
                        "[session_state] session=%s SIGNATURE-CONTINUE (old=%s new=%s)",
                        session_id,
                        entry.get("signature"),
                        signature,
                    )
                    entry["signature"] = signature
                    entry["last_activity"] = now
                    self._save_entry_atomic(session_id, entry)
                else:
                    logger.info(
                        "[session_state] session=%s SIGNATURE-RESET (old=%s new=%s)",
                        session_id,
                        entry.get("signature"),
                        signature,
                    )
                    self._save_entry_atomic(session_id, self._fresh_entry(signature))
                    return empty
            projected = self._projected_prefix_tokens(
                entry,
                new_prompt_est=new_prompt_est,
                max_context_turns=context_turns,
            )
            if projected > self.max_prefix_tokens:
                dropped_turns = 0
                while projected > self.max_prefix_tokens and entry.get("codec_tokens"):
                    if not self._trim_oldest_complete_turn(entry):
                        logger.info(
                            "[session_state] session=%s CAP-RESET (projected=%d > max=%d, no per-turn trim index)",
                            session_id,
                            projected,
                            self.max_prefix_tokens,
                        )
                        self._save_entry_atomic(session_id, self._fresh_entry(signature))
                        return empty
                    dropped_turns += 1
                    projected = self._projected_prefix_tokens(
                        entry,
                        new_prompt_est=new_prompt_est,
                        max_context_turns=context_turns,
                    )
                if projected > self.max_prefix_tokens:
                    logger.info(
                        "[session_state] session=%s CAP-RESET (projected=%d > max=%d)",
                        session_id,
                        projected,
                        self.max_prefix_tokens,
                    )
                    self._save_entry_atomic(session_id, self._fresh_entry(signature))
                    return empty
                entry["last_activity"] = now
                self._save_entry_atomic(session_id, entry)
                selected_after_trim = self._select_prior_context(
                    entry,
                    max_context_turns=context_turns,
                )
                logger.info(
                    "[session_state] session=%s CAP-TRIM "
                    "(dropped_turns=%d projected=%d max=%d frames=%d "
                    "selected_turns=%d total_complete_turns=%d)",
                    session_id,
                    dropped_turns,
                    projected,
                    self.max_prefix_tokens,
                    len(selected_after_trim.codec_tokens),
                    selected_after_trim.selected_turns,
                    selected_after_trim.total_complete_turns,
                )

            # Touch mtime for LRU
            try:
                os.utime(self._session_path(session_id), (now, now))
            except FileNotFoundError:
                pass

            context = self._select_prior_context(
                entry,
                max_context_turns=context_turns,
            )
            context_uses = int(entry.get("context_uses_since_refresh", 0))
            if self.max_context_uses > 0 and context.selected_turns > 0 and context_uses >= self.max_context_uses:
                logger.info(
                    "[session_state] session=%s REFRESH-RESET (context_uses=%d max_context_uses=%d)",
                    session_id,
                    context_uses,
                    self.max_context_uses,
                )
                self._save_entry_atomic(session_id, self._fresh_entry(signature))
                return empty
            if context.selected_turns > 0:
                context_uses += 1
                entry["context_uses_since_refresh"] = context_uses
                entry["last_activity"] = now
                self._save_entry_atomic(session_id, entry)
            logger.info(
                "[session_state] session=%s HIT "
                "(frames=%d selected_turns=%d total_complete_turns=%d "
                "max_context_turns=%d context_uses=%d max_context_uses=%d)",
                session_id,
                len(context.codec_tokens),
                context.selected_turns,
                context.total_complete_turns,
                context_turns,
                context_uses,
                self.max_context_uses,
            )
            return context

    def get_prior_texts(self, session_id: str) -> list[str]:
        """Return accumulated prior-turn texts for ``session_id`` in order."""
        if not session_id:
            return []
        with _FileLock(self._lock_path(session_id)):
            entry = self._load_entry(session_id)
            if entry is None:
                return []
            return list(entry.get("prior_texts", []))

    def append_prior_text(self, session_id: str, text: str) -> None:
        """Append a just-submitted turn's text to the session.

        Called from the API-server ingress path *before* handing the
        request to the engine, so that the NEXT request's
        :meth:`get_prior_texts` call reflects this turn.
        """
        if not session_id or not text:
            return
        with _FileLock(self._lock_path(session_id)):
            entry = self._load_entry(session_id)
            if entry is None:
                return
            entry.setdefault("prior_texts", [])
            entry.setdefault("prompt_lens", [])
            text_str = str(text)
            entry["prior_texts"].append(text_str)
            entry["prompt_lens"].append(len(text_str))
            entry["last_activity"] = time.time()
            self._save_entry_atomic(session_id, entry)
            logger.info(
                "[session_state] session=%s text-appended (turns=%d chars=%d)",
                session_id,
                len(entry["prior_texts"]),
                len(text_str),
            )

    def append_codec_frames(self, session_id: str, frames: list[list[int]]) -> None:
        """Append emitted codec frames for a completed turn.

        Called from the engine side (chunk transfer adapter's
        ``cleanup_sender`` hook) at request completion. Does not
        auto-create a missing session — if the session was evicted mid-request
        the frames are dropped with a warning.
        """
        if not session_id or not frames:
            return
        with _FileLock(self._lock_path(session_id)):
            entry = self._load_entry(session_id)
            if entry is None:
                logger.warning(
                    "[session_state] append_codec_frames for missing session=%s (dropping %d frames)",
                    session_id,
                    len(frames),
                )
                return
            entry.setdefault("codec_tokens", [])
            entry.setdefault("codec_lens", [])
            if self.max_turn_frames > 0 and len(frames) > self.max_turn_frames:
                logger.warning(
                    "[session_state] session=%s TURN-RESET (frames=%d > max_turn_frames=%d)",
                    session_id,
                    len(frames),
                    self.max_turn_frames,
                )
                self._save_entry_atomic(
                    session_id,
                    self._fresh_entry(str(entry.get("signature", ""))),
                )
                return
            entry["codec_tokens"].extend(frames)
            entry["codec_lens"].append(len(frames))
            entry["last_activity"] = time.time()
            self._save_entry_atomic(session_id, entry)
            logger.info(
                "[session_state] session=%s APPENDED %d frames (total=%d)",
                session_id,
                len(frames),
                len(entry["codec_tokens"]),
            )

    def debug_state(self, session_id: str) -> dict[str, Any] | None:
        """Return a shallow snapshot of the session for diagnostics."""
        if not session_id:
            return None
        with _FileLock(self._lock_path(session_id)):
            entry = self._load_entry(session_id)
            if entry is None:
                return None
            return {
                "signature": entry.get("signature"),
                "codec_frames": len(entry.get("codec_tokens", [])),
                "prior_texts_count": len(entry.get("prior_texts", [])),
                "prompt_lens_total_chars": sum(entry.get("prompt_lens", [])),
                "context_uses_since_refresh": int(entry.get("context_uses_since_refresh", 0)),
                "last_activity": entry.get("last_activity"),
            }

    def reset(self, session_id: str) -> None:
        """Drop the session entirely."""
        if not session_id:
            return
        with _FileLock(self._lock_path(session_id)):
            try:
                os.unlink(self._session_path(session_id))
            except FileNotFoundError:
                pass

    def reset_to_fresh(self, session_id: str, *, reason: str) -> None:
        """Clear accumulated state while preserving the current signature."""
        if not session_id:
            return
        with _FileLock(self._lock_path(session_id)):
            entry = self._load_entry(session_id)
            if entry is None:
                return
            signature = str(entry.get("signature", ""))
            logger.warning(
                "[session_state] session=%s %s (clearing session state)",
                session_id,
                reason,
            )
            self._save_entry_atomic(session_id, self._fresh_entry(signature))

    # ---- request <-> session binding ---------------------------------

    def _req_binding_path(self, request_id: str) -> str:
        safe = hashlib.sha256(request_id.encode("utf-8")).hexdigest()[:32]
        return os.path.join(self.root, "req_bindings", safe)

    def register_request(self, request_id: str, session_id: str) -> None:
        """Bind a vLLM request id to the originating session id."""
        if not request_id or not session_id:
            return
        path = self._req_binding_path(request_id)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w") as fp:
            fp.write(session_id)
        os.replace(tmp, path)

    def pop_request_session(self, request_id: str) -> str | None:
        """Resolve and remove the binding for ``request_id``."""
        if not request_id:
            return None
        path = self._req_binding_path(request_id)
        try:
            with open(path) as fp:
                sid = fp.read().strip() or None
        except FileNotFoundError:
            return None
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
        return sid


# ---- module-level singleton + convenience wrappers ---------------------

_SINGLETON: FileSessionStore | None = None
_SINGLETON_LOCK = threading.Lock()


def get_session_store() -> FileSessionStore:
    """Return the process-wide :class:`FileSessionStore` singleton."""
    global _SINGLETON
    if _SINGLETON is None:
        with _SINGLETON_LOCK:
            if _SINGLETON is None:
                _SINGLETON = FileSessionStore()
    return _SINGLETON


def register_request_session(request_id: str, session_id: str) -> None:
    """Bind ``request_id`` to ``session_id`` in the singleton store."""
    try:
        get_session_store().register_request(request_id, session_id)
    except Exception:
        logger.warning("[session_state] register_request_session failed", exc_info=True)


def drop_request_binding(request_id: str) -> None:
    """Remove the ``request_id -> session`` binding if present.

    Called when a request is cancelled or errored, to prevent stale
    bindings from leaking.
    """
    try:
        store = get_session_store()
        path = store._req_binding_path(request_id)
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
    except Exception:
        pass


def on_request_complete(request_id: str, codec_frames: list[list[int]]) -> None:
    """Append the emitted codec frames to the owning session, if any.

    Called from the engine side (chunk transfer adapter) at request
    completion. Resolves the session id via the request binding and
    delegates to :meth:`FileSessionStore.append_codec_frames`.
    """
    if not request_id:
        return
    try:
        store = get_session_store()
        session_id = store.pop_request_session(request_id)
        if session_id is None:
            return
        if not codec_frames:
            store.reset_to_fresh(session_id, reason="EMPTY-RESET")
            return
        normalised: list[list[int]] = []
        for frame in codec_frames:
            if hasattr(frame, "tolist"):
                values = frame.tolist()
            else:
                values = list(frame)
            try:
                normalised.append([int(x) for x in values])
            except (TypeError, ValueError):
                continue
        if normalised:
            store.append_codec_frames(session_id, normalised)
        else:
            store.reset_to_fresh(session_id, reason="EMPTY-RESET")
    except Exception:
        logger.warning("[session_state] on_request_complete failed", exc_info=True)
