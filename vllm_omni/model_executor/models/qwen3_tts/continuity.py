"""Small Qwen3-TTS continuity helpers.

The helpers in this module are intentionally policy-free.  They only parse
request metadata and normalize codec frames so callers can decide whether to
use static talker ICL or Code2Wav left context.
"""

from __future__ import annotations

import os
from collections import OrderedDict
from threading import RLock
from typing import Any

import numpy as np
import torch

CONTINUITY_TALKER_ICL = "talker_icl"
CONTINUITY_CODE2WAV_CONTEXT = "code2wav_context"
CONTINUITY_OFF = "off"
CONTINUITY_BOTH = f"{CONTINUITY_TALKER_ICL}+{CONTINUITY_CODE2WAV_CONTEXT}"
CONTINUITY_MODES = frozenset(
    (
        CONTINUITY_OFF,
        CONTINUITY_TALKER_ICL,
        CONTINUITY_CODE2WAV_CONTEXT,
        CONTINUITY_BOTH,
    )
)


def continuity_mode_enabled(value: object, mode: str) -> bool:
    """Return whether a canonical continuity mode enables one feature."""
    value = unwrap_singleton(value)
    if value is None:
        return False
    if not isinstance(value, str):
        raise ValueError("continuity_mode must be a string")
    if value not in CONTINUITY_MODES:
        allowed = ", ".join(sorted(CONTINUITY_MODES))
        raise ValueError(f"Unsupported continuity_mode value: {value!r}; expected one of: {allowed}")
    if mode not in (CONTINUITY_TALKER_ICL, CONTINUITY_CODE2WAV_CONTEXT):
        raise ValueError(f"Unsupported continuity feature: {mode!r}")
    return value == mode or value == CONTINUITY_BOTH


def unwrap_singleton(value: object) -> object:
    """Unwrap the one-element list convention used by additional_information."""
    if isinstance(value, list) and len(value) == 1 and not isinstance(value[0], (int, np.integer)):
        return value[0]
    return value


def additional_value_from_request(request: Any, key: str) -> Any:
    """Extract one value from a request's serialized additional_information."""
    additional_information = getattr(request, "additional_information", None)
    if isinstance(additional_information, dict):
        return unwrap_singleton(additional_information.get(key))

    entries = getattr(additional_information, "entries", None)
    if not isinstance(entries, dict):
        return None
    entry = entries.get(key)
    if entry is None:
        return None
    list_data = getattr(entry, "list_data", None)
    if list_data is not None:
        return unwrap_singleton(list_data)
    return getattr(entry, "scalar_data", None)


def normalize_codec_frames(value: object, *, device: torch.device | None = None) -> torch.Tensor | None:
    """Normalize codec frames to a contiguous int64 tensor shaped [frames, quantizers]."""
    value = unwrap_singleton(value)
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        tensor = value.detach()
    elif isinstance(value, np.ndarray):
        tensor = torch.from_numpy(value)
    elif isinstance(value, list) and value:
        try:
            tensor = torch.tensor(value, dtype=torch.long)
        except (TypeError, ValueError):
            return None
    else:
        return None

    if tensor.ndim == 3 and int(tensor.shape[0]) == 1:
        tensor = tensor[0]
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 2 or int(tensor.shape[0]) <= 0 or int(tensor.shape[1]) <= 0:
        return None
    if device is None:
        return tensor.to(dtype=torch.long).contiguous()
    return tensor.to(device=device, dtype=torch.long).contiguous()


def codec_frame_count(value: object) -> int | None:
    """Return the number of codec frames in a request value, if it is valid."""
    tensor = normalize_codec_frames(value)
    if tensor is None:
        return None
    return int(tensor.shape[0])


def normalize_cache_key(value: object) -> str | None:
    """Normalize an optional opaque continuity cache key."""
    value = unwrap_singleton(value)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("continuity_cache_key must be a string")
    key = value.strip()
    if not key:
        raise ValueError("continuity_cache_key must be non-empty when provided")
    return key


def continuity_max_sessions_from_env() -> int:
    """Read the bounded in-process continuity cache size."""
    env_name = "VLLM_OMNI_QWEN3_TTS_CONTINUITY_MAX_SESSIONS"
    raw = os.environ.get(env_name)
    if raw is None or not raw.strip():
        raise ValueError(f"{env_name} must be set to a positive integer")
    try:
        max_sessions = int(raw)
    except ValueError as exc:
        raise ValueError(f"{env_name} must be a positive integer") from exc
    if max_sessions <= 0:
        raise ValueError(f"{env_name} must be a positive integer")
    return max_sessions


class CodecFrameLRUCache:
    """Bounded in-process cache for the latest generated codec frames per key."""

    def __init__(self, max_sessions: int) -> None:
        if max_sessions <= 0:
            raise ValueError("max_sessions must be positive")
        self.max_sessions = int(max_sessions)
        self._items: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._lock = RLock()

    def get(self, key: str) -> torch.Tensor | None:
        cache_key = normalize_cache_key(key)
        if cache_key is None:
            return None
        with self._lock:
            frames = self._items.get(cache_key)
            if frames is None:
                return None
            self._items.move_to_end(cache_key)
            return frames.clone()

    def put(self, key: str, frames: object) -> bool:
        cache_key = normalize_cache_key(key)
        if cache_key is None:
            return False
        normalized = normalize_codec_frames(frames)
        if normalized is None:
            return False
        normalized = normalized.cpu().contiguous()
        with self._lock:
            self._items[cache_key] = normalized
            self._items.move_to_end(cache_key)
            while len(self._items) > self.max_sessions:
                self._items.popitem(last=False)
        return True

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

    def keys(self) -> list[str]:
        with self._lock:
            return list(self._items.keys())


def continuity_cache_for_transfer_manager(transfer_manager: Any) -> CodecFrameLRUCache:
    """Return the per-process Qwen3-TTS continuity cache."""
    cache = getattr(transfer_manager, "_qwen3_tts_continuity_cache", None)
    if isinstance(cache, CodecFrameLRUCache):
        return cache
    cache = CodecFrameLRUCache(max_sessions=continuity_max_sessions_from_env())
    transfer_manager._qwen3_tts_continuity_cache = cache
    return cache
