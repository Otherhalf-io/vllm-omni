"""Small Qwen3-TTS continuity helpers.

This module stays policy-free: it validates canonical request metadata, loads
named talker anchors from local files, manages the bounded Code2Wav codec cache,
and keeps continuity state in-process. Callers decide when to use each
primitive; service telemetry is owned by the System1 sidecar.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any

import numpy as np
import torch

CONTINUITY_TALKER_ICL = "talker_icl"
CONTINUITY_CODE2WAV_CONTEXT = "code2wav_context"
CONTINUITY_OFF = "off"
CONTINUITY_BOTH = f"{CONTINUITY_TALKER_ICL}+{CONTINUITY_CODE2WAV_CONTEXT}"
CONTINUITY_MODES = (
    CONTINUITY_OFF,
    CONTINUITY_TALKER_ICL,
    CONTINUITY_CODE2WAV_CONTEXT,
    CONTINUITY_BOTH,
)

_ANCHORS_DIR_ENV = "VLLM_OMNI_QWEN3_TTS_CONTINUITY_ANCHORS_DIR"
_ANCHORS_LOADED = False
_ANCHORS: dict[str, ContinuityAnchor] = {}
_ANCHORS_LOCK = RLock()


@dataclass(frozen=True, slots=True)
class ContinuityAnchor:
    """A startup-loaded talker ICL anchor."""

    name: str
    ref_text: str
    ref_code: torch.Tensor
    source_kind: str | None = None

    @property
    def frame_count(self) -> int:
        return int(self.ref_code.shape[0])

    @property
    def quantizer_count(self) -> int:
        return int(self.ref_code.shape[1])

    def code_to(self, *, device: torch.device | None = None) -> torch.Tensor:
        return self.ref_code.to(device=device, dtype=torch.long).contiguous()


def continuity_mode_enabled(value: object, mode: str) -> bool:
    """Return whether a canonical continuity mode enables one feature."""
    value = unwrap_singleton(value)
    if value is None:
        return False
    if not isinstance(value, str):
        raise ValueError("continuity_mode must be a string")
    if value not in CONTINUITY_MODES:
        raise ValueError(
            f"Unsupported continuity_mode value: {value!r}; expected one of: {', '.join(CONTINUITY_MODES)}"
        )
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


def normalize_anchor_name(value: object) -> str | None:
    """Normalize an optional continuity anchor name."""
    value = unwrap_singleton(value)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("continuity_anchor_name must be a string")
    name = value.strip()
    if not name:
        raise ValueError("continuity_anchor_name must be non-empty when provided")
    if "/" in name or "\\" in name or Path(name).name != name:
        raise ValueError("continuity_anchor_name must be a simple anchor id")
    return name


def _read_json_file(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Failed to read continuity anchor JSON: {path}") from exc


def _read_text_file(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise ValueError(f"Failed to read continuity anchor text: {path}") from exc
    if not text:
        raise ValueError(f"Continuity anchor text is empty: {path}")
    return text


def _relative_anchor_path(anchor_dir: Path, raw_path: object, *, field: str) -> Path:
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError(f"Continuity anchor manifest field {field!r} must be a non-empty relative path")
    candidate = Path(raw_path)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"Continuity anchor manifest field {field!r} must stay inside the anchor directory")
    return anchor_dir / candidate


def _load_anchor_metadata(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    if not path.exists():
        return {}
    payload = _read_json_file(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Continuity anchor metadata must be a JSON object: {path}")
    return payload


def load_continuity_anchors_from_dir(anchor_dir: str | os.PathLike[str]) -> dict[str, ContinuityAnchor]:
    """Load all Qwen3-TTS talker anchors from a local synced anchor directory."""
    root = Path(anchor_dir)
    manifest_path = root / "anchors.json"
    if not manifest_path.is_file():
        raise ValueError(f"Continuity anchor manifest not found: {manifest_path}")

    manifest = _read_json_file(manifest_path)
    if not isinstance(manifest, dict) or not isinstance(manifest.get("anchors"), dict):
        raise ValueError("Continuity anchor manifest must be an object with an 'anchors' object")

    anchors: dict[str, ContinuityAnchor] = {}
    for raw_name, raw_entry in manifest["anchors"].items():
        name = normalize_anchor_name(raw_name)
        if name is None:
            raise ValueError("Continuity anchor manifest contains an empty anchor name")
        if name in anchors:
            raise ValueError(f"Duplicate continuity anchor name after normalization: {name}")
        if not isinstance(raw_entry, dict):
            raise ValueError(f"Continuity anchor entry must be an object: {name}")

        text_path = _relative_anchor_path(root, raw_entry.get("ref_text_path"), field=f"{name}.ref_text_path")
        codec_path = _relative_anchor_path(root, raw_entry.get("codec_path"), field=f"{name}.codec_path")
        metadata_path_raw = raw_entry.get("metadata_path")
        metadata_path = (
            _relative_anchor_path(root, metadata_path_raw, field=f"{name}.metadata_path")
            if metadata_path_raw is not None
            else None
        )
        ref_text = _read_text_file(text_path)
        ref_code = normalize_codec_frames(_read_json_file(codec_path))
        if ref_code is None:
            raise ValueError(f"Continuity anchor codec frames are invalid: {codec_path}")
        metadata = _load_anchor_metadata(metadata_path)
        anchors[name] = ContinuityAnchor(
            name=name,
            ref_text=ref_text,
            ref_code=ref_code.cpu().contiguous(),
            source_kind=metadata.get("source_kind") if isinstance(metadata.get("source_kind"), str) else None,
        )

    return anchors


def load_continuity_anchors_from_env() -> dict[str, ContinuityAnchor]:
    """Load configured anchors once per process."""
    global _ANCHORS, _ANCHORS_LOADED
    with _ANCHORS_LOCK:
        if _ANCHORS_LOADED:
            return _ANCHORS
        anchor_dir = os.environ.get(_ANCHORS_DIR_ENV)
        if not anchor_dir:
            _ANCHORS = {}
            _ANCHORS_LOADED = True
            return _ANCHORS
        _ANCHORS = load_continuity_anchors_from_dir(anchor_dir)
        _ANCHORS_LOADED = True
        return _ANCHORS


def reset_continuity_anchors_for_test() -> None:
    """Clear the process-local anchor registry for focused tests."""
    global _ANCHORS, _ANCHORS_LOADED
    with _ANCHORS_LOCK:
        _ANCHORS = {}
        _ANCHORS_LOADED = False


def get_continuity_anchor(name: object) -> ContinuityAnchor:
    """Return a configured talker ICL anchor by request-provided name."""
    anchor_name = normalize_anchor_name(name)
    if anchor_name is None:
        raise ValueError("talker_icl continuity requires continuity_anchor_name")
    anchors = load_continuity_anchors_from_env()
    anchor = anchors.get(anchor_name)
    if anchor is None:
        available = ", ".join(sorted(anchors)) or "<none loaded>"
        raise ValueError(f"Unknown Qwen3-TTS continuity anchor {anchor_name!r}; available anchors: {available}")
    return anchor


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


def cache_key_hash(key: str) -> str:
    """Return a stable low-cardinality identifier for diagnostics."""
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]


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
