import importlib.util
import json
import logging
import sys
from pathlib import Path
from types import ModuleType

import torch
from safetensors.torch import save_file

_VLLM = ModuleType("vllm")
_VLLM_LOGGER = ModuleType("vllm.logger")
_VLLM_LOGGER.init_logger = logging.getLogger
sys.modules.setdefault("vllm", _VLLM)
sys.modules.setdefault("vllm.logger", _VLLM_LOGGER)

_SPEAKER_CACHE_PATH = Path(__file__).parents[2] / "vllm_omni/utils/speaker_cache.py"
_SPEC = importlib.util.spec_from_file_location("color_speaker_cache", _SPEAKER_CACHE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_SPEAKER_CACHE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_SPEAKER_CACHE)


def test_maya_icl_profile_loads_without_a_speaker_encoder_embedding(tmp_path: Path) -> None:
    save_file({"ref_code": torch.arange(12, dtype=torch.int32).reshape(3, 4)}, tmp_path / "maya.safetensors")
    (tmp_path / "custom_voice_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model_type": "qwen3_tts",
                "voices": {
                    "Maya": {
                        "file": "maya.safetensors",
                        "mode": "icl",
                        "ref_text": "reference transcript",
                        "speaker_anchor_voice": " MAYA_WARM ",
                    }
                },
            }
        )
    )

    profiles = _SPEAKER_CACHE.iter_custom_voice_profiles(tmp_path, expected_model_type="qwen3_tts")
    tensors = _SPEAKER_CACHE.load_validated_profile_tensors(
        profiles[0],
        expected_model_type="qwen3_tts",
        qwen3_embedding_dim=2048,
    )

    assert tensors is not None
    assert profiles[0]["voice_name_lower"] == "maya"
    assert profiles[0]["speaker_anchor_voice"] == "maya_warm"
    assert profiles[0]["ref_code_length"] == 3
    assert "embedding_dim" not in profiles[0]


def test_anchor_only_profile_is_rejected_outside_icl_mode(tmp_path: Path) -> None:
    save_file({"ref_code": torch.ones((1, 4), dtype=torch.int32)}, tmp_path / "maya.safetensors")
    (tmp_path / "custom_voice_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model_type": "qwen3_tts",
                "voices": {
                    "Maya": {
                        "file": "maya.safetensors",
                        "mode": "xvec",
                        "speaker_anchor_voice": "maya_warm",
                    }
                },
            }
        )
    )

    profile = _SPEAKER_CACHE.iter_custom_voice_profiles(tmp_path, expected_model_type="qwen3_tts")[0]
    assert (
        _SPEAKER_CACHE.load_validated_profile_tensors(
            profile,
            expected_model_type="qwen3_tts",
            qwen3_embedding_dim=2048,
        )
        is None
    )
