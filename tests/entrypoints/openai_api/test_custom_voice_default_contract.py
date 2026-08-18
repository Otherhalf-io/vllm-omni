# SPDX-License-Identifier: Apache-2.0
"""CustomVoice speaker-resolution contract.

Regression guard for a process-wide fault. The CustomVoice speaker name was
only checked against the checkpoint's ``spk_id`` map inside the model's
``preprocess`` on the GPU worker, where a ``ValueError`` kills EngineCore for
every in-flight session rather than failing the one offending request.

The name reaching the worker was not always the caller's. When ``voice`` was
omitted the server substituted upstream's ``"Vivian"`` default, and that
substituted value bypassed validation entirely -- so on a fine-tuned checkpoint
(Maya ships only ``maya_conv``/``maya_warm``) a single request with no ``voice``
took TTS down for every session until the pod restarted. Reproduced manually:
one such request, then 0 of 3 subsequent valid requests succeeded.

These assert the *admission* rules, which is what keeps such a request from
reaching the worker. They cannot assert "the engine survives" -- that needs a
live engine -- so treat the manual reproduction as the companion check.

The module under test is deliberately dependency-free and loaded by path, so
these run without vLLM or torch installed.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
_MODULE_PATH = REPO_ROOT / "vllm_omni/entrypoints/openai/tts_adapters/voice_resolution.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("color_voice_resolution", _MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_VR = _load_module()
resolve_default_custom_voice = _VR.resolve_default_custom_voice
resolve_custom_voice = _VR.resolve_custom_voice

MAYA = {"maya_conv", "maya_warm"}
STOCK = {"vivian", "ryan", "aiden"}


# --- resolving a deployment's default ---------------------------------------


def test_finetuned_checkpoint_has_no_usable_default() -> None:
    """Maya's map lacks the stock default, so there is no default to substitute."""
    default, problem = resolve_default_custom_voice(MAYA)

    assert default is None
    assert problem is not None
    assert "maya_warm" in problem


def test_stock_checkpoint_keeps_the_upstream_default() -> None:
    default, problem = resolve_default_custom_voice(STOCK)

    assert default == "vivian"
    assert problem is None


def test_operator_can_configure_a_supported_default() -> None:
    default, problem = resolve_default_custom_voice(MAYA, configured="  MAYA_WARM  ")

    assert default == "maya_warm"
    assert problem is None


def test_configured_default_outside_the_map_is_refused() -> None:
    default, problem = resolve_default_custom_voice(MAYA, configured="vivian")

    assert default is None
    assert problem is not None
    assert "vivian" in problem


def test_model_without_speakers_reports_no_problem() -> None:
    """No speaker map at all is a different condition, flagged at validation."""
    default, problem = resolve_default_custom_voice(set())

    assert default is None
    assert problem is None


# --- validating an individual request ---------------------------------------


def test_omitted_voice_is_rejected_when_no_default_resolves() -> None:
    """The crash case, now refused at admission."""
    effective, error = resolve_custom_voice(None, None, MAYA)

    assert effective is None
    assert error is not None
    assert "requires 'voice'" in error
    assert "maya_conv" in error


def test_unresolvable_default_is_never_forwarded() -> None:
    """A default outside the map must fail here, not on the GPU worker."""
    effective, error = resolve_custom_voice(None, "vivian", MAYA)

    assert effective is None
    assert error is not None
    assert "Invalid voice 'vivian'" in error


def test_omitted_voice_is_accepted_when_default_resolves() -> None:
    assert resolve_custom_voice(None, "vivian", STOCK) == ("vivian", None)


@pytest.mark.parametrize("voice", ("maya_warm", "MAYA_WARM", "  maya_conv  "))
def test_supported_voice_accepted_case_and_space_insensitively(voice: str) -> None:
    effective, error = resolve_custom_voice(voice, None, MAYA)

    assert effective == voice.strip().lower()
    assert error is None


@pytest.mark.parametrize("voice", ("ryan", "Vivian", "aiden", "nope"))
def test_unknown_supplied_voice_is_rejected(voice: str) -> None:
    effective, error = resolve_custom_voice(voice, None, MAYA)

    assert effective is None
    assert error is not None
    assert "Invalid voice" in error


def test_supplied_voice_takes_precedence_over_default() -> None:
    assert resolve_custom_voice("maya_conv", "maya_warm", MAYA) == ("maya_conv", None)


@pytest.mark.parametrize("voice", (None, "", "   "))
def test_absent_voice_returns_the_canonical_default(voice: str | None) -> None:
    assert resolve_custom_voice(voice, "MAYA_WARM", MAYA) == ("maya_warm", None)


@pytest.mark.parametrize(
    ("supplied", "expected"),
    (("   ", "maya_warm"), ("  MAYA_CONV  ", "maya_conv"), ("MAYA_WARM", "maya_warm")),
)
def test_adapter_assigns_canonical_voice_during_admission(monkeypatch, supplied: str, expected: str) -> None:
    """Admission surfaces errors early and writes the canonical speaker."""
    import types

    logger_module = types.ModuleType("vllm.logger")
    logger_module.init_logger = lambda _name: types.SimpleNamespace()
    monkeypatch.setitem(sys.modules, "vllm", types.ModuleType("vllm"))
    monkeypatch.setitem(sys.modules, "vllm.logger", logger_module)

    adapters_module = types.ModuleType("vllm_omni.entrypoints.openai.tts_adapters")
    adapters_module.register_tts_adapter = lambda cls: cls
    monkeypatch.setitem(sys.modules, "vllm_omni.entrypoints.openai.tts_adapters", adapters_module)

    base_module = types.ModuleType("vllm_omni.entrypoints.openai.tts_adapters.base")

    class ARTTSAdapter:
        def __init__(self, ctx):
            self.ctx = ctx

    base_module.ARTTSAdapter = ARTTSAdapter
    base_module.PreparedRequest = object
    monkeypatch.setitem(sys.modules, "vllm_omni.entrypoints.openai.tts_adapters.base", base_module)
    monkeypatch.setitem(
        sys.modules,
        "vllm_omni.entrypoints.openai.tts_adapters.voice_resolution",
        _VR,
    )

    module_path = REPO_ROOT / "vllm_omni/entrypoints/openai/tts_adapters/qwen3_tts.py"
    spec = importlib.util.spec_from_file_location("color_qwen3_tts_adapter", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class Request:
        input = "Hello"
        voice = supplied
        task_type = None
        ref_audio = None
        ref_text = None
        language = None
        speaker_embedding = None
        x_vector_only_mode = None
        instructions = None
        max_new_tokens = None
        initial_codec_chunk_frames = None
        non_streaming_mode = None

    server = types.SimpleNamespace(
        default_custom_voice="maya_warm",
        supported_speakers=MAYA,
        precomputed_speakers={},
        uploaded_speakers={},
        supported_languages=set(),
        _max_instructions_length=500,
    )
    adapter = module.Qwen3TTSAdapter(types.SimpleNamespace(server=server))
    request = Request()

    assert adapter.validate(request) is None
    assert request.voice == expected


def test_no_speakers_configured_is_reported_distinctly() -> None:
    effective, error = resolve_custom_voice("maya_warm", None, set())

    assert effective is None
    assert error is not None
    assert "does not support CustomVoice" in error
