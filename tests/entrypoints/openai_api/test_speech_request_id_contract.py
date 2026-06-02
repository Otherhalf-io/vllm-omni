import importlib.util
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

REPO_ROOT = Path(__file__).resolve().parents[3]
_AUDIO_PROTOCOL_PATH = REPO_ROOT / "vllm_omni/entrypoints/openai/protocol/audio.py"


def _load_audio_protocol():
    spec = importlib.util.spec_from_file_location("speech_contract_audio_protocol", _AUDIO_PROTOCOL_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_AUDIO = _load_audio_protocol()
BatchSpeechRequest = _AUDIO.BatchSpeechRequest
OpenAICreateSpeechRequest = _AUDIO.OpenAICreateSpeechRequest
SpeechBatchItem = _AUDIO.SpeechBatchItem


def test_speech_request_requires_caller_request_id() -> None:
    with pytest.raises(ValidationError):
        OpenAICreateSpeechRequest(input="hello")

    with pytest.raises(ValidationError, match="identifier fields must not be blank"):
        OpenAICreateSpeechRequest(input="hello", request_id=" ")

    request = OpenAICreateSpeechRequest(input="hello", request_id="tts-123")
    assert request.request_id == "tts-123"


def test_batch_item_requires_caller_request_id() -> None:
    with pytest.raises(ValidationError):
        SpeechBatchItem(input="hello")

    with pytest.raises(ValidationError):
        BatchSpeechRequest(items=[{"input": "hello"}])

    batch = BatchSpeechRequest(items=[{"input": "hello", "request_id": "tts-batch-1"}])
    assert batch.items[0].request_id == "tts-batch-1"


def test_vllm_omni_does_not_configure_otel_or_langfuse() -> None:
    continuity = (REPO_ROOT / "vllm_omni/model_executor/models/qwen3_tts/continuity.py").read_text()
    serving = (REPO_ROOT / "vllm_omni/entrypoints/openai/serving_speech.py").read_text()

    assert "opentelemetry" not in continuity
    assert "OTEL_TRACES_EXPORTER" not in continuity
    assert "LANGFUSE" not in continuity
    assert "OTLP" not in continuity
    assert "telemetry_span" not in continuity
    assert "set_span_attributes" not in continuity
    assert "telemetry_span" not in serving
    assert "set_span_attributes" not in serving
    assert "speech-{random_uuid()}" not in serving
    assert "from contextlib import aclosing" not in serving
    assert (
        "from contextlib import aclosing"
        not in (REPO_ROOT / "vllm_omni/entrypoints/openai/serving_speech_stream.py").read_text()
    )


def test_client_validation_failures_do_not_increment_health_error_count() -> None:
    serving = (REPO_ROOT / "vllm_omni/entrypoints/openai/serving_speech.py").read_text()

    assert 'response_format not in ["pcm", "wav"]' in serving
    assert "if request.speed is not None and request.speed != 1.0:" in serving
    assert serving.count("self._speech_health_stats.finish_request(started_at, error=False)") >= 3


def test_omni_health_route_replaces_upstream_app_route() -> None:
    api_server = (REPO_ROOT / "vllm_omni/entrypoints/openai/api_server.py").read_text()
    remove_health = '_remove_route_from_app(app, "/health", {"GET"})'
    include_router = "app.include_router(router)"

    assert remove_health in api_server
    assert api_server.index(remove_health) < api_server.index(include_router)
    assert 'content["tts"] = speech_handler.get_speech_health_snapshot()' in api_server


def test_pr_ci_runs_added_lightweight_tests_without_vllm() -> None:
    workflow = (REPO_ROOT / ".github/workflows/ci.yml").read_text()
    assert "test_speech_request_id_contract.py" in workflow
    assert "--with vllm" not in workflow
