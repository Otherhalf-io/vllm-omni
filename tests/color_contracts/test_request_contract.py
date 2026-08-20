import importlib.util
from pathlib import Path

import pytest
from pydantic import ValidationError

_AUDIO_PROTOCOL = Path(__file__).parents[2] / "vllm_omni/entrypoints/openai/protocol/audio.py"
_SPEC = importlib.util.spec_from_file_location("color_audio_protocol", _AUDIO_PROTOCOL)
assert _SPEC is not None and _SPEC.loader is not None
_AUDIO = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_AUDIO)

BatchSpeechRequest = _AUDIO.BatchSpeechRequest
OpenAICreateSpeechRequest = _AUDIO.OpenAICreateSpeechRequest
StreamingSpeechSessionConfig = _AUDIO.StreamingSpeechSessionConfig
resolve_speech_request_id = _AUDIO.resolve_speech_request_id


def test_http_contract_normalizes_caller_identifiers() -> None:
    request = OpenAICreateSpeechRequest(
        input="Hello",
        request_id=" request-color-42 ",
        session_id=" session-color-7 ",
    )

    assert request.request_id == "request-color-42"
    assert request.session_id == "session-color-7"
    assert resolve_speech_request_id(request) == "request-color-42"


def test_internal_contract_accepts_an_explicit_request_id() -> None:
    request = OpenAICreateSpeechRequest(input="Hello")

    assert resolve_speech_request_id(request, " internal-color-7 ") == "internal-color-7"


def test_speech_generation_fails_loud_without_any_request_id() -> None:
    with pytest.raises(ValueError, match="request_id must be a non-empty string"):
        resolve_speech_request_id(OpenAICreateSpeechRequest(input="Hello"))


@pytest.mark.parametrize("field", ("request_id", "session_id"))
def test_http_contract_rejects_blank_identifiers(field: str) -> None:
    with pytest.raises(ValidationError, match=f"{field} must not be blank"):
        OpenAICreateSpeechRequest(input="Hello", **{field: "  "})


def test_batch_contract_requires_a_caller_prefix_and_rejects_item_ids() -> None:
    with pytest.raises(ValidationError, match="request_id"):
        BatchSpeechRequest(items=[{"input": "Hello"}])

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        BatchSpeechRequest(
            request_id="batch-color-1",
            items=[{"input": "Hello", "request_id": "must-not-be-overwritten"}],
        )


def test_websocket_contract_requires_and_normalizes_a_caller_prefix() -> None:
    with pytest.raises(ValidationError, match="request_id"):
        StreamingSpeechSessionConfig()

    config = StreamingSpeechSessionConfig(
        request_id=" ws-color-1 ",
        session_id=" session-color-1 ",
    )
    assert config.request_id == "ws-color-1"
    assert config.session_id == "session-color-1"


def test_server_sources_do_not_restore_generated_speech_ids() -> None:
    repository = Path(__file__).parents[2]
    serving = (repository / "vllm_omni/entrypoints/openai/serving_speech.py").read_text()
    websocket = (repository / "vllm_omni/entrypoints/openai/serving_speech_stream.py").read_text()

    assert "speech-internal-" not in serving
    assert "speech-batch-" not in serving
    assert "random_uuid" not in websocket
    assert 'request_id=f"speech-ws-' not in websocket


def test_public_quick_start_conforms_to_the_required_identity_contract() -> None:
    documentation = (Path(__file__).parents[2] / "docs/serving/speech_api.md").read_text()
    curl_example = documentation.split("**Using curl:**", maxsplit=1)[1].split("**Using Python:**", maxsplit=1)[0]

    assert '"request_id": "speech-example-001"' in curl_example
    assert '"session_id": "session-example"' in curl_example
