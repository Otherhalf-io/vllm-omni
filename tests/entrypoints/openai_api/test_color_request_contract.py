import pytest
from pydantic import ValidationError

from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest
from vllm_omni.entrypoints.openai.serving_speech import _resolve_speech_request_id


def test_color_body_contract_normalizes_caller_identifiers() -> None:
    request = OpenAICreateSpeechRequest(
        input="Hello",
        request_id=" request-color-42 ",
        session_id=" session-color-7 ",
    )

    assert request.request_id == "request-color-42"
    assert request.session_id == "session-color-7"


@pytest.mark.parametrize("field", ("request_id", "session_id"))
def test_color_body_contract_rejects_blank_identifiers(field: str) -> None:
    with pytest.raises(ValidationError, match="must not be blank"):
        OpenAICreateSpeechRequest(input="Hello", **{field: "  "})


def test_request_id_resolution_prefers_explicit_then_body() -> None:
    request = OpenAICreateSpeechRequest(input="Hello", request_id="request-color-42")

    assert _resolve_speech_request_id(request) == "request-color-42"
    assert _resolve_speech_request_id(request, "request-explicit-7") == "request-explicit-7"


def test_external_speech_boundary_requires_caller_request_id() -> None:
    request = OpenAICreateSpeechRequest(input="Hello")

    with pytest.raises(ValueError, match="request_id must be a non-empty string"):
        _resolve_speech_request_id(request)
