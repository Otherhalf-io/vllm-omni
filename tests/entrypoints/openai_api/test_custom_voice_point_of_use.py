# SPDX-License-Identifier: Apache-2.0
"""Point-of-use CustomVoice resolution regressions."""

from types import SimpleNamespace

import pytest

from vllm_omni.entrypoints.openai.serving_speech import OmniOpenAIServingSpeech

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _serving() -> OmniOpenAIServingSpeech:
    serving = OmniOpenAIServingSpeech.__new__(OmniOpenAIServingSpeech)
    serving.default_custom_voice = "maya_warm"
    serving.supported_speakers = {"maya_conv", "maya_warm"}
    serving.uploaded_speakers = {}
    serving.precomputed_speakers = {}
    serving._voice_created_at = lambda _voice: 0
    return serving


def _request(voice: str | None) -> SimpleNamespace:
    return SimpleNamespace(
        input="Hello",
        voice=voice,
        task_type=None,
        language=None,
        ref_audio=None,
        ref_text=None,
        speaker_embedding=None,
        x_vector_only_mode=None,
        instructions=None,
        max_new_tokens=None,
        initial_codec_chunk_frames=None,
        non_streaming_mode=None,
    )


@pytest.mark.parametrize(
    ("supplied", "expected"),
    ((None, "maya_warm"), ("", "maya_warm"), ("   ", "maya_warm"), ("  MAYA_CONV  ", "maya_conv")),
)
def test_build_tts_params_resolves_canonical_custom_voice_at_point_of_use(supplied: str | None, expected: str) -> None:
    request = _request(supplied)

    params = _serving()._build_tts_params(request)

    assert request.voice == expected
    assert params["speaker"] == [expected]


def test_build_tts_params_rejects_unsupported_custom_voice() -> None:
    with pytest.raises(ValueError, match="Invalid voice 'vivian'"):
        _serving()._build_tts_params(_request("vivian"))
