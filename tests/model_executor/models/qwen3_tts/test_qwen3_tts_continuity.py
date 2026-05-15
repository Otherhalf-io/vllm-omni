import importlib.util
import sys
import types
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from pydantic import ValidationError

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_REPO_ROOT = Path(__file__).resolve().parents[4]


def _read_repo_file(path: str) -> str:
    return (_REPO_ROOT / path).read_text()


def _load_continuity_module():
    continuity_path = _REPO_ROOT / "vllm_omni/model_executor/models/qwen3_tts/continuity.py"
    spec = importlib.util.spec_from_file_location("qwen3_tts_continuity_under_test", continuity_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_CONTINUITY = _load_continuity_module()
CONTINUITY_CODE2WAV_CONTEXT = _CONTINUITY.CONTINUITY_CODE2WAV_CONTEXT
CONTINUITY_TALKER_ICL = _CONTINUITY.CONTINUITY_TALKER_ICL
CodecFrameLRUCache = _CONTINUITY.CodecFrameLRUCache
codec_frame_count = _CONTINUITY.codec_frame_count
continuity_max_sessions_from_env = _CONTINUITY.continuity_max_sessions_from_env
continuity_mode_enabled = _CONTINUITY.continuity_mode_enabled
normalize_codec_frames = _CONTINUITY.normalize_codec_frames


def _load_module_from_repo(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, _REPO_ROOT / path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _install_stage_processor_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Logger:
        def info(self, *_args, **_kwargs):
            pass

        def warning(self, *_args, **_kwargs):
            pass

    class _CodesStruct:
        def __init__(self, audio=None):
            self.audio = audio

    class _MetaStruct:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class _OmniPayloadStruct:
        def __init__(self, codes=None, meta=None, speaker=None, language=None):
            self.codes = codes
            self.meta = meta
            self.speaker = speaker
            self.language = language

    modules = {
        "vllm": types.ModuleType("vllm"),
        "vllm.logger": types.ModuleType("vllm.logger"),
        "vllm_omni": types.ModuleType("vllm_omni"),
        "vllm_omni.data_entry_keys": types.ModuleType("vllm_omni.data_entry_keys"),
        "vllm_omni.model_executor": types.ModuleType("vllm_omni.model_executor"),
        "vllm_omni.model_executor.models": types.ModuleType("vllm_omni.model_executor.models"),
        "vllm_omni.model_executor.models.qwen3_tts": types.ModuleType("vllm_omni.model_executor.models.qwen3_tts"),
        "vllm_omni.model_executor.stage_input_processors": types.ModuleType(
            "vllm_omni.model_executor.stage_input_processors"
        ),
        "vllm_omni.model_executor.stage_input_processors.chunk_size_utils": types.ModuleType(
            "vllm_omni.model_executor.stage_input_processors.chunk_size_utils"
        ),
        "vllm_omni.model_executor.stage_input_processors.tts_utils": types.ModuleType(
            "vllm_omni.model_executor.stage_input_processors.tts_utils"
        ),
    }
    modules["vllm.logger"].init_logger = lambda _name: _Logger()
    modules["vllm_omni.data_entry_keys"].CodesStruct = _CodesStruct
    modules["vllm_omni.data_entry_keys"].MetaStruct = _MetaStruct
    modules["vllm_omni.data_entry_keys"].OmniPayload = dict
    modules["vllm_omni.data_entry_keys"].OmniPayloadStruct = _OmniPayloadStruct
    modules["vllm_omni.data_entry_keys"].to_dict = lambda value: value
    modules["vllm_omni.model_executor.models.qwen3_tts.continuity"] = _CONTINUITY
    modules["vllm_omni.model_executor.stage_input_processors.chunk_size_utils"].compute_dynamic_initial_chunk_size = (
        lambda _active, _capacity, max_ic: max_ic
    )
    modules["vllm_omni.model_executor.stage_input_processors.chunk_size_utils"].max_ic_for_chunk_size = (
        lambda chunk_size: chunk_size
    )
    modules["vllm_omni.model_executor.stage_input_processors.tts_utils"].extract_language_from_prompt = lambda _prompt: (
        None
    )
    modules["vllm_omni.model_executor.stage_input_processors.tts_utils"].extract_language_from_request = (
        lambda _request: None
    )
    modules["vllm_omni.model_executor.stage_input_processors.tts_utils"].extract_speaker_from_prompt = lambda _prompt: (
        None
    )
    modules["vllm_omni.model_executor.stage_input_processors.tts_utils"].extract_speaker_from_request = (
        lambda _request: None
    )
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)


def _load_stage_processor_module(monkeypatch: pytest.MonkeyPatch):
    _install_stage_processor_stubs(monkeypatch)
    return _load_module_from_repo(
        "qwen3_tts_stage_processor_under_test",
        "vllm_omni/model_executor/stage_input_processors/qwen3_tts.py",
    )


def _request(
    request_id: str,
    *,
    finished: bool,
    initial_codec_chunk_frames: int | None = None,
    continuity_mode: str | None = None,
    continuity_cache_key: str | None = None,
):
    entries = {}
    if initial_codec_chunk_frames is not None:
        entries["initial_codec_chunk_frames"] = SimpleNamespace(list_data=[initial_codec_chunk_frames])
    if continuity_mode is not None:
        entries["continuity_mode"] = SimpleNamespace(list_data=[continuity_mode])
    if continuity_cache_key is not None:
        entries["continuity_cache_key"] = SimpleNamespace(list_data=[continuity_cache_key])
    return SimpleNamespace(
        external_req_id=request_id,
        is_finished=lambda: finished,
        additional_information=SimpleNamespace(entries=entries),
    )


def _transfer_manager():
    return SimpleNamespace(
        code_prompt_token_ids=defaultdict(list),
        connector=SimpleNamespace(
            config={
                "extra": {
                    "codec_chunk_frames": 10,
                    "codec_left_context_frames": 25,
                    "initial_codec_chunk_frames": 10,
                }
            }
        ),
        scheduler_max_num_seqs=8,
    )


def test_normalize_codec_frames_accepts_serialized_wrapper_and_2d_frames():
    frames = [[[1, 2], [3, 4], [5, 6]]]

    tensor = normalize_codec_frames(frames)

    assert tensor is not None
    assert tensor.dtype == torch.long
    assert tensor.tolist() == [[1, 2], [3, 4], [5, 6]]
    assert codec_frame_count(frames) == 3


def test_normalize_codec_frames_accepts_single_frame():
    tensor = normalize_codec_frames([[1, 2]])

    assert tensor is not None
    assert tensor.tolist() == [[1, 2]]


def test_continuity_mode_checks_exact_canonical_values():
    assert not continuity_mode_enabled(None, CONTINUITY_TALKER_ICL)
    assert not continuity_mode_enabled("off", CONTINUITY_TALKER_ICL)
    assert continuity_mode_enabled(CONTINUITY_TALKER_ICL, CONTINUITY_TALKER_ICL)
    assert not continuity_mode_enabled(CONTINUITY_TALKER_ICL, CONTINUITY_CODE2WAV_CONTEXT)
    assert continuity_mode_enabled("talker_icl+code2wav_context", CONTINUITY_TALKER_ICL)
    assert continuity_mode_enabled("talker_icl+code2wav_context", CONTINUITY_CODE2WAV_CONTEXT)

    with pytest.raises(ValueError, match="Unsupported"):
        continuity_mode_enabled("unknown", CONTINUITY_TALKER_ICL)


def test_codec_frame_lru_cache_returns_prior_frames_and_evicts_oldest_key():
    cache = CodecFrameLRUCache(max_sessions=2)

    assert cache.put("session-a", [[1, 2], [3, 4]])
    assert cache.put("session-b", [[5, 6]])
    assert cache.get("session-a").tolist() == [[1, 2], [3, 4]]
    assert cache.put("session-c", [[7, 8]])

    assert cache.get("session-b") is None
    assert cache.keys() == ["session-a", "session-c"]


def test_continuity_max_sessions_env_is_strict(monkeypatch):
    env_name = "VLLM_OMNI_QWEN3_TTS_CONTINUITY_MAX_SESSIONS"

    monkeypatch.delenv(env_name, raising=False)
    with pytest.raises(ValueError, match=env_name):
        continuity_max_sessions_from_env()

    monkeypatch.setenv(env_name, "7")
    assert continuity_max_sessions_from_env() == 7

    monkeypatch.setenv(env_name, "0")
    with pytest.raises(ValueError, match=env_name):
        continuity_max_sessions_from_env()

    monkeypatch.setenv(env_name, "not-an-int")
    with pytest.raises(ValueError, match=env_name):
        continuity_max_sessions_from_env()


def test_openai_speech_request_validates_continuity_cache_key_and_static_anchor():
    protocol = _load_module_from_repo(
        "qwen3_tts_audio_protocol_under_test",
        "vllm_omni/entrypoints/openai/protocol/audio.py",
    )

    request = protocol.OpenAICreateSpeechRequest(
        input="Hello",
        continuity_mode="talker_icl+code2wav_context",
        continuity_ref_text="Reference.",
        continuity_ref_code=[[1, 2], [3, 4]],
        continuity_cache_key=" session-a ",
    )

    assert request.continuity_mode == "talker_icl+code2wav_context"
    assert request.continuity_cache_key == "session-a"

    off_request = protocol.OpenAICreateSpeechRequest(input="Hello", continuity_mode="off")
    assert off_request.continuity_mode == "off"

    with pytest.raises(ValidationError, match="continuity_mode"):
        protocol.OpenAICreateSpeechRequest(input="Hello", continuity_mode="code2wav_context+talker_icl")

    with pytest.raises(ValidationError, match="continuity_cache_key"):
        protocol.OpenAICreateSpeechRequest(input="Hello", continuity_mode="code2wav_context")

    with pytest.raises(ValidationError, match="continuity_ref_code"):
        protocol.OpenAICreateSpeechRequest(
            input="Hello",
            continuity_mode="talker_icl",
            continuity_ref_text="Reference.",
        )


def test_openai_speech_request_forwards_continuity_fields_to_qwen3_tts_metadata():
    protocol = _read_repo_file("vllm_omni/entrypoints/openai/protocol/audio.py")
    serving = _read_repo_file("vllm_omni/entrypoints/openai/serving_speech.py")

    for field in (
        "continuity_mode",
        "continuity_ref_text",
        "continuity_ref_code",
        "continuity_cache_key",
    ):
        assert field in protocol
        assert f'params["{field}"] = [request.{field}]' in serving

    assert "Unsupported continuity_mode value:" in protocol
    assert "'continuity_ref_text' is required when continuity_mode includes 'talker_icl'" in protocol
    assert "'continuity_ref_code' is required when continuity_mode includes 'talker_icl'" in protocol
    assert "'continuity_cache_key' is required when continuity_mode includes 'code2wav_context'" in protocol


def test_customvoice_talker_icl_source_keeps_reference_on_talker_path_only():
    source = _read_repo_file("vllm_omni/model_executor/models/qwen3_tts/qwen3_tts_talker.py")

    assert "CONTINUITY_TALKER_ICL" in source
    assert 'info.get("continuity_mode"), CONTINUITY_TALKER_ICL' in source
    assert "continuity_ref_text" in source
    assert "continuity_ref_code" in source
    assert "talker_icl continuity requires continuity_ref_text" in source
    assert "talker_icl continuity requires continuity_ref_code" in source
    assert "Qwen3-TTS talker continuity ICL active" in source
    assert "ref_code=continuity_ref_code" in source
    assert "codec_lens = 1 + int(continuity_ref_code_len)" in source
    assert "prompt_len += text_lens + codec_lens if non_streaming_mode else codec_lens" in source


def test_code2wav_context_cache_source_contracts_are_present():
    source = _read_repo_file("vllm_omni/model_executor/stage_input_processors/qwen3_tts.py")

    assert "CONTINUITY_CODE2WAV_CONTEXT" in source
    assert '"continuity_cache_key"' in source
    assert "continuity_cache_for_transfer_manager(transfer_manager)" in source
    assert "code2wav_context continuity requires continuity_cache_key" in source
    assert "window_frames = context_frames + window_frames" in source
    assert "left_context_size += len(context_frames)" in source
    assert "left_context_size=left_context_size" in source


def test_code2wav_context_prepends_cached_frames_to_first_window(monkeypatch):
    monkeypatch.setenv("VLLM_OMNI_QWEN3_TTS_CONTINUITY_MAX_SESSIONS", "2")
    stage = _load_stage_processor_module(monkeypatch)
    transfer_manager = _transfer_manager()
    request_id = "request-window"
    current_frame = [1, 2, 3, 4]
    transfer_manager.code_prompt_token_ids[request_id] = [current_frame[:] for _ in range(10)]
    assert stage.continuity_cache_for_transfer_manager(transfer_manager).put(
        "session-a",
        [[9, 9, 9, 9], [8, 8, 8, 8]],
    )

    payload = stage.talker2code2wav_async_chunk(
        transfer_manager=transfer_manager,
        pooling_output={"codes": {"audio": torch.zeros((0,))}},
        request=_request(
            request_id,
            finished=False,
            initial_codec_chunk_frames=10,
            continuity_mode="code2wav_context",
            continuity_cache_key="session-a",
        ),
        is_finished=False,
    )

    assert payload is not None
    assert payload.meta.left_context_size == 2
    assert len(payload.codes.audio) == 4 * 12
    assert payload.codes.audio[:2].tolist() == [9, 8]


def test_code2wav_context_uses_prior_cache_before_storing_finished_request(monkeypatch):
    monkeypatch.setenv("VLLM_OMNI_QWEN3_TTS_CONTINUITY_MAX_SESSIONS", "2")
    stage = _load_stage_processor_module(monkeypatch)
    transfer_manager = _transfer_manager()
    request_id = "request-1"
    current_frame = [1, 2, 3, 4]
    transfer_manager.code_prompt_token_ids[request_id] = [current_frame[:] for _ in range(3)]
    assert stage.continuity_cache_for_transfer_manager(transfer_manager).put(
        "session-a",
        [[9, 9, 9, 9], [8, 8, 8, 8]],
    )

    payload = stage.talker2code2wav_async_chunk(
        transfer_manager=transfer_manager,
        pooling_output=None,
        request=_request(
            request_id,
            finished=True,
            continuity_mode="code2wav_context",
            continuity_cache_key="session-a",
        ),
        is_finished=True,
    )

    assert payload is not None
    assert payload.meta.left_context_size == 2
    assert payload.codes.audio[:2].tolist() == [9, 8]
    assert stage.continuity_cache_for_transfer_manager(transfer_manager).get("session-a").tolist() == [
        current_frame,
        current_frame,
        current_frame,
    ]


def test_code2wav_context_requires_cache_key_at_stage_processor(monkeypatch):
    stage = _load_stage_processor_module(monkeypatch)
    transfer_manager = _transfer_manager()
    request_id = "request-2"
    transfer_manager.code_prompt_token_ids[request_id] = [[1, 2, 3, 4] for _ in range(3)]

    with pytest.raises(ValueError, match="continuity_cache_key"):
        stage.talker2code2wav_async_chunk(
            transfer_manager=transfer_manager,
            pooling_output=None,
            request=_request(
                request_id,
                finished=True,
                continuity_mode="code2wav_context",
            ),
            is_finished=True,
        )
