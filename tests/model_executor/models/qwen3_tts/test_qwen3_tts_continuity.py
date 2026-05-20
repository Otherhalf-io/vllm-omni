import builtins
import importlib.util
import json
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
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_CONTINUITY = _load_continuity_module()
CONTINUITY_CODE2WAV_CONTEXT = _CONTINUITY.CONTINUITY_CODE2WAV_CONTEXT
CONTINUITY_TALKER_ICL = _CONTINUITY.CONTINUITY_TALKER_ICL
CodecFrameLRUCache = _CONTINUITY.CodecFrameLRUCache
cache_key_hash = _CONTINUITY.cache_key_hash
codec_frame_count = _CONTINUITY.codec_frame_count
collect_memory_attributes = _CONTINUITY.collect_memory_attributes
continuity_max_sessions_from_env = _CONTINUITY.continuity_max_sessions_from_env
continuity_mode_enabled = _CONTINUITY.continuity_mode_enabled
get_continuity_anchor = _CONTINUITY.get_continuity_anchor
load_continuity_anchors_from_dir = _CONTINUITY.load_continuity_anchors_from_dir
normalize_codec_frames = _CONTINUITY.normalize_codec_frames
reset_continuity_anchors_for_test = _CONTINUITY.reset_continuity_anchors_for_test
set_span_attributes = _CONTINUITY.set_span_attributes
telemetry_span = _CONTINUITY.telemetry_span


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


def test_load_continuity_anchors_from_manifest(tmp_path):
    anchor_dir = tmp_path / "anchors"
    anchor_dir.mkdir()
    (anchor_dir / "anchors.json").write_text(
        json.dumps(
            {
                "version": 1,
                "anchors": {
                    "maya_s2_0028_jsonl": {
                        "ref_text_path": "maya_s2_0028_jsonl/ref.txt",
                        "codec_path": "maya_s2_0028_jsonl/codec.json",
                        "metadata_path": "maya_s2_0028_jsonl/metadata.json",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    (anchor_dir / "maya_s2_0028_jsonl").mkdir()
    (anchor_dir / "maya_s2_0028_jsonl/ref.txt").write_text("Stable reference.", encoding="utf-8")
    (anchor_dir / "maya_s2_0028_jsonl/codec.json").write_text(
        json.dumps([[1, 2], [3, 4]]),
        encoding="utf-8",
    )
    (anchor_dir / "maya_s2_0028_jsonl/metadata.json").write_text(
        json.dumps({"source_kind": "jsonl"}),
        encoding="utf-8",
    )

    anchors = load_continuity_anchors_from_dir(anchor_dir)

    anchor = anchors["maya_s2_0028_jsonl"]
    assert anchor.ref_text == "Stable reference."
    assert anchor.ref_code.tolist() == [[1, 2], [3, 4]]
    assert anchor.source_kind == "jsonl"
    assert anchor.frame_count == 2
    assert anchor.quantizer_count == 2


def test_load_continuity_anchors_rejects_duplicate_normalized_names(tmp_path):
    anchor_dir = tmp_path / "anchors"
    anchor_dir.mkdir()
    (anchor_dir / "ref").mkdir()
    (anchor_dir / "anchors.json").write_text(
        json.dumps(
            {
                "anchors": {
                    "ref": {
                        "ref_text_path": "ref/ref.txt",
                        "codec_path": "ref/codec.json",
                    },
                    " ref ": {
                        "ref_text_path": "ref/ref.txt",
                        "codec_path": "ref/codec.json",
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    (anchor_dir / "ref/ref.txt").write_text("Reference.", encoding="utf-8")
    (anchor_dir / "ref/codec.json").write_text(json.dumps([[1, 2]]), encoding="utf-8")

    with pytest.raises(ValueError, match="Duplicate continuity anchor name"):
        load_continuity_anchors_from_dir(anchor_dir)


def test_get_continuity_anchor_uses_env_registry_and_rejects_unknown(tmp_path, monkeypatch):
    anchor_dir = tmp_path / "anchors"
    (anchor_dir / "ref").mkdir(parents=True)
    (anchor_dir / "anchors.json").write_text(
        json.dumps(
            {
                "anchors": {
                    "ref": {
                        "ref_text_path": "ref/ref.txt",
                        "codec_path": "ref/codec.json",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    (anchor_dir / "ref/ref.txt").write_text("Reference.", encoding="utf-8")
    (anchor_dir / "ref/codec.json").write_text(json.dumps([[1, 2]]), encoding="utf-8")

    reset_continuity_anchors_for_test()
    monkeypatch.setenv("VLLM_OMNI_QWEN3_TTS_CONTINUITY_ANCHORS_DIR", str(anchor_dir))

    assert get_continuity_anchor("ref").ref_text == "Reference."
    with pytest.raises(ValueError, match="Unknown Qwen3-TTS continuity anchor"):
        get_continuity_anchor("missing")

    reset_continuity_anchors_for_test()


def test_anchor_lookup_telemetry_records_anchor_metadata(tmp_path, monkeypatch):
    spans = []

    class FakeSpan:
        def __init__(self, name, attributes):
            self.name = name
            self.attributes = dict(attributes or {})

        def __enter__(self):
            spans.append(self)
            return self

        def __exit__(self, _exc_type, _exc, _tb):
            return False

        def is_recording(self):
            return True

        def set_attribute(self, key, value):
            self.attributes[key] = value

    class FakeTracer:
        def start_as_current_span(self, name, attributes=None):
            return FakeSpan(name, attributes)

    class FakeTrace:
        def get_tracer(self, _name):
            return FakeTracer()

    anchor_dir = tmp_path / "anchors"
    (anchor_dir / "ref").mkdir(parents=True)
    (anchor_dir / "anchors.json").write_text(
        json.dumps(
            {
                "anchors": {
                    "ref": {
                        "ref_text_path": "ref/ref.txt",
                        "codec_path": "ref/codec.json",
                        "metadata_path": "ref/metadata.json",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    (anchor_dir / "ref/ref.txt").write_text("Reference.", encoding="utf-8")
    (anchor_dir / "ref/codec.json").write_text(json.dumps([[1, 2], [3, 4]]), encoding="utf-8")
    (anchor_dir / "ref/metadata.json").write_text(json.dumps({"source_kind": "jsonl"}), encoding="utf-8")

    monkeypatch.delenv("OTEL_TRACES_EXPORTER", raising=False)
    monkeypatch.setattr(_CONTINUITY, "_OTEL_CONFIGURED", False)
    monkeypatch.setattr(_CONTINUITY, "_otel_trace", FakeTrace())
    monkeypatch.setenv("VLLM_OMNI_QWEN3_TTS_CONTINUITY_ANCHORS_DIR", str(anchor_dir))
    reset_continuity_anchors_for_test()

    assert get_continuity_anchor("ref").ref_text == "Reference."

    assert spans[-1].name == "qwen3_tts.continuity.anchor.lookup"
    assert spans[-1].attributes["qwen3_tts.continuity.anchor.name"] == "ref"
    assert spans[-1].attributes["qwen3_tts.continuity.anchor.loaded"] is True
    assert spans[-1].attributes["qwen3_tts.continuity.anchor.source_kind"] == "jsonl"
    assert spans[-1].attributes["qwen3_tts.continuity.codec.frames"] == 2
    assert "qwen3_tts.continuity.anchor.lookup.duration_us" in spans[-1].attributes

    reset_continuity_anchors_for_test()


def test_load_continuity_anchors_rejects_malformed_codecs(tmp_path):
    anchor_dir = tmp_path / "anchors"
    (anchor_dir / "bad").mkdir(parents=True)
    (anchor_dir / "anchors.json").write_text(
        json.dumps(
            {
                "anchors": {
                    "bad": {
                        "ref_text_path": "bad/ref.txt",
                        "codec_path": "bad/codec.json",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    (anchor_dir / "bad/ref.txt").write_text("Reference.", encoding="utf-8")
    (anchor_dir / "bad/codec.json").write_text(json.dumps([[1, 2], [3]]), encoding="utf-8")

    with pytest.raises(ValueError, match="Continuity anchor codec frames are invalid"):
        load_continuity_anchors_from_dir(anchor_dir)


def test_codec_frame_lru_cache_returns_prior_frames_and_evicts_oldest_key():
    cache = CodecFrameLRUCache(max_sessions=2)

    assert cache.put("session-a", [[1, 2], [3, 4]])
    assert cache.put("session-b", [[5, 6]])
    assert cache.get("session-a").tolist() == [[1, 2], [3, 4]]
    assert cache.put("session-c", [[7, 8]])

    assert cache.get("session-b") is None
    assert cache.keys() == ["session-a", "session-c"]


def test_telemetry_helpers_are_optional_and_low_cardinality():
    assert cache_key_hash("maya_warm:room-42") == cache_key_hash("maya_warm:room-42")
    assert cache_key_hash("maya_warm:room-42") != "maya_warm:room-42"

    attrs = collect_memory_attributes("probe")
    assert isinstance(attrs, dict)
    assert all(key.startswith("probe.") for key in attrs)
    assert all(value >= 0 for value in attrs.values())

    with telemetry_span("qwen3_tts.test", {"test.attribute": 1}, capture_memory=True) as span:
        set_span_attributes(span, {"test.attribute.after": 2})


def test_cache_telemetry_records_read_write_duration_and_cache_attributes(monkeypatch):
    spans = []

    class FakeSpan:
        def __init__(self, name, attributes):
            self.name = name
            self.attributes = dict(attributes or {})

        def __enter__(self):
            spans.append(self)
            return self

        def __exit__(self, _exc_type, _exc, _tb):
            return False

        def is_recording(self):
            return True

        def set_attribute(self, key, value):
            self.attributes[key] = value

    class FakeTracer:
        def start_as_current_span(self, name, attributes=None):
            return FakeSpan(name, attributes)

    class FakeTrace:
        def get_tracer(self, _name):
            return FakeTracer()

    monkeypatch.delenv("OTEL_TRACES_EXPORTER", raising=False)
    monkeypatch.setattr(_CONTINUITY, "_OTEL_CONFIGURED", False)
    monkeypatch.setattr(_CONTINUITY, "_otel_trace", FakeTrace())

    cache = CodecFrameLRUCache(max_sessions=2)
    assert cache.put("session-a", [[1, 2], [3, 4]])
    assert cache.get("session-a").tolist() == [[1, 2], [3, 4]]

    write_span = next(span for span in spans if span.name == "qwen3_tts.continuity.cache.write")
    read_span = next(span for span in spans if span.name == "qwen3_tts.continuity.cache.read")
    for span in (write_span, read_span):
        assert isinstance(span.attributes["qwen3_tts.continuity.cache.duration_us"], int)
        assert span.attributes["qwen3_tts.continuity.cache.duration_us"] >= 0
        assert span.attributes["qwen3_tts.continuity.cache.key_hash"] == cache_key_hash("session-a")
        assert "session-a" not in span.attributes.values()

    assert write_span.attributes["qwen3_tts.continuity.cache.evicted"] == 0
    assert write_span.attributes["qwen3_tts.continuity.codec.frames"] == 2
    assert read_span.attributes["qwen3_tts.continuity.cache.hit"] is True
    assert read_span.attributes["qwen3_tts.continuity.cache.size"] == 1


def test_console_telemetry_does_not_require_otlp_exporter(monkeypatch):
    providers = []

    class FakeResource:
        @staticmethod
        def create(attributes):
            return attributes

    class FakeTracerProvider:
        def __init__(self, resource):
            self.resource = resource
            self.span_processors = []

        def add_span_processor(self, span_processor):
            self.span_processors.append(span_processor)

    class FakeConsoleSpanExporter:
        pass

    class FakeSimpleSpanProcessor:
        def __init__(self, exporter):
            self.exporter = exporter

    class FakeTrace:
        def set_tracer_provider(self, provider):
            providers.append(provider)

    modules = {
        "opentelemetry": types.ModuleType("opentelemetry"),
        "opentelemetry.sdk": types.ModuleType("opentelemetry.sdk"),
        "opentelemetry.sdk.resources": types.ModuleType("opentelemetry.sdk.resources"),
        "opentelemetry.sdk.trace": types.ModuleType("opentelemetry.sdk.trace"),
        "opentelemetry.sdk.trace.export": types.ModuleType("opentelemetry.sdk.trace.export"),
    }
    modules["opentelemetry.sdk.resources"].Resource = FakeResource
    modules["opentelemetry.sdk.trace"].TracerProvider = FakeTracerProvider
    modules["opentelemetry.sdk.trace.export"].ConsoleSpanExporter = FakeConsoleSpanExporter
    modules["opentelemetry.sdk.trace.export"].SimpleSpanProcessor = FakeSimpleSpanProcessor
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    real_import = builtins.__import__

    def import_without_otlp(name, *args, **kwargs):
        if name.startswith("opentelemetry.exporter"):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_otlp)
    monkeypatch.setenv("OTEL_TRACES_EXPORTER", "console")
    monkeypatch.setattr(_CONTINUITY, "_OTEL_CONFIGURED", False)
    monkeypatch.setattr(_CONTINUITY, "_otel_trace", FakeTrace())

    _CONTINUITY.configure_telemetry_from_env()

    assert _CONTINUITY._OTEL_CONFIGURED is True
    assert len(providers) == 1
    assert len(providers[0].span_processors) == 1


def test_failed_telemetry_exporter_import_is_not_sticky(monkeypatch):
    providers = []

    class FakeResource:
        @staticmethod
        def create(attributes):
            return attributes

    class FakeTracerProvider:
        def __init__(self, resource):
            self.resource = resource
            self.span_processors = []

        def add_span_processor(self, span_processor):
            self.span_processors.append(span_processor)

    class FakeTrace:
        def set_tracer_provider(self, provider):
            providers.append(provider)

    modules = {
        "opentelemetry": types.ModuleType("opentelemetry"),
        "opentelemetry.sdk": types.ModuleType("opentelemetry.sdk"),
        "opentelemetry.sdk.resources": types.ModuleType("opentelemetry.sdk.resources"),
        "opentelemetry.sdk.trace": types.ModuleType("opentelemetry.sdk.trace"),
    }
    modules["opentelemetry.sdk.resources"].Resource = FakeResource
    modules["opentelemetry.sdk.trace"].TracerProvider = FakeTracerProvider
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    real_import = builtins.__import__

    def import_without_otlp(name, *args, **kwargs):
        if name.startswith("opentelemetry.exporter"):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_otlp)
    monkeypatch.setenv("OTEL_TRACES_EXPORTER", "otlp")
    monkeypatch.setattr(_CONTINUITY, "_OTEL_CONFIGURED", False)
    monkeypatch.setattr(_CONTINUITY, "_otel_trace", FakeTrace())

    _CONTINUITY.configure_telemetry_from_env()

    assert _CONTINUITY._OTEL_CONFIGURED is False
    assert providers == []


def test_telemetry_source_records_partial_progress_and_keeps_cache_spans_light():
    serving = _read_repo_file("vllm_omni/entrypoints/openai/serving_speech.py")
    continuity = _read_repo_file("vllm_omni/model_executor/models/qwen3_tts/continuity.py")

    assert "configure_telemetry_from_env()" in continuity
    assert "OTEL_TRACES_EXPORTER" in continuity
    assert "LANGFUSE_BASE_URL" in continuity
    assert "OTEL_EXPORTER_OTLP_ENDPOINT" in continuity
    assert "BatchSpanProcessor(OTLPSpanExporter())" in continuity
    assert '"service.name": os.environ.get("OTEL_SERVICE_NAME", "vllm-omni")' in continuity
    assert "time.perf_counter_ns()" in continuity
    assert 'duration_attribute="qwen3_tts.continuity.cache.duration_us"' in continuity

    assert "finally:\n                    set_span_attributes(\n                        span," in serving
    assert '"vllm_omni.audio.chunk_count": chunk_count' in serving
    assert 'audio_bytes: bytes | str = b""' in serving
    assert 'attrs = {"vllm_omni.audio.bytes": len(audio_bytes)}' in serving
    assert '"vllm_omni.continuity.mode": continuity_mode' in serving
    assert '"vllm_omni.continuity.cache_key.present": request.continuity_cache_key is not None' in serving
    assert '"vllm_omni.continuity.anchor_name.present": request.continuity_anchor_name is not None' in serving
    assert 'telemetry_attrs["vllm_omni.continuity.anchor_name"] = request.continuity_anchor_name' in serving

    assert "def _read_proc_kib_field" in continuity
    assert "process.memory.max_rss_bytes" not in continuity
    assert "torch.accelerator.current_device_index()" in continuity
    assert "except (AttributeError, RuntimeError):" in continuity
    assert "capture_memory=False" in continuity


def test_tokenizer_decoder_forward_documents_cache_position_for_clean_startup_logs():
    tokenizer_v2 = _read_repo_file(
        "vllm_omni/model_executor/models/qwen3_tts/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py"
    )

    assert "@auto_docstring\n    def forward(\n        self,\n        input_ids=None," in tokenizer_v2
    assert "cache_position=None,\n        **kwargs,\n    ) -> BaseModelOutputWithPast:" in tokenizer_v2
    assert "cache_position (`torch.LongTensor`, *optional*):" in tokenizer_v2
    assert "Absolute positions for the current tokens in the cache." in tokenizer_v2


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
        continuity_anchor_name=" maya_s2_0028_jsonl ",
        continuity_cache_key=" session-a ",
    )

    assert request.continuity_mode == "talker_icl+code2wav_context"
    assert request.continuity_anchor_name == "maya_s2_0028_jsonl"
    assert request.continuity_cache_key == "session-a"

    off_request = protocol.OpenAICreateSpeechRequest(input="Hello", continuity_mode="off")
    assert off_request.continuity_mode == "off"

    with pytest.raises(ValidationError, match="continuity_mode"):
        protocol.OpenAICreateSpeechRequest(input="Hello", continuity_mode="code2wav_context+talker_icl")

    with pytest.raises(ValidationError, match="continuity_cache_key"):
        protocol.OpenAICreateSpeechRequest(input="Hello", continuity_mode="code2wav_context")

    with pytest.raises(ValidationError, match="continuity_anchor_name"):
        protocol.OpenAICreateSpeechRequest(
            input="Hello",
            continuity_mode="talker_icl",
        )


def test_openai_speech_request_forwards_continuity_fields_to_qwen3_tts_metadata():
    protocol = _read_repo_file("vllm_omni/entrypoints/openai/protocol/audio.py")
    serving = _read_repo_file("vllm_omni/entrypoints/openai/serving_speech.py")

    for field in (
        "continuity_mode",
        "continuity_anchor_name",
        "continuity_cache_key",
    ):
        assert field in protocol
        assert f'params["{field}"] = [request.{field}]' in serving
    assert "continuity_ref_text" not in protocol
    assert "continuity_ref_code" not in protocol
    assert 'params["continuity_ref_text"]' not in serving
    assert 'params["continuity_ref_code"]' not in serving

    assert "Unsupported continuity_mode value:" in protocol
    assert "'continuity_anchor_name' is required when continuity_mode includes 'talker_icl'" in protocol
    assert "'continuity_cache_key' is required when continuity_mode includes 'code2wav_context'" in protocol


def test_customvoice_talker_icl_source_keeps_reference_on_talker_path_only():
    source = _read_repo_file("vllm_omni/model_executor/models/qwen3_tts/qwen3_tts_talker.py")

    assert "CONTINUITY_TALKER_ICL" in source
    assert 'info.get("continuity_mode"), CONTINUITY_TALKER_ICL' in source
    assert "get_continuity_anchor" in source
    assert "continuity_anchor_name" in source
    assert "Qwen3-TTS talker continuity ICL active" in source
    assert "ref_code=continuity_ref_code" in source
    assert "codec_lens = 1 + continuity_anchor.frame_count" in source
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


def test_code2wav_model_skips_terminal_token_without_malformed_warning():
    source = _read_repo_file("vllm_omni/model_executor/models/qwen3_tts/qwen3_tts_code2wav.py")

    assert "if n == 1:" in source
    assert "Code2Wav received one terminal token" in source
    assert "logger.debug(" in source
    assert "elif n > 0:" in source
    assert "logger.warning(" in source
    assert "not divisible by num_quantizers" in source


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
