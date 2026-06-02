import math
from typing import Any, Literal

import numpy as np
from pydantic import AliasChoices, BaseModel, Field, field_validator, model_validator

_MAX_EMBEDDING_DIM = 8192


class OpenAICreateSpeechRequest(BaseModel):
    input: str
    request_id: str = Field(
        description="Caller-provided request id used as the vLLM engine request id.",
    )
    session_id: str | None = Field(
        default=None,
        description="Caller-provided continuity session id. Omit for stateless speech.",
    )
    model: str | None = None
    # Accept both "voice" (OpenAI convention) and "speaker" (model/internal
    # convention) as input keys.  Intentionally global — all TTS backends
    # (Qwen3-TTS, Voxtral, Fish Speech) use this field for the speaker name.
    voice: str | None = Field(
        default=None,
        validation_alias=AliasChoices("voice", "speaker"),
        description="Speaker/voice to use. For Qwen3-TTS: vivian, ryan, aiden, etc.",
    )
    instructions: str | None = Field(
        default=None,
        description="Instructions for voice style/emotion (maps to 'instruct' for Qwen3-TTS)",
    )
    response_format: Literal["wav", "pcm", "flac", "mp3", "aac", "opus"] = "wav"
    speed: float | None = Field(
        default=1.0,
        ge=0.25,
        le=4.0,
    )
    stream_format: Literal["sse", "audio"] | None = "audio"
    stream: bool = Field(
        default=False,
        description=(
            "If true, stream raw PCM audio chunks as they are decoded. "
            "Requires response_format='pcm'. Speed adjustment is not supported when streaming."
        ),
    )

    # Qwen3-TTS specific parameters
    task_type: Literal["CustomVoice", "VoiceDesign", "Base"] | None = Field(
        default=None,
        description="TTS task type: CustomVoice, VoiceDesign, or Base (voice clone)",
    )
    language: str | None = Field(
        default=None,
        description="Language code (e.g., 'Chinese', 'English', 'Auto')",
    )
    ref_audio: str | None = Field(
        default=None,
        description="Reference audio for voice cloning (Base task). URL, base64, or file URI.",
    )
    ref_text: str | None = Field(
        default=None,
        description="Transcript of reference audio for voice cloning (Base task)",
    )
    x_vector_only_mode: bool | None = Field(
        default=None,
        description="Use speaker embedding only without in-context learning (Base task)",
    )
    speaker_embedding: list[float] | None = Field(
        default=None,
        max_length=_MAX_EMBEDDING_DIM,
        description="Pre-computed speaker embedding vector (1024-dim for 0.6B, "
        "2048-dim for 1.7B). Skips speaker encoder extraction from ref_audio. "
        "Implies x_vector_only_mode=True. Mutually exclusive with ref_audio.",
    )
    max_new_tokens: int | None = Field(
        default=None,
        description="Maximum tokens to generate",
    )
    seed: int | None = Field(
        default=None,
        ge=0,
        le=2**63 - 1,
        description="Random seed for reproducible generation. When set, ensures "
        "deterministic output for the same input text and seed value.",
    )
    initial_codec_chunk_frames: int | None = Field(
        default=None,
        ge=0,
        description="Per-request initial chunk size override. If null, computed dynamically based on server load.",
    )
    continuity_mode: str | None = Field(
        default=None,
        description=(
            "Qwen3-TTS continuity primitive to enable for this request. Supported values are "
            "'talker_icl', 'code2wav_context', and 'talker_icl+code2wav_context'."
        ),
    )
    continuity_anchor_name: str | None = Field(
        default=None,
        description="Qwen3-TTS startup-loaded continuity anchor id for talker ICL.",
    )
    continuity_cache_key: str | None = Field(
        default=None,
        description="Opaque Qwen3-TTS continuity cache key used for Code2Wav left context.",
    )
    extra_params: dict[str, Any] | None = Field(
        default=None,
        description=("Optional model-specific parameters passed directly to the model's extra_args."),
    )

    @field_validator("stream_format")
    @classmethod
    def validate_stream_format(cls, v: str) -> str:
        if v == "sse":
            raise ValueError("'sse' is not a supported stream_format yet. Please use 'audio'.")
        return v

    @field_validator("request_id", "session_id")
    @classmethod
    def validate_identifier(cls, v: str | None) -> str | None:
        if v is None:
            return None
        normalized = v.strip()
        if not normalized:
            raise ValueError("identifier fields must not be blank")
        return normalized

    @field_validator("speaker_embedding")
    @classmethod
    def validate_speaker_embedding(cls, v: list[float] | None) -> list[float] | None:
        if v is not None and not all(math.isfinite(x) for x in v):
            raise ValueError("'speaker_embedding' values must be finite (no NaN or Inf)")
        return v

    @field_validator("continuity_mode")
    @classmethod
    def validate_continuity_mode(cls, v: str | None) -> str | None:
        if v is None:
            return None
        allowed = ("off", "talker_icl", "code2wav_context", "talker_icl+code2wav_context")
        if v not in allowed:
            raise ValueError(f"Unsupported continuity_mode value: {v!r}; expected one of: {', '.join(allowed)}")
        return v

    @field_validator("continuity_anchor_name")
    @classmethod
    def validate_continuity_anchor_name(cls, v: str | None) -> str | None:
        if v is None:
            return None
        name = v.strip()
        if not name:
            raise ValueError("'continuity_anchor_name' must be non-empty when provided")
        if "/" in name or "\\" in name:
            raise ValueError("'continuity_anchor_name' must be a simple anchor id")
        return name

    @field_validator("continuity_cache_key")
    @classmethod
    def validate_continuity_cache_key(cls, v: str | None) -> str | None:
        if v is None:
            return None
        key = v.strip()
        if not key:
            raise ValueError("'continuity_cache_key' must be non-empty when provided")
        return key

    @model_validator(mode="after")
    def validate_embedding_constraints(self) -> "OpenAICreateSpeechRequest":
        if self.speaker_embedding is not None:
            if self.ref_audio is not None:
                raise ValueError("'speaker_embedding' and 'ref_audio' are mutually exclusive")
        return self

    @model_validator(mode="after")
    def validate_continuity_constraints(self) -> "OpenAICreateSpeechRequest":
        modes = set(self.continuity_mode.split("+")) if self.continuity_mode else set()
        if "talker_icl" in modes and self.continuity_anchor_name is None:
            raise ValueError("'continuity_anchor_name' is required when continuity_mode includes 'talker_icl'")
        if "code2wav_context" in modes and self.continuity_cache_key is None:
            raise ValueError("'continuity_cache_key' is required when continuity_mode includes 'code2wav_context'")
        return self

    @model_validator(mode="after")
    def validate_streaming_constraints(self) -> "OpenAICreateSpeechRequest":
        if self.stream:
            if self.response_format not in ("pcm", "wav"):
                raise ValueError(
                    "Streaming (stream=true) requires response_format='pcm' or 'wav'. "
                    f"Got response_format='{self.response_format}'."
                )
            if self.speed is None:
                self.speed = 1.0
            elif self.speed != 1.0:
                raise ValueError(
                    "Speed adjustment is not supported when streaming (stream=true). Set speed=1.0 or omit it."
                )
        return self


class OpenAICreateAudioGenerateRequest(BaseModel):
    """Request model for audio generation via diffusion models (e.g. Stable Audio)."""

    input: str = Field(
        description="Text prompt describing the audio to generate",
    )
    model: str | None = None
    response_format: Literal["wav", "pcm", "flac", "mp3", "aac", "opus"] = "wav"
    speed: float | None = Field(
        default=1.0,
        ge=0.25,
        le=4.0,
    )
    stream_format: Literal["sse", "audio"] | None = "audio"
    audio_length: float | None = Field(
        default=None,
        description="Audio length in seconds",
    )
    audio_start: float | None = Field(
        default=0.0,
        description="Audio start time in seconds",
    )
    negative_prompt: str | None = Field(
        default=None,
        description="Negative prompt for classifier-free guidance",
    )
    guidance_scale: float | None = Field(
        default=None,
        description="Guidance scale for diffusion models",
    )
    num_inference_steps: int | None = Field(
        default=None,
        description="Number of inference steps",
    )
    seed: int | None = Field(
        default=None,
        description="Random seed for reproducibility",
    )

    @field_validator("stream_format")
    @classmethod
    def validate_stream_format(cls, v: str) -> str:
        if v == "sse":
            raise ValueError("'sse' is not a supported stream_format yet. Please use 'audio'.")
        return v


class CreateAudio(BaseModel):
    audio_tensor: np.ndarray
    sample_rate: int = 24000
    response_format: str = "wav"
    speed: float = 1.0
    stream_format: Literal["sse", "audio"] | None = "audio"
    base64_encode: bool = True

    class Config:
        arbitrary_types_allowed = True


class AudioResponse(BaseModel):
    audio_data: bytes | str
    media_type: str


# --- Batch Speech Models ---


class SpeechBatchItem(BaseModel):
    """Per-item input for batch speech.

    `input` and `request_id` are required; all other fields override the
    batch-level defaults when set.
    """

    input: str
    request_id: str
    session_id: str | None = None
    voice: str | None = None
    instructions: str | None = None
    response_format: Literal["wav", "pcm", "flac", "mp3", "aac", "opus"] | None = None
    speed: float | None = Field(default=None, ge=0.25, le=4.0)
    task_type: Literal["CustomVoice", "VoiceDesign", "Base"] | None = None
    language: str | None = None
    ref_audio: str | None = None
    ref_text: str | None = None
    x_vector_only_mode: bool | None = None
    max_new_tokens: int | None = None
    initial_codec_chunk_frames: int | None = Field(default=None, ge=0)

    @field_validator("request_id", "session_id")
    @classmethod
    def validate_identifier(cls, v: str | None) -> str | None:
        return OpenAICreateSpeechRequest.validate_identifier(v)


class BatchSpeechRequest(BaseModel):
    """Top-level request for batch speech generation.
    Fields here act as shared defaults; per-item overrides win."""

    model: str | None = None
    session_id: str | None = None
    items: list[SpeechBatchItem] = Field(..., min_length=1)
    voice: str | None = None
    instructions: str | None = None
    response_format: Literal["wav", "pcm", "flac", "mp3", "aac", "opus"] = "wav"
    speed: float | None = Field(default=1.0, ge=0.25, le=4.0)
    task_type: Literal["CustomVoice", "VoiceDesign", "Base"] | None = None
    language: str | None = None
    ref_audio: str | None = None
    ref_text: str | None = None
    x_vector_only_mode: bool | None = None
    max_new_tokens: int | None = None
    initial_codec_chunk_frames: int | None = Field(default=None, ge=0)

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, v: str | None) -> str | None:
        return OpenAICreateSpeechRequest.validate_identifier(v)


class SpeechBatchItemResult(BaseModel):
    index: int
    status: Literal["success", "error"]
    audio_data: str | None = None
    media_type: str | None = None
    error: str | None = None


class BatchSpeechResponse(BaseModel):
    id: str
    results: list[SpeechBatchItemResult]
    total: int
    succeeded: int
    failed: int


class StreamingSpeechSessionConfig(BaseModel):
    """Configuration sent as the first WebSocket message for streaming TTS."""

    request_id: str
    session_id: str | None = None
    model: str | None = None
    voice: str | None = None
    task_type: Literal["CustomVoice", "VoiceDesign", "Base"] | None = None
    language: str | None = None
    instructions: str | None = None
    response_format: Literal["wav", "pcm", "flac", "mp3", "aac", "opus"] = "wav"
    speed: float | None = Field(default=1.0, ge=0.25, le=4.0)
    max_new_tokens: int | None = Field(default=None, ge=1)
    initial_codec_chunk_frames: int | None = Field(
        default=None,
        ge=0,
        description="Initial chunk size for reduced TTFA. Overrides stage config for this session.",
    )
    ref_audio: str | None = None
    ref_text: str | None = None
    x_vector_only_mode: bool | None = None
    speaker_embedding: list[float] | None = Field(
        default=None,
        max_length=_MAX_EMBEDDING_DIM,
        description="Pre-computed speaker embedding vector. Mutually exclusive with ref_audio.",
    )
    stream_audio: bool = Field(
        default=False,
        description=(
            "If true, send raw PCM audio chunks progressively over WebSocket. "
            "Requires response_format='pcm'. Speed adjustment is not supported when streaming."
        ),
    )
    split_granularity: Literal["sentence", "clause"] = Field(
        default="sentence",
        description=(
            "Text splitting granularity: 'sentence' splits on .!?。！？, "
            "'clause' also splits on CJK commas ， and semicolons ；."
        ),
    )

    @field_validator("request_id", "session_id")
    @classmethod
    def validate_identifier(cls, v: str | None) -> str | None:
        return OpenAICreateSpeechRequest.validate_identifier(v)

    @model_validator(mode="after")
    def validate_streaming_constraints(self) -> "StreamingSpeechSessionConfig":
        if self.stream_audio:
            if self.response_format != "pcm":
                raise ValueError(
                    "WebSocket streaming audio (stream_audio=true) requires response_format='pcm'. "
                    f"Got response_format='{self.response_format}'."
                )
            if self.speed is None:
                self.speed = 1.0
            elif self.speed != 1.0:
                raise ValueError("Speed adjustment is not supported when stream_audio=true. Set speed=1.0 or omit it.")
        return self
