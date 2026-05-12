# Qwen3-TTS Capture And Session Continuity

This fork carries two experimental diagnostics for System1 TTS evaluation on top of upstream vLLM-Omni Qwen3-TTS.

## Audio Capture

Set `VLLM_OMNI_TTS_CAPTURE_DIR` to enable per-request audio capture. When unset, no capture files are written.

Optional settings:

- `VLLM_OMNI_TTS_CAPTURE_MAX_FILES`: maximum metadata/audio pairs to retain in the capture directory. Default: `200`.

Each request writes one audio file (`.pcm`, `.wav`, or `.bin`) and one JSON metadata file containing request id, voice, instructions, input text, response format, stream mode, session id, byte count, duration estimate, and completion/error status.

## Session Continuity

Requests tagged with the `X-Session-Id` header can use prior generated Qwen3-TTS codec frames as bounded in-context reference for later turns in the same session. Requests without this header remain stateless.

Environment:

- `VLLM_OMNI_SESSION_STATE_DIR`: file-backed session state directory. Default: `/tmp/vllm_session_state`.
- `VLLM_OMNI_SESSION_CONTEXT_MODE`: `talker_icl` or `code2wav`. Default: `talker_icl`.
- `VLLM_OMNI_SESSION_MAX_CONTEXT_TURNS`: recent complete turns selected for context. Default: `1`; `0` keeps full rolling context subject to caps.
- `VLLM_OMNI_SESSION_MAX_CONTEXT_USES`: positive values force a stateless refresh after that many continued turns. Default: `0`.
- `VLLM_OMNI_SESSION_MAX_PREFIX_TOKENS`: hard cap for selected prior context plus current prompt estimate. Default: `500`.
- `VLLM_OMNI_SESSION_MAX_TURN_FRAMES`: rejects anomalously long generated turns. Default: `256`.
- `VLLM_OMNI_SESSION_ALLOW_CROSS_SIGNATURE_CONTEXT`: keeps context across voice/instruction changes for experiments. Default: `false`.
- `VLLM_OMNI_SESSION_CODE2WAV_REF_CONTEXT`: additionally passes ICL ref code as Code2Wav decoder context. Default: `false`.

The implementation records request text at API ingress, records codec frames from the chunk-transfer cleanup hook, and uses those paired prior text/codec turns on the next request. It intentionally logs binding misses, signature resets, cap trims/resets, and empty-output resets so continuity failures are visible during evaluation.
