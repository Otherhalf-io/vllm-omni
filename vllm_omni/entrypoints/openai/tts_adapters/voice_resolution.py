# SPDX-License-Identifier: Apache-2.0
"""Pure CustomVoice speaker-resolution rules.

Deliberately dependency-free (no vLLM, no torch) so the rules can be unit
tested in a lightweight environment, and so the serving layer and the adapter
share one implementation instead of each carrying its own copy.

The rules exist because an unresolvable speaker name is not a benign fallback.
The name is only checked against the checkpoint's ``spk_id`` map inside the
model's ``preprocess`` on the GPU worker, where a ``ValueError`` kills
EngineCore for every in-flight session instead of failing one request. Both the
caller-supplied name and any substituted default must therefore be settled at
request admission.
"""

from collections.abc import Iterable

#: Upstream's CustomVoice fallback speaker. Present in stock Qwen3-TTS, absent
#: from fine-tuned checkpoints that ship their own ``spk_id`` map.
STOCK_DEFAULT_CUSTOM_VOICE = "vivian"


def normalize_speaker(value: str | None) -> str | None:
    """Lowercase/strip a speaker name, mapping blank to None."""
    if value is None:
        return None
    name = value.strip().lower()
    return name or None


def resolve_default_custom_voice(
    supported: Iterable[str],
    *,
    configured: str | None = None,
    stock_default: str = STOCK_DEFAULT_CUSTOM_VOICE,
) -> tuple[str | None, str | None]:
    """Resolve the CustomVoice default speaker for a deployment.

    Returns ``(default, problem)``. ``default`` is None when this model has no
    usable default, in which case CustomVoice requests must name a voice
    explicitly. ``problem`` is a human-readable diagnostic to log, set only when
    a default was expected but is unusable.
    """
    supported_set = {s.strip().lower() for s in supported if isinstance(s, str) and s.strip()}
    requested_default = normalize_speaker(configured)

    if requested_default is not None:
        if requested_default in supported_set:
            return requested_default, None
        return None, (
            f"configured default speaker {requested_default!r} is not supported by this model "
            f"({', '.join(sorted(supported_set)) or 'none'}); "
            "CustomVoice requests must specify `voice` explicitly"
        )

    stock = normalize_speaker(stock_default)
    if stock is not None and stock in supported_set:
        return stock, None

    if not supported_set:
        return None, None

    return None, (
        f"default CustomVoice speaker {stock!r} is not in this checkpoint's speaker map "
        f"({', '.join(sorted(supported_set))}); CustomVoice requests must specify `voice` explicitly"
    )


def resolve_custom_voice(
    requested: str | None,
    default: str | None,
    supported: Iterable[str],
) -> tuple[str | None, str | None]:
    """Resolve and validate the canonical CustomVoice speaker.

    Returns ``(effective, problem)``. ``effective`` is the normalized speaker
    name that must be written back to the admitted request before any later
    parameter builder consumes it. Keeping resolution and validation together
    prevents a blank or padded supplied value from being accepted via the
    default while the original, unusable value is forwarded to the model.
    """
    supported_set = {s.strip().lower() for s in supported if isinstance(s, str) and s.strip()}
    if not supported_set:
        return None, (
            "This model does not support CustomVoice task (no speakers configured). "
            "Use task_type='Base' with ref_audio/ref_text for voice cloning, "
            "or use a CustomVoice model."
        )

    effective = normalize_speaker(requested) or normalize_speaker(default)
    if effective is None:
        return None, (
            "CustomVoice task requires 'voice'; this model has no usable default speaker. "
            f"Supported: {', '.join(sorted(supported_set))}"
        )
    if effective not in supported_set:
        return None, f"Invalid voice '{effective}'. Supported: {', '.join(sorted(supported_set))}"
    return effective, None
