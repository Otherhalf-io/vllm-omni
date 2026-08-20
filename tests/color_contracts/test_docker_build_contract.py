import importlib.util
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).parents[2]
_RESOLVER_PATH = _REPO_ROOT / ".github/scripts/resolve_color_release.py"
_SPEC = importlib.util.spec_from_file_location("color_release", _RESOLVER_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_RESOLVER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_RESOLVER)


def test_cuda_image_requires_explicit_source_identity() -> None:
    dockerfile = (_REPO_ROOT / "docker/Dockerfile.cuda").read_text()
    cuda_installation = (_REPO_ROOT / "docs/getting_started/installation/gpu/cuda.inc.md").read_text()

    assert "ARG VLLM_OMNI_VERSION_OVERRIDE" in dockerfile
    assert "ARG VLLM_OMNI_SOURCE_SHA" in dockerfile
    assert 'test -n "${VLLM_OMNI_VERSION_OVERRIDE}"' in dockerfile
    assert 'test -n "${VLLM_OMNI_SOURCE_SHA}"' in dockerfile
    assert 'org.opencontainers.image.revision="${VLLM_OMNI_SOURCE_SHA}"' in dockerfile
    assert cuda_installation.count("--build-arg VLLM_OMNI_VERSION_OVERRIDE=") == 2
    assert cuda_installation.count("--build-arg VLLM_OMNI_SOURCE_SHA=") == 2


def test_release_workflow_uses_repository_provenance_and_immutable_tags() -> None:
    workflow = (_REPO_ROOT / ".github/workflows/release-image.yml").read_text()

    assert ".github/scripts/resolve_color_release.py" in workflow
    assert "git describe" not in workflow
    assert "latest_image" not in workflow
    assert "latest-preview" not in workflow
    assert "BASE_IMAGE=${{ steps.release.outputs.base_image }}" in workflow
    assert "VLLM_OMNI_SOURCE_SHA=${{ steps.release.outputs.full_sha }}" in workflow
    assert "RELEASE_IMAGE: ${{ steps.release.outputs.immutable_image }}" in workflow
    assert "RELEASE_DIGEST: ${{ steps.image.outputs.digest }}" in workflow
    assert '"${RELEASE_IMAGE}" "${RELEASE_DIGEST}"' in workflow


def test_release_version_is_derived_from_owned_v026_provenance() -> None:
    resolved = _RESOLVER.resolve_release(
        _REPO_ROOT,
        channel="preview",
        image="registry.example/color/vllm-omni",
    )

    assert resolved["upstream_base_version"] == "0.26.0"
    assert resolved["upstream_base_commit"] == "a4ea67a21b20054dacc6e83952f9bd407e8ee4e7"
    assert resolved["base_image"].endswith("@sha256:ffb2d59b1c059a5bd8d781320c9f5189de8293693b7d95da54befddaa54abf52")
    assert resolved["version_override"].startswith("0.26.1.dev")
    assert resolved["version_override"].endswith(f"+g{resolved['short_sha']}.color.preview")
    assert resolved["immutable_image"].endswith(f":preview-{resolved['short_sha']}")


@pytest.mark.parametrize(
    ("base", "distance", "sha", "channel", "message"),
    [
        ("v0.26.0", 1, "12345678", "preview", "base version"),
        ("0.26.0", -1, "12345678", "preview", "negative"),
        ("0.26.0", 1, "short", "preview", "source SHA"),
        ("0.26.0", 1, "12345678", "Preview!", "channel"),
    ],
)
def test_release_version_rejects_ambiguous_identity(base, distance, sha, channel, message) -> None:
    with pytest.raises(ValueError, match=message):
        _RESOLVER.resolve_pep440_version(base, distance, sha, channel)
