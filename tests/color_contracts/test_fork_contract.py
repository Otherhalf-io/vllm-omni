from pathlib import Path

REPO_ROOT = Path(__file__).parents[2]
EXPECTED_UPSTREAM_BASE = "a4ea67a21b20054dacc6e83952f9bd407e8ee4e7"
EXPECTED_BASE_IMAGE = (
    "docker.io/vllm/vllm-openai@sha256:ffb2d59b1c059a5bd8d781320c9f5189de8293693b7d95da54befddaa54abf52"
)


def _release_contract() -> dict[str, str]:
    result: dict[str, str] = {}
    for line in (REPO_ROOT / ".github" / "color-release.env").read_text().splitlines():
        key, value = line.split("=", maxsplit=1)
        result[key] = value
    return result


def test_release_contract_pins_the_tested_upstream_and_base_image() -> None:
    contract = _release_contract()

    assert contract == {
        "VLLM_OMNI_UPSTREAM_BASE_VERSION": "0.26.0",
        "VLLM_OMNI_UPSTREAM_BASE_COMMIT": EXPECTED_UPSTREAM_BASE,
        "VLLM_BASE_IMAGE": EXPECTED_BASE_IMAGE,
    }
    dockerfile = (REPO_ROOT / "docker" / "Dockerfile.cuda").read_text()
    assert f"ARG BASE_IMAGE={EXPECTED_BASE_IMAGE}" in dockerfile
    assert "ARG VLLM_OMNI_VERSION_OVERRIDE" in dockerfile
    assert 'test -n "${VLLM_OMNI_VERSION_OVERRIDE}"' in dockerfile
    assert 'VLLM_OMNI_VERSION_OVERRIDE="${VLLM_OMNI_VERSION_OVERRIDE}"' in dockerfile


def test_pull_request_workflows_are_not_limited_to_main() -> None:
    for workflow_name in ("ci.yml", "pre-commit.yml", "build_wheel.yml"):
        workflow = (REPO_ROOT / ".github" / "workflows" / workflow_name).read_text()
        assert "  pull_request:\n    branches:" not in workflow
