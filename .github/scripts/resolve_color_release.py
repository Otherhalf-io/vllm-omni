#!/usr/bin/env python3
"""Resolve Color's immutable vLLM-Omni image identity without Git tags."""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

_VERSION_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")
_CHANNEL_RE = re.compile(r"^[a-z][a-z0-9-]*$")
_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_DIGEST_IMAGE_RE = re.compile(r"^.+@sha256:[0-9a-f]{64}$")


def _read_provenance(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if not separator or not key or not value:
            raise ValueError(f"Malformed provenance line: {raw_line!r}")
        values[key] = value
    required = {
        "VLLM_OMNI_UPSTREAM_BASE_VERSION",
        "VLLM_OMNI_UPSTREAM_BASE_COMMIT",
        "VLLM_BASE_IMAGE",
    }
    missing = sorted(required - values.keys())
    if missing:
        raise ValueError(f"Missing provenance fields: {', '.join(missing)}")
    return values


def resolve_pep440_version(base_version: str, distance: int, short_sha: str, channel: str) -> str:
    match = _VERSION_RE.fullmatch(base_version)
    if match is None:
        raise ValueError(f"Invalid upstream base version: {base_version!r}")
    if distance < 0:
        raise ValueError("Commit distance cannot be negative")
    if re.fullmatch(r"[0-9a-f]{8}", short_sha) is None:
        raise ValueError(f"Invalid short source SHA: {short_sha!r}")
    if _CHANNEL_RE.fullmatch(channel) is None:
        raise ValueError(f"Invalid release channel: {channel!r}")

    major, minor, patch = (int(part) for part in match.groups())
    public = f"{major}.{minor}.{patch}" if distance == 0 else f"{major}.{minor}.{patch + 1}.dev{distance}"
    return f"{public}+g{short_sha}.color.{channel.replace('-', '.')}"


def _git(repository: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repository), *args], text=True).strip()


def resolve_release(repository: Path, *, channel: str, image: str) -> dict[str, str]:
    provenance = _read_provenance(repository / ".github/color-release.env")
    full_sha = _git(repository, "rev-parse", "HEAD")
    base_commit = provenance["VLLM_OMNI_UPSTREAM_BASE_COMMIT"]
    base_image = provenance["VLLM_BASE_IMAGE"]
    if _SHA_RE.fullmatch(full_sha) is None or _SHA_RE.fullmatch(base_commit) is None:
        raise ValueError("Source and upstream base commits must be full 40-character SHAs")
    if _DIGEST_IMAGE_RE.fullmatch(base_image) is None:
        raise ValueError("VLLM_BASE_IMAGE must be pinned by sha256 digest")
    ancestor = subprocess.run(
        ["git", "-C", str(repository), "merge-base", "--is-ancestor", base_commit, full_sha],
        check=False,
    )
    if ancestor.returncode != 0:
        raise ValueError(f"Configured upstream base {base_commit} is not an ancestor of {full_sha}")

    distance = int(_git(repository, "rev-list", "--count", f"{base_commit}..{full_sha}"))
    short_sha = full_sha[:8]
    version = resolve_pep440_version(
        provenance["VLLM_OMNI_UPSTREAM_BASE_VERSION"],
        distance,
        short_sha,
        channel,
    )
    return {
        "full_sha": full_sha,
        "short_sha": short_sha,
        "version_override": version,
        "immutable_image": f"{image}:{channel}-{short_sha}",
        "base_image": base_image,
        "upstream_base_version": provenance["VLLM_OMNI_UPSTREAM_BASE_VERSION"],
        "upstream_base_commit": base_commit,
        "commit_distance": str(distance),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--channel", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--github-output", type=Path)
    args = parser.parse_args()

    resolved = resolve_release(args.repository.resolve(), channel=args.channel, image=args.image)
    rendered = "".join(f"{key}={value}\n" for key, value in resolved.items())
    if args.github_output is None:
        print(rendered, end="")
    else:
        with args.github_output.open("a", encoding="utf-8") as output:
            output.write(rendered)


if __name__ == "__main__":
    main()
