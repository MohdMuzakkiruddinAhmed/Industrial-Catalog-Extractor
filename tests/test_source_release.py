import zipfile
from pathlib import Path, PurePosixPath

import pytest

from scripts.build_source_release import build_archive, validate_public_payload


def test_public_release_rejects_private_paths_and_tokens() -> None:
    separator = bytes((92,))
    with pytest.raises(ValueError, match="absolute Windows user path"):
        validate_public_payload(
            PurePosixPath("docs/private.md"),
            b"Local checkout: C:"
            + separator
            + b"Users"
            + separator
            + b"researcher"
            + separator
            + b"catalog-project",
        )

    with pytest.raises(ValueError, match="NVIDIA API token"):
        validate_public_payload(
            PurePosixPath("docs/token.txt"),
            b"nv" + b"api-" + b"abcdefghijklmnopqrstuvwxyz012345",
        )


def test_source_archive_is_deterministic_and_manifested(tmp_path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"

    first_count, first_digest = build_archive(project_root, first)
    second_count, second_digest = build_archive(project_root, second)

    assert first_count == second_count
    assert first_digest == second_digest
    assert first.read_bytes() == second.read_bytes()

    with zipfile.ZipFile(first) as archive:
        names = archive.namelist()
        root = names[0].split("/", 1)[0] + "/"
        manifest = archive.read(root + "SOURCE_MANIFEST.sha256").decode("utf-8")

    manifested_paths = {line.split("  ", 1)[1] for line in manifest.splitlines()}
    archived_paths = {
        name.removeprefix(root)
        for name in names
        if name != root + "SOURCE_MANIFEST.sha256"
    }
    assert manifested_paths == archived_paths
