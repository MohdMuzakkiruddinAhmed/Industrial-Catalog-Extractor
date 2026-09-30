"""Build a deterministic, allowlisted public source archive.

The project workspace can contain private catalogs, extraction artifacts, model
outputs, databases, and transfer bundles.  This command packages only the public
research repository surface and writes a manifest plus an archive checksum.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import re
import zipfile
from pathlib import Path, PurePosixPath

import pymupdf

ROOT_FILES = (
    ".dockerignore",
    ".editorconfig",
    ".env.example",
    ".gitattributes",
    ".gitignore",
    ".pre-commit-config.yaml",
    "CHANGELOG.md",
    "CITATION.cff",
    "CONTRIBUTING.md",
    "DATA_AVAILABILITY.md",
    "Dockerfile",
    "LICENSE",
    "Makefile",
    "NOTICE",
    "PUBLISHING_CHECKLIST.md",
    "README.md",
    "SECURITY.md",
    "THIRD_PARTY_MODELS.md",
    "THIRD_PARTY_NOTICES.md",
    "package.json",
    "pnpm-lock.yaml",
    "pyproject.toml",
    "requirements-dev.lock",
    "requirements-model.txt",
    "requirements.in",
    "requirements.lock",
)

ROOT_DIRECTORIES = (
    ".github",
    "configs",
    "docs",
    "examples",
    "reports",
    "reproducibility",
    "scripts",
    "src",
    "tests",
)

EXCLUDED_PARTS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "node_modules",
    "qa",
    "tmp",
    "transfer_staging",
}

DISALLOWED_SUFFIXES = {
    ".arrow",
    ".db",
    ".engine",
    ".gguf",
    ".jsonl",
    ".lance",
    ".log",
    ".onnx",
    ".parquet",
    ".part",
    ".pt",
    ".pth",
    ".safetensors",
    ".sqlite",
    ".sqlite3",
    ".tar",
    ".tgz",
    ".zip",
}

DISALLOWED_NAMES = {".env", "id_ed25519", "id_rsa"}
MAX_PUBLIC_FILE_BYTES = 25 * 1024 * 1024
SENSITIVE_PATTERNS = (
    (
        "absolute Windows user path",
        re.compile(rb"[A-Za-z]:[\\/]+Users[\\/]+[^\\/\s]+", re.IGNORECASE),
    ),
    (
        "private-key material",
        re.compile(b"-----BEGIN " + b"(?:RSA |OPENSSH |EC )?PRIVATE KEY-----", re.IGNORECASE),
    ),
    ("NVIDIA API token", re.compile(rb"nvapi-[A-Za-z0-9_-]{20,}", re.IGNORECASE)),
    ("Hugging Face token", re.compile(rb"hf_[A-Za-z0-9]{20,}", re.IGNORECASE)),
)

FIXED_ZIP_TIME = (2026, 8, 7, 0, 0, 0)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def project_version(project_root: Path) -> str:
    init_text = (project_root / "src" / "industrial_catalog" / "__init__.py").read_text(
        encoding="utf-8"
    )
    match = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', init_text, re.MULTILINE)
    if match is None:
        raise ValueError("Could not read __version__ from the package")
    return match.group(1)


def is_public_file(path: Path, project_root: Path) -> bool:
    relative = path.relative_to(project_root)
    return (
        path.is_file()
        and not path.is_symlink()
        and not path.name.startswith("~$")
        and not (path.name.startswith(".~lock.") and path.name.endswith("#"))
        and not EXCLUDED_PARTS.intersection(relative.parts)
    )


def collect_public_files(project_root: Path) -> list[Path]:
    files: set[Path] = set()
    for name in ROOT_FILES:
        path = project_root / name
        if not path.is_file():
            raise FileNotFoundError(f"Required public-release file is missing: {name}")
        files.add(path)

    for name in ROOT_DIRECTORIES:
        directory = project_root / name
        if not directory.is_dir():
            raise FileNotFoundError(f"Required public-release directory is missing: {name}")
        files.update(path for path in directory.rglob("*") if is_public_file(path, project_root))

    return sorted(files, key=lambda path: path.relative_to(project_root).as_posix())


def searchable_payload(relative: PurePosixPath, payload: bytes) -> bytes:
    """Return plain content suitable for public-release privacy scans."""

    suffix = relative.suffix.lower()
    if suffix == ".docx":
        try:
            with zipfile.ZipFile(io.BytesIO(payload)) as document:
                return b"\n".join(
                    document.read(name)
                    for name in document.namelist()
                    if name.endswith((".xml", ".rels"))
                )
        except zipfile.BadZipFile as error:
            raise ValueError(f"Invalid DOCX in public release: {relative}") from error
    if suffix == ".pdf":
        try:
            with pymupdf.open(stream=payload, filetype="pdf") as document:
                return "\n".join(page.get_text() for page in document).encode("utf-8")
        except Exception as error:
            raise ValueError(f"Unreadable PDF in public release: {relative}") from error
    return payload


def validate_public_payload(relative: PurePosixPath, payload: bytes) -> None:
    """Fail closed on risky file types, oversized files, paths, and token shapes."""

    if relative.name.lower() in DISALLOWED_NAMES:
        raise ValueError(f"Disallowed secret-bearing filename in public release: {relative}")
    if relative.suffix.lower() in DISALLOWED_SUFFIXES:
        raise ValueError(f"Disallowed runtime/data file in public release: {relative}")
    if len(payload) > MAX_PUBLIC_FILE_BYTES:
        raise ValueError(
            f"Public-release file exceeds {MAX_PUBLIC_FILE_BYTES} bytes: {relative}"
        )

    searchable = searchable_payload(relative, payload)
    findings = [label for label, pattern in SENSITIVE_PATTERNS if pattern.search(searchable)]
    if findings:
        raise ValueError(f"Sensitive content in {relative}: {', '.join(findings)}")


def zip_info(name: str, executable: bool = False) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=FIXED_ZIP_TIME)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    mode = 0o100755 if executable else 0o100644
    info.external_attr = mode << 16
    return info


def build_archive(project_root: Path, output: Path) -> tuple[int, str]:
    version = project_version(project_root)
    archive_root = PurePosixPath(f"industrial-catalog-extractor-{version}")
    files = collect_public_files(project_root)
    manifest_lines: list[str] = []

    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in files:
            relative = PurePosixPath(path.relative_to(project_root).as_posix())
            payload = path.read_bytes()
            validate_public_payload(relative, payload)
            manifest_lines.append(f"{sha256_bytes(payload)}  {relative}")
            executable = path.suffix == ".sh"
            archive.writestr(zip_info(str(archive_root / relative), executable), payload)

        manifest = ("\n".join(manifest_lines) + "\n").encode("utf-8")
        archive.writestr(zip_info(str(archive_root / "SOURCE_MANIFEST.sha256")), manifest)

    archive_digest = sha256_file(output)
    checksum_path = output.with_suffix(output.suffix + ".sha256")
    checksum_path.write_text(f"{archive_digest}  {output.name}\n", encoding="ascii")
    return len(files), archive_digest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a deterministic allowlisted source ZIP for public release."
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output ZIP (default: releases/industrial-catalog-extractor-<version>-source.zip)",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    version = project_version(project_root)
    output = args.output or (
        project_root / "releases" / f"industrial-catalog-extractor-{version}-source.zip"
    )
    output = output.resolve()
    count, digest = build_archive(project_root, output)
    print(f"archive={output}")
    print(f"files={count}")
    print(f"sha256={digest}")


if __name__ == "__main__":
    main()
