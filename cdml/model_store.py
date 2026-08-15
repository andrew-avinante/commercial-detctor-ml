"""Download and locate the released checkpoint for installed CDML commands."""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import shutil
from urllib.error import URLError
from urllib.request import urlopen


MODEL_URL = "https://huggingface.co/andrew-avinante/cdml-fade-detector/resolve/main/fade_detector.pt"
MODEL_SHA256 = "4af08510774e7eae496a4e413bba54f915a1c13fe912fee70857a862c5def300"
MODEL_FILENAME = "fade_detector.pt"


def default_cache_path() -> Path:
    """Return the platform-independent user-cache location for the checkpoint."""
    base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return base / "cdml" / MODEL_FILENAME


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_model(destination: Path | None = None, *, force: bool = False) -> Path:
    """Download the published checkpoint atomically and verify its SHA-256."""
    destination = (destination or default_cache_path()).expanduser()
    if destination.exists() and not force:
        if sha256(destination) == MODEL_SHA256:
            return destination
        raise RuntimeError(
            f"cached model {destination} failed its SHA-256 check; rerun with --force")

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    partial.unlink(missing_ok=True)
    try:
        with urlopen(MODEL_URL) as response, partial.open("wb") as output:
            shutil.copyfileobj(response, output)
        actual = sha256(partial)
        if actual != MODEL_SHA256:
            raise RuntimeError(
                f"downloaded model SHA-256 mismatch: expected {MODEL_SHA256}, got {actual}")
        os.replace(partial, destination)
        return destination
    except (OSError, URLError) as exc:
        raise RuntimeError(f"could not download the default CDML model: {exc}") from exc
    finally:
        partial.unlink(missing_ok=True)


def resolve_model_path(model: str | None) -> Path:
    """Resolve an explicit model path or download the verified default model."""
    if model:
        path = Path(model).expanduser()
        if not path.is_file():
            raise RuntimeError(f"model checkpoint not found: {path}")
        return path
    return download_model()


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Download CDML's default fade-detector checkpoint.")
    parser.add_argument("--output", type=Path, help="destination (default: the user cache)")
    parser.add_argument("--force", action="store_true", help="replace an existing cached file")
    args = parser.parse_args(argv)
    path = download_model(args.output, force=args.force)
    print(path)
