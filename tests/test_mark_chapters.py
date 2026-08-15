from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from cdml.chapter_planner import parse_endpoint
from cdml.mark_chapters import build_chapters, probe_existing, process, remux
import cdml.mark_chapters as mark_chapters


def _make_video(path) -> None:
    subprocess.run([
        "ffmpeg", "-v", "error", "-nostdin", "-y", "-f", "lavfi", "-i",
        "color=c=black:s=32x32:r=24:d=3", "-c:v", "libx264", str(path),
    ], check=True)


@pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
                    reason="ffmpeg and ffprobe are required")
def test_remux_writes_readable_chapters_and_dry_run_leaves_source_unchanged(tmp_path) -> None:
    source = tmp_path / "episode.mkv"
    output = tmp_path / "episode.chapters.mkv"
    _make_video(source)
    remux(source, output, build_chapters([1.0, 2.0], 3.0))
    chapters, _duration = probe_existing(output)
    assert [chapter["title"] for chapter in chapters] == ["Act 1", "Act 2", "Act 3"]
    assert [round(chapter["start"], 2) for chapter in chapters] == [0.0, 1.0, 2.0]

    before = hashlib.sha256(source.read_bytes()).digest()
    args = argparse.Namespace(
        existing="replace", anchor="mid", min_gap=0.0, auto_cap=0,
        title_format="Act {i}", dry_run=True, start_margin=0.0, end_margin=0.0,
        hop=1, batch_size=1, in_place=False, out_dir=None, suffix=".chapters",
        overwrite=False,
    )
    row = process(source, None, None, None, args, [parse_endpoint("start:1")], [])
    assert row["action"] == "dry-run"
    assert hashlib.sha256(source.read_bytes()).digest() == before


def test_failed_remux_removes_partial_destination(tmp_path, monkeypatch) -> None:
    source = tmp_path / "source.mkv"
    destination = tmp_path / "partial.mkv"
    destination.write_bytes(b"partial output")
    monkeypatch.setattr(
        mark_chapters.subprocess, "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=1, stderr="ffmpeg failure"),
    )
    with pytest.raises(RuntimeError, match="ffmpeg failed"):
        remux(source, destination, build_chapters([], 1.0))
    assert not destination.exists()
    assert not destination.with_name(destination.name + ".ffmetadata").exists()
