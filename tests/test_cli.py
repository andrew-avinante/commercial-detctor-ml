from __future__ import annotations

import pytest

from cdml import cli


@pytest.mark.parametrize(("argv", "module", "attribute", "expected"), [
    (["infer", "--video", "episode.mkv"], "cdml.infer", "main", ["--video", "episode.mkv"]),
    (["chapters", "mark", "episode.mkv"], "cdml.mark_chapters", "main", ["episode.mkv"]),
    (["model", "download", "--force"], "cdml.model_store", "main", ["--force"]),
])
def test_routes_each_public_subcommand(argv, module, attribute, expected, monkeypatch) -> None:
    received = []
    target = __import__(module, fromlist=[attribute])
    monkeypatch.setattr(target, attribute, received.append)
    cli.main(argv)
    assert received == [expected]


def test_root_help_describes_the_three_command_groups(capsys) -> None:
    cli.main([])
    out = capsys.readouterr().out
    assert "chapters mark" in out
    assert "model download" in out


def test_rejects_unknown_command() -> None:
    with pytest.raises(SystemExit, match="2"):
        cli.main(["nope"])
