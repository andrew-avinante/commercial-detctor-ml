from __future__ import annotations

import pytest

from cdml.chapter_planner import AutoRule, parse_endpoint, parse_seconds, plan_boundaries


@pytest.mark.parametrize(("text", "expected"), [
    ("60", 60), ("60s", 60), ("01:02", 62), ("01:02:03", 3723),
    ("1h2m3s", 3723), ("60:00", 3600),
])
def test_parse_seconds_supported_forms(text: str, expected: float) -> None:
    assert parse_seconds(text) == expected


@pytest.mark.parametrize("text", ["", "-1", "1:60", "start:5", "nonsense"])
def test_parse_seconds_rejects_invalid_values(text: str) -> None:
    with pytest.raises(ValueError):
        parse_seconds(text)


def test_parse_endpoint_requires_side_and_time() -> None:
    assert parse_endpoint("end:01:30").side == "end"
    with pytest.raises(ValueError, match="endpoint"):
        parse_endpoint("middle:10")


@pytest.mark.parametrize(("start", "end", "expected"), [
    ("start:60", "start:900", (60, 900)),
    ("start:960", "end:900", (960, 1100)),
    ("end:1040", "start:1160", (960, 1160)),
    ("end:840", "end:60", (1160, 1940)),
])
def test_resolves_all_endpoint_pairings(start: str, end: str,
                                        expected: tuple[float, float]) -> None:
    plan = plan_boundaries(
        fixed=[], auto_rules=[AutoRule(parse_endpoint(start), parse_endpoint(end), 0)],
        events=[], duration=2000, min_gap=0)
    rule = plan["auto_rules"][0]
    assert (rule["start"], rule["end"]) == expected


def test_rejects_empty_reversed_and_outside_ranges() -> None:
    with pytest.raises(ValueError, match="empty or reversed"):
        plan_boundaries(
            fixed=[], auto_rules=[AutoRule(parse_endpoint("start:20"), parse_endpoint("end:90"), 0)],
            events=[], duration=100, min_gap=0)
    with pytest.raises(ValueError, match="outside"):
        plan_boundaries(
            fixed=[parse_endpoint("start:101")], auto_rules=[], events=[], duration=100)


def test_fixed_markers_deduplicate_but_never_silently_violate_gap() -> None:
    plan = plan_boundaries(
        fixed=[parse_endpoint("start:60"), parse_endpoint("start:60")],
        auto_rules=[], events=[], duration=300, min_gap=30)
    assert plan["fixed"] == [60]
    with pytest.raises(ValueError, match="fixed markers"):
        plan_boundaries(
            fixed=[parse_endpoint("start:60"), parse_endpoint("start:80")],
            auto_rules=[], events=[], duration=300, min_gap=30)


def test_auto_selection_is_confidence_ranked_capped_and_global() -> None:
    events = [
        {"start": 100, "end": 102, "confidence": 0.7},
        {"start": 200, "end": 202, "confidence": 0.95},
        {"start": 300, "end": 302, "confidence": 0.8},
    ]
    plan = plan_boundaries(
        fixed=[parse_endpoint("start:50")],
        auto_rules=[AutoRule(parse_endpoint("start:0"), parse_endpoint("end:0"), 0)],
        events=events, duration=400, min_gap=0, auto_cap=2)
    assert [item["time"] for item in plan["accepted"]] == [50, 201, 301]
    assert any(item["reason"] == "range cap" and item["time"] == 101
               for item in plan["rejected"])


def test_fixed_priority_min_gap_and_overlapping_ranges_are_explained() -> None:
    events = [
        {"start": 95, "end": 105, "confidence": 0.99},
        {"start": 200, "end": 210, "confidence": 0.9},
        {"start": 220, "end": 230, "confidence": 0.8},
    ]
    plan = plan_boundaries(
        fixed=[parse_endpoint("start:100")],
        auto_rules=[
            AutoRule(parse_endpoint("start:0"), parse_endpoint("start:250"), 0),
            AutoRule(parse_endpoint("start:180"), parse_endpoint("start:250"), 1),
        ],
        events=events, duration=300, min_gap=30, auto_cap=0)
    assert [item["origin"] for item in plan["accepted"]] == ["fixed", "auto"]
    reasons = {item["reason"] for item in plan["rejected"]}
    assert "within min-gap of fixed marker" in reasons
    assert "within min-gap of higher confidence auto candidate" in reasons
    assert "duplicate overlap" in reasons


def test_zero_gap_and_zero_cap_leave_candidates_unrestricted() -> None:
    events = [{"start": 100, "end": 101, "confidence": 0.9},
              {"start": 101, "end": 102, "confidence": 0.8}]
    plan = plan_boundaries(
        fixed=[], auto_rules=[AutoRule(parse_endpoint("start:0"), parse_endpoint("end:0"), 0)],
        events=events, duration=200, min_gap=0, auto_cap=0)
    assert [item["time"] for item in plan["accepted"]] == [100.5, 101.5]
