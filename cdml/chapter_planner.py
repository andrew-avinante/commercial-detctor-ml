"""Pure parsing and selection logic for chapter-marking rules.

This module deliberately knows nothing about ffmpeg, PyTorch, or command-line
arguments.  Keeping the rule engine separate makes its conflict decisions
testable and lets callers explain every automatic boundary that was rejected.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Iterable


_CLOCK = re.compile(r"^(?:(\d+):)?(\d{1,2}):(\d{1,2}(?:\.\d+)?)$")
_UNITS = re.compile(
    r"^(?:(?P<h>\d+(?:\.\d+)?)h)?(?:(?P<m>\d+(?:\.\d+)?)m)?"
    r"(?:(?P<s>\d+(?:\.\d+)?)s)?$"
)


@dataclass(frozen=True)
class Endpoint:
    """A time offset measured from the start or end of a media file."""

    side: str
    seconds: float
    text: str

    def resolve(self, duration: float) -> float:
        if not 0 <= self.seconds <= duration:
            raise ValueError(
                f"{self.text!r} is outside a media duration of {duration:g}s")
        return self.seconds if self.side == "start" else duration - self.seconds


@dataclass(frozen=True)
class AutoRule:
    """An automatic-detection interval expressed by two endpoints."""

    start: Endpoint
    end: Endpoint
    index: int


@dataclass(frozen=True)
class ResolvedAutoRule:
    start: float
    end: float
    index: int
    source: tuple[str, str]


def parse_seconds(text: str) -> float:
    """Parse seconds, a ``MM:SS``/``HH:MM:SS`` clock, or compact units."""
    value = text.strip().lower()
    if not value:
        raise ValueError("time value is empty")
    try:
        seconds = float(value[:-1] if value.endswith("s") else value)
    except ValueError:
        seconds = math.nan
    if math.isfinite(seconds):
        if seconds < 0:
            raise ValueError(f"time must not be negative: {text!r}")
        return seconds

    clock = _CLOCK.fullmatch(value)
    if clock:
        hours = int(clock.group(1) or 0)
        minutes = int(clock.group(2))
        secs = float(clock.group(3))
        if (clock.group(1) is not None and minutes >= 60) or secs >= 60:
            raise ValueError(f"invalid clock time: {text!r}")
        return hours * 3600 + minutes * 60 + secs

    units = _UNITS.fullmatch(value)
    if units and any(units.groupdict().values()):
        seconds = sum(float(units.group(name) or 0) * factor
                      for name, factor in (("h", 3600), ("m", 60), ("s", 1)))
        return seconds
    raise ValueError(
        f"invalid time {text!r}; use seconds, MM:SS, HH:MM:SS, or 1h2m3s")


def parse_endpoint(text: str) -> Endpoint:
    """Parse ``start:<time>`` or ``end:<time>``."""
    side, delimiter, value = text.strip().lower().partition(":")
    if delimiter != ":" or side not in {"start", "end"} or not value:
        raise ValueError(f"invalid endpoint {text!r}; use start:<time> or end:<time>")
    return Endpoint(side=side, seconds=parse_seconds(value), text=text)


def resolve_auto_rules(rules: Iterable[AutoRule], duration: float) -> list[ResolvedAutoRule]:
    """Resolve rules against duration and reject empty or reversed intervals."""
    resolved: list[ResolvedAutoRule] = []
    for rule in rules:
        start, end = rule.start.resolve(duration), rule.end.resolve(duration)
        if start >= end:
            raise ValueError(
                f"--auto {rule.start.text!r} {rule.end.text!r} resolves to an "
                f"empty or reversed interval ({start:g}s >= {end:g}s)")
        resolved.append(ResolvedAutoRule(
            start=start, end=end, index=rule.index,
            source=(rule.start.text, rule.end.text)))
    return resolved


def _decision(*, time: float, origin: str, reason: str | None = None,
              confidence: float | None = None, rule: ResolvedAutoRule | None = None) -> dict:
    decision = {"time": round(time, 3), "origin": origin}
    if confidence is not None:
        decision["confidence"] = confidence
    if rule is not None:
        decision["auto_rule"] = {
            "index": rule.index, "start": round(rule.start, 3), "end": round(rule.end, 3),
            "source": list(rule.source),
        }
    if reason is not None:
        decision["reason"] = reason
    return decision


def plan_boundaries(*, fixed: Iterable[Endpoint], auto_rules: Iterable[AutoRule],
                    events: Iterable[dict], duration: float, anchor: str = "mid",
                    min_gap: float = 30.0, auto_cap: int = 0) -> dict:
    """Select fixed and ML chapter boundaries, returning an auditable plan.

    Automatic candidates are considered by descending confidence.  A candidate
    selected through an overlapping range is represented once, while every
    skipped candidate carries the reason that prevented selection.
    """
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("media duration must be a positive finite number")
    if min_gap < 0:
        raise ValueError("--min-gap must be zero or greater")
    if auto_cap < 0:
        raise ValueError("--auto-cap must be zero or greater")
    if anchor not in {"start", "mid", "end"}:
        raise ValueError(f"unknown event anchor: {anchor!r}")

    events = list(events)
    raw_fixed = sorted(endpoint.resolve(duration) for endpoint in fixed)
    fixed_times: list[float] = []
    for value in raw_fixed:
        if not fixed_times or value != fixed_times[-1]:
            fixed_times.append(value)
    if min_gap:
        for previous, current in zip(fixed_times, fixed_times[1:]):
            if current - previous < min_gap:
                raise ValueError(
                    f"fixed markers at {previous:g}s and {current:g}s are closer "
                    f"than --min-gap {min_gap:g}s")

    resolved_rules = resolve_auto_rules(auto_rules, duration)
    accepted = [_decision(time=value, origin="fixed") for value in fixed_times]
    rejected: list[dict] = []
    candidates: list[tuple[float, float, int, dict, ResolvedAutoRule]] = []
    for rule in resolved_rules:
        for event_index, event in enumerate(events):
            start, end = float(event["start"]), float(event["end"])
            time = start if anchor == "start" else end if anchor == "end" else (start + end) / 2
            confidence = float(event.get("confidence", 0.0))
            if not rule.start <= time <= rule.end:
                rejected.append(_decision(time=time, origin="auto", confidence=confidence,
                                          rule=rule, reason="outside range"))
                continue
            candidates.append((confidence, time, event_index, event, rule))

    # Stable tie breakers make a report reproducible regardless of scan order.
    candidates.sort(key=lambda item: (-item[0], item[1], item[4].index, item[2]))
    accepted_auto: list[dict] = []
    selected_events: set[tuple[float, float]] = set()
    accepted_per_rule: dict[int, int] = {}
    for confidence, time, _event_index, event, rule in candidates:
        event_key = (float(event["start"]), float(event["end"]))
        if event_key in selected_events:
            rejected.append(_decision(time=time, origin="auto", confidence=confidence,
                                      rule=rule, reason="duplicate overlap"))
            continue
        if min_gap and any(abs(time - fixed_time) < min_gap for fixed_time in fixed_times):
            rejected.append(_decision(time=time, origin="auto", confidence=confidence,
                                      rule=rule, reason="within min-gap of fixed marker"))
            continue
        if min_gap and any(abs(time - item["time"]) < min_gap for item in accepted_auto):
            rejected.append(_decision(time=time, origin="auto", confidence=confidence,
                                      rule=rule,
                                      reason="within min-gap of higher confidence auto candidate"))
            continue
        if auto_cap and accepted_per_rule.get(rule.index, 0) >= auto_cap:
            rejected.append(_decision(time=time, origin="auto", confidence=confidence,
                                      rule=rule, reason="range cap"))
            continue
        decision = _decision(time=time, origin="auto", confidence=confidence, rule=rule)
        accepted_auto.append(decision)
        accepted_per_rule[rule.index] = accepted_per_rule.get(rule.index, 0) + 1
        selected_events.add(event_key)

    accepted.extend(accepted_auto)
    accepted.sort(key=lambda item: item["time"])
    return {
        "duration": round(duration, 3),
        "min_gap": min_gap,
        "auto_cap": auto_cap,
        "fixed": [round(value, 3) for value in fixed_times],
        "auto_rules": [
            {"index": rule.index, "start": round(rule.start, 3), "end": round(rule.end, 3),
             "source": list(rule.source)} for rule in resolved_rules
        ],
        "accepted": accepted,
        "rejected": rejected,
    }
