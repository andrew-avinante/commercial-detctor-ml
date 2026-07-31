"""Read commercial breaks out of container chapter markers.

    python -m cdml.chapters --shows "/media/Authorized/Example Show"

Chapter markers in these rips sit on the act breaks, so they are ground truth
and no hand labelling is needed. Verified by decoding around every boundary in
a sample of all four shows: each one lands on a real fade -- black runs of
28-144 frames with audio dropping to between -63 dBFS and true digital
silence.

This module is the cheap half of the pipeline. `ffprobe` reads chapters from
the container without decoding a single frame, so the CLI here answers "what
would be labelled, and does the structure look sane?" in seconds, before
`cdml.build_dataset` spends an hour decoding 259 GB.

Two things the markers do NOT give you, which the geometry helpers add:

* **Precision.** A chapter mark is keyframe-rounded and names an instant; the
  event is a 1-6 second fade with a ramp either side. `snap_to_fade` widens a
  mark into the real [start, end] of the fade, so the labels line up with what
  the model can actually see.
* **A sanity check.** A mark with no fade near it is either not a break or a
  bad rip. Those get reported as unmatched rather than silently trusted, since
  a mislabelled positive is worse than a missing one.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

import numpy as np

VIDEO_EXT = {".mkv", ".mp4", ".avi", ".m4v", ".ts"}

# s1e1.mkv / S01E02.mkv / show.s3e14.mkv
EPISODE_RE = re.compile(r"s(\d{1,2})[._ ]?e(\d{1,3})", re.IGNORECASE)


def find_videos(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*")
                  if p.suffix.lower() in VIDEO_EXT and not p.name.startswith("."))


def episode_id(path: Path) -> str:
    """`s1e1` from the filename, else the stem. This is the split group key."""
    m = EPISODE_RE.search(path.stem)
    return f"s{int(m.group(1))}e{int(m.group(2))}" if m else path.stem


def probe(path: Path) -> tuple[list[float], float]:
    """Internal chapter boundaries (seconds) and duration. No decoding."""
    out = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json",
         "-show_chapters", "-show_format", str(path)],
        capture_output=True, text=True).stdout
    if not out:
        return [], 0.0
    j = json.loads(out)
    duration = float(j.get("format", {}).get("duration", 0) or 0)
    # chapters[0] starts at 0, which is the head of the file, not a break
    return [float(c["start_time"]) for c in j.get("chapters", [])[1:]], duration


def usable_marks(bounds: list[float], duration: float,
                 end_margin: float) -> tuple[list[float], int]:
    """Drop marks in the tail: the last fade runs into the end credits or the
    black leader, and has no programme after it, so it is not a break the
    detector should ever fire on."""
    keep = [b for b in bounds if b < duration - end_margin]
    return keep, len(bounds) - len(keep)


# --- fade geometry -----------------------------------------------------------

def black_events(lum: np.ndarray, black_thr: float = 0.06,
                 dark_thr: float = 0.25, min_core: int = 6,
                 max_ramp: int = 12) -> list[tuple[int, int, int, int]]:
    """Locate fades as (start, end, core_start, core_end) frame indices.

    Hysteresis, the same shape of rule the detector uses at inference: a run of
    genuinely black frames is the core, then the event grows outwards while
    frames stay merely dark. That captures the ramp down and back up, which is
    what separates a fade from a cut straight to a dark shot.

    `max_ramp` bounds that growth, and is not optional. End credits are white
    text on black, so their mean luminance sits under `dark_thr` for the whole
    roll -- unbounded growth swallows the entire credit sequence into one
    "fade" and labels a third of the episode positive. A real fade ramp is
    well under a second.
    """
    core, dark = lum < black_thr, lum < dark_thr
    events, n, i = [], len(lum), 0
    while i < n:
        if not core[i]:
            i += 1
            continue
        j = i
        while j < n and core[j]:
            j += 1
        if j - i >= min_core:
            s, lo = i, max(0, i - max_ramp)
            while s > lo and dark[s - 1]:
                s -= 1
            e, hi = j, min(n, j + max_ramp)
            while e < hi and dark[e]:
                e += 1
            events.append((s, e, i, j))
        i = j
    return events


def snap_to_fade(events, t_frame: int, tol: int):
    """The fade whose core sits nearest `t_frame`, or None if none is within tol."""
    best, best_d = None, tol + 1
    for ev in events:
        _, _, cs, ce = ev
        d = 0 if cs <= t_frame <= ce else min(abs(cs - t_frame), abs(ce - t_frame))
        if d < best_d:
            best, best_d = ev, d
    return best


def label_episode(lum: np.ndarray, bounds: list[float], duration: float, fps: int,
                  black_thr: float, dark_thr: float, min_core: int,
                  tol_s: float, end_margin: float, max_ramp: int = 12) -> dict:
    """Match chapter marks to fades. Returns breaks, unmatched marks, and the
    fades that are NOT breaks -- those are the hard negatives."""
    marks, dropped = usable_marks(bounds, duration, end_margin)
    evs = black_events(lum, black_thr, dark_thr, min_core, max_ramp)
    tol = int(round(tol_s * fps))

    breaks, unmatched, claimed = [], [], set()
    for b in marks:
        ev = snap_to_fade(evs, int(round(b * fps)), tol)
        if ev is None:
            unmatched.append(round(b, 2))
            continue
        claimed.add(ev)
        s, e, cs, ce = ev
        breaks.append({"mark": round(b, 2), "start": s, "end": e,
                       "core_start": cs, "core_end": ce})
    # two marks can snap onto the same fade; keep one copy
    seen, uniq = set(), []
    for br in sorted(breaks, key=lambda r: r["core_start"]):
        key = (br["core_start"], br["core_end"])
        if key not in seen:
            seen.add(key)
            uniq.append(br)
    return {"breaks": uniq, "unmatched_marks": unmatched,
            "dropped_tail_marks": dropped,
            "hard_negative_fades": [e for e in evs if e not in claimed],
            "n_black_events": len(evs)}


# --- CLI: fast structural preview, no decoding -------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shows", nargs="+", required=True)
    ap.add_argument("--out", default="", help="optional JSON dump of the survey")
    ap.add_argument("--end-margin", type=float, default=20.0,
                    help="ignore marks this close to the end of the file")
    args = ap.parse_args()

    shows = [Path(s) for s in args.shows]
    missing = [s for s in shows if not s.is_dir()]
    if missing:
        raise SystemExit("not a directory: " + ", ".join(str(m) for m in missing))

    rows, total_marks, total_usable = [], 0, 0
    for show_dir in shows:
        vids = find_videos(show_dir)
        print(f"\n=== {show_dir.name}  ({len(vids)} files)")
        for path in vids:
            bounds, duration = probe(path)
            if not bounds:
                print(f"  {path.name:30s} no chapters -- will be skipped")
                continue
            keep, dropped = usable_marks(bounds, duration, args.end_margin)
            total_marks += len(bounds)
            total_usable += len(keep)
            rows.append({"show": show_dir.name, "episode": episode_id(path),
                         "path": str(path), "duration": round(duration, 2),
                         "chapter_bounds": [round(b, 2) for b in bounds],
                         "usable_marks": [round(b, 2) for b in keep],
                         "dropped_tail_marks": dropped})
            marks = "  ".join(f"{b:7.1f}" for b in keep)
            print(f"  {path.name:30s} {duration / 60:5.1f}m  "
                  f"keep {len(keep)}/{len(bounds)}  {marks}")

    print(f"\n{'-' * 68}")
    print(f"episodes with chapters : {len(rows)}")
    print(f"shows                  : {len(set(r['show'] for r in rows))}")
    print(f"chapter marks          : {total_marks}")
    print(f"usable as breaks       : {total_usable}   "
          f"({total_marks - total_usable} dropped as end-of-file tails)")
    for show in sorted(set(r["show"] for r in rows)):
        sr = [r for r in rows if r["show"] == show]
        print(f"  {show:34s} {len(sr):3d} eps  "
              f"{sum(len(r['usable_marks']) for r in sr):3d} breaks")
    print("\nThese are chapter positions only -- no frames decoded. "
          "Run cdml.build_dataset to\nsnap them onto the real fades and cut clips.")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rows, indent=2))
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
