"""Detect the act breaks in a full episode and write them back as chapters.

    python -m cdml.mark_chapters "/media/Shows/Thundarr the Barbarian" --in-place

This is `cdml.infer` plus a remux: the detector produces fade events, the fades
become chapter boundaries, and ffmpeg stream-copies the file with an
`FFMETADATA1` chapter list attached. Nothing is re-encoded -- the video and
audio bitstreams are copied verbatim, so the operation is lossless and runs at
disk speed.

Three details that matter:

* **The boundary is the middle of the fade, not its edge.** A DVD chapter mark
  sits inside the fade, not at the first dark frame: on the episode in the
  README the marks are at 63.4 / 708.3 / 1367.4 s and the detected fades are
  61.5-65.1 / 705.1-710.5 / 1364.5-1369.8 -- midpoints to within a few tenths
  of a second. `--anchor start|end` overrides this.

* **Marks near the end of the file are dropped.** The last fade of an episode
  runs into the end credits and has no programme after it, so it is not a
  break. `cdml.chapters` drops those when building labels, for the same reason,
  and with the same 20 s default.

* **A file that already has chapters is left alone** unless you say otherwise.
  These rips are the training corpus's own ground truth; clobbering them with
  predictions would be destroying the labels. `--existing replace` overrides,
  `--existing compare` reports the delta and writes nothing.

`--in-place` replaces the source file, but only after ffmpeg exits clean and
the chapters read back out of the new file; the swap itself is an atomic
`os.replace` within the same directory. Without it, output goes to a sibling
`<name>.chapters<ext>`.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import torch

from .chapters import VIDEO_EXT, find_videos
from .config import CLIP_FRAMES, FPS
from .features import probe_duration
from .infer import fmt, load_model, scan, to_events

# Containers whose muxer can carry chapters. AVI and raw TS cannot.
CHAPTER_EXT = {".mkv", ".mp4", ".m4v", ".mov", ".webm"}


# --- probing -----------------------------------------------------------------

def probe_existing(path: Path) -> tuple[list[dict], float]:
    """(chapters, duration). Chapters as {start, end, title} in seconds."""
    out = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json",
         "-show_chapters", "-show_format", str(path)],
        capture_output=True, text=True).stdout
    if not out:
        return [], 0.0
    j = json.loads(out)
    duration = float(j.get("format", {}).get("duration", 0) or 0)
    chapters = [{"start": float(c["start_time"]), "end": float(c["end_time"]),
                 "title": c.get("tags", {}).get("title", "")}
                for c in j.get("chapters", [])]
    return chapters, duration


# --- events -> chapter boundaries --------------------------------------------

def anchor_time(event: dict, anchor: str) -> float:
    if anchor == "start":
        return event["start"]
    if anchor == "end":
        return event["end"]
    return (event["start"] + event["end"]) / 2.0


def boundaries(events: list[dict], duration: float, anchor: str = "mid",
               start_margin: float = 5.0, end_margin: float = 20.0,
               min_gap: float = 30.0) -> tuple[list[float], list[dict]]:
    """Fade events -> sorted split points, plus the events that were rejected.

    Each rejected event carries a `reason`, so a surprising chapter count can be
    explained without re-running the scan.
    """
    kept: list[float] = []
    dropped: list[dict] = []
    for ev in sorted(events, key=lambda e: e["start"]):
        t = anchor_time(ev, anchor)
        if t < start_margin:
            dropped.append({**ev, "reason": "head of file"})
        elif t > duration - end_margin:
            dropped.append({**ev, "reason": "tail of file"})
        elif kept and t - kept[-1] < min_gap:
            dropped.append({**ev, "reason": f"within {min_gap:g}s of previous"})
        else:
            kept.append(t)
    return kept, dropped


def build_chapters(splits: list[float], duration: float,
                   title_format: str = "Act {i}") -> list[dict]:
    edges = [0.0] + splits + [duration]
    return [{"start": a, "end": b, "title": title_format.format(i=i)}
            for i, (a, b) in enumerate(zip(edges, edges[1:]), start=1)]


# --- ffmetadata --------------------------------------------------------------

def _escape(text: str) -> str:
    """ffmetadata treats = ; # \\ and newline as special."""
    for ch in ("\\", "=", ";", "#", "\n"):
        text = text.replace(ch, "\\" + ch)
    return text


def ffmetadata(chapters: list[dict]) -> str:
    lines = [";FFMETADATA1"]
    for c in chapters:
        # Millisecond timebase: finer than the frame grid we detect on, and the
        # form every muxer accepts.
        lines += ["[CHAPTER]", "TIMEBASE=1/1000",
                  f"START={int(round(c['start'] * 1000))}",
                  f"END={int(round(c['end'] * 1000))}",
                  f"title={_escape(c['title'])}"]
    return "\n".join(lines) + "\n"


def remux(src: Path, dst: Path, chapters: list[dict]) -> None:
    """Copy `src` to `dst` with `chapters` attached. No re-encode."""
    meta = dst.with_name(dst.name + ".ffmetadata")
    meta.write_text(ffmetadata(chapters), encoding="utf-8")
    try:
        p = subprocess.run(
            ["ffmpeg", "-v", "error", "-nostdin", "-y",
             "-i", str(src), "-f", "ffmetadata", "-i", str(meta),
             "-map", "0",              # every stream: video, audio, subs, attachments
             "-map_metadata", "0",     # keep the source's global tags ...
             "-map_chapters", "1",     # ... but take chapters from the metadata input
             "-c", "copy", str(dst)],
            capture_output=True, text=True)
        if p.returncode != 0:
            dst.unlink(missing_ok=True)
            tail = " | ".join(p.stderr.strip().splitlines()[-4:])
            raise RuntimeError(f"ffmpeg failed on {src.name}: {tail}")
    finally:
        meta.unlink(missing_ok=True)


def verify(path: Path, expected: int) -> None:
    got, _ = probe_existing(path)
    if len(got) != expected:
        raise RuntimeError(
            f"{path.name}: wrote {expected} chapters but read back {len(got)}")


# --- one file ----------------------------------------------------------------

def compare_to_existing(existing: list[dict], splits: list[float],
                        duration: float, args, tol: float = 3.0) -> None:
    """Score the detections against the file's own marks. Read-only.

    Both directions, because either one alone flatters the detector: a
    prediction with no mark near it is a false positive, and a mark with no
    prediction near it is a missed break. Marks the boundary filter would have
    discarded anyway (head, tail) are excluded rather than counted as misses.
    """
    marks = [c["start"] for c in existing[1:]           # chapters[0] is the head
             if args.start_margin <= c["start"] <= duration - args.end_margin]
    print(f"    existing marks : {'  '.join(f'{m:.1f}' for m in marks) or '-'}")
    print(f"    detected       : {'  '.join(f'{s:.1f}' for s in splits) or '-'}")

    hits = 0
    for s in splits:
        near = min(marks, key=lambda m: abs(m - s)) if marks else None
        if near is not None and abs(s - near) <= tol:
            hits += 1
            print(f"    {fmt(s)}  matches mark {fmt(near)}  delta {s - near:+.2f}s")
        else:
            print(f"    {fmt(s)}  FALSE POSITIVE"
                  + (f" (nearest mark {fmt(near)}, {s - near:+.1f}s)" if near else ""))
    for m in marks:
        if not splits or min(abs(m - s) for s in splits) > tol:
            print(f"    {fmt(m)}  MISSED mark")
    print(f"    {hits}/{len(marks)} marks found, "
          f"{len(splits) - hits} extra detection(s)")


def process(path: Path, model, cfg, device, args) -> dict:
    duration = probe_duration(str(path))
    existing, _ = probe_existing(path)

    row: dict = {"path": str(path), "duration": round(duration, 3),
                 "existing_chapters": len(existing)}

    if existing and args.existing == "fail":
        raise RuntimeError(f"{path.name}: already has {len(existing)} chapters")
    if existing and args.existing == "skip":
        print(f"  {path.name}: {len(existing)} chapters already -- skipped "
              f"(--existing replace|compare to override)")
        return {**row, "action": "skipped"}

    scores = scan(str(path), model, cfg, device, hop=args.hop,
                  batch_size=args.batch_size)
    events = to_events(scores, cfg)
    splits, dropped = boundaries(events, duration, args.anchor,
                                 args.start_margin, args.end_margin,
                                 args.min_gap)
    chapters = build_chapters(splits, duration, args.title_format)

    print(f"  {path.name}  {fmt(duration)}  ->  {len(events)} fade(s), "
          f"{len(chapters)} chapter(s)")
    for c in chapters:
        print(f"    {fmt(c['start'])} -> {fmt(c['end'])}  {c['title']}")
    for d in dropped:
        print(f"    dropped {fmt(anchor_time(d, args.anchor))} "
              f"(p={d['confidence']:.3f}): {d['reason']}")

    row |= {"events": events, "splits": [round(s, 3) for s in splits],
            "dropped": [{"start": d["start"], "end": d["end"],
                         "reason": d["reason"]} for d in dropped],
            "chapters": [{**c, "start": round(c["start"], 3),
                          "end": round(c["end"], 3)} for c in chapters]}

    if args.existing == "compare" and existing:
        compare_to_existing(existing, splits, duration, args)
        return {**row, "action": "compared"}
    if args.dry_run:
        return {**row, "action": "dry-run"}
    if len(splits) == 0:
        print("    no breaks found -- nothing to write")
        return {**row, "action": "no-breaks"}
    if path.suffix.lower() not in CHAPTER_EXT:
        print(f"    {path.suffix} cannot carry chapters -- "
              f"use --out-dir with an .mkv name, or remux first")
        return {**row, "action": "unsupported-container"}

    if args.in_place:
        tmp = path.with_name(f".{path.stem}.cdml-tmp{path.suffix}")
        remux(path, tmp, chapters)
        verify(tmp, len(chapters))
        os.replace(tmp, path)
        out = path
    else:
        out = args.out_dir / f"{path.stem}{args.suffix}{path.suffix}" \
            if args.out_dir else \
            path.with_name(f"{path.stem}{args.suffix}{path.suffix}")
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists() and not args.overwrite:
            print(f"    {out} exists -- skipped (--overwrite to replace)")
            return {**row, "action": "output-exists", "output": str(out)}
        remux(path, out, chapters)
        verify(out, len(chapters))
    print(f"    wrote {len(chapters)} chapters -> {out}")
    return {**row, "action": "written", "output": str(out)}


# --- CLI ---------------------------------------------------------------------

def collect(inputs: list[str]) -> list[Path]:
    paths: list[Path] = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            paths += find_videos(p)
        elif p.is_file():
            paths.append(p)
        else:
            raise SystemExit(f"not found: {item}")
    seen, uniq = set(), []
    for p in paths:
        if p.suffix.lower() in VIDEO_EXT and p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+",
                    help="episode files and/or show directories (searched recursively)")
    ap.add_argument("--model", default="models/fade_detector_v2.pt")

    out = ap.add_argument_group("output")
    out.add_argument("--in-place", action="store_true",
                     help="replace the source file (atomic; only after verification)")
    out.add_argument("--out-dir", type=Path, default=None,
                     help="write next to the source by default")
    out.add_argument("--suffix", default=".chapters",
                     help="filename suffix when not writing in place")
    out.add_argument("--overwrite", action="store_true")
    out.add_argument("--dry-run", action="store_true", help="detect, write nothing")
    out.add_argument("--json", help="write a report of every file here")
    out.add_argument("--existing", choices=["skip", "replace", "compare", "fail"],
                     default="skip",
                     help="what to do with a file that already has chapters "
                          "(default: skip)")

    ch = ap.add_argument_group("chapter geometry")
    ch.add_argument("--anchor", choices=["mid", "start", "end"], default="mid",
                    help="where in the fade the boundary goes (default: mid)")
    ch.add_argument("--start-margin", type=float, default=5.0,
                    help="ignore fades this close to the start (default: 5s)")
    ch.add_argument("--end-margin", type=float, default=20.0,
                    help="ignore fades this close to the end (default: 20s)")
    ch.add_argument("--min-gap", type=float, default=30.0,
                    help="minimum spacing between two breaks (default: 30s)")
    ch.add_argument("--title-format", default="Act {i}")

    det = ap.add_argument_group("detector")
    det.add_argument("--hop", type=int, default=CLIP_FRAMES // 2)
    det.add_argument("--batch-size", type=int, default=16)
    det.add_argument("--high", type=float, default=None)
    det.add_argument("--low", type=float, default=None)
    det.add_argument("--min-frames", type=int, default=None)
    det.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    args = ap.parse_args()

    videos = collect(args.inputs)
    if not videos:
        raise SystemExit("no video files found")
    if args.in_place and args.out_dir:
        raise SystemExit("--in-place and --out-dir are mutually exclusive")

    device = torch.device(args.device or
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    model, cfg, thr = load_model(args.model, device)

    # Same defaulting as cdml.infer: trigger on the threshold tuned on
    # validation, not on 0.5.
    cfg.hyst_high = args.high if args.high is not None else max(thr, cfg.hyst_low)
    if args.low is not None:
        cfg.hyst_low = args.low
    if args.min_frames is not None:
        cfg.min_event_frames = args.min_frames

    print(f"{Path(args.model).name} on {device.type}  "
          f"high={cfg.hyst_high:.3f} low={cfg.hyst_low:.3f}  "
          f"{len(videos)} file(s)")

    rows, failed = [], 0
    for path in videos:
        try:
            rows.append(process(path, model, cfg, device, args))
        except Exception as exc:                       # keep going through a batch
            failed += 1
            print(f"  {path.name}: FAILED -- {exc}", file=sys.stderr)
            rows.append({"path": str(path), "action": "failed", "error": str(exc)})

    counts: dict[str, int] = {}
    for r in rows:
        counts[r["action"]] = counts.get(r["action"], 0) + 1
    print("\n" + "-" * 68)
    print("  ".join(f"{k}: {v}" for k, v in sorted(counts.items())))

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(
            {"model": args.model, "fps": FPS, "anchor": args.anchor,
             "files": rows}, indent=2))
        print(f"wrote {args.json}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
