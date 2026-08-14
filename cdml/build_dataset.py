"""Cut a training cache straight from chapter-marked episodes.

    # 1. look before you leap -- seconds, no decoding
    python -m cdml.chapters --shows "/media/Authorized/Example Show" ...

    # 2. cut clips (hours; resumable, safe to ctrl-C and rerun)
    python -m cdml.build_dataset --shows "/media/Authorized/Example Show" \
                                 "REDACTED_LOCAL_PATH Show B" \
                                 "REDACTED_LOCAL_PATH Show C" \
                                 "REDACTED_LOCAL_PATH Show D" \
                                 --out data/chapters --workers 4

    # 3. glue the shards into a cache train.py can read
    python -m cdml.build_dataset --out data/chapters --assemble --cache cache

Labels come from chapter markers that align with commercial-break fades. The
builder creates three kinds of windows: chapter-marked breaks, non-break fades,
and ordinary footage. Fade windows are placed at random offsets so training
matches sliding-window inference, while their natural durations are preserved.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .chapters import episode_id, find_videos, label_episode, probe
from .config import AUDIO_SR, CLIP_FRAMES, FPS, HOP, IMG_SIZE, N_AUDIO
from .features import DecodeError, audio_features

_print_lock = threading.Lock()


def log(msg: str) -> None:
    with _print_lock:
        print(msg, flush=True)


# --- decoding ----------------------------------------------------------------

def decode_episode(path: str, size: int, fps: int, sr: int):
    """Decode video and audio in ONE pass. Returns (frames uint8, wav float32).

    One `ffmpeg` invocation, not two: these are 1-5 GB files on a network
    mount, and reading each one twice doubles the only part of this job that
    is actually slow.
    """
    with tempfile.TemporaryDirectory() as td:
        apath = Path(td) / "a.f32"
        cmd = [
            "ffmpeg", "-v", "error", "-nostdin", "-i", path,
            "-map", "0:v:0",
            "-vf", f"fps={fps},scale={size}:{size}:flags=area,format=gray",
            "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1",
            "-map", "0:a:0?", "-ac", "1", "-ar", str(sr),
            "-f", "f32le", "-acodec", "pcm_f32le", str(apath),
        ]
        p = subprocess.run(cmd, capture_output=True)
        if p.returncode != 0:
            raise DecodeError(p.stderr.decode("utf8", "replace")[-400:])
        px = size * size
        n = len(p.stdout) // px
        if n == 0:
            raise DecodeError("no video frames decoded")
        frames = np.frombuffer(p.stdout[: n * px], np.uint8).reshape(n, size, size)
        wav = (np.fromfile(apath, np.float32) if apath.exists()
               else np.zeros(0, np.float32))

    need = n * (sr // fps)
    if len(wav) < need:                     # silent or short track: pad
        wav = np.concatenate([wav, np.zeros(need - len(wav), np.float32)])
    return frames, wav[:need]


# --- window selection --------------------------------------------------------

def place(rng, span: tuple[int, int], n: int, T: int, margin: int) -> int | None:
    """Random window start that contains [s, e) with `margin` frames to spare.

    Randomising this is the point: a fade pinned to the middle of every clip is
    a cue that exists in training and nowhere else.
    """
    s, e = span
    lo, hi = max(0, e - T + margin), min(s - margin, n - T)
    if lo > hi:                              # fade longer than the window
        c = (s + e) // 2 - T // 2
        return max(0, min(c, n - T)) if n >= T else None
    return int(rng.integers(lo, hi + 1))


def cut(frames, wav, f0: int, T: int, label_span=None):
    """One window -> (frames, audio features, per-frame labels)."""
    fr = frames[f0:f0 + T]
    aud = audio_features(wav[f0 * HOP:(f0 + T) * HOP], T)
    y = np.zeros(T, np.uint8)
    if label_span is not None:
        s, e = label_span
        y[max(0, s - f0):max(0, e - f0)] = 1
    return fr, aud, y


def build_episode(path: Path, show: str, args, rng) -> dict | None:
    bounds, duration = probe(path)
    if not bounds:
        return None

    frames, wav = decode_episode(str(path), args.size, args.fps, AUDIO_SR)
    n = len(frames)
    lum = frames.reshape(n, -1).mean(1) / 255.0

    lab = label_episode(lum, bounds, duration, args.fps, args.black_thr,
                        args.dark_thr, args.min_core, args.tol, args.end_margin,
                        args.max_ramp)
    T, M = args.clip_frames, args.margin
    F, A, Y, kinds = [], [], [], []

    # -- positives: the chapter-marked fades ---------------------------------
    for br in lab["breaks"]:
        span = (br["start"], br["end"])
        for _ in range(args.pos_per_break):
            f0 = place(rng, span, n, T, M)
            if f0 is None:
                continue
            f, a, y = cut(frames, wav, f0, T, span)
            F.append(f); A.append(a); Y.append(y); kinds.append("positive")

    # -- hard negatives: fades that are NOT at a chapter mark ----------------
    hard = lab["hard_negative_fades"]
    if len(hard) > args.hard_per_episode:
        idx = rng.choice(len(hard), args.hard_per_episode, replace=False)
        hard = [hard[i] for i in sorted(idx)]
    for s, e, _cs, _ce in hard:
        f0 = place(rng, (s, e), n, T, M)
        if f0 is None:
            continue
        f, a, y = cut(frames, wav, f0, T, None)
        F.append(f); A.append(a); Y.append(y); kinds.append("hard_negative")

    # -- easy negatives: ordinary footage, nowhere near a fade ---------------
    blocked = np.zeros(n, bool)
    for s, e, _cs, _ce in lab["hard_negative_fades"]:
        blocked[max(0, s - T):min(n, e + T)] = True
    for br in lab["breaks"]:
        blocked[max(0, br["start"] - T):min(n, br["end"] + T)] = True
    free = [i for i in range(0, max(1, n - T), T // 2) if not blocked[i:i + T].any()]
    if free:
        k = min(args.easy_per_episode, len(free))
        for i in rng.choice(len(free), k, replace=False):
            f0 = int(free[i])
            f, a, y = cut(frames, wav, f0, T, None)
            F.append(f); A.append(a); Y.append(y); kinds.append("easy_negative")

    if not F:
        return None
    return {
        "frames": np.stack(F), "audio": np.stack(A).astype(np.float32),
        "labels": np.stack(Y), "kinds": kinds,
        "meta": {"show": show, "episode": episode_id(path), "path": str(path),
                 "duration": duration, "n_frames": n,
                 "chapter_bounds": [round(b, 2) for b in bounds],
                 "n_breaks": len(lab["breaks"]),
                 "unmatched_marks": lab["unmatched_marks"],
                 "dropped_tail_marks": lab["dropped_tail_marks"],
                 "n_black_events": lab["n_black_events"],
                 "counts": {k: kinds.count(k) for k in set(kinds)}},
    }


# --- driver ------------------------------------------------------------------

def shard_name(show: str, ep: str) -> str:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in f"{show}__{ep}")
    return safe + ".npz"


def run_build(args) -> None:
    out = Path(args.out)
    (out / "shards").mkdir(parents=True, exist_ok=True)

    jobs = []
    for show_dir in (Path(s) for s in args.shows):
        if not show_dir.is_dir():
            raise SystemExit(f"not a directory: {show_dir}")
        for p in find_videos(show_dir):
            jobs.append((show_dir.name, p))
    if args.limit:
        jobs = jobs[:args.limit]

    todo = [(s, p) for s, p in jobs
            if not (out / "shards" / shard_name(s, episode_id(p))).exists()]
    log(f"{len(jobs)} episodes, {len(jobs) - len(todo)} already done, "
        f"{len(todo)} to do, {args.workers} workers\n")

    done = [0]

    def work(job):
        show, path = job
        dest = out / "shards" / shard_name(show, episode_id(path))
        rng = np.random.default_rng(abs(hash((show, path.name))) % (2**32))
        try:
            r = build_episode(path, show, args, rng)
        except DecodeError as e:
            log(f"  !! {path.name}: decode failed -- {e}")
            return
        except Exception as e:                      # noqa: BLE001
            log(f"  !! {path.name}: {type(e).__name__}: {e}")
            return
        if r is None:
            log(f"  -- {path.name}: no chapters / no windows, skipped")
            return
        tmp = dest.with_suffix(".tmp.npz")          # atomic: a ctrl-C mid-write
        np.savez_compressed(tmp, frames=r["frames"], audio=r["audio"],
                            labels=r["labels"],
                            kinds=np.array(r["kinds"]),
                            meta=json.dumps(r["meta"]))
        tmp.replace(dest)                           # must not look complete
        done[0] += 1
        c = r["meta"]["counts"]
        log(f"  [{done[0]:3d}/{len(todo)}] {show[:22]:22s} {r['meta']['episode']:7s} "
            f"pos {c.get('positive', 0):2d}  hard {c.get('hard_negative', 0):2d}  "
            f"easy {c.get('easy_negative', 0):2d}  "
            f"(fades {r['meta']['n_black_events']:3d}, "
            f"unmatched {len(r['meta']['unmatched_marks'])})")

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(work, todo))

    log(f"\nshards in {out / 'shards'}  -- now run with --assemble")


def run_assemble(args) -> None:
    shards = sorted((Path(args.out) / "shards").glob("*.npz"))
    if not shards:
        raise SystemExit(f"no shards in {Path(args.out) / 'shards'}")

    F, A, Y, meta, kinds = [], [], [], [], []
    dropped = 0
    for sh in shards:
        z = np.load(sh, allow_pickle=False)
        m = json.loads(str(z["meta"]))
        k = [str(x) for x in z["kinds"]]
        lab = z["labels"]
        # A window labelled positive edge to edge is a break whose black runs
        # longer than the window itself: nothing but black, no transition
        # visible, so there is no "where" for the model to learn. Neighbouring
        # windows still cover the fade in and out, so nothing is lost by
        # dropping these -- and keeping them teaches "all black => break",
        # which is the false positive we least want on a dark scene.
        keep = np.flatnonzero(lab.sum(1) < args.max_label_frac * lab.shape[1])
        dropped += len(lab) - len(keep)
        if len(keep) == 0:
            continue
        F.append(z["frames"][keep]); A.append(z["audio"][keep]); Y.append(lab[keep])
        for i in keep:
            meta.append({"index": len(meta), "num": len(meta),
                         "show": m["show"], "episode": f"{m['show']}/{m['episode']}",
                         "path": m["path"], "kind": k[i],
                         "positive_frames": int(lab[i].sum())})
        kinds += [k[i] for i in keep]

    frames = np.concatenate(F); audio = np.concatenate(A); labels = np.concatenate(Y)
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    np.save(cache / "frames.npy", frames)
    np.save(cache / "audio.npy", audio)
    np.save(cache / "labels.npy", labels)
    (cache / "meta.json").write_text(json.dumps({
        "clips": meta, "size": frames.shape[-1], "clip_frames": frames.shape[1],
        "n_audio": audio.shape[-1],
        "source": "chapter markers", "shards": len(shards),
        "kind_counts": {k: kinds.count(k) for k in sorted(set(kinds))},
    }, indent=2))

    eps = sorted({m["episode"] for m in meta})
    shows = sorted({m["show"] for m in meta})
    print(f"\nwrote {len(labels)} windows to {cache}")
    print(f"  shows          : {len(shows)}   {', '.join(s[:24] for s in shows)}")
    print(f"  episodes       : {len(eps)}  (these are the split groups)")
    for k in sorted(set(kinds)):
        print(f"  {k:15s}: {kinds.count(k)}")
    print(f"  positive frames: {int(labels.sum())}/{labels.size} "
          f"({100 * labels.sum() / labels.size:.2f}%)")

    # A window labelled positive end-to-end teaches nothing about *where* the
    # fade is, and is the signature of a runaway ramp. Surface it.
    plen = labels.sum(1)[labels.sum(1) > 0]
    if len(plen):
        T = labels.shape[1]
        print(f"  label length   : median {np.median(plen):.0f}/{T} frames, "
              f"p10 {np.percentile(plen, 10):.0f}, p90 {np.percentile(plen, 90):.0f}")
    if dropped:
        print(f"  dropped        : {dropped} all-black window(s) "
              f"(label >= {args.max_label_frac:.0%} of the window)")
    print(f"  cache size     : "
          f"{sum(p.stat().st_size for p in cache.glob('*.npy')) / 1e6:.0f} MB")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shows", nargs="*", default=[])
    ap.add_argument("--out", default="data/chapters", help="shard directory")
    ap.add_argument("--assemble", action="store_true",
                    help="concatenate shards into --cache instead of building")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--max-label-frac", type=float, default=0.95,
                    help="assemble: drop windows labelled positive for at least "
                         "this fraction of their length (all-black, no context)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="first N episodes only")

    ap.add_argument("--size", type=int, default=IMG_SIZE)
    ap.add_argument("--fps", type=int, default=FPS)
    ap.add_argument("--clip-frames", type=int, default=CLIP_FRAMES)
    ap.add_argument("--margin", type=int, default=8,
                    help="frames of slack kept between a fade and the window edge")

    ap.add_argument("--black-thr", type=float, default=0.06)
    ap.add_argument("--dark-thr", type=float, default=0.25)
    ap.add_argument("--max-ramp", type=int, default=12,
                    help="cap on fade-ramp growth each side, in frames. Without "
                         "it the dark end credits merge into one huge 'fade'")
    ap.add_argument("--min-core", type=int, default=6)
    ap.add_argument("--tol", type=float, default=6.0,
                    help="seconds a chapter mark may sit from its fade")
    ap.add_argument("--end-margin", type=float, default=20.0)

    ap.add_argument("--pos-per-break", type=int, default=3,
                    help="windows cut per break, at different offsets")
    ap.add_argument("--hard-per-episode", type=int, default=6,
                    help="non-break fades kept per episode")
    ap.add_argument("--easy-per-episode", type=int, default=8,
                    help="ordinary-footage windows per episode")
    args = ap.parse_args()

    # `features.audio_features` frames the waveform on the module constant HOP
    # (= AUDIO_SR // FPS). Decoding video at any other rate would slide the
    # audio against the picture by a growing offset, silently, so refuse.
    if args.fps != FPS:
        ap.error(f"--fps must stay {FPS}: the audio framing in features.py is "
                 f"built on HOP={HOP} = AUDIO_SR/{FPS}, and changing one "
                 f"without the other misaligns audio from video.")

    if args.assemble:
        run_assemble(args)
    elif args.shows:
        run_build(args)
    else:
        ap.error("give --shows to build, or --assemble to collect shards")


if __name__ == "__main__":
    main()
