"""Build the training cache.

    python -m cdml.preprocess --vids commercial_detector/training_vids \
                              --labels label_json --out cache

Writes four files into --out:

    frames.npy   (N, 192, 64, 64) uint8   memory-mappable
    audio.npy    (N, 192, 17)     float32
    labels.npy   (N, 192)         uint8
    meta.json    per-clip id, episode, source path

The whole 311-clip set lands at ~250 MB instead of 5.3 GB, because frames stay
uint8 at 64x64 instead of float64 at 128x128 flattened. That is what lets
`train.py` hold the entire dataset in VRAM and drop the data loader entirely.

Two correctness fixes live here rather than in the model:

* Clips are paired with their labels by parsed clip number. The old pipeline
  built three independent `glob.glob` lists and zipped them positionally, which
  silently mistrains the moment the three orderings disagree.
* Per-frame video scalars are NOT cached. They are derived from the frames on
  GPU at training time, so they can never go stale relative to an augmentation
  that changes pixel values.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np

from .config import CLIP_FRAMES, FPS, IMG_SIZE, N_AUDIO
from .features import DecodeError, clip_features

# train_00007_A Creepy Tangle in the Bermuda Triangle s03e02.mp4  -> (7, "...s03e02")
# train_00000.mp4                                                 -> (0, None)
CLIP_RE = re.compile(r"(?:train|test)_0*(\d+)(?:_(.+?))?\.mp4$", re.IGNORECASE)
EPISODE_RE = re.compile(r"^(.*?s\d{1,2}e\d{1,2})", re.IGNORECASE)


def parse_clip(path: str) -> tuple[int, str | None]:
    m = CLIP_RE.search(os.path.basename(path))
    if not m:
        return -1, None
    num = int(m.group(1))
    tail = m.group(2)
    if not tail:
        return num, None
    ep = EPISODE_RE.match(tail)
    return num, (ep.group(1) if ep else tail).strip()


def load_episode_map(path: str | None) -> dict[int, str]:
    """Optional `{clip_num: episode}` sidecar, for clips already renamed.

    `rename.py` rewrites `train_00007_A Creepy Tangle... s03e02.mp4` to
    `train_00007.mp4`, discarding the only record of which episode a clip came
    from. Without it, train/val/test can share an episode and every score is
    optimistic -- the model gets credit for recognising a background it already
    saw. This lets that mapping be supplied out of band.

    Accepts either `{"7": "...s03e02"}` or `[{"num": 7, "episode": "..."}]`.
    """
    if not path:
        return {}
    raw = json.loads(Path(path).read_text())
    if isinstance(raw, dict):
        return {int(k): str(v) for k, v in raw.items() if v}
    return {int(r["num"]): str(r["episode"]) for r in raw if r.get("episode")}


def timestamp_to_frame(ts: str) -> int:
    ts = ts.replace(",", ".")
    parts = [float(p) for p in ts.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0.0)
    h, m, s = parts
    return int(round((h * 3600 + m * 60 + s) * FPS))


def load_labels(label_dir: str) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for f in sorted(Path(label_dir).glob("*.json")):
        for item in json.loads(f.read_text()):
            out[int(item["num"])] = item
    return out


def build_label_vector(item: dict, n_frames: int = CLIP_FRAMES) -> np.ndarray:
    """Per-frame 0/1 fade mask, clamped to the clip so it can never go ragged."""
    start = max(0, min(n_frames, timestamp_to_frame(item["start"])))
    end = max(0, min(n_frames, timestamp_to_frame(item["end"])))
    if end < start:
        start, end = end, start
    y = np.zeros(n_frames, np.uint8)
    y[start:end] = 1
    return y


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vids", required=True, help="directory of clip .mp4 files")
    ap.add_argument("--labels", default="label_json")
    ap.add_argument("--out", default="cache")
    ap.add_argument("--size", type=int, default=IMG_SIZE)
    ap.add_argument("--keep-pruned", action="store_true")
    ap.add_argument("--episodes", help="optional {clip_num: episode} JSON, used "
                                       "to group the split when filenames no "
                                       "longer carry the episode name")
    args = ap.parse_args()

    episode_map = load_episode_map(args.episodes)
    labels = load_labels(args.labels)
    vids = sorted(Path(args.vids).glob("*.mp4"))
    if not vids:
        raise SystemExit(f"no .mp4 files under {args.vids}")

    kept, skipped = [], {"unlabelled": 0, "pruned": 0, "decode_error": 0}
    for path in vids:
        num, episode = parse_clip(str(path))
        item = labels.get(num)
        if num < 0 or item is None:
            skipped["unlabelled"] += 1
            continue
        if item.get("prune") and not args.keep_pruned:
            skipped["pruned"] += 1
            continue
        # sidecar wins: filenames are the lossy source here, not the sidecar
        kept.append((num, episode_map.get(num, episode), str(path), item))

    n = len(kept)
    if n == 0:
        raise SystemExit("nothing to preprocess after filtering")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    frames = np.lib.format.open_memmap(
        out / "frames.npy", mode="w+", dtype=np.uint8,
        shape=(n, CLIP_FRAMES, args.size, args.size))
    audio = np.zeros((n, CLIP_FRAMES, N_AUDIO), np.float32)
    ylab = np.zeros((n, CLIP_FRAMES), np.uint8)
    meta = []

    write = 0
    for i, (num, episode, path, item) in enumerate(kept):
        try:
            f, a = clip_features(path, CLIP_FRAMES, size=args.size)
        except DecodeError as e:
            skipped["decode_error"] += 1
            print(f"  !! {path}: {e}")
            continue
        frames[write] = f
        audio[write] = a
        ylab[write] = build_label_vector(item)
        meta.append({
            "index": write, "num": num, "episode": episode, "path": path,
            "positive_frames": int(ylab[write].sum()),
        })
        write += 1
        if write % 25 == 0 or i == n - 1:
            print(f"  {write}/{n} clips", flush=True)

    frames.flush()
    if write != n:                      # trim the tail left by decode failures
        del frames
        full = np.load(out / "frames.npy", mmap_mode="r")
        trimmed = np.array(full[:write])        # materialise before reopening
        del full                                # drop the mmap, then overwrite
        np.save(out / "frames.npy", trimmed)
        del trimmed
    np.save(out / "audio.npy", audio[:write])
    np.save(out / "labels.npy", ylab[:write])

    y = ylab[:write]
    episodes = {m["episode"] for m in meta if m["episode"]}
    (out / "meta.json").write_text(json.dumps({
        "clips": meta, "size": args.size, "clip_frames": CLIP_FRAMES,
        "n_audio": N_AUDIO, "skipped": skipped,
    }, indent=2))

    print(f"\nwrote {write} clips to {out}")
    print(f"  positive clips : {int((y.sum(1) > 0).sum())}/{write}")
    print(f"  positive frames: {int(y.sum())}/{y.size} "
          f"({100 * y.sum() / y.size:.2f}%)")
    print(f"  episodes known : {len(episodes)}"
          + ("" if episodes else "  (grouping will fall back to clip id)"))
    print(f"  skipped        : {skipped}")
    print(f"  cache size     : "
          f"{sum(p.stat().st_size for p in out.glob('*.npy')) / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
