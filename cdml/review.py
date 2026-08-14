"""Contact sheets for eyeballing a built cache.

    python -m cdml.review --cache cache_v2 --out review

Automatic labels are only worth having if you can check them, and a table of
counts will not tell you that a fade was labelled two seconds late. This
renders one PNG per window kind: each row is a clip, sampled evenly across its
frames, with two bars underneath --

    label bar   white where the frame is labelled a break
    audio bar   height is loudness, so a fade-out reads as a dip to nothing

A correct positive row shows the picture going black, the audio bar collapsing
underneath it, and the label bar bracketing both. A hard negative shows the
same black with an empty label bar. An easy negative shows neither.

PNG is written by hand (zlib + a CRC table) rather than pulling in Pillow --
this repo has no image dependency and does not need one for grey tiles.
"""
from __future__ import annotations

import argparse
import json
import struct
import zlib
from pathlib import Path

import numpy as np


def write_png(path: Path, img: np.ndarray) -> None:
    """8-bit greyscale PNG from a (h, w) uint8 array."""
    h, w = img.shape
    raw = b"".join(b"\x00" + img[y].tobytes() for y in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b""))


def sheet(frames, audio, labels, cols: int, tile: int,
          bar: int, gap: int) -> np.ndarray:
    """Tile n clips into one image, each row = sampled frames + label/audio bars."""
    n, T = labels.shape
    idx = np.linspace(0, T - 1, cols).round().astype(int)
    row_h = tile + 2 * bar + gap
    out = np.full((n * row_h, cols * tile), 32, np.uint8)

    for i in range(n):
        y0 = i * row_h
        for c, f in enumerate(idx):
            src = frames[i, f]
            if src.shape[0] != tile:                 # nearest-neighbour resize
                s = (np.arange(tile) * src.shape[0] // tile)
                src = src[s][:, s]
            out[y0:y0 + tile, c * tile:(c + 1) * tile] = src

        lab = labels[i, idx]
        y1 = y0 + tile
        for c in range(cols):
            out[y1:y1 + bar, c * tile:(c + 1) * tile] = 255 if lab[c] else 60

        # audio: dBFS in [-1, 0] -> bar height
        amp = np.clip(audio[i, idx, 0] + 1.0, 0.0, 1.0)
        y2 = y1 + bar
        for c in range(cols):
            hgt = int(round(amp[c] * bar))
            out[y2:y2 + bar, c * tile:(c + 1) * tile] = 20
            if hgt:
                out[y2 + bar - hgt:y2 + bar, c * tile:(c + 1) * tile] = 200
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default="cache_v2")
    ap.add_argument("--out", default="review")
    ap.add_argument("--per-kind", type=int, default=12, help="clips per sheet")
    ap.add_argument("--cols", type=int, default=24, help="frames sampled per clip")
    ap.add_argument("--tile", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cache = Path(args.cache)
    frames = np.load(cache / "frames.npy", mmap_mode="r")
    audio = np.load(cache / "audio.npy", mmap_mode="r")
    labels = np.load(cache / "labels.npy")
    meta = json.loads((cache / "meta.json").read_text())["clips"]
    kinds = [m.get("kind", "clip") for m in meta]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    for kind in sorted(set(kinds)):
        pool = np.flatnonzero(np.array(kinds) == kind)
        pick = rng.choice(pool, min(args.per_kind, len(pool)), replace=False)
        pick.sort()
        img = sheet(np.asarray(frames[pick]), np.asarray(audio[pick]),
                    labels[pick], args.cols, args.tile, bar=8, gap=6)
        dest = out / f"{kind}.png"
        write_png(dest, img)
        print(f"{dest}   {len(pick)} clips")
        for i in pick[:args.per_kind]:
            m = meta[i]
            print(f"    [{i:5d}] {m.get('episode', '?'):44s} "
                  f"labelled {int(labels[i].sum()):3d}/{labels.shape[1]}")

    print(f"\nRows are clips, columns are frames across the window.\n"
          f"Upper bar = label (white = break), lower bar = loudness.\n"
          f"A positive should go black with the audio bar collapsing and the\n"
          f"label bar covering both.")


if __name__ == "__main__":
    main()
