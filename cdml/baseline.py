"""A no-model baseline, to check the network is earning its keep.

    python -m cdml.baseline --cache cache

Scores each frame by how dark it is and how quiet it is, with a couple of
thresholds fitted on the training split only. If the network cannot clearly
beat this, the extra 230k parameters are not buying anything and the honest
answer is to ship the threshold rule.

Worth running before any training run: the original clip miner used ffmpeg
`blackdetect` to mine candidate clips, so the model's real job was the narrower
one of separating commercial-break fades from ordinary in-episode black frames.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from .config import TrainConfig
from .dataset import Cache, grouped_split
from .metrics import evaluate
from .model import video_scalars


def baseline_scores(cache: Cache) -> np.ndarray:
    """(N, T) score in [0,1]: dark picture AND quiet audio."""
    out = []
    for i in range(0, cache.n, 32):
        frames = cache.frames[i:i + 32].float().div_(255.0)
        scal = video_scalars(frames).cpu().numpy()
        rms = cache.audio[i:i + 32, :, 0].cpu().numpy()   # dB/80, in [-1, 0]

        darkness = scal[..., 3]                            # frac pixels < 16/255
        quiet = np.clip(-rms, 0.0, 1.0)                    # 1.0 == -80 dBFS
        out.append(darkness * quiet)
    return np.concatenate(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = Cache(args.cache, device)
    cfg = TrainConfig(val_frac=args.val_frac, test_frac=args.test_frac,
                      seed=args.seed)
    tr, va, te = grouped_split(cache.groups(), args.val_frac, args.test_frac,
                               args.seed)

    scores = baseline_scores(cache)
    labels = cache.labels.cpu().numpy()

    # Tune the operating threshold on train, report on val and test, exactly as
    # the network is evaluated -- so the comparison is like for like.
    tuned = evaluate(scores[tr], labels[tr], cfg)
    thr = tuned["threshold"]
    print(f"threshold tuned on train: {thr:.3f}\n")

    report = {"threshold": thr, "train": tuned}
    for name, idx in (("val", va), ("test", te)):
        if len(idx) == 0:
            continue
        r = evaluate(scores[idx], labels[idx], cfg, threshold=thr)
        report[name] = r
        print(f"{name}: AP {r['ap']:.4f}  frame_f1 {r['frame_f1']:.4f}  "
              f"event_f1 {r['event_f1']:.4f} "
              f"(tp {r['event_tp']} fp {r['event_fp']} fn {r['event_fn']})")

    print("\n" + json.dumps(report, indent=2)[:0] or "", end="")


if __name__ == "__main__":
    main()
