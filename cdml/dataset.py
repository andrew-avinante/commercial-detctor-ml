"""Cache loading, leakage-free splitting, and on-GPU augmentation.

The whole cache is ~250 MB, so it is loaded once and parked in VRAM. This keeps
training compute-bound without worker processes or per-sample disk reads.

Splits are grouped by episode when the filename records it, falling back to clip
id. Augmentation happens on the fly, so transformed copies cannot cross splits.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .config import BLACK_LEVEL  # noqa: F401  (kept for downstream imports)


class Cache:
    """Everything `preprocess.py` wrote, resident on `device`."""

    def __init__(self, path: str, device: torch.device):
        p = Path(path)
        meta = json.loads((p / "meta.json").read_text())
        self.meta = meta
        self.clips = meta["clips"]

        frames = np.load(p / "frames.npy")          # (N,T,H,W) uint8
        audio = np.load(p / "audio.npy")            # (N,T,A)  float32
        labels = np.load(p / "labels.npy")          # (N,T)    uint8

        self.device = device
        self.frames = torch.from_numpy(frames).to(device)          # stays uint8
        self.audio = torch.from_numpy(audio).to(device)
        self.labels = torch.from_numpy(labels).to(device).float()
        self.n, self.t = self.labels.shape

    def bytes_on_device(self) -> int:
        return sum(x.element_size() * x.nelement()
                   for x in (self.frames, self.audio, self.labels))

    def groups(self) -> list[str]:
        """Split key per clip: episode when known, else the clip's own id."""
        return [c["episode"] or f"clip::{c['num']}" for c in self.clips]


def grouped_split(groups: list[str], val_frac: float, test_frac: float,
                  seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Partition indices so no group spans two splits.

    Groups are shuffled then greedily assigned to whichever split is furthest
    below its target share, which keeps sizes close to the requested fractions
    even when group sizes are very uneven (some episodes contribute one clip,
    others a dozen).
    """
    rng = np.random.default_rng(seed)
    by_group: dict[str, list[int]] = {}
    for i, g in enumerate(groups):
        by_group.setdefault(g, []).append(i)

    names = sorted(by_group)
    rng.shuffle(names)
    # Largest groups placed first: they constrain the balance the most.
    names.sort(key=lambda g: -len(by_group[g]))

    n = len(groups)
    targets = np.array([1.0 - val_frac - test_frac, val_frac, test_frac]) * n
    filled = np.zeros(3)
    buckets: list[list[int]] = [[], [], []]

    for g in names:
        idx = by_group[g]
        deficit = targets - filled
        # Never route into a split with a zero target.
        deficit[targets <= 0] = -np.inf
        k = int(np.argmax(deficit))
        buckets[k].extend(idx)
        filled[k] += len(idx)

    return tuple(np.array(sorted(b), dtype=np.int64) for b in buckets)


def pos_weight_for(labels: torch.Tensor, cap: float) -> float:
    """neg/pos ratio, capped. ~12.5 on this dataset (7.4% positive frames)."""
    pos = float(labels.sum())
    neg = float(labels.numel() - pos)
    if pos <= 0:
        return 1.0
    return float(min(neg / pos, cap))


class BatchSampler:
    """Shuffled index batches over one split."""

    def __init__(self, indices: np.ndarray, batch_size: int, seed: int,
                 shuffle: bool = True):
        self.indices = indices
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return max(1, int(np.ceil(len(self.indices) / self.batch_size)))

    def __iter__(self):
        idx = self.indices.copy()
        if self.shuffle:
            self.rng.shuffle(idx)
        for i in range(0, len(idx), self.batch_size):
            yield torch.from_numpy(idx[i:i + self.batch_size])


def fetch(cache: Cache, batch_idx: torch.Tensor
          ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gather one batch as (frames float [0,1], audio, labels)."""
    bi = batch_idx.to(cache.device)
    frames = cache.frames.index_select(0, bi).float().div_(255.0)
    audio = cache.audio.index_select(0, bi)
    labels = cache.labels.index_select(0, bi)
    return frames, audio, labels


def augment(frames: torch.Tensor, audio: torch.Tensor, labels: torch.Tensor,
            cfg, generator: torch.Generator | None = None
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batch-level augmentation, entirely on GPU.

    Each batch receives horizontal flips, temporal jitter, bounded pixel noise,
    gamma jitter, and an additive dB audio gain. Frame, audio, and label jitter
    always use the same temporal offsets.
    """
    b, t = labels.shape
    dev = frames.device
    g = generator

    def rand(*shape):
        return torch.rand(*shape, device=dev, generator=g)

    # --- horizontal flip -----------------------------------------------------
    if cfg.aug_hflip > 0:
        flip = rand(b) < cfg.aug_hflip
        if flip.any():
            frames[flip] = torch.flip(frames[flip], dims=(-1,))

    # --- temporal jitter (frames, audio and labels shift together) -----------
    if cfg.aug_shift_frames > 0:
        shifts = torch.randint(-cfg.aug_shift_frames, cfg.aug_shift_frames + 1,
                               (b,), device=dev, generator=g)
        ar = torch.arange(t, device=dev)
        # Clamped gather => edge frames are held, not wrapped. Wrapping would
        # teleport post-fade black to the head of the clip and invent an event.
        src = (ar[None, :] - shifts[:, None]).clamp_(0, t - 1)
        fi = src[:, :, None, None].expand(-1, -1, frames.shape[2], frames.shape[3])
        frames = frames.gather(1, fi)
        audio = audio.gather(1, src[:, :, None].expand(-1, -1, audio.shape[2]))
        labels = labels.gather(1, src)

    # --- photometric ---------------------------------------------------------
    if cfg.aug_gamma > 0:
        gamma = 1.0 + (rand(b, 1, 1, 1) * 2.0 - 1.0) * cfg.aug_gamma
        frames = frames.clamp_(1e-4, 1.0).pow_(gamma)

    if cfg.aug_noise_std > 0:
        noise = torch.randn(frames.shape, device=dev, generator=g)
        frames = frames.add_(noise * cfg.aug_noise_std).clamp_(0.0, 1.0)

    # --- audio gain, additive in dB -----------------------------------------
    if cfg.aug_gain_db > 0:
        # features are dB/80, so a g dB gain is + g/80 on every energy channel
        offset = (rand(b, 1, 1) * 2.0 - 1.0) * (cfg.aug_gain_db / 80.0)
        audio = (audio + offset).clamp_(-1.0, 0.0)

    return frames, audio, labels
