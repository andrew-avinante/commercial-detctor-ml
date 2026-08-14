"""Per-frame fade detector: shared CNN frame encoder -> bidirectional GRU.

Three changes from the Keras model, in descending order of expected impact.

1. The frame encoder is a small 2-D CNN with shared weights across time,
   instead of feeding a flattened 16384-vector straight into an LSTM. The old
   first layer alone was 4*(16384+128+1)*128 ~= 8.4M parameters -- 97% of the
   network, spent on a dense projection that throws away spatial structure. On
   296 training clips that is not a model, it is a memoriser.

2. Six explicit luminance statistics are concatenated to the CNN embedding.
   A fade to black *is* a luminance trajectory; per-frame mean, spread, peak,
   dark-pixel fractions and inter-frame motion hand the temporal model the
   signal directly rather than making it rediscover `frame.mean()` through a
   recurrent bottleneck. They are computed here, on GPU, from whatever pixels
   arrive -- so pixel augmentations stay consistent with them for free.

3. The GRU is bidirectional. This is offline detection over a complete 8-second
   window, so the second half of a fade is legitimately available when scoring
   the first half. A causal LSTM gives that up for nothing.

Result is ~230k parameters (about 1 MB) against the original ~8.5M / 109 MB.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .config import BLACK_LEVEL, DARK_LEVEL, N_AUDIO, N_VIDEO_SCALAR


def video_scalars(frames: torch.Tensor) -> torch.Tensor:
    """(B, T, H, W) float in [0,1] -> (B, T, 6) luminance statistics.

    Deliberately built from cheap reductions only (no `torch.quantile`, which
    refuses inputs past ~16M elements in the reduced dimension and would break
    at larger batch sizes).
    """
    b, t = frames.shape[:2]
    flat = frames.reshape(b, t, -1)

    mean = flat.mean(-1)
    std = flat.std(-1)
    peak = flat.amax(-1)
    frac_black = (flat < BLACK_LEVEL).to(frames.dtype).mean(-1)
    frac_dark = (flat < DARK_LEVEL).to(frames.dtype).mean(-1)

    motion = torch.zeros_like(mean)
    if t > 1:
        motion[:, 1:] = (flat[:, 1:] - flat[:, :-1]).abs().mean(-1)

    return torch.stack([mean, std, peak, frac_black, frac_dark, motion], dim=-1)


class FrameEncoder(nn.Module):
    """64x64 -> embedding, applied with shared weights to every frame."""

    def __init__(self, width: int = 16, out_dim: int = 64):
        super().__init__()
        c1, c2, c3, c4 = width, width * 2, width * 4, out_dim

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.SiLU(inplace=True),
            )

        self.net = nn.Sequential(
            block(1, c1),        # 64 -> 32
            block(c1, c2),       # 32 -> 16
            block(c2, c3),       # 16 -> 8
            block(c3, c4),       # 8  -> 4
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.out_dim = c4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AudioEncoder(nn.Module):
    """Widen the 17 dB-scaled energy channels before fusion.

    Measured on this dataset, ~90% of *negative* clips also contain a full
    fade to black -- because the original clip miner selected every clip from an ffmpeg
    `blackdetect` hit. So the picture barely discriminates, and audio carries
    most of the signal. Fusing 17 audio dims straight against a 64-dim video
    embedding lets the video branch dominate the projection by width alone.
    """

    def __init__(self, n_audio: int = N_AUDIO, out_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_audio, out_dim),
            nn.LayerNorm(out_dim),
            nn.SiLU(inplace=True),
            nn.Linear(out_dim, out_dim),
        )
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FadeDetector(nn.Module):
    def __init__(self, cnn_width: int = 16, embed_dim: int = 128,
                 gru_hidden: int = 64, gru_layers: int = 2,
                 dropout: float = 0.25, n_audio: int = N_AUDIO,
                 audio_dim: int = 32, use_video: bool = True,
                 use_audio: bool = True):
        super().__init__()
        if not (use_video or use_audio):
            raise ValueError("at least one of use_video/use_audio must be set")
        self.use_video, self.use_audio = use_video, use_audio

        fused = 0
        if use_video:
            self.encoder = FrameEncoder(width=cnn_width, out_dim=64)
            fused += self.encoder.out_dim + N_VIDEO_SCALAR
        if use_audio:
            self.audio_encoder = AudioEncoder(n_audio, audio_dim)
            fused += audio_dim

        self.proj = nn.Sequential(
            nn.Linear(fused, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.gru = nn.GRU(
            embed_dim, gru_hidden, num_layers=gru_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(gru_hidden * 2, 1),
        )

    def forward(self, frames: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        """frames (B,T,H,W) float in [0,1], audio (B,T,N_AUDIO) -> logits (B,T).

        One logit per frame with BCE, not a 2-way softmax: identical decision
        boundary, half the head, and it makes threshold tuning on the validation
        set a direct operation on the output.
        """
        b, t, h, w = frames.shape
        parts = []
        if self.use_video:
            emb = self.encoder(frames.reshape(b * t, 1, h, w)).reshape(b, t, -1)
            parts += [emb, video_scalars(frames)]
        if self.use_audio:
            parts.append(self.audio_encoder(audio))

        x = self.proj(torch.cat(parts, dim=-1))
        x, _ = self.gru(x)
        return self.head(x).squeeze(-1)


def build_model(cfg) -> FadeDetector:
    return FadeDetector(
        cnn_width=cfg.cnn_width, embed_dim=cfg.embed_dim,
        gru_hidden=cfg.gru_hidden, gru_layers=cfg.gru_layers,
        dropout=cfg.dropout, audio_dim=cfg.audio_dim,
        use_video=cfg.use_video, use_audio=cfg.use_audio,
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
