"""Shared constants and the training config.

Everything downstream keys off these, so a change here (clip length, image size,
audio rate) propagates through preprocessing, training and inference together.
"""
from dataclasses import dataclass

# --- clip geometry -----------------------------------------------------------
FPS = 24
CLIP_SECONDS = 8
CLIP_FRAMES = FPS * CLIP_SECONDS          # 192
IMG_SIZE = 64                              # 64x64 grayscale is ample for fades

# --- audio -------------------------------------------------------------------
# 24000 / 24 = 1000 samples per video frame exactly, so every audio feature row
# lines up with a video frame with no resampling drift.
AUDIO_SR = 24_000
HOP = AUDIO_SR // FPS                      # 1000
N_FFT = 2048
N_MELS = 16
FMIN, FMAX = 20.0, 10_000.0
DB_FLOOR = -80.0                           # dBFS floor; also the "silence" value

# --- feature widths ----------------------------------------------------------
N_VIDEO_SCALAR = 6                         # computed on GPU from frames
N_AUDIO = 1 + N_MELS                       # log-RMS dBFS + mel band energies
BLACK_LEVEL = 16.0 / 255.0                 # "this pixel is black"
DARK_LEVEL = 48.0 / 255.0                  # "this pixel is dark"


@dataclass
class TrainConfig:
    cache: str = "cache"
    out: str = "runs/baseline"

    # split (grouped by episode when known, else by source clip)
    val_frac: float = 0.15
    test_frac: float = 0.15
    seed: int = 1337
    # Put every clip of one show in test, and nothing of it in train. A normal
    # grouped split only proves the model generalises to a new *episode* of a
    # show it has already studied; this proves -- or disproves -- that it
    # generalises to a show it has never seen, which is what you need before
    # pointing it at anything that is not in the training corpus.
    holdout_show: str = ""
    # A second cache folded into TRAIN only, never val or test. Use it to add
    # an older or dirtier corpus as extra training material while keeping the
    # evaluation set clean and comparable to a run without it.
    extra_cache: str = ""

    # optimisation
    epochs: int = 120
    batch_size: int = 16
    lr: float = 3e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 5
    grad_clip: float = 1.0
    patience: int = 25                     # early stop on val average precision
    amp: bool = True

    # model
    cnn_width: int = 16                    # first conv channels; doubles 3x
    embed_dim: int = 128
    gru_hidden: int = 64
    gru_layers: int = 2
    dropout: float = 0.25
    audio_dim: int = 32
    # Ablation switches. Run with --use-video false to measure how much the
    # picture actually contributes; on this dataset almost every clip (positive
    # or negative) contains a fade to black, so audio does most of the work.
    use_video: bool = True
    use_audio: bool = True

    # loss
    pos_weight: float = -1.0               # <0 => derive from train label stats
    pos_weight_cap: float = 12.0

    # augmentation (applied on GPU, per batch)
    aug_hflip: float = 0.5
    aug_shift_frames: int = 12             # temporal jitter, edge padded
    aug_noise_std: float = 0.02            # in [0,1] pixel units
    aug_gamma: float = 0.15                # +/- gamma exponent jitter
    aug_gain_db: float = 6.0               # +/- audio gain, additive in dB

    # inference post-processing
    hyst_high: float = 0.60
    hyst_low: float = 0.35
    min_event_frames: int = 12             # 0.5 s
    smooth_frames: int = 5                 # moving average over the score trace
