"""Decoding and audio feature extraction.

Decoding goes through ffmpeg pipes rather than OpenCV/librosa: it is one
dependency instead of two, it is faster for the sequential reads we do, and it
lets inference stream a whole episode without seeking.

The audio side is where the old pipeline lost the signal. `librosa.load(sr=24)`
resampled the *waveform* to 24 Hz, which low-passes everything above 12 Hz into
noise and leaves a zero-centred signal that carries no loudness. Fades are a
loudness phenomenon, so we keep the audio at 24 kHz and reduce it to per-frame
energy: log-RMS in dBFS plus 16 mel bands, also in dB.

dB matters twice over. It is the scale on which a fade-out is roughly linear,
and it makes a gain change a simple additive offset, which is what makes the
audio augmentation in `dataset.py` a one-line add.
"""
from __future__ import annotations

import json
import subprocess

import numpy as np

from .config import (
    AUDIO_SR, DB_FLOOR, FMAX, FMIN, FPS, HOP, IMG_SIZE, N_FFT, N_MELS,
)


class DecodeError(RuntimeError):
    pass


def _run(cmd: list[str]) -> bytes:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        tail = p.stderr.decode("utf-8", "replace").strip().splitlines()[-4:]
        raise DecodeError(f"{' '.join(cmd[:6])}...: {' | '.join(tail)}")
    return p.stdout


def probe_duration(path: str) -> float:
    out = _run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "json", path,
    ])
    return float(json.loads(out)["format"]["duration"])


def decode_gray(path: str, size: int = IMG_SIZE, fps: int = FPS,
                start: float | None = None,
                duration: float | None = None) -> np.ndarray:
    """Decode to (T, size, size) uint8 grayscale at a fixed frame rate."""
    cmd = ["ffmpeg", "-v", "error", "-nostdin"]
    if start is not None:
        cmd += ["-ss", f"{start:.6f}"]
    cmd += ["-i", path]
    if duration is not None:
        cmd += ["-t", f"{duration:.6f}"]
    cmd += [
        "-map", "0:v:0",
        "-vf", f"fps={fps},scale={size}:{size}:flags=area,format=gray",
        "-f", "rawvideo", "-pix_fmt", "gray", "-",
    ]
    raw = _run(cmd)
    n = len(raw) // (size * size)
    if n == 0:
        raise DecodeError(f"no video frames decoded from {path}")
    return np.frombuffer(raw[: n * size * size], np.uint8).reshape(n, size, size)


def decode_audio(path: str, sr: int = AUDIO_SR, start: float | None = None,
                 duration: float | None = None) -> np.ndarray:
    """Decode to mono float32 in [-1, 1]. Returns silence if there is no track."""
    cmd = ["ffmpeg", "-v", "error", "-nostdin"]
    if start is not None:
        cmd += ["-ss", f"{start:.6f}"]
    cmd += ["-i", path]
    if duration is not None:
        cmd += ["-t", f"{duration:.6f}"]
    cmd += ["-map", "0:a:0?", "-ac", "1", "-ar", str(sr),
            "-f", "f32le", "-acodec", "pcm_f32le", "-"]
    raw = _run(cmd)
    if not raw:
        return np.zeros(0, np.float32)
    return np.frombuffer(raw, np.float32)


# --- mel filterbank ----------------------------------------------------------

def _hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)


def _mel_to_hz(m):
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def mel_filterbank(sr: int = AUDIO_SR, n_fft: int = N_FFT, n_mels: int = N_MELS,
                   fmin: float = FMIN, fmax: float = FMAX) -> np.ndarray:
    """(n_mels, n_fft//2 + 1) triangular, area-normalised filterbank."""
    edges = _mel_to_hz(np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels + 2))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    fb = np.zeros((n_mels, freqs.size), np.float32)
    for i in range(n_mels):
        lo, mid, hi = edges[i], edges[i + 1], edges[i + 2]
        rise = (freqs - lo) / max(mid - lo, 1e-9)
        fall = (hi - freqs) / max(hi - mid, 1e-9)
        fb[i] = np.maximum(0.0, np.minimum(rise, fall))
    fb /= np.maximum(fb.sum(axis=1, keepdims=True), 1e-9)
    return fb


_FB = mel_filterbank()
_WIN = np.hanning(N_FFT).astype(np.float32)


def _frame(x: np.ndarray, n_frames: int) -> np.ndarray:
    """Centred framing: row i is the N_FFT window centred on sample i*HOP."""
    pad = N_FFT // 2
    need = (n_frames - 1) * HOP + 1 + pad
    if x.size < need:
        x = np.pad(x, (0, need - x.size))
    x = np.pad(x, (pad, 0), mode="reflect" if x.size > pad else "constant")
    idx = np.arange(n_frames)[:, None] * HOP + np.arange(N_FFT)[None, :]
    return x[idx]


def audio_features(wav: np.ndarray, n_frames: int) -> np.ndarray:
    """(n_frames, 1 + N_MELS) float32 of dB-scaled energies, normalised to ~[-1, 0].

    Absolute level is preserved on purpose. The old pipeline called
    `librosa.util.normalize`, which rescales every clip to peak 1.0 and so
    destroys the difference between "quiet scene" and "faded to silence" -- the
    exact distinction this model exists to make.
    """
    if wav.size == 0:
        return np.full((n_frames, 1 + N_MELS), DB_FLOOR / -DB_FLOOR * -1.0, np.float32)

    frames = _frame(wav.astype(np.float32), n_frames) * _WIN
    rms = np.sqrt(np.maximum((frames ** 2).mean(axis=1), 1e-20))
    rms_db = 20.0 * np.log10(rms)

    spec = np.abs(np.fft.rfft(frames, n=N_FFT, axis=1)).astype(np.float32) ** 2
    mel = spec @ _FB.T                                   # (n_frames, N_MELS)
    mel_db = 10.0 * np.log10(np.maximum(mel, 1e-20))

    feats = np.concatenate([rms_db[:, None], mel_db], axis=1)
    feats = np.clip(feats, DB_FLOOR, 0.0) / -DB_FLOOR     # -> [-1, 0]
    return feats.astype(np.float32)


def clip_features(path: str, n_frames: int, size: int = IMG_SIZE,
                  start: float | None = None,
                  duration: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Decode one clip to (frames uint8 (T,size,size), audio (T, N_AUDIO))."""
    frames = decode_gray(path, size=size, start=start, duration=duration)
    if frames.shape[0] >= n_frames:
        frames = frames[:n_frames]
    else:
        # Pad short clips by holding the last frame. The old code padded with
        # pure *white* frames, injecting a hard luminance step at the tail that
        # looks nothing like anything in real footage.
        hold = np.repeat(frames[-1:], n_frames - frames.shape[0], axis=0)
        frames = np.concatenate([frames, hold], axis=0)

    wav = decode_audio(path, start=start, duration=duration)
    audio = audio_features(wav, n_frames)
    return np.ascontiguousarray(frames), audio
