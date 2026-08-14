"""Scan a full episode for commercial-break fades.

    python -m cdml.infer --model runs/baseline/best.pt --video episode.mp4

The previous scanner had two serious problems:

* It called `librosa.load(VIDEO_PATH, ...)` once per 4-second window. On a
  22-minute episode that is ~330 separate decodes of the same file, and librosa
  decodes from the start each time. Here the episode is decoded exactly once,
  streaming, for both video and audio.
* It merged overlapping windows with `if 1 in local_result[:overlap]` on
  hard `argmax` labels -- an OR over binarised predictions, which only ever adds
  detections and cannot recover from a confident false positive. Here the raw
  probabilities are averaged across every window covering a frame, then
  thresholded once with hysteresis.

"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import torch

from .config import AUDIO_SR, CLIP_FRAMES, FPS, IMG_SIZE, TrainConfig
from .features import audio_features, decode_audio, probe_duration
from .metrics import hysteresis, to_segments
from .model import build_model


def stream_gray(path: str, size: int, fps: int, chunk: int = 512):
    """Yield (chunk, size, size) uint8 blocks from a single ffmpeg process."""
    cmd = [
        "ffmpeg", "-v", "error", "-nostdin", "-i", path, "-map", "0:v:0",
        "-vf", f"fps={fps},scale={size}:{size}:flags=area,format=gray",
        "-f", "rawvideo", "-pix_fmt", "gray", "-",
    ]
    frame_bytes = size * size
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                         bufsize=frame_bytes * chunk)
    try:
        while True:
            raw = p.stdout.read(frame_bytes * chunk)
            if not raw:
                break
            n = len(raw) // frame_bytes
            if n == 0:
                break
            yield np.frombuffer(raw[: n * frame_bytes], np.uint8).reshape(
                n, size, size)
    finally:
        p.stdout.close()
        p.wait()


def load_model(path: str, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = TrainConfig(**ckpt["config"])
    model = build_model(cfg).to(device).eval()
    model.load_state_dict(ckpt["model"])
    return model, cfg, ckpt.get("threshold", 0.5)


@torch.no_grad()
def scan(video: str, model, cfg, device, size: int = IMG_SIZE,
         hop: int = CLIP_FRAMES // 2, batch_size: int = 16,
         amp: bool = True, progress=None) -> np.ndarray:
    """Return a per-frame fade probability for the whole file.

    `progress`, if given, is called as `progress(frames_done, total_frames)`
    after each batch. A full episode on CPU is minutes of silence otherwise,
    which is indistinguishable from a hang to anything supervising this.
    """
    frames = np.concatenate(list(stream_gray(video, size, FPS)), axis=0)
    n = frames.shape[0]

    wav = decode_audio(video, sr=AUDIO_SR)
    audio = audio_features(wav, n)

    if n < CLIP_FRAMES:                       # pad a very short file
        pad = CLIP_FRAMES - n
        frames = np.concatenate([frames, np.repeat(frames[-1:], pad, 0)], 0)
        audio = np.concatenate([audio, np.repeat(audio[-1:], pad, 0)], 0)

    starts = list(range(0, max(1, len(frames) - CLIP_FRAMES + 1), hop))
    if starts[-1] + CLIP_FRAMES < len(frames):
        starts.append(len(frames) - CLIP_FRAMES)

    total = np.zeros(len(frames), np.float64)
    count = np.zeros(len(frames), np.float64)
    use_amp = amp and device.type == "cuda"

    for i in range(0, len(starts), batch_size):
        block = starts[i:i + batch_size]
        fb = np.stack([frames[s:s + CLIP_FRAMES] for s in block])
        ab = np.stack([audio[s:s + CLIP_FRAMES] for s in block])

        ft = torch.from_numpy(fb).to(device).float().div_(255.0)
        at = torch.from_numpy(ab).to(device)
        with torch.autocast("cuda", torch.float16, enabled=use_amp):
            logits = model(ft, at)
        probs = torch.sigmoid(logits.float()).cpu().numpy()

        # Average probabilities over every window covering a frame, rather than
        # letting the last (or any OR-ing) window win.
        for s, p in zip(block, probs):
            total[s:s + CLIP_FRAMES] += p
            count[s:s + CLIP_FRAMES] += 1.0

        if progress is not None:
            progress(min(block[-1] + CLIP_FRAMES, n), n)

    return (total / np.maximum(count, 1.0))[:n]


def to_events(scores: np.ndarray, cfg, fps: int = FPS) -> list[dict]:
    mask = hysteresis(scores, cfg.hyst_high, cfg.hyst_low,
                      cfg.min_event_frames, cfg.smooth_frames)
    events = []
    for a, b in to_segments(mask):
        events.append({
            "start_frame": a, "end_frame": b,
            "start": round(a / fps, 3), "end": round(b / fps, 3),
            "duration": round((b - a) / fps, 3),
            "confidence": round(float(scores[a:b].max()), 4),
        })
    return events


def fmt(seconds: float) -> str:
    m, s = divmod(seconds, 60)
    h, m = divmod(int(m), 60)
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--json", help="write events here instead of stdout only")
    ap.add_argument("--scores", help="save the raw per-frame probabilities (.npy)")
    ap.add_argument("--hop", type=int, default=CLIP_FRAMES // 2)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--high", type=float, default=None)
    ap.add_argument("--low", type=float, default=None)
    ap.add_argument("--min-frames", type=int, default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, thr = load_model(args.model, device)

    # Default the trigger to the threshold tuned on validation, not to 0.5.
    cfg.hyst_high = args.high if args.high is not None else max(thr, cfg.hyst_low)
    if args.low is not None:
        cfg.hyst_low = args.low
    if args.min_frames is not None:
        cfg.min_event_frames = args.min_frames

    duration = probe_duration(args.video)
    scores = scan(args.video, model, cfg, device, hop=args.hop,
                  batch_size=args.batch_size)
    events = to_events(scores, cfg)

    print(f"{Path(args.video).name}  {fmt(duration)}  "
          f"{len(scores)} frames  ->  {len(events)} fade(s)")
    for e in events:
        print(f"  {fmt(e['start'])} -> {fmt(e['end'])}  "
              f"({e['duration']:.2f}s, p={e['confidence']:.3f})")

    if args.scores:
        np.save(args.scores, scores)
    if args.json:
        Path(args.json).write_text(json.dumps(
            {"video": args.video, "duration": duration,
             "fps": FPS, "events": events}, indent=2))


if __name__ == "__main__":
    main()
