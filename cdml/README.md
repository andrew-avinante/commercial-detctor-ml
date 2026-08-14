# CDML

CDML detects commercial breaks from the paired fade-to-black and fade-to-silence
signals in episode video. It provides dataset construction, model training,
full-episode inference, and chapter writing.

## Install

```bash
pip install -r requirements.txt
# CUDA 12.x on the 3060:
#   pip install torch --index-url https://download.pytorch.org/whl/cu124
```

`ffmpeg` and `ffprobe` must be available on `PATH`.

## Detect breaks

```bash
python -m cdml.infer --model models/fade_detector.pt --video episode.mkv
```

Inference decodes an episode once, combines overlapping-window probabilities,
then applies smoothing, hysteresis, and a minimum event duration to produce
stable fade segments.

## Write chapters

`cdml.mark_chapters` detects breaks and writes their midpoint as chapter marks
without re-encoding media streams.

```bash
python -m cdml.mark_chapters "/media/Shows/Example Show" --dry-run
python -m cdml.mark_chapters "/media/Shows/Example Show" --in-place
```

Files that already contain chapters are skipped by default. Use
`--existing replace` to replace them or `--existing compare` to score detected
breaks against existing marks.

## Build a dataset

Chapter markers in DVD and Blu-ray rips provide labels for internal act breaks.
The builder creates positive break windows, hard-negative non-break fades, and
easy-negative ordinary footage.

```bash
# Survey chapters without decoding video.
python -m cdml.chapters --shows "/media/Raw/Show A"

# Create resumable episode shards.
python -m cdml.build_dataset --shows "/media/Raw/Show A" \
    --out data/chapters --workers 5

# Assemble shards into a training cache.
python -m cdml.build_dataset --out data/chapters --assemble --cache cache

# Inspect labels, establish a threshold baseline, and train.
python -m cdml.review --cache cache --out review
python -m cdml.baseline --cache cache
python -m cdml.train --cache cache --out runs/detector
```

## Model and evaluation

The detector uses a shared 2-D CNN for grayscale video, luminance statistics,
and log-RMS plus mel-band audio energies. A bidirectional GRU produces one fade
probability per frame.

The shipped evaluation uses 2,032 windows from 83 episodes across four shows,
with splits grouped by episode. The detector achieved test AP `0.9823`, frame
F1 `0.9654`, and event F1 `0.9840`; the threshold baseline achieved event F1
`0.7782` on the same evaluation.

For an honest comparison, tune operating thresholds on validation data and
evaluate the held-out test split once. `cdml.baseline` and the
`--use-video` / `--use-audio` switches support that workflow.
