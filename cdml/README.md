# cdml — PyTorch commercial-break fade detector

The `cdml` package is the supported commercial-break detection pipeline.

```bash
pip install -r requirements.txt          # numpy + torch; needs ffmpeg on PATH
# CUDA 12.x on the 3060:
#   pip install torch --index-url https://download.pytorch.org/whl/cu124
```

**Building a dataset from chapter markers (preferred).** Chapter marks in
DVD/BD rips sit on the act breaks, so they are free, exact labels:

```bash
# 1. survey -- seconds, decodes nothing
python -m cdml.chapters --shows "/media/Raw/Thundarr the Barbarian" ...

# 2. cut clips -- hours, resumable, safe to ctrl-C and rerun
python -m cdml.build_dataset --shows "/media/Raw/Thundarr the Barbarian" ... \
                             --out data/chapters --workers 5

# 3. collect shards into a cache
python -m cdml.build_dataset --out data/chapters --assemble --cache cache_v2
```

Then:

```bash
python -m cdml.baseline  --cache cache_v2                 # no-model reference
python -m cdml.train     --cache cache_v2 --out runs/v1
python -m cdml.infer     --model runs/v1/best.pt --video episode.mp4 --json out.json
```

**Writing the breaks back into the file.** `cdml.mark_chapters` is `cdml.infer`
plus a stream-copy remux: fades become chapter markers, nothing is re-encoded.
It takes files or show directories, skips files that already have chapters, and
can score itself against them with `--existing compare`.

```bash
python -m cdml.mark_chapters "/media/Shows/Courage the Cowardly Dog" --dry-run
python -m cdml.mark_chapters "/media/Shows/Courage the Cowardly Dog" --in-place
```

---

## The finding that should drive the next round of work

The previous corpus builder mined training clips from ffmpeg `blackdetect d=1` hits, so **a
clip only enters the dataset if ffmpeg already found ≥1 second of black in
it.** 289 of the 296 cached clips contain fully-dark frames. Both classes are
genuine fades; the negatives are scene transitions, not dark scenes.

So the model is never asked "is this a fade?" — it is asked "is this
*particular* fade an act break?" Measured on the 296-clip cache, clip-level,
AUC 0.5 = coin flip:

| discriminator | AUC |
|---|---|
| deepest silence (min RMS dBFS) | 0.553 |
| frames below −65 dB | 0.568 |
| longest silent run | 0.545 |
| longest black run | 0.570 |
| deepest fade ramp | 0.544 |
| energy burst before silence ("the sting") | 0.388 |
| **all 8 combined, logistic regression, 5-fold** | **0.518 ± 0.033** |

Nothing separates them, because the deciding information is not inside the
8-second window. Three artefacts compound it:

* **No easy negatives.** 98% of training clips contain a fade; at scan time
  over 99% of windows do not. The model never sees ordinary footage.
* **The black is always centred**, at 0.49 ± 0.05 of the clip in *both*
  classes, because the window was centred on the black run.
  Nothing at inference looks like that.
* **`d=1` clamps black duration**, destroying the one feature that really does
  differ between an act break and a scene transition.

Ablations agree: a video-only model found **zero** events (best epoch 0), and
the full network beat a no-model threshold rule by too little to matter.

**`cdml.chapters` + `cdml.build_dataset` are the answer to this**, and are now
the preferred way to build a corpus — see the quick start above.

---

## What changed, and why

### 1. The audio features carried no loudness (the big one)

```python
audio, sr = librosa.load(audio_path, sr=24)      # previous preprocessing
audio = librosa.util.normalize(audio)            # previous preprocessing
```

`sr=24` resamples the **waveform** to 24 Hz. Anti-alias filtering discards
everything above 12 Hz, leaving a zero-centred signal with no relationship to
loudness — the cached `processed_audio_000.npy` runs `[0.441, 0.321, 0.05,
0.139, -0.023, ...]`. Then `librosa.util.normalize` rescales every clip to peak
1.0, erasing the difference between "quiet scene" and "faded to silence".

So the one channel that actually distinguishes a commercial break was noise.

Now: audio stays at 24 kHz and is reduced to per-frame **energy** — log-RMS in
dBFS plus 16 mel bands, also in dB, on an absolute scale. 24000/24 = 1000
samples per video frame exactly, so audio rows align with video frames with no
resampling drift. On the same clip the fade is now plainly visible:

```
luminance  0.627  0.362  0.341  0.188  0.003  0.000  0.000  0.015  0.388
audio dB  -26.9  -24.8  -25.9  -25.1  -38.4  -49.3  -80.0  -27.9  -25.4
```

dB scale pays twice: a fade-out is roughly linear in dB, and a gain change
becomes a simple additive offset — which is what makes the audio augmentation a
one-line add rather than a resynthesis.

### 2. The model spent 97% of its parameters on a dense pixel projection

The old video branch fed a flattened 16384-vector into an LSTM. That first layer
alone is `4*(16384+128+1)*128 ≈ 8.4M` parameters, out of ~8.5M total — on 296
training clips. It also discards spatial structure entirely.

Now: a small shared-weight 2-D CNN encodes each frame, and six explicit
luminance statistics (mean, spread, peak, two dark-pixel fractions, inter-frame
motion) are concatenated to it. A fade *is* a luminance trajectory; there is no
reason to make a recurrent net rediscover `frame.mean()` through a bottleneck.

**~225k parameters, ~1 MB, against ~8.5M and 109 MB.**

The scalars are computed on GPU from whatever pixels arrive, so pixel
augmentation can never desynchronise them from a cached copy.

### 3. Validation was measuring memorisation

- The previous pipeline split randomly over clips while it wrote four
  augmented **copies** of every clip to disk. Copies of the same clip landed on
  both sides of the split.
- `EarlyStopping(monitor='loss')` watched *training* loss, so
  `restore_best_weights` restored the most overfit epoch.
- `accuracy` on a problem with 7.8% positive frames: predicting "never" scores
  92.2%.

Now: grouped splitting (by episode where the filename records it, else by clip,
so augmented views can't span the split), augmentation generated on the fly
instead of duplicated to disk, early stopping on validation average precision,
and a held-out test split scored once at the end with the threshold tuned on
validation. There was no test split at all before.

### 4. Silent data-misalignment risk

The previous pipeline kept video, audio, and labels in separate filename-globbed
arrays.

Three independent globs, zipped positionally. `glob` does not guarantee ordering,
and the augmented-copy suffixes (`_augmented_0`) interleave differently across
the three patterns. Any disagreement trains video against another clip's labels,
with no error. Now clips are paired by parsed clip number.

Related, in the previous batch generator:
`to_categorical(label_batch)` without `num_classes` infers the class count from
the batch maximum — an all-negative batch yields shape `(B,192,1)` against a
2-unit softmax. The model now emits one logit per frame with BCE, so the failure
mode does not exist.

### 5. 5.0 GB of cache for 250 MB of data

Frames were stored as `float64` at 128×128 flattened: 25 MB per 8-second clip,
and the generator re-read that from disk once per clip per epoch. The run was
disk-bound.

Now frames are `uint8` at 64×64 — **237 MB total, measured.** The entire dataset
fits in VRAM on the 3060, so there is no DataLoader and no per-sample I/O.

### 6. Augmentation bugs

```python
noise = np.random.normal(mean, stddev, frame.shape).astype('uint8')   # old
```
Every negative sample wraps to ~246–255, and `cv2.add` saturates it to white.
That is salt noise, not Gaussian noise. It also flipped **vertically**
(`orientation` 0), producing upside-down footage that never occurs at inference,
and pitch-shifted the 24 Hz signal, which is not a meaningful operation.

Now: horizontal flip only, noise added in float and clamped, gamma jitter,
temporal jitter (frames/audio/labels shifted together, edge-held rather than
wrapped so black never teleports to the head of a clip), and audio gain as an
additive dB offset.

### 7. Learning rate died two-fifths of the way in

`lr * 0.1` every 10 epochs reaches 1e-6 by epoch 30 of 50 — the last 20 epochs
did nothing. Replaced with warmup + cosine decay.

### 8. Inference

The previous scanner called `librosa.load(VIDEO_PATH, ...)` **once per 4-second window** —
roughly 330 separate decodes of the same file for a 22-minute episode, each
starting from the beginning. `cdml.infer` decodes once, streaming.

It also merged overlapping windows with `if 1 in local_result[:overlap]` over
hard `argmax` labels — an OR over binarised predictions, which can only ever add
detections. Now raw probabilities are averaged across every window covering a
frame, then thresholded once with hysteresis (smooth → dual threshold → minimum
duration), which yields stable segments instead of the flicker a per-frame
`argmax` produces.

---

## Evaluate honestly

`python -m cdml.baseline` scores a no-model "dark AND quiet" rule with the same
threshold tuning and event matching as the network. **If the network does not
clearly beat it, the 225k parameters are not earning their place** and the
threshold rule is the honest thing to ship.

`--use-video false` / `--use-audio false` ablate a modality. Given the table at
the top, run these — they tell you where the signal actually is.

Metrics reported: average precision (threshold-free), per-frame F1 with the
threshold that produced it, and event-level precision/recall via greedy IoU≥0.5
matching. The event numbers are the ones that reflect the product: whether each
real break is found once, at roughly the right time.

---
