# legacy — the Keras/TensorFlow pipeline, superseded by `cdml/`

Kept for reference and still runnable from here. Nothing in `cdml/` imports
any of it.

Two of these actively caused the problems the rewrite exists to fix, and
should not be used to build a new corpus:

**`get_black.py`** mined every training clip from an ffmpeg `blackdetect d=1`
hit. A clip could only enter the dataset if ffmpeg had already found ≥1 second
of black in it, so 289 of 296 clips contained a fade and *both* classes looked
alike — across eight hand-built features (black depth and duration, silence
depth and duration, fade slope) a logistic regression separated real breaks
from other fades at AUC 0.518, a coin flip. It also produced no ordinary
footage at all, while over 99% of windows at scan time are exactly that. And
`generate_random_clip` centred the window on the black run, pinning the fade
at 0.49 ± 0.05 of every clip in both classes — a cue that exists in training
and nowhere else.

**`rename.py`** rewrote `train_00007_A Creepy Tangle... s03e02.mp4` to
`train_00007.mp4`, discarding the only record of which episode a clip came
from. Without it the train/val/test split cannot be grouped, so clips from one
episode land on both sides and every score is optimistic.

Use `cdml.chapters` + `cdml.build_dataset` instead: chapter markers in a rip
sit on the act breaks, which makes them exact labels for free, and the builder
samples ordinary footage and non-break fades as separate negative classes.
Measured on the corpus that produced: the same separability test scores AUC
0.875 ± 0.023 under episode-grouped cross-validation.

The rest are inference and utility scripts against `model.h5`:

| file | note |
|---|---|
| `train.py` | Keras trainer. ~8.5M params, 97% of them in one dense pixel projection; early stopping watched *training* loss; no test split. |
| `preprocess.py` | Wrote `librosa.load(..., sr=24)` audio — 24 Hz resampling of the waveform, which destroys the loudness envelope the detector depends on. |
| `scan.py` | Called `librosa.load` once per 4-second window (~330 decodes of the same file per episode) and merged overlaps with a logical OR over `argmax` labels. |
| `run.py`, `search.py` | Inference wrappers. Pass `custom_objects={'KerasLayer': hub.KerasLayer}` for a hub layer the model does not contain. |

See `cdml/README.md` for the full before/after.
