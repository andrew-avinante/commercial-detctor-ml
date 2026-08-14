# commercial-detector-ml

Finds commercial breaks in an episode by detecting the fade to black paired
with the audio fading out.

```bash
pip install -r requirements.txt        # numpy + torch; needs ffmpeg on PATH
# CUDA 12.x:
#   pip install torch --index-url https://download.pytorch.org/whl/cu124

python -m cdml.infer --model models/fade_detector.pt --video episode.mkv
```

```
episode.mkv  00:23:46.186  34228 frames  ->  3 fade(s)
  00:01:01.500 -> 00:01:05.125  (3.62s, p=0.989)
  00:11:45.125 -> 00:11:50.542  (5.42s, p=0.991)
  00:22:44.542 -> 00:22:49.750  (5.21s, p=0.990)
```

All three land on the episode's real chapter marks (63.4s, 708.3s, 1367.4s).
That episode was held out of training.

## Writing the breaks back as chapters

`cdml.mark_chapters` runs the same detector over a whole episode — or a whole
show directory — and remuxes the file with chapter markers on the breaks. The
streams are copied, never re-encoded.

```bash
python -m cdml.mark_chapters "/media/Shows/Example Show" --dry-run
python -m cdml.mark_chapters "/media/Shows/Example Show" --in-place
```

The boundary goes in the *middle* of the fade, which is where a DVD mark sits
(`--anchor start|end` to move it). Fades in the last 20 s are ignored — that
one runs into the end credits and is not a break. A file that already has
chapters is skipped, since in this corpus those chapters are the ground truth;
`--existing replace` overrides it and `--existing compare` scores the detector
against them instead of writing:

```
  s1e11.mkv  00:23:31.520  ->  3 fade(s), 4 chapter(s)
    existing marks : 248.5  806.5  1379.0
    detected       : 61.4  806.2  1378.7
    00:13:26.167  matches mark 00:13:26.489  delta -0.32s
    00:22:58.729  matches mark 00:22:58.978  delta -0.25s
    00:01:01.396  FALSE POSITIVE (nearest mark 00:04:08.515, -187.1s)
    00:04:08.515  MISSED mark
    2/3 marks found, 1 extra detection(s)
```

Without `--in-place` the output is a sibling `<name>.chapters<ext>`;
`--in-place` swaps the source atomically, and only after ffmpeg exits clean
*and* the chapters read back out of the new file.

## Layout

| path | |
|---|---|
| `cdml/` | detection and training package — see [`cdml/README.md`](cdml/README.md) |
| `models/fade_detector.pt` | the shipped detector, 225k parameters, 892 KB |
| `results/` | run reports, split, training history, chapter survey |
| `review/` | contact sheets for eyeballing labels |

## Results

Trained on 2,032 windows from 83 episodes across 4 shows, labelled
automatically from container chapter markers. Split is grouped by episode, so
no episode spans train and test.

| | test AP | frame F1 | event F1 | missed |
|---|---|---|---|---|
| threshold baseline | 0.7792 | 0.7255 | 0.7782 | 9 of 123 |
| **CDML detector** | **0.9823** | **0.9654** | **0.9840** | **0 of 123** |

Generalisation to a show that was never trained on, leave-one-show-out:

| held out | test AP | event F1 | recall |
|---|---|---|---|
| Show A | 0.9709 | 0.9545 | 1.000 |
| Show B | 0.9634 | 0.9067 | 0.872 |
| Show C *(live action)* | 0.9449 | 0.9061 | 0.967 |
| Show D | 0.9313 | 0.7606 | 0.643 |

Show C is the informative row: trained on nothing but cartoons and tested
on a live-action sitcom, it still recovers 96.7% of breaks. The model learned
"sustained black plus sustained silence", not "cartoon".

Show D is the known weak spot, and it is not a threshold problem —
re-tuning the threshold on that show recovers nothing. Its breaks are the
longest in the corpus (68 frames of silence against 33-38 everywhere else), so
a model that never saw one may not recognise it. It scores normally when the
show is in training.

## Building a corpus

Chapter markers in a DVD/BD rip sit on the act breaks, so they are exact
labels for free. Verified across four shows: every internal boundary lands on
a real fade.

```bash
python -m cdml.chapters --shows "/media/Raw/Show A"   # survey, seconds
python -m cdml.build_dataset --shows "/media/Raw/..." --out data/chapters --workers 5
python -m cdml.build_dataset --out data/chapters --assemble --cache cache
python -m cdml.review  --cache cache --out review       # check the labels
python -m cdml.baseline --cache cache                   # no-model reference
python -m cdml.train   --cache cache --out runs/detector
```

`cdml.baseline` keeps the model honest: if it does not clearly beat a tuned
threshold rule, the parameters are not earning their place.
