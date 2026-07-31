# commercial-detector-ml

Finds commercial breaks in an episode by detecting the fade to black paired
with the audio fading out.

```bash
pip install -r requirements.txt        # numpy + torch; needs ffmpeg on PATH
# CUDA 12.x:
#   pip install torch --index-url https://download.pytorch.org/whl/cu124

python -m cdml.infer --model models/fade_detector_v2.pt --video episode.mkv
```

```
episode.mkv  00:23:46.186  34228 frames  ->  3 fade(s)
  00:01:01.500 -> 00:01:05.125  (3.62s, p=0.989)
  00:11:45.125 -> 00:11:50.542  (5.42s, p=0.991)
  00:22:44.542 -> 00:22:49.750  (5.21s, p=0.990)
```

All three land on the episode's real chapter marks (63.4s, 708.3s, 1367.4s).
That episode was held out of training.

## Layout

| path | |
|---|---|
| `cdml/` | the PyTorch pipeline — see [`cdml/README.md`](cdml/README.md) |
| `models/fade_detector_v2.pt` | the shipped detector, 225k parameters, 892 KB |
| `results/` | run reports, split, training history, chapter survey |
| `review/` | contact sheets for eyeballing labels |
| `legacy/` | the superseded Keras pipeline — see [`legacy/README.md`](legacy/README.md) |

## Results

Trained on 2,032 windows from 83 episodes across 4 shows, labelled
automatically from container chapter markers. Split is grouped by episode, so
no episode spans train and test.

| | test AP | frame F1 | event F1 | missed |
|---|---|---|---|---|
| previous corpus, same model | 0.3633 | 0.4471 | 0.4681 | 3 of 14 |
| this corpus, **no model** (threshold rule) | 0.7792 | 0.7255 | 0.7782 | 9 of 123 |
| **this corpus, model** | **0.9823** | **0.9654** | **0.9840** | **0 of 123** |

Generalisation to a show that was never trained on, leave-one-show-out:

| held out | test AP | event F1 | recall |
|---|---|---|---|
| Thundarr the Barbarian | 0.9709 | 0.9545 | 1.000 |
| The 13 Ghost of Scooby-Doo | 0.9634 | 0.9067 | 0.872 |
| The Brady Bunch *(live action)* | 0.9449 | 0.9061 | 0.967 |
| Scooby's All-Star Laff-A-Lympics | 0.9313 | 0.7606 | 0.643 |

Brady Bunch is the informative row: trained on nothing but cartoons and tested
on a live-action sitcom, it still recovers 96.7% of breaks. The model learned
"sustained black plus sustained silence", not "cartoon".

Laff-A-Lympics is the known weak spot, and it is not a threshold problem —
re-tuning the threshold on that show recovers nothing. Its breaks are the
longest in the corpus (68 frames of silence against 33-38 everywhere else), so
a model that never saw one may not recognise it. It scores normally when the
show is in training.

## Building a corpus

Chapter markers in a DVD/BD rip sit on the act breaks, so they are exact
labels for free. Verified across four shows: every internal boundary lands on
a real fade.

```bash
python -m cdml.chapters --shows "/media/Raw/Thundarr the Barbarian"   # survey, seconds
python -m cdml.build_dataset --shows "/media/Raw/..." --out data/chapters --workers 5
python -m cdml.build_dataset --out data/chapters --assemble --cache cache_v2
python -m cdml.review  --cache cache_v2 --out review    # check the labels
python -m cdml.baseline --cache cache_v2                # no-model reference
python -m cdml.train   --cache cache_v2 --out runs/v2
```

`cdml.baseline` exists to keep the model honest: if it does not clearly beat a
tuned threshold rule, the parameters are not earning their place. On the
previous corpus it did not, which is what prompted the rewrite.
