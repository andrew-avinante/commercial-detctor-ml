# Graph Report - commercial-detector-ml  (2026-08-13)

## Corpus Check
- 26 files · ~96,193 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 201 nodes · 391 edges · 10 communities (8 shown, 2 thin omitted)
- Extraction: 99% EXTRACTED · 1% INFERRED · 0% AMBIGUOUS · INFERRED: 3 edges (avg confidence: 0.8)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `2a5eaa02`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- train.py
- mark_chapters.py
- build_dataset.py
- features.py
- infer.py
- AudioEncoder
- write_png
- CDML
- __init__.py
- cdml

## God Nodes (most connected - your core abstractions)
1. `process()` - 15 edges
2. `main()` - 14 edges
3. `Cache` - 11 edges
4. `evaluate()` - 11 edges
5. `scan()` - 10 edges
6. `audio_features()` - 9 edges
7. `build_episode()` - 8 edges
8. `label_episode()` - 8 edges
9. `TrainConfig` - 8 edges
10. `BatchSampler` - 8 edges

## Surprising Connections (you probably didn't know these)
- `main()` --calls--> `evaluate()`  [EXTRACTED]
  cdml/baseline.py → cdml/metrics.py
- `decode_episode()` --calls--> `DecodeError`  [EXTRACTED]
  cdml/build_dataset.py → cdml/features.py
- `cut()` --calls--> `audio_features()`  [EXTRACTED]
  cdml/build_dataset.py → cdml/features.py
- `collect()` --calls--> `find_videos()`  [EXTRACTED]
  cdml/mark_chapters.py → cdml/chapters.py
- `load_model()` --calls--> `TrainConfig`  [EXTRACTED]
  cdml/infer.py → cdml/config.py

## Import Cycles
- None detected.

## Communities (10 total, 2 thin omitted)

### Community 0 - "train.py"
Cohesion: 0.08
Nodes (38): baseline_scores(), main(), ndarray, A no-model baseline, to check the network is earning its keep. python -m…, (N, T) score in [0,1]: dark picture AND quiet audio., Shared constants and the training config. Everything downstream keys off these,…, TrainConfig, augment() (+30 more)

### Community 1 - "mark_chapters.py"
Cohesion: 0.19
Nodes (20): anchor_time(), boundaries(), build_chapters(), collect(), compare_to_existing(), _escape(), ffmetadata(), main() (+12 more)

### Community 2 - "build_dataset.py"
Cohesion: 0.12
Nodes (31): build_episode(), cut(), decode_episode(), log(), main(), place(), Path, Cut a training cache straight from chapter-marked episodes. # 1. look before… (+23 more)

### Community 3 - "features.py"
Cohesion: 0.12
Nodes (28): audio_features(), clip_features(), decode_audio(), decode_gray(), DecodeError, _frame(), _hz_to_mel(), mel_filterbank() (+20 more)

### Community 4 - "infer.py"
Cohesion: 0.11
Nodes (30): probe_duration(), fmt(), load_model(), main(), device, ndarray, no_grad, Scan a full episode for commercial-break fades. python -m cdml.infer --model… (+22 more)

### Community 5 - "AudioEncoder"
Cohesion: 0.27
Nodes (5): AudioEncoder, FrameEncoder, Tensor, 64x64 -> embedding, applied with shared weights to every frame., Widen the 17 dB-scaled energy channels before fusion. Fusing 17 audio…

### Community 6 - "write_png"
Cohesion: 0.33
Nodes (8): main(), ndarray, Path, Contact sheets for eyeballing a built cache. python -m cdml.review --cache…, 8-bit greyscale PNG from a (h, w) uint8 array., Tile n clips into one image, each row = sampled frames + label/audio bars., sheet(), write_png()

### Community 7 - "CDML"
Cohesion: 0.15
Nodes (11): Build a dataset, CDML, Detect breaks, Install, Model and evaluation, Write chapters, Building a corpus, commercial-detector-ml (+3 more)

## Knowledge Gaps
- **10 isolated node(s):** `cdml`, `Writing the breaks back as chapters`, `Layout`, `Results`, `Building a corpus` (+5 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **2 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `audio_features()` connect `features.py` to `build_dataset.py`, `infer.py`?**
  _High betweenness centrality (0.037) - this node is a cross-community bridge._
- **What connects `cdml`, `Writing the breaks back as chapters`, `Layout` to the rest of the system?**
  _10 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `train.py` be split into smaller, more focused modules?**
  _Cohesion score 0.07591836734693877 - nodes in this community are weakly interconnected._
- **Should `build_dataset.py` be split into smaller, more focused modules?**
  _Cohesion score 0.11742424242424243 - nodes in this community are weakly interconnected._
- **Should `features.py` be split into smaller, more focused modules?**
  _Cohesion score 0.11954022988505747 - nodes in this community are weakly interconnected._
- **Should `infer.py` be split into smaller, more focused modules?**
  _Cohesion score 0.11491935483870967 - nodes in this community are weakly interconnected._