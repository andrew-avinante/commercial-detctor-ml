# Graph Report - .  (2026-08-14)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 188 nodes · 379 edges · 9 communities (7 shown, 2 thin omitted)
- Extraction: 99% EXTRACTED · 1% INFERRED · 0% AMBIGUOUS · INFERRED: 3 edges (avg confidence: 0.8)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `e6723662`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- train.py
- build_dataset.py
- features.py
- infer.py
- mark_chapters.py
- metrics.py
- write_png
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

## Communities (9 total, 2 thin omitted)

### Community 0 - "train.py"
Cohesion: 0.08
Nodes (34): baseline_scores(), main(), ndarray, A no-model baseline, to check the network is earning its keep. python -m…, (N, T) score in [0,1]: dark picture AND quiet audio., Shared constants and the training config. Everything downstream keys off these,…, TrainConfig, augment() (+26 more)

### Community 1 - "build_dataset.py"
Cohesion: 0.12
Nodes (31): build_episode(), cut(), decode_episode(), log(), main(), place(), Path, Cut a training cache straight from chapter-marked episodes. # 1. look before… (+23 more)

### Community 2 - "features.py"
Cohesion: 0.12
Nodes (28): audio_features(), clip_features(), decode_audio(), decode_gray(), DecodeError, _frame(), _hz_to_mel(), mel_filterbank() (+20 more)

### Community 3 - "infer.py"
Cohesion: 0.11
Nodes (21): probe_duration(), load_model(), main(), device, ndarray, no_grad, Scan a full episode for commercial-break fades. python -m cdml.infer --model…, Yield (chunk, size, size) uint8 blocks from a single ffmpeg process. (+13 more)

### Community 4 - "mark_chapters.py"
Cohesion: 0.19
Nodes (21): fmt(), anchor_time(), boundaries(), build_chapters(), collect(), compare_to_existing(), _escape(), ffmetadata() (+13 more)

### Community 5 - "metrics.py"
Cohesion: 0.20
Nodes (17): average_precision(), best_f1(), evaluate(), event_scores(), _greedy_match(), hysteresis(), moving_average(), ndarray (+9 more)

### Community 6 - "write_png"
Cohesion: 0.33
Nodes (8): main(), ndarray, Path, Contact sheets for eyeballing a built cache. python -m cdml.review --cache…, 8-bit greyscale PNG from a (h, w) uint8 array., Tile n clips into one image, each row = sampled frames + label/audio bars., sheet(), write_png()

## Knowledge Gaps
- **1 isolated node(s):** `cdml`
  These have ≤1 connection - possible missing edges or undocumented components.
- **2 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `audio_features()` connect `features.py` to `build_dataset.py`, `infer.py`?**
  _High betweenness centrality (0.042) - this node is a cross-community bridge._
- **What connects `cdml` to the rest of the system?**
  _1 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `train.py` be split into smaller, more focused modules?**
  _Cohesion score 0.08350951374207188 - nodes in this community are weakly interconnected._
- **Should `build_dataset.py` be split into smaller, more focused modules?**
  _Cohesion score 0.11742424242424243 - nodes in this community are weakly interconnected._
- **Should `features.py` be split into smaller, more focused modules?**
  _Cohesion score 0.11954022988505747 - nodes in this community are weakly interconnected._
- **Should `infer.py` be split into smaller, more focused modules?**
  _Cohesion score 0.10837438423645321 - nodes in this community are weakly interconnected._