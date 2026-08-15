# CDML fade detector

`fade_detector.pt` is the released checkpoint for the CDML commercial-break
boundary detector.

The published model repository is
[andrew-avinante/cdml-fade-detector](https://huggingface.co/andrew-avinante/cdml-fade-detector).

## License

The checkpoint is licensed under [Apache-2.0](../LICENSE), the same license as
the repository code and documentation. This grant applies to the checkpoint;
it does not grant any rights in the underlying training media, which is not
included or distributed.

## Purpose and output

The model detects commercial-break boundaries represented by a paired
fade-to-black and fade-to-silence signal. It accepts grayscale video-frame and
audio features and produces a fade score for each frame. CDML then combines
those scores into timestamped events or chapter markers.

The checkpoint does not generate, reproduce, or distribute video, audio, or
images from the training corpus.

## Usage

Install the project dependencies, including `ffmpeg`, `ffprobe`, and PyTorch,
then run:

```bash
python -m cdml.infer --model models/fade_detector.pt --video episode.mkv
```

## Training-data boundary

The checkpoint was trained on privately held commercial episodic media with
existing commercial-break chapter markers. Source media, extracted clips,
training caches, contact sheets, source paths, and source identifiers are not
part of this release.

## Limitations

Performance may be weaker for breaks without the paired black-and-silent
transition, unusually long breaks, end credits, unfamiliar video formats, or
unfamiliar audio mixes. The published aggregate and leave-one-show-out metrics
are in [`../results/`](../results/).

## Integrity and reproducibility

- SHA-256: `4af08510774e7eae496a4e413bba54f915a1c13fe912fee70857a862c5def300`
- Architecture: grayscale CNN and luminance features plus audio-energy features,
  followed by a bidirectional GRU.
- Training and inference instructions: [`../README.md`](../README.md)
