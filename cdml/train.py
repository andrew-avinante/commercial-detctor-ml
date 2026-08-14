"""Train the fade detector.

    python -m cdml.train --cache cache --out runs/baseline

Training uses grouped train/validation/test splits, validation average precision
for early stopping, warmup plus cosine learning-rate decay, and `pos_weight` for
class imbalance. The held-out test split is scored once with a threshold chosen
on validation. AMP and a VRAM-resident cache keep epochs short on compatible
GPUs.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .config import TrainConfig
from .dataset import BatchSampler, Cache, augment, fetch, grouped_split, pos_weight_for
from .metrics import evaluate
from .model import build_model, count_parameters


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def lr_lambda(cfg: TrainConfig):
    def fn(epoch: int) -> float:
        if epoch < cfg.warmup_epochs:
            return (epoch + 1) / max(1, cfg.warmup_epochs)
        span = max(1, cfg.epochs - cfg.warmup_epochs)
        t = (epoch - cfg.warmup_epochs) / span
        return 0.5 * (1.0 + math.cos(math.pi * min(t, 1.0)))
    return fn


@torch.no_grad()
def predict(model, cache, indices, batch_size, device, amp):
    model.eval()
    scores, labels = [], []
    for bi in BatchSampler(indices, batch_size, seed=0, shuffle=False):
        frames, audio, y = fetch(cache, bi)
        with torch.autocast("cuda", torch.float16, enabled=amp and device.type == "cuda"):
            logits = model(frames, audio)
        scores.append(torch.sigmoid(logits.float()).cpu().numpy())
        labels.append(y.cpu().numpy())
    return np.concatenate(scores), np.concatenate(labels)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    for f in dataclasses.fields(TrainConfig):
        arg = f"--{f.name.replace('_', '-')}"
        if f.type is bool or isinstance(f.default, bool):
            ap.add_argument(arg, type=lambda s: s.lower() in ("1", "true", "yes"),
                            default=f.default)
        else:
            ap.add_argument(arg, type=type(f.default), default=f.default)
    args = ap.parse_args()
    cfg = TrainConfig(**vars(args))

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)

    cache = Cache(cfg.cache, device)
    groups = cache.groups()

    if cfg.holdout_show:
        shows = [c.get("show", "") for c in cache.clips]
        held = np.array([s == cfg.holdout_show for s in shows])
        if not held.any():
            raise SystemExit(
                f"--holdout-show {cfg.holdout_show!r} matches nothing. "
                f"Known: {sorted({s for s in shows if s})}")
        rest = np.flatnonzero(~held)
        # split the remaining shows into train/val only; test IS the held show
        sub = grouped_split([groups[i] for i in rest],
                            cfg.val_frac / (1.0 - cfg.test_frac), 0.0, cfg.seed)
        tr, va = rest[sub[0]], rest[sub[1]]
        te = np.flatnonzero(held)
        print(f"holdout show: {cfg.holdout_show} -> test only")
    else:
        tr, va, te = grouped_split(groups, cfg.val_frac, cfg.test_frac, cfg.seed)

    if cfg.extra_cache:
        extra = Cache(cfg.extra_cache, device)
        same = (extra.t == cache.t
                and extra.frames.shape[2:] == cache.frames.shape[2:]
                and extra.audio.shape[2] == cache.audio.shape[2])
        if not same:
            raise SystemExit(
                f"--extra-cache geometry {tuple(extra.frames.shape[1:])}/"
                f"{extra.audio.shape[2]} does not match the main cache "
                f"{tuple(cache.frames.shape[1:])}/{cache.audio.shape[2]}")
        off = cache.n
        cache.frames = torch.cat([cache.frames, extra.frames])
        cache.audio = torch.cat([cache.audio, extra.audio])
        cache.labels = torch.cat([cache.labels, extra.labels])
        cache.clips = cache.clips + extra.clips
        cache.n = cache.labels.shape[0]
        tr = np.concatenate([tr, np.arange(off, cache.n)])
        print(f"extra cache : +{extra.n} clips -> train only "
              f"(val/test untouched, so this run is comparable to one without it)")

    n_groups = len(set(groups))
    has_episodes = any(c["episode"] for c in cache.clips)
    print(f"device      : {device}")
    print(f"clips       : {cache.n}  ({n_groups} groups)")
    print(f"split       : train {len(tr)} / val {len(va)} / test {len(te)}")
    print(f"cache in VRAM: {cache.bytes_on_device() / 1e6:.0f} MB")

    if not has_episodes:
        print("\n!! No episode names in the cache, so the split is grouped by\n"
              "   clip only -- clips from the SAME episode are almost certainly\n"
              "   spread across train/val/test. Every score below is therefore\n"
              "   optimistic: the model gets credit for recognising a background\n"
              "   it has already seen. Re-run cdml.preprocess --episodes <map>\n"
              "   before trusting these numbers for a modelling decision.")

    if len(va) == 0 or cache.labels[va].sum() == 0:
        print("\n!! validation split has no positive frames -- "
              "model selection will be meaningless. Re-run with a different "
              "--seed or a larger --val-frac.")

    model = build_model(cfg).to(device)
    print(f"parameters  : {count_parameters(model):,}")

    pw = cfg.pos_weight if cfg.pos_weight > 0 else pos_weight_for(
        cache.labels[tr], cfg.pos_weight_cap)
    print(f"pos_weight  : {pw:.2f}\n")
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pw, device=device))

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(cfg))
    use_amp = cfg.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    gen = torch.Generator(device=device)
    gen.manual_seed(cfg.seed)

    best_ap, best_epoch, history = -1.0, -1, []
    t0 = time.time()

    for epoch in range(cfg.epochs):
        model.train()
        sampler = BatchSampler(tr, cfg.batch_size, seed=cfg.seed + epoch)
        total, nb = 0.0, 0

        for bi in sampler:
            frames, audio, y = fetch(cache, bi)
            frames, audio, y = augment(frames, audio, y, cfg, generator=gen)

            with torch.autocast("cuda", torch.float16, enabled=use_amp):
                loss = criterion(model(frames, audio), y)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
            total += float(loss.detach())
            nb += 1

        sched.step()
        train_loss = total / max(nb, 1)

        vs, vy = predict(model, cache, va, cfg.batch_size, device, use_amp)
        m = evaluate(vs, vy, cfg)
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "lr": sched.get_last_lr()[0], **m})

        flag = ""
        if not math.isnan(m["ap"]) and m["ap"] > best_ap:
            best_ap, best_epoch = m["ap"], epoch
            torch.save({"model": model.state_dict(),
                        "config": dataclasses.asdict(cfg),
                        "threshold": m["threshold"],
                        "epoch": epoch, "val_ap": m["ap"]},
                       out / "best.pt")
            flag = "  *"

        # flush: keeps `tee`/redirected logs live instead of buffering to the end
        print(f"epoch {epoch:3d}  loss {train_loss:.4f}  "
              f"val_ap {m['ap']:.4f}  frame_f1 {m['frame_f1']:.4f}  "
              f"event_f1 {m['event_f1']:.4f}  thr {m['threshold']:.2f}{flag}",
              flush=True)

        if epoch - best_epoch >= cfg.patience:
            print(f"\nearly stop: no val AP improvement in {cfg.patience} epochs")
            break

    print(f"\nbest epoch {best_epoch} (val AP {best_ap:.4f}) "
          f"in {time.time() - t0:.0f}s")

    # --- final report on the untouched test split ---------------------------
    ckpt = torch.load(out / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    thr = ckpt["threshold"]

    report = {"best_epoch": best_epoch, "val_ap": best_ap, "threshold": thr}
    for name, idx in (("val", va), ("test", te)):
        if len(idx) == 0:
            continue
        s, y = predict(model, cache, idx, cfg.batch_size, device, use_amp)
        report[name] = evaluate(s, y, cfg, threshold=thr)
        r = report[name]
        print(f"\n{name}: AP {r['ap']:.4f}  frame_f1 {r['frame_f1']:.4f} "
              f"(P {r['frame_precision']:.3f} R {r['frame_recall']:.3f})")
        print(f"      events  f1 {r['event_f1']:.4f} "
              f"(P {r['event_precision']:.3f} R {r['event_recall']:.3f})  "
              f"tp {r['event_tp']} fp {r['event_fp']} fn {r['event_fn']}")

    (out / "history.json").write_text(json.dumps(history, indent=2))
    (out / "report.json").write_text(json.dumps(report, indent=2))
    (out / "split.json").write_text(json.dumps(
        {"train": tr.tolist(), "val": va.tolist(), "test": te.tolist()}))
    print(f"\nwrote {out}/best.pt, report.json, history.json, split.json")


if __name__ == "__main__":
    main()
