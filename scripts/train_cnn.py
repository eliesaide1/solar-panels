"""Train the patch CNN on a capture with verified (or transferred) labels.

Patches are sampled from inside labelled panels (positive) and from elsewhere
in the same tiles (negative). Negatives are drawn from the same imagery rather
than a generic pool so the network learns to separate panels from *these*
rooftops, tarmac and vegetation.

Evaluation holds out whole tiles. Splitting by patch would leak: neighbouring
patches from one array are near-identical, so a random split scores its own
training data.

    python scripts/train_cnn.py --capture jbeil-mb
"""

import argparse
import json

import _bootstrap  # noqa: F401

import cv2
import numpy as np
import torch
import torch.nn as nn

from solarmap.config import Config
from solarmap.model.patchnet import PATCH, PatchNet, normalise


def sample_tile(img, panels, rng, n_pos=120, n_neg=240):
    """Return (patches, labels) sampled from one tile."""
    H, W = img.shape[:2]
    half = PATCH // 2
    occ = np.zeros((H, W), bool)
    for b in panels:
        occ[max(0, b["y1"]):b["y2"], max(0, b["x1"]):b["x2"]] = True

    out_x, out_y = [], []

    # Positives: centred inside a panel, jittered so the net does not learn a
    # fixed alignment.
    if panels:
        for _ in range(n_pos):
            b = panels[rng.integers(0, len(panels))]
            cx = rng.integers(b["x1"], max(b["x1"] + 1, b["x2"]))
            cy = rng.integers(b["y1"], max(b["y1"] + 1, b["y2"]))
            x, y = int(np.clip(cx - half, 0, W - PATCH)), int(np.clip(cy - half, 0, H - PATCH))
            patch = img[y:y + PATCH, x:x + PATCH]
            # Require the crop to be mostly panel, or it teaches the wrong thing.
            if occ[y:y + PATCH, x:x + PATCH].mean() < 0.6:
                continue
            out_x.append(patch); out_y.append(1)

    # Negatives: anywhere with no panel in the crop at all.
    tries = 0
    got = 0
    while got < n_neg and tries < n_neg * 12:
        tries += 1
        x = int(rng.integers(0, max(1, W - PATCH)))
        y = int(rng.integers(0, max(1, H - PATCH)))
        if occ[y:y + PATCH, x:x + PATCH].any():
            continue
        out_x.append(img[y:y + PATCH, x:x + PATCH]); out_y.append(0)
        got += 1

    return out_x, out_y


def augment(patch, rng):
    k = rng.integers(0, 4)
    if k:
        patch = np.rot90(patch, k)
    if rng.random() < 0.5:
        patch = patch[:, ::-1]
    if rng.random() < 0.5:
        patch = patch[::-1]
    # Overhead imagery varies in exposure between captures and sun angles.
    if rng.random() < 0.7:
        patch = np.clip(patch.astype(np.float32) * rng.uniform(0.75, 1.3)
                        + rng.uniform(-18, 18), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(patch)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--capture", required=True)
    ap.add_argument("--labels", default="labels_transferred.json")
    ap.add_argument("--out", default="models/patchnet.pt")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--val-frac", type=float, default=0.25)
    ap.add_argument("--config")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    cap = cfg.path("captures") / args.capture
    labels = json.loads((cap / args.labels).read_text(encoding="utf-8"))
    manifest = json.loads((cap / "manifest.json").read_text(encoding="utf-8"))
    rng = np.random.default_rng(0)

    tiles_with = [t for t in manifest["tiles"] if labels["tiles"].get(t["tile_id"])]
    rng.shuffle(tiles_with)
    n_val = max(1, int(len(tiles_with) * args.val_frac))
    val_ids = {t["tile_id"] for t in tiles_with[:n_val]}
    print(f"{len(tiles_with)} labelled tiles -> {len(val_ids)} held out for validation")

    Xtr, ytr, Xva, yva = [], [], [], []
    for t in manifest["tiles"]:
        tid = t["tile_id"]
        panels = labels["tiles"].get(tid, [])
        if not panels:
            continue
        img = cv2.cvtColor(cv2.imread(str(cap / "tiles" / f"{tid}.jpg")), cv2.COLOR_BGR2RGB)
        xs, ys = sample_tile(img, panels, rng)
        if tid in val_ids:
            Xva += xs; yva += ys
        else:
            Xtr += xs; ytr += ys

    print(f"train patches {len(Xtr)} ({sum(ytr)} positive) | "
          f"val patches {len(Xva)} ({sum(yva)} positive)")
    if sum(ytr) < 50:
        raise SystemExit("Too few positive patches; check label alignment.")

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = PatchNet().to(dev)
    n_par = sum(p.numel() for p in net.parameters())
    print(f"PatchNet {n_par/1e3:.0f}k parameters on {dev}")

    pos_w = torch.tensor([(len(ytr) - sum(ytr)) / max(sum(ytr), 1)], device=dev)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    yva_t = torch.tensor(yva, dtype=torch.float32)
    Xva_t = torch.stack([normalise(p) for p in Xva])

    best = -1.0
    for ep in range(1, args.epochs + 1):
        net.train()
        order = rng.permutation(len(Xtr))
        tot = 0.0
        for i in range(0, len(order), args.batch):
            idx = order[i:i + args.batch]
            xb = torch.stack([normalise(augment(Xtr[j], rng)) for j in idx]).to(dev)
            yb = torch.tensor([ytr[j] for j in idx], dtype=torch.float32, device=dev)
            opt.zero_grad(set_to_none=True)
            loss = crit(net.classify(xb), yb)
            loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        sched.step()

        net.eval()
        with torch.no_grad():
            logits = torch.cat([net.classify(Xva_t[i:i + 128].to(dev)).cpu()
                                for i in range(0, len(Xva_t), 128)])
        prob = torch.sigmoid(logits)
        pred = (prob > 0.5).float()
        tp = float((pred * yva_t).sum()); fp = float((pred * (1 - yva_t)).sum())
        fn = float(((1 - pred) * yva_t).sum())
        P = tp / max(tp + fp, 1); R = tp / max(tp + fn, 1)
        f1 = 2 * P * R / max(P + R, 1e-9)
        print(f"epoch {ep:3d}  loss {tot/len(order):.4f}  val P={P:.3f} R={R:.3f} F1={f1:.3f}")
        if f1 > best:
            best = f1
            out = cfg.path("checkpoints").parent / args.out
            torch.save({"state_dict": net.state_dict(), "patch": PATCH,
                        "gsd": float(manifest["tiles"][0]["gsd_m"]), "val_f1": f1}, out)

    print(f"\nbest val F1 {best:.3f} -> {cfg.path('checkpoints').parent / args.out}")


if __name__ == "__main__":
    main()
