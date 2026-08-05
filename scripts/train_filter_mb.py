"""Train the box classifier on a capture using transferred labels.

Proposals are generated on the target imagery and labelled automatically by
overlap with known panel rectangles, so no manual review is needed when the
same ground has already been labelled at another resolution.

Positive/negative is decided by IoU rather than one-sided containment: a
proposal that merely sits inside a large panel is not a good detection box, and
neither is one that swallows a panel along with half a roof.

    python scripts/train_filter_mb.py --capture jbeil-mb
"""

import argparse
import json
import pickle

import _bootstrap  # noqa: F401

import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import precision_recall_curve

from solarmap.config import Config
from solarmap.infer.cvfilter import box_features, propose, texture_map


def iou(a, b) -> float:
    ix = max(0, min(a[2], b["x2"]) - max(a[0], b["x1"]))
    iy = max(0, min(a[3], b["y2"]) - max(a[1], b["y1"]))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b["x2"] - b["x1"]) * (b["y2"] - b["y1"]) - inter
    return inter / ua if ua > 0 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--capture", required=True)
    ap.add_argument("--labels", default="labels_transferred.json")
    ap.add_argument("--out", default="models/box_filter_mb.pkl")
    ap.add_argument("--pos-iou", type=float, default=0.25)
    ap.add_argument("--neg-iou", type=float, default=0.05)
    ap.add_argument("--config")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    cap = cfg.path("captures") / args.capture
    labels = json.loads((cap / args.labels).read_text(encoding="utf-8"))
    manifest = json.loads((cap / "manifest.json").read_text(encoding="utf-8"))

    X, y, groups = [], [], []
    n_prop = n_amb = 0
    prop_recall_hit = prop_recall_tot = 0

    for t in manifest["tiles"]:
        tid = t["tile_id"]
        panels = labels["tiles"].get(tid, [])
        img = cv2.imread(str(cap / "tiles" / f"{tid}.jpg"), cv2.IMREAD_COLOR)
        if img is None:
            continue
        gsd = float(t["gsd_m"])
        props = propose(img, gsd)
        n_prop += len(props)
        tex = texture_map(img, gsd)

        for g in panels:
            box = (g["x1"], g["y1"], g["x2"], g["y2"])
            prop_recall_tot += 1
            prop_recall_hit += any(iou(box, p) >= args.pos_iou for p in props)

        for p in props:
            box = (p["x1"], p["y1"], p["x2"], p["y2"])
            best = max((iou((g["x1"], g["y1"], g["x2"], g["y2"]), p) for g in panels),
                       default=0.0)
            if args.neg_iou < best < args.pos_iou:
                n_amb += 1
                continue
            f = box_features(img, tex, p, gsd)
            if f is None:
                continue
            X.append(f); y.append(1 if best >= args.pos_iou else 0); groups.append(tid)

    X = np.array(X, np.float32); y = np.array(y); groups = np.array(groups)
    print(f"proposals: {n_prop} | training boxes {len(X)} "
          f"({y.sum()} positive, {len(y)-y.sum()} negative, {n_amb} ambiguous dropped)")
    print(f"proposal recall (IoU>={args.pos_iou}): "
          f"{prop_recall_hit}/{prop_recall_tot} = {prop_recall_hit/max(prop_recall_tot,1):.1%}")
    if y.sum() < 20:
        raise SystemExit("Too few positives to train; check label alignment.")

    params = dict(n_estimators=500, min_samples_leaf=2,
                  class_weight="balanced_subsample", n_jobs=-1, random_state=0)

    oof = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        oof[te] = RandomForestClassifier(**params).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]

    prec, rec, thr = precision_recall_curve(y, oof)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    best = int(np.argmax(f1[:-1]))
    print("\ncross-validated, whole tiles held out:")
    print(f"  best F1 {f1[best]:.3f} @ thr {thr[best]:.2f} -> "
          f"precision {prec[best]:.3f}, recall {rec[best]:.3f}")
    for tgt in (0.70, 0.80, 0.90):
        idx = np.where(rec[:-1] >= tgt)[0]
        if len(idx):
            i = idx[np.argmax(prec[:-1][idx])]
            print(f"  recall >= {tgt:.0%}: precision {prec[i]:.3f} @ thr {thr[i]:.2f}")

    clf = RandomForestClassifier(**params).fit(X, y)
    out = cfg.path("checkpoints").parent / args.out
    with open(out, "wb") as fh:
        pickle.dump({"model": clf, "threshold": float(thr[best]),
                     "gsd": float(manifest["tiles"][0]["gsd_m"])}, fh)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
