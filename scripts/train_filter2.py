"""Train the box classifier on richly-generated proposals.

The first version trained only on boxes a human had explicitly reviewed, which
tied the classifier to whatever the original (tight) proposal settings happened
to emit. Those settings capped recall at ~81% -- panels the proposer never
suggested could never be found, no matter how good the classifier got.

Relaxing the proposal thresholds (saturation < 60, texture > 10, min 4 m2)
reaches 100% recall on the verified panels at Jbeil, at the cost of many more
candidates. That is the right trade here, because candidates are cheap to
generate and we already have ground truth to label them automatically: a
proposal is positive if it substantially overlaps a verified panel, negative
otherwise. The classifier's job becomes the whole problem, and it has far more
training data than before.

    python scripts/train_filter2.py --capture jbeil-nds
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
from solarmap.infer.cvfilter import box_features, propose_rich, texture_map


def overlap_frac(a, b) -> float:
    """Fraction of box ``a`` covered by box ``b``."""
    ix = max(0, min(a[2], b["x2"]) - max(a[0], b["x1"]))
    iy = max(0, min(a[3], b["y2"]) - max(a[1], b["y1"]))
    area = (a[2] - a[0]) * (a[3] - a[1])
    return (ix * iy) / area if area > 0 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--capture", required=True)
    ap.add_argument("--out", default="models/box_filter2.pkl")
    ap.add_argument("--pos-overlap", type=float, default=0.45,
                    help="a proposal is positive if this much of it lies on a verified panel")
    ap.add_argument("--neg-overlap", type=float, default=0.15,
                    help="proposals between neg and pos overlap are ambiguous and dropped")
    ap.add_argument("--config")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    cap = cfg.path("captures") / args.capture
    labels = json.loads((cap / "labels.json").read_text(encoding="utf-8"))
    manifest = json.loads((cap / "manifest.json").read_text(encoding="utf-8"))
    gsd = float(manifest["gsd_m"])

    X, y, groups = [], [], []
    n_prop = n_amb = 0
    for t in manifest["tiles"]:
        tid = t["tile_id"]
        boxes = labels["tiles"].get(tid, [])
        if not boxes or any(b.get("verified") is None for b in boxes):
            continue  # only fully-reviewed tiles: an unreviewed panel would
                      # be labelled negative and poison the classifier
        panels = [b for b in boxes if b.get("verified") is True]

        img = cv2.imread(str(cap / "tiles" / f"{tid}.jpg"), cv2.IMREAD_COLOR)
        if img is None:
            continue
        tex = texture_map(img)
        props = propose_rich(img, gsd)
        n_prop += len(props)

        for p in props:
            box = (p["x1"], p["y1"], p["x2"], p["y2"])
            best = max((overlap_frac(box, g) for g in panels), default=0.0)
            if args.neg_overlap < best < args.pos_overlap:
                n_amb += 1
                continue  # ambiguous: partly on a panel, partly not
            f = box_features(img, tex, p, gsd)
            if f is None:
                continue
            X.append(f); y.append(1 if best >= args.pos_overlap else 0); groups.append(tid)

    X = np.array(X, np.float32); y = np.array(y); groups = np.array(groups)
    print(f"{n_prop} proposals -> {len(X)} training boxes "
          f"({y.sum()} positive, {len(y)-y.sum()} negative, {n_amb} ambiguous dropped) "
          f"across {len(set(groups))} tiles")

    params = dict(n_estimators=500, min_samples_leaf=2, class_weight="balanced_subsample",
                  n_jobs=-1, random_state=0)

    oof = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        mdl = RandomForestClassifier(**params).fit(X[tr], y[tr])
        oof[te] = mdl.predict_proba(X[te])[:, 1]

    prec, rec, thr = precision_recall_curve(y, oof)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    best = int(np.argmax(f1[:-1]))
    print("\ncross-validated, whole tiles held out:")
    print(f"  best F1 {f1[best]:.3f} @ thr {thr[best]:.2f} -> "
          f"precision {prec[best]:.3f}, recall {rec[best]:.3f}")
    for target in (0.80, 0.90, 0.95):
        idx = np.where(rec[:-1] >= target)[0]
        if len(idx):
            i = idx[np.argmax(prec[:-1][idx])]
            print(f"  recall >= {target:.0%}: precision {prec[i]:.3f} @ thr {thr[i]:.2f}")

    clf = RandomForestClassifier(**params).fit(X, y)
    out = cfg.path("checkpoints").parent / args.out
    with open(out, "wb") as fh:
        pickle.dump({"model": clf, "threshold": float(thr[best]), "gsd": gsd,
                     "proposals": "rich"}, fh)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
