"""Train a box classifier to filter classical-CV panel proposals.

The proposal generator in propose_labels.py has high recall (~76% of verified
panels at Jbeil) but poor precision (~18%) -- it fires on any drab, textured
rectangle, including concrete roofs and greenhouse glazing. A detector trained
from scratch on a few hundred boxes does far worse on recall, because detection
must *localise* as well as classify.

Splitting the problem is much easier: let classical CV localise, and train a
classifier only to answer "is this box a panel?" -- a task with 1,100+ labelled
examples available and no localisation burden.

Scored by grouped cross-validation over tiles, so a tile's boxes never appear in
both train and test. Random splitting would leak: boxes from the same rooftop
are near-duplicates and would inflate the score.

    python scripts/train_filter.py --capture jbeil-nds
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


def box_features(bgr: np.ndarray, tex: np.ndarray, b: dict, gsd: float) -> list[float]:
    """Describe one candidate box: its own appearance and how it differs from context."""
    H, W = bgr.shape[:2]
    x1, y1 = max(0, b["x1"]), max(0, b["y1"])
    x2, y2 = min(W, b["x2"]), min(H, b["y2"])
    if x2 <= x1 or y2 <= y1:
        return None

    roi = bgr[y1:y2, x1:x2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV).astype(np.float32)
    t = tex[y1:y2, x1:x2]

    # Surrounding ring, for contrast against context. A panel differs from its
    # roof; a plain bright roof does not differ from the roof around it.
    pad = 12
    ox1, oy1 = max(0, x1 - pad), max(0, y1 - pad)
    ox2, oy2 = min(W, x2 + pad), min(H, y2 + pad)
    outer = cv2.cvtColor(bgr[oy1:oy2, ox1:ox2], cv2.COLOR_BGR2HSV).astype(np.float32)

    w_m, h_m = (x2 - x1) * gsd, (y2 - y1) * gsd
    long_s, short_s = max(w_m, h_m), max(min(w_m, h_m), 1e-3)

    return [
        hsv[:, :, 0].mean(), hsv[:, :, 0].std(),
        hsv[:, :, 1].mean(), hsv[:, :, 1].std(),
        hsv[:, :, 2].mean(), hsv[:, :, 2].std(),
        t.mean(), t.std(), np.percentile(t, 90),
        hsv[:, :, 1].mean() - outer[:, :, 1].mean(),   # saturation vs context
        hsv[:, :, 2].mean() - outer[:, :, 2].mean(),   # brightness vs context
        w_m * h_m, long_s, long_s / short_s,
        float(b.get("score", 0.0)),
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--capture", required=True)
    ap.add_argument("--out", default="models/box_filter.pkl")
    ap.add_argument("--config")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    cap = cfg.path("captures") / args.capture
    labels = json.loads((cap / "labels.json").read_text(encoding="utf-8"))
    gsd = float(json.loads((cap / "manifest.json").read_text(encoding="utf-8"))["gsd_m"])

    X, y, groups = [], [], []
    for tid, boxes in labels["tiles"].items():
        reviewed = [b for b in boxes if b.get("verified") is not None]
        if not reviewed:
            continue
        img = cv2.imread(str(cap / "tiles" / f"{tid}.jpg"), cv2.IMREAD_COLOR)
        if img is None:
            continue
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        mean = cv2.blur(g, (7, 7))
        tex = np.sqrt(np.maximum(cv2.blur(g * g, (7, 7)) - mean * mean, 0))
        for b in reviewed:
            f = box_features(img, tex, b, gsd)
            if f is None:
                continue
            X.append(f); y.append(1 if b["verified"] else 0); groups.append(tid)

    X = np.array(X, np.float32); y = np.array(y); groups = np.array(groups)
    print(f"{len(X)} labelled boxes ({y.sum()} panels, {len(y)-y.sum()} negatives) "
          f"across {len(set(groups))} tiles")

    clf = RandomForestClassifier(
        n_estimators=400, max_depth=None, min_samples_leaf=3,
        class_weight="balanced", n_jobs=-1, random_state=0,
    )

    # Grouped CV: hold out whole tiles, never individual boxes.
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        m = RandomForestClassifier(**clf.get_params())
        m.fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]

    prec, rec, thr = precision_recall_curve(y, oof)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    best = int(np.argmax(f1[:-1]))
    print(f"\ncross-validated (whole tiles held out):")
    print(f"  best F1 {f1[best]:.3f} at threshold {thr[best]:.2f} "
          f"-> precision {prec[best]:.3f}, recall {rec[best]:.3f}")
    for target in (0.60, 0.70, 0.80):
        idx = np.where(rec[:-1] >= target)[0]
        if len(idx):
            i = idx[np.argmax(prec[:-1][idx])]
            print(f"  at recall >= {target:.0%}: precision {prec[i]:.3f} "
                  f"(threshold {thr[i]:.2f})")

    clf.fit(X, y)
    order = np.argsort(-clf.feature_importances_)
    names = ["hue_mean","hue_std","sat_mean","sat_std","val_mean","val_std",
             "tex_mean","tex_std","tex_p90","sat_vs_ctx","val_vs_ctx",
             "area_m2","long_side_m","aspect","cv_score"]
    print("\ntop features: " + ", ".join(
        f"{names[i]} {clf.feature_importances_[i]:.2f}" for i in order[:6]))

    out = cfg.path("checkpoints").parent / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as fh:
        pickle.dump({"model": clf, "threshold": float(thr[best]), "gsd": gsd}, fh)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
