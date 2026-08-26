"""Proposal-stage recall: the hard ceiling on detection recall.
A verified panel with no candidate box can never be recovered downstream."""
import json, pathlib, sys
import cv2, numpy as np
import _bootstrap  # noqa
from solarmap.config import Config
from solarmap.infer.cvfilter import propose, propose_rich

def iou(a, b):
    ix = max(0, min(a["x2"], b["x2"]) - max(a["x1"], b["x1"]))
    iy = max(0, min(a["y2"], b["y2"]) - max(a["y1"], b["y1"]))
    inter = ix*iy
    if inter <= 0: return 0.0
    ua = (a["x2"]-a["x1"])*(a["y2"]-a["y1"]) + (b["x2"]-b["x1"])*(b["y2"]-b["y1"]) - inter
    return inter/ua if ua>0 else 0.0

def cover(gt, b):  # fraction of the GT box covered by a proposal
    ix = max(0, min(gt["x2"], b["x2"]) - max(gt["x1"], b["x1"]))
    iy = max(0, min(gt["y2"], b["y2"]) - max(gt["y1"], b["y1"]))
    a = (gt["x2"]-gt["x1"])*(gt["y2"]-gt["y1"])
    return ix*iy/a if a>0 else 0.0

cfg = Config.load(None)
for capname, labelfile in [("jbeil-nds","labels.json"), ("jbeil-mb","labels.json"),
                           ("jbeil-mb","labels_transferred.json"),
                           ("jbeil-mb-081","labels.json"), ("jbeil-mb-104","labels.json")]:
    cap = cfg.path("captures")/capname
    man = json.loads((cap/"manifest.json").read_text(encoding="utf-8"))
    lab = json.loads((cap/labelfile).read_text(encoding="utf-8"))
    for pname, pf in [("propose", propose), ("propose_rich", propose_rich)]:
        n_gt=n_hit25=n_hit50=n_prop=0
        for t in man["tiles"]:
            tid=t["tile_id"]; g=float(t["gsd_m"])
            panels=[b for b in lab["tiles"].get(tid,[]) if b.get("verified") is True]
            if not panels: continue
            img=cv2.imread(str(cap/"tiles"/f"{tid}.jpg"), cv2.IMREAD_COLOR)
            if img is None: continue
            props=pf(img,g); n_prop+=len(props)
            for gt in panels:
                n_gt+=1
                best_i=max((iou(gt,p) for p in props), default=0.0)
                best_c=max((cover(gt,p) for p in props), default=0.0)
                if best_i>=0.25: n_hit25+=1
                if best_c>=0.50: n_hit50+=1
        print(f"{capname:14} {labelfile:26} {pname:13} "
              f"gsd={float(man['tiles'][0]['gsd_m']):.4f} "
              f"proposals={n_prop:6d}  recall@IoU.25={n_hit25:4d}/{n_gt:<4d}={n_hit25/max(n_gt,1):6.1%}  "
              f"recall@cover.50={n_hit50/max(n_gt,1):6.1%}")
        sys.stdout.flush()
