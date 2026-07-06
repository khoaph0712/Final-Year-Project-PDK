#!/usr/bin/env python
"""Pseudo-label PlastOPol class-agnostic GT boxes with the deployed ConvNeXt classifier,
so the new field data can train the 6-class detector. Detector material is de-emphasized
(alpha cap 0.40), and the classifier is the 92.9% material authority - so classifier-derived
labels are the right source. Writes 6-class YOLO labels to a parallel dir.
Run: python scripts/pseudo_label_plastopol.py"""
import sys, numpy as np, cv2
from pathlib import Path
sys.path.insert(0,'web'); import server
ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/"external_datasets/plastopol_clean_v1/train"
OUTLBL=ROOT/"external_datasets/plastopol_clean_v1/train/labels_6class"
OUTLBL.mkdir(exist_ok=True)
models=server.get_models()
imgs=sorted((SRC/"images").glob("*.jpg"))
n_box=0
for i,p in enumerate(imgs):
    im=cv2.imread(str(p));h,w=im.shape[:2]
    lf=SRC/"labels"/(p.stem+".txt")
    if not lf.exists(): continue
    rows=[];crops=[];geoms=[]
    for ln in lf.read_text().splitlines():
        t=ln.split()
        if len(t)!=5: continue
        _,cx,cy,bw,bh=(float(v) for v in t)
        x1=int((cx-bw/2)*w);y1=int((cy-bh/2)*h);x2=int((cx+bw/2)*w);y2=int((cy+bh/2)*h)
        crop=im[max(0,y1):y2,max(0,x1):x2]
        if crop.size==0 or crop.shape[0]<12 or crop.shape[1]<12:
            geoms.append(None); continue
        crops.append(crop); geoms.append((cx,cy,bw,bh,len(crops)-1))
    if not crops: continue
    probs=np.concatenate([server.classify_bgr_batch(crops[k:k+64],models) for k in range(0,len(crops),64)],0)
    mat=probs[:,:6].argmax(1)  # 6-class material, ignore Background(6)
    lines=[]
    for g in geoms:
        if g is None: continue
        cx,cy,bw,bh,ci=g
        lines.append(f"{int(mat[ci])} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    (OUTLBL/(p.stem+".txt")).write_text("\n".join(lines))
    n_box+=len(lines)
    if (i+1)%400==0: print(f"  {i+1}/{len(imgs)}")
print(f"pseudo-labeled {len(imgs)} imgs, {n_box} boxes -> {OUTLBL}")
