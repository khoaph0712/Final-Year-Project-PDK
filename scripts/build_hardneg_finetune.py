#!/usr/bin/env python
"""Build the hard-negative fine-tune set for small-object recall.
Field-heavy, small-FN-emphasized, with a studio anchor to resist forgetting:
  studio (hardcase non-taco, sample) x1  [real 6-class GT]
  taco train x1 + taco hard imgs xK extra [real 6-class GT]
  pp   train x1 + pp   hard imgs xK extra [pseudo 6-class GT from classifier]
val = taco_field_clean/val (real 6-class GT) for field early-stop.
Run: python scripts/build_hardneg_finetune.py --k 3 --studio 4000"""
from __future__ import annotations
import argparse, json, os, random, shutil
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
HC=ROOT/"external_datasets/yolo26_hardcase_dataset_v1/train"
TACO=ROOT/"external_datasets/taco_field_clean_v1"
PP=ROOT/"external_datasets/plastopol_clean_v1"
OUT=ROOT/"external_datasets/hardneg_smallobj_v1"
IMG_EXT={".jpg",".jpeg",".png"}
def link(src,dst):
    dst.parent.mkdir(parents=True,exist_ok=True)
    if dst.exists(): return
    try: os.link(src,dst)
    except OSError: shutil.copy2(src,dst)
def emit(img,lbl,di,dl,stem):
    if not lbl.exists() or not lbl.read_text().strip(): return 0
    link(img,di/f"{stem}{img.suffix}"); (dl/f"{stem}.txt").write_text(lbl.read_text()); return 1
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--k",type=int,default=3); ap.add_argument("--studio",type=int,default=4000)
    a=ap.parse_args(); random.seed(42)
    if OUT.exists(): shutil.rmtree(OUT)
    ti,tl=OUT/"train/images",OUT/"train/labels"; vi,vl=OUT/"val/images",OUT/"val/labels"
    for d in (ti,tl,vi,vl): d.mkdir(parents=True,exist_ok=True)
    c=Counter()
    # studio sample x1
    studio=[p for p in (HC/"images").iterdir() if p.suffix.lower() in IMG_EXT and not p.name.startswith("taco")]
    random.shuffle(studio)
    for p in studio[:a.studio]: c["studio"]+=emit(p,HC/"labels"/(p.stem+".txt"),ti,tl,f"st_{p.stem}")
    # taco x1 (real GT) + hard extra
    tacohard=set(json.load(open(ROOT/"runs/audits/small_fn_mine_taco.json"))["all_hard"])
    for p in (TACO/"train/images").iterdir():
        if p.suffix.lower() not in IMG_EXT: continue
        c["taco"]+=emit(p,TACO/"train/labels"/(p.stem+".txt"),ti,tl,f"tc_{p.stem}")
        if p.name in tacohard:
            for r in range(a.k): c["taco_hard"]+=emit(p,TACO/"train/labels"/(p.stem+".txt"),ti,tl,f"tc_{p.stem}_h{r}")
    # pp x1 (pseudo GT) + hard extra
    pphard=set(json.load(open(ROOT/"runs/audits/small_fn_mine_pp.json"))["all_hard"])
    pplbl=PP/"train/labels_6class"
    for p in (PP/"train/images").iterdir():
        if p.suffix.lower() not in IMG_EXT: continue
        c["pp"]+=emit(p,pplbl/(p.stem+".txt"),ti,tl,f"pp_{p.stem}")
        if p.name in pphard:
            for r in range(a.k): c["pp_hard"]+=emit(p,pplbl/(p.stem+".txt"),ti,tl,f"pp_{p.stem}_h{r}")
    # val = taco field val (real GT)
    for p in (TACO/"val/images").iterdir():
        if p.suffix.lower() in IMG_EXT: c["val"]+=emit(p,TACO/"val/labels"/(p.stem+".txt"),vi,vl,f"tc_{p.stem}")
    (OUT/"data.yaml").write_text(
        f"path: {OUT}\ntrain: train/images\nval: val/images\n"
        "nc: 6\nnames: ['plastic','glass','metal','paper','cardboard','organic']\n")
    tot=sum(v for k,v in c.items() if k!="val")
    print(dict(c)); print(f"train total={tot} field_share={(c['taco']+c['taco_hard']+c['pp']+c['pp_hard'])/tot:.3f} val={c['val']}")
if __name__=="__main__": main()
