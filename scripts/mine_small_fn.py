#!/usr/bin/env python
"""Mine small-object false negatives: images where the deployed detector misses
small GT litter. Class-agnostic IoU>=0.5 match at serving conf. Small = box area <1%.
Drives the hard-negative fine-tune set. Run: python scripts/mine_small_fn.py"""
from __future__ import annotations
import json, numpy as np, cv2
from pathlib import Path
from ultralytics import YOLO
ROOT = Path(__file__).resolve().parents[1]
def iou_mat(a,b):
    if len(a)==0 or len(b)==0: return np.zeros((len(a),len(b)))
    x1=np.maximum(a[:,None,0],b[None,:,0]);y1=np.maximum(a[:,None,1],b[None,:,1])
    x2=np.minimum(a[:,None,2],b[None,:,2]);y2=np.minimum(a[:,None,3],b[None,:,3])
    inter=np.clip(x2-x1,0,None)*np.clip(y2-y1,0,None)
    aa=(a[:,2]-a[:,0])*(a[:,3]-a[:,1]);ab=(b[:,2]-b[:,0])*(b[:,3]-b[:,1])
    return inter/(aa[:,None]+ab[None,:]-inter+1e-9)
def gt(lbl,w,h):
    B=[];S=[]
    if lbl.exists():
        for ln in lbl.read_text().splitlines():
            p=ln.split()
            if len(p)==5:
                _,cx,cy,bw,bh=(float(v) for v in p)
                B.append([(cx-bw/2)*w,(cy-bh/2)*h,(cx+bw/2)*w,(cy+bh/2)*h]);S.append(bw*bh<0.01)
    return np.array(B,np.float32).reshape(-1,4),np.array(S,bool)
m=YOLO(str(ROOT/"models/trained/yolov11_detector/best.pt"))
for tag,ds in (("taco",ROOT/"external_datasets/taco_field_clean_v1/train"),
               ("pp",ROOT/"external_datasets/plastopol_clean_v1/train")):
    hard=[]; tot_sgt=tot_sfn=tot_gt=tot_fn=0
    imgs=sorted((ds/"images").glob("*.jpg"))
    for i in range(0,len(imgs),16):
        batch=imgs[i:i+16]
        res=m.predict([str(p) for p in batch],conf=0.04,imgsz=640,iou=0.55,max_det=80,verbose=False)
        for p,r in zip(batch,res):
            im=cv2.imread(str(p));h,w=im.shape[:2]
            g,small=gt(ds/"labels"/(p.stem+".txt"),w,h)
            if len(g)==0: continue
            pb=r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.zeros((0,4))
            matched=np.zeros(len(g),bool)
            if len(pb):
                M=iou_mat(g,pb);used=np.zeros(len(pb),bool)
                for gi in np.argsort(-M.max(axis=1)):
                    c=np.where((M[gi]>=0.5)&~used)[0]
                    if len(c): j=c[np.argmax(M[gi][c])];matched[gi]=True;used[j]=True
            sfn=int((small&~matched).sum()); fn=int((~matched).sum())
            tot_sgt+=int(small.sum());tot_sfn+=sfn;tot_gt+=len(g);tot_fn+=fn
            if sfn>0: hard.append({"img":p.name,"small_fn":sfn,"fn":fn,"gt":len(g)})
    hard.sort(key=lambda x:-x["small_fn"])
    out={"dataset":tag,"images":len(imgs),"gt_boxes":tot_gt,"fn_boxes":tot_fn,
         "small_gt":tot_sgt,"small_fn":tot_sfn,
         "small_recall":1-tot_sfn/tot_sgt if tot_sgt else 0,
         "hard_images":len(hard),"top_hard":hard[:10],"all_hard":[h["img"] for h in hard]}
    json.dump(out,open(ROOT/f"runs/audits/small_fn_mine_{tag}.json","w"),indent=1)
    print(f'{tag}: imgs={len(imgs)} small_gt={tot_sgt} small_fn={tot_sfn} '
          f'small_recall={out["small_recall"]:.3f} hard_imgs={len(hard)}')
