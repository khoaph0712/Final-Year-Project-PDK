import { CLASSES, type WasteClass } from "./classes";

export type RawBox = {
  x: number;
  y: number;
  w: number;
  h: number;
  score: number;
  classIdx: number;
  cls: WasteClass;
};

/**
 * Decode a YOLOv8 detection output tensor.
 *
 * Ultralytics TFLite export emits a tensor of shape
 *   [1, 4 + num_classes, num_anchors]   (NCHW-like)     OR
 *   [1, num_anchors, 4 + num_classes]   (transposed)
 *
 * We detect the layout by shape and decode accordingly.
 * Coordinates returned are normalised to 0..1 (centre x/y, w, h).
 */
export function decodeYoloOutput(
  output: Float32Array | number[],
  shape: readonly number[],
  numClasses: number,
  confThreshold: number,
): RawBox[] {
  if (shape.length !== 3) return [];
  const [_, d1, d2] = shape;
  const stride = 4 + numClasses;

  let numAnchors: number;
  let transposed: boolean;
  if (d1 === stride) {
    transposed = false; // [1, stride, A]
    numAnchors = d2;
  } else if (d2 === stride) {
    transposed = true; // [1, A, stride]
    numAnchors = d1;
  } else {
    return [];
  }

  const out: RawBox[] = [];
  for (let i = 0; i < numAnchors; i++) {
    let cx: number, cy: number, w: number, h: number;
    let bestScore = 0;
    let bestIdx = -1;

    if (transposed) {
      const base = i * stride;
      cx = output[base];
      cy = output[base + 1];
      w = output[base + 2];
      h = output[base + 3];
      for (let c = 0; c < numClasses; c++) {
        const s = output[base + 4 + c];
        if (s > bestScore) {
          bestScore = s;
          bestIdx = c;
        }
      }
    } else {
      cx = output[0 * numAnchors + i];
      cy = output[1 * numAnchors + i];
      w = output[2 * numAnchors + i];
      h = output[3 * numAnchors + i];
      for (let c = 0; c < numClasses; c++) {
        const s = output[(4 + c) * numAnchors + i];
        if (s > bestScore) {
          bestScore = s;
          bestIdx = c;
        }
      }
    }

    if (bestScore < confThreshold || bestIdx < 0) continue;

    out.push({
      x: cx,
      y: cy,
      w,
      h,
      score: bestScore,
      classIdx: bestIdx,
      cls: CLASSES[bestIdx] ?? "other",
    });
  }
  return out;
}

export type Xyxy = { x1: number; y1: number; x2: number; y2: number };

function toXyxy(b: RawBox): Xyxy {
  return {
    x1: b.x - b.w / 2,
    y1: b.y - b.h / 2,
    x2: b.x + b.w / 2,
    y2: b.y + b.h / 2,
  };
}

function iou(a: Xyxy, b: Xyxy): number {
  const x1 = Math.max(a.x1, b.x1);
  const y1 = Math.max(a.y1, b.y1);
  const x2 = Math.min(a.x2, b.x2);
  const y2 = Math.min(a.y2, b.y2);
  const iw = Math.max(0, x2 - x1);
  const ih = Math.max(0, y2 - y1);
  const inter = iw * ih;
  const areaA = Math.max(0, a.x2 - a.x1) * Math.max(0, a.y2 - a.y1);
  const areaB = Math.max(0, b.x2 - b.x1) * Math.max(0, b.y2 - b.y1);
  const uni = areaA + areaB - inter;
  return uni <= 0 ? 0 : inter / uni;
}

/** Class-aware non-max suppression. */
export function nms(boxes: RawBox[], iouThreshold: number, topK = 50): RawBox[] {
  const sorted = [...boxes].sort((a, b) => b.score - a.score);
  const kept: RawBox[] = [];
  const xyxy = sorted.map(toXyxy);
  const suppressed = new Array<boolean>(sorted.length).fill(false);

  for (let i = 0; i < sorted.length && kept.length < topK; i++) {
    if (suppressed[i]) continue;
    const base = sorted[i];
    kept.push(base);
    for (let j = i + 1; j < sorted.length; j++) {
      if (suppressed[j]) continue;
      if (sorted[j].classIdx !== base.classIdx) continue;
      if (iou(xyxy[i], xyxy[j]) > iouThreshold) suppressed[j] = true;
    }
  }
  return kept;
}

export function summariseDetections(boxes: RawBox[]) {
  if (boxes.length === 0) return null;
  const counts = new Map<WasteClass, { count: number; maxScore: number }>();
  for (const b of boxes) {
    const cur = counts.get(b.cls) ?? { count: 0, maxScore: 0 };
    cur.count += 1;
    cur.maxScore = Math.max(cur.maxScore, b.score);
    counts.set(b.cls, cur);
  }
  const top = [...counts.entries()].sort((a, b) => b[1].maxScore - a[1].maxScore)[0];
  return {
    topClass: top[0],
    topScore: top[1].maxScore,
    perClass: Object.fromEntries(counts),
  };
}
