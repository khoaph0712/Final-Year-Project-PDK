# 6. Datasets: Acquisition, Auditing, and Rebuilds

## 6.1 Raw Sources

Four public sources supply every image used in this project. No image in any
training, validation, or test split originates from a model prediction; all
labels trace to human annotation, which was verified by reading the ingestion
scripts rather than assumed.
<!-- src: runs/audits/MODEL_RISK_AUDIT.md#model-inbreeding-CLEAR -->

**TACO (Trash Annotations in Context)** provides in-the-wild litter photographed
against uncontrolled backgrounds, annotated as COCO instance segmentations
across 60 fine-grained categories. TACO is the project's reference for realistic
acquisition conditions. Boxes are derived from the segmentation fields and the
60 categories collapsed onto the project taxonomy by
`scripts/export_taco_yolo_hardcase.py`.
<!-- src: runs/external_dataset_registry/DATASET_CANDIDATE_SUMMARY.md#TACO -->

**Roboflow Universe community projects** contribute the bulk of the detection
data. Four projects are ingested — a garbage classification set, a TACO-derived
trash set, a food-waste set, and a cigarette set — mirrored locally as
`rf_garbage_cls`, `rf_taco_trash`, `rf_food_waste`, and `rf_cigarettes`. Their
upstream annotation quality is community-sourced and therefore not independently
verifiable, which is recorded here as a standing limitation rather than
discovered later.
<!-- src: scripts/archive/build_merged_dataset.py#docstring -->

**Kaggle Garbage Classification** supplies studio-condition classification
imagery: single objects, centred, on clean backgrounds, across twelve classes
(battery, biological, brown-glass, cardboard, clothes, green-glass, metal,
paper, plastic, shoes, trash, white-glass). All twelve appear in the
classification dataset's filename stems.
<!-- src: data/merged_dataset_v5/train (filename stems: clothes, biological, brown-glass, green-glass, white-glass, shoes, battery, trash) -->

A further six datasets were evaluated and rejected, with reasons recorded in the
candidate registry: MJU-Waste and GINI are effectively single-class; TrashCan is
underwater; ZeroWaste is industrial rather than household; TrashNet has staged
clean backgrounds unsuitable for detector training.
<!-- src: runs/external_dataset_registry/DATASET_CANDIDATE_SUMMARY.md#Risks -->

> **Evidence-integrity note.** The file `runs/dataset_eda/external_datasets_stats.md`
> reports class distributions for TrashNet, TACO, "SortWaste 2026", and WaDaBa.
> Its numbers for all but TrashNet are hard-coded constants from the
> `SIMULATED_STATS` dictionary in `scripts/archive/download_and_structure_datasets.py`,
> emitted when the script runs without network access; neither SortWaste nor
> WaDaBa exists anywhere in the project's data directories. Nothing in this
> report is sourced from that file.
> <!-- src: scripts/archive/download_and_structure_datasets.py#SIMULATED_STATS -->

## 6.2 Dataset Generations

The datasets were rebuilt five times. Each rebuild responded to a specific
measured failure rather than to a general desire for more data.

The detection line begins with `super_yolo_dataset`, assembled from the Roboflow
projects and TACO exports. Version 3 dropped the noisy `BIODEGRADABLE` and
`food waste` categories, replaced them with a clean organic class sourced from
the food-waste project, and narrowed an over-broad `other` class down to
styrofoam and cigarettes, discarding ambiguous battery, cap, and utensil
categories. It also re-split classes that had ended up with zero test images.
<!-- src: scripts/archive/build_merged_dataset.py#docstring -->

The current detection dataset, `yolo26_hardcase_dataset_v1`, merges
`super_yolo_dataset` with a TACO hard-case export. Its manifest records the
provenance of every one of its 26,100 images: 24,711 from `super_yolo` and 1,389
TACO hard cases, split 21,199 train / 3,538 validation / 1,363 test. Ingestion
is by hardlink, so no image is silently re-encoded between generations.
<!-- src: external_datasets/yolo26_hardcase_dataset_v1/source_manifest.csv#source_id,split -->

The classification line converges on `merged_dataset_v5`: 24,039 training and
5,600 test images across seven classes (the six material classes plus
Background). It deliberately mixes the two acquisition conditions described in
§6.1 — studio imagery from Kaggle Garbage Classification, and field crops cut
from the detection dataset — because a classifier trained on studio imagery
alone collapses on field input. Of the 24,039 training images, 11,659 are field
crops and 12,380 are studio images.
<!-- src: runs/audits/model_risk_audit.json#datasets.merged.domains.train (field/studio columns sum) -->

Table 6.1 summarises the four audited datasets.

| Dataset | Role | Train | Val | Test |
|---|---|---:|---:|---:|
| `super_yolo_dataset` | detection, superseded | 20,165 | 3,354 | 1,192 |
| `yolo26_hardcase_dataset_v1` | detection, current | 21,199 | 3,538 | 1,363 |
| `merged_dataset_v5` | classification, current | 24,039 | — | 5,600 |
| `hard_case_classifier_v1` | classification, hard cases | 11,163 | 2,680 | 2,696 |

<!-- src: runs/audits/model_risk_audit.json#datasets.*.labels.*.label_files; on-disk image counts for merged_dataset_v5 and hard_case_classifier_v1 -->

*Table 6.1 — Audited dataset generations and split sizes. `merged_dataset_v5`
has no validation split; model selection for the classifier uses a held-out
portion of train.*

## 6.3 Exploratory Data Analysis

Three properties of the detection data materially shaped later design decisions.

**Class balance is severely skewed, and skewed differently in every split.** In
`yolo26_hardcase_dataset_v1`, organic dominates training with 33,871 boxes while
paper has 4,750; in the test split that ordering inverts, with paper at 1,424
boxes and organic at only 50. Glass is the extreme case: 7,324 training boxes
against 77 in test.
<!-- src: runs/audits/MODEL_RISK_AUDIT.md#Dataset:-hardcase per-class table -->
Any single headline accuracy over this data is therefore dominated by whichever
classes happen to be frequent in the split being reported, which is part of why
§8 reports per-class metrics rather than a single figure.

**Objects are small, and smallest in exactly the classes that matter.** Median
box area as a fraction of image area is 1.14% for organic and 2.17% for plastic
in training, against 22.3% for paper. In the training split 47.0% of organic
boxes and 35.8% of plastic boxes occupy under 1% of their image.
<!-- src: runs/audits/MODEL_RISK_AUDIT.md#Dataset:-hardcase per-class table (median area, tiny <1% img) -->
This directly motivated the tiled-training and higher-resolution experiments
reported in §9.

**Annotation format is mixed.** Labels arrive as a mixture of bounding boxes and
polygons: the training split contains 6,099 polygon rows, validation 624, and
test 10, all converted to axis-aligned boxes at ingestion. A small number of
degenerate annotations — 10 zero-area rows in train, 6 in validation, 1 in test
— were also present. Additionally, 71.6% of training images contain only a
single class, meaning most images offer no within-image class contrast.
<!-- src: runs/audits/model_risk_audit.json#datasets.hardcase.labels.*.noise, .single_class_image_pct -->

## 6.4 Cross-Split Leakage Audit

This audit is the report's principal methodological contribution. It began as a
routine data-hygiene check and ended by invalidating every evaluation metric
produced before it.

### 6.4.1 Method

Two images leak across a split boundary if they are byte-identical or
perceptually near-identical. Byte identity is tested by MD5 over the raw file.
Perceptual similarity uses a 64-bit difference hash.

Each image is converted to greyscale and resized to a 9×8 grid. The hash records
whether each pixel is brighter than its right-hand neighbour, giving 8×8 = 64
bits:

$$
b_{i,j} = \begin{cases} 1 & \text{if } I(i,\, j+1) > I(i,\, j) \\ 0 & \text{otherwise} \end{cases}
\qquad 0 \le i \le 7,\; 0 \le j \le 7
\tag{1}
$$

$$
h(x) = \sum_{i=0}^{7} \sum_{j=0}^{7} b_{i,j} \cdot 2^{\,8i + j}
\tag{2}
$$

<!-- src: scripts/audit_model_risks.py#dhash64 -->

Because the hash encodes only relative brightness ordering, it is invariant to
rescaling and to global brightness or contrast shifts — precisely the
transformations that distinguish a re-exported duplicate from a genuinely
different photograph. Similarity between two hashes is their Hamming distance,
the population count of their XOR:

$$
d_H(a, b) = \operatorname{popcount}(a \oplus b)
\tag{3}
$$

<!-- src: scripts/audit_model_risks.py#hamming -->

An image pair is counted as leaked when either test succeeds, with the
near-duplicate threshold set at 4 bits:

$$
L(x, y) = \bigl[\,\mathrm{md5}(x) = \mathrm{md5}(y)\,\bigr] \;\lor\; \bigl[\, d_H\bigl(h(x), h(y)\bigr) \le 4 \,\bigr]
\tag{4}
$$

The threshold is validated by an assertion pair that runs on every invocation: a
rescaled copy of an image must hash within 4 bits of the original, and an
unrelated image must not.
<!-- src: scripts/audit_model_risks.py#positive-control-and-negative-control-asserts -->

### 6.4.2 Making the audit tractable

Evaluating $L$ over all cross-split pairs means 406,242,619 comparisons across
the four datasets — 134.6 million for `merged_dataset_v5` alone. Decoding is the
dominant cost, so each image is decoded exactly once and its MD5 and hash cached.

The Hamming search is reduced by a pigeonhole argument. The 64-bit hash is
partitioned into five chunks of 13, 13, 13, 13 and 12 bits. If two hashes differ
in at most 4 bits, those differing bits can occupy at most four of the five
chunks, so **at least one chunk must match exactly**. Indexing every hash by each
of its five chunk values and querying only the five matching buckets therefore
returns a superset of the true neighbours, and exhaustive comparison is avoided
without any loss of recall.
<!-- src: scripts/audit_model_risks.py#NearDupIndex.CHUNKS -->

### 6.4.3 Findings

Every dataset audited leaked. Table 6.2 reports, per dataset, the leaked pairs
found and the resulting fraction of test images that also appear in training.

| Dataset | Exact pairs | Near pairs | Test images | Leaked test images | **Leakage rate** |
|---|---:|---:|---:|---:|---:|
| `merged_dataset_v5` | 1,723 | 10,289 | 5,600 | 2,406 | **43.0%** |
| `yolo26_hardcase_dataset_v1` | 1 | 4,808 | 1,363 | 457 | **33.5%** |
| `hard_case_classifier_v1` | 516 | 2,775 | 2,696 | 545 | **20.2%** |
| `super_yolo_dataset` | 1 | 3,880 | 1,192 | 236 | **19.8%** |

<!-- src: runs/audits/model_risk_audit.json#datasets.*.leakage (exact/near pair counts; leaked test images = unique test-side paths in exact+near) -->

*Table 6.2 — Cross-split leakage. Pair counts include all split directions;
leaked test images counts unique test-split images sharing an MD5 or falling
within Hamming 4 of any training image. No image in any dataset failed to
decode.*

The classification dataset is worst affected: 43.0% of `merged_dataset_v5`'s test
set is a duplicate or near-duplicate of something the model trained on, and
1,723 of those are byte-identical files. The cause is visible in the filenames —
`c5_682_shoes1950.jpg` in test and `c5_1061_shoes1950.jpg` in train are the same
source image ingested twice under different sequence numbers, so a split
performed on the merged output could never have separated them.
<!-- src: runs/audits/MODEL_RISK_AUDIT.md#Dataset:-merged EXACT examples -->

Two consequences follow, and both are carried forward rather than set aside.
First, every accuracy figure measured on these test splits before the audit is
inflated by an unknown but substantial margin, and none is quoted as a headline
result in §8. Second, the remedy is not resplitting the same images — that
merely relocates the duplicates. Clean evaluation splits were rebuilt with
duplicates removed across the boundary (`data/merged_dataset_v5_clean_test`,
`external_datasets/yolo26_hardcase_clean_eval`), and §8 reports against those.

One negative result is worth recording: the filename-marker check for
studio/field contamination in `merged_dataset_v5` returned zero in both
directions. The acquisition-domain split is clean; it is the image content that
leaks.
<!-- src: runs/audits/MODEL_RISK_AUDIT.md#Filename-marker-cross-split-contamination-merged_v5-CLEAR -->

![Cross-split leakage audit workflow](docs/diagrams/png/wf_leakage_audit.png)

*Figure 6.1 — Leakage audit pipeline: single-pass decode with MD5 and dHash
caching, chunk-indexed near-duplicate lookup, and per-split reporting.*
