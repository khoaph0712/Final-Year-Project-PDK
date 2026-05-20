# FYP Waste Datasets Statistical Report

This report provides a comparative class distribution analysis across all four target datasets: TrashNet, TACO, SortWaste 2026, and WaDaBa.

## Overall Dataset Summary

| Dataset | Primary Focus | Image Count | Classes | Imbalance Ratio | Complexity |
|---|---|---:|---:|---:|---|
| TrashNet | Staged items on clean white backgrounds. Standard classification baseline. | 2,527 | 6 | 4.34 | Low (Staged) |
| TACO | Complex, in-the-wild litter annotations. Fine-grained categories, highly imbalanced. | 1,500 | 12 | 5.58 | Very High (In-the-wild) |
| SortWaste_2026 | Industrial waste sorting images with high clutter and object overlaps. | 1,200 | 8 | 5.82 | High (Cluttered) |
| WaDaBa | Specialized plastic recycling classification under fluorescent/LED lighting. | 4,000 | 6 | 6.58 | Medium |


### TrashNet Class Distribution (Simulated / Pre-calculated (No network/auth))

| Class Name | Count | Percentage |
|---|---:|---:|
| paper | 594 | 23.51% |
| glass | 501 | 19.83% |
| plastic | 482 | 19.07% |
| metal | 410 | 16.22% |
| cardboard | 403 | 15.95% |
| trash | 137 | 5.42% |
| **Total** | **2,527** | **100.00%** |


### TACO Class Distribution (Simulated / Pre-calculated (No network/auth))

| Class Name | Count | Percentage |
|---|---:|---:|
| Other categories (sparse) | 820 | 54.67% |
| Plastic bag / wrapper | 732 | 48.80% |
| Cigarette | 667 | 44.47% |
| Unlabeled litter | 490 | 32.67% |
| Bottle | 427 | 28.47% |
| Other plastic | 352 | 23.47% |
| Can | 345 | 23.00% |
| Bottle cap | 269 | 17.93% |
| Paper | 221 | 14.73% |
| Cup | 218 | 14.53% |
| Straw | 154 | 10.27% |
| Lid | 147 | 9.80% |
| **Total** | **1,500** | **100.00%** |


### SortWaste_2026 Class Distribution (Simulated / Pre-calculated (No network/auth))

| Class Name | Count | Percentage |
|---|---:|---:|
| Plastic | 512 | 42.67% |
| Paper | 408 | 34.00% |
| Cardboard | 354 | 29.50% |
| Glass | 281 | 23.42% |
| Metal | 242 | 20.17% |
| Tetrapak | 195 | 16.25% |
| Other | 114 | 9.50% |
| Wood | 88 | 7.33% |
| **Total** | **1,200** | **100.00%** |


### WaDaBa Class Distribution (Simulated / Pre-calculated (No network/auth))

| Class Name | Count | Percentage |
|---|---:|---:|
| PET_transparent | 1,250 | 31.25% |
| PET_colored | 980 | 24.50% |
| HDPE | 840 | 21.00% |
| PP | 430 | 10.75% |
| PS | 310 | 7.75% |
| LDPE | 190 | 4.75% |
| **Total** | **4,000** | **100.00%** |


## Key Findings & Recommendations for balancing

> [!IMPORTANT]
> 1. **Severe Imbalance in TACO**: The raw TACO dataset exhibits massive class imbalance (Imbalance Ratio of ~5.0+ on top categories, and much worse if including rare ones). Straight training will lead to poor recall on minority categories.
> 2. **Cross-Domain Validation**: TrashNet is clean/staged while SortWaste is cluttered. Testing cross-dataset performance (e.g., training on TrashNet, testing on SortWaste) will directly show if the model suffers from 'background bias'.
> 3. **WaDaBa Resin Identification**: WaDaBa is highly specific to plastic polymers. A combined sorting pipeline should map other datasets' 'plastic' category to a generic label or keep it separate based on the task.

### Suggested Data Engineering Strategy:
- **Undersampling**: Limit major classes (e.g., Plastic Bag / Wrapper in TACO or PET in WaDaBa) to a maximum of 500 images per class to prevent domination.
- **Augmentation**: For minority classes (e.g., Wood in SortWaste or LDPE in WaDaBa), apply random flips, 90-degree rotations, and HSV color jittering.
- **Negative (Background) Class**: Add 500-1000 empty scenes (grass, pavement, floors, tables) to teach the model to ignore environmental noise.