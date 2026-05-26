# PCA Feature-Space Dimensionality Reduction Report

This report analyzes the impact of Principal Component Analysis (PCA) dimensionality reduction on our 637 handcrafted texture/edge feature classifier.

| Components Count | Explained Variance (%) | Validation Accuracy (%) | Weighted F1-Score | Inference Latency (ms) | Train Time (s) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **637** | 100.00% | 73.24% | 0.7319 | 0.0533 ms | 7.31 s |
| **32** | 99.60% | 60.71% | 0.6062 | 0.0233 ms | 6.16 s |
| **64** | 99.78% | 67.48% | 0.6736 | 0.0284 ms | 6.22 s |
| **128** | 99.90% | 68.71% | 0.6863 | 0.0314 ms | 6.46 s |
| **256** | 99.97% | 66.95% | 0.6691 | 0.0296 ms | 6.81 s |
| **512** | 100.00% | 65.05% | 0.6498 | 0.0281 ms | 5.66 s |
