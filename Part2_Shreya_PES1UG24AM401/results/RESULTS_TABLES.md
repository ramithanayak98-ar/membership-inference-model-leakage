# Experimental Results Tables: Membership Inference & Defense Evaluation

**Team Members:** Ramitha Nayak MV (`PES1UG24AM392`) & Shreya N Hegde (`PES1UG24AM401`)

## Table 1: Utility, Runtime Cost, and Privacy Leakage Across Defenses

| Defense Mechanism | Privacy ($\epsilon$) | Train Acc (%) | Test Acc (%) | Gen. Gap (%) | Acc. Cost (pp) | Train Time (s) | Time Cost | Shadow MIA AUC | MIA Adv. | TPR @ 1% FPR | Risk Assessment |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (No Defense)** | `Inf` | 95.67% | 56.17% | 39.50% | +0.00 pp | 4.74s | 1.00x | **0.7390** | 0.4083 | 3.00% | **CRITICAL / HIGH** |
| **L2 Regularization (Moderate, lambda=8e-3)** | `Inf` | 88.33% | 59.33% | 29.00% | -3.17 pp | 4.76s | 1.00x | **0.6611** | 0.2917 | 1.83% | **CRITICAL / HIGH** |
| **L2 Regularization (Strong, lambda=2.5e-2)** | `Inf` | 77.67% | 60.33% | 17.33% | -4.17 pp | 4.84s | 1.02x | **0.5974** | 0.1967 | 0.83% | **MODERATE** |
| **Dropout (p=0.35) + Confidence Smoothing** | `Inf` | 74.83% | 59.17% | 15.67% | -3.00 pp | 3.89s | 0.82x | **0.5839** | 0.1733 | 2.17% | **LOW-MODERATE** |
| **DP-SGD (Moderate Privacy, eps=5.30, sigma=1.38)** | `5.30` | 59.00% | 49.00% | 10.00% | +7.17 pp | 6.26s | 1.32x | **0.5637** | 0.1100 | 1.83% | **LOW-MODERATE** |
| **DP-SGD (Strict Privacy, eps=2.76, sigma=2.45)** | `2.86` | 54.17% | 47.33% | 6.83% | +8.83 pp | 6.30s | 1.33x | **0.5445** | 0.0800 | 0.67% | **LOW-MODERATE** |

## Table 2: Comparison of Attack Methodologies Across Defenses

| Defense Configuration | Loss-Threshold AUC | Confidence-Entropy AUC | Shadow-Supervised AUC | Shadow Attack Acc (%) | Shadow F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline (No Defense)** | 0.6708 | 0.6568 | **0.7390** | 70.08% | 0.7595 |
| **L2 Regularization (Moderate, lambda=8e-3)** | 0.6455 | 0.6253 | **0.6611** | 64.25% | 0.7088 |
| **L2 Regularization (Strong, lambda=2.5e-2)** | 0.6005 | 0.5794 | **0.5974** | 58.67% | 0.6522 |
| **Dropout (p=0.35) + Confidence Smoothing** | 0.5851 | 0.5651 | **0.5839** | 57.75% | 0.6381 |
| **DP-SGD (Moderate Privacy, eps=5.30, sigma=1.38)** | 0.5549 | 0.5361 | **0.5637** | 55.00% | 0.5666 |
| **DP-SGD (Strict Privacy, eps=2.76, sigma=2.45)** | 0.5430 | 0.5306 | **0.5445** | 53.42% | 0.5353 |

## Table 3: Subgroup Disparity & Loss Separation Analysis

| Defense Configuration | Member Mean Loss | Non-Member Mean Loss | Loss Ratio (Out/In) | Rare Subgroup MIA AUC | Majority Subgroup MIA AUC | Subgroup Disparity ($\Delta$ AUC) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (No Defense)** | 0.2486 | 1.6844 | 6.78x | **0.8832** | 0.7043 | +0.1789 |
| **L2 Regularization (Moderate, lambda=8e-3)** | 0.4616 | 1.2430 | 2.69x | **0.7927** | 0.6372 | +0.1555 |
| **L2 Regularization (Strong, lambda=2.5e-2)** | 0.7416 | 1.1244 | 1.52x | **0.6947** | 0.5852 | +0.1095 |
| **Dropout (p=0.35) + Confidence Smoothing** | 0.8260 | 1.1361 | 1.38x | **0.6778** | 0.5729 | +0.1049 |
| **DP-SGD (Moderate Privacy, eps=5.30, sigma=1.38)** | 1.1779 | 1.4907 | 1.27x | **0.5824** | 0.5658 | +0.0166 |
| **DP-SGD (Strict Privacy, eps=2.76, sigma=2.45)** | 1.3691 | 1.5616 | 1.14x | **0.5685** | 0.5435 | +0.0250 |
