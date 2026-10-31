# Model Card: Clinical Phenotype Classifier (`ClinicalMLPClassifier`)

- **Model Name:** `ClinicalMLPClassifier` (Standard & Differentially Private Variants)
- **Version:** 1.0.0
- **Developers / Authors:**
  - **Ramitha Nayak MV** (`PES1UG24AM392`)
  - **Shreya N Hegde** (`PES1UG24AM401`)
- **Course:** Big Data Systems and Responsible AI Engineering (PES University)
- **Model Type:** Two-Layer Feedforward Neural Network (Multi-Layer Perceptron) with ReLU hidden units and Softmax classification head.

---

## 1. Intended Use & Clinical Scope

### Primary Intended Use
- Multi-class clinical phenotype risk stratification across $C = 6$ diagnostic categories from standardized electronic health record (EHR) vitals, metabolic panels, and genomic loci ($d = 28$ features).
- Auditing **Membership Inference Attack (MIA)** vulnerability and evaluating privacy-preserving training mechanisms ($L_2$ Weight Decay, Dropout + Confidence Smoothing, and **DP-SGD**).

### Out-of-Scope Uses
- Unprotected deployment of the **Baseline (No Defense)** variant on real patient cohorts without Differential Privacy or output confidence masking.
- Automated clinical decision-making without human clinician oversight.

---

## 2. Architecture & Training Configurations

- **Input Layer:** $d = 28$ standardized continuous clinical/genomic biomarkers.
- **Hidden Layer:** $H = 64$ units with ReLU activation $\phi(z) = \max(0, z)$ and optional inverted Dropout ($p \in \{0.0, 0.35\}$).
- **Output Layer:** $C = 6$ logits mapped via temperature-calibrated Softmax $P(y=c \mid x) = \frac{e^{z_c}}{\sum_{j} e^{z_j}}$.
- **Optimization:** Mini-batch Stochastic Gradient Descent ($B = 30$, $E = 38$ epochs) with learning rate decay.

### Evaluated Privacy Defense Variants

| Variant ID | Defense Mechanism | Key Hyperparameters | Formal Privacy Guarantee |
| :--- | :--- | :--- | :--- |
| `baseline` | None (Standard Empirical Risk Minimization) | $\lambda = 0, p_{\text{drop}} = 0$ | None ($\epsilon = \infty$) |
| `l2_moderate` | Moderate $L_2$ Weight Decay | $\lambda = 8 \times 10^{-3}$ | Empirical heuristic ($\epsilon = \infty$) |
| `l2_strong` | Strong $L_2$ Weight Decay | $\lambda = 2.5 \times 10^{-2}$ | Empirical heuristic ($\epsilon = \infty$) |
| `dropout_smooth` | Inverted Dropout + Label Smoothing | $p = 0.35, \alpha = 0.10, \lambda = 6 \times 10^{-3}$ | Empirical heuristic ($\epsilon = \infty$) |
| `dpsgd_moderate` | Differentially Private SGD (Moderate) | Clip $C = 1.1$, Noise $\sigma = 0.75$ | $(\epsilon = 5.39, \delta = 10^{-5})$-DP |
| `dpsgd_strict` | Differentially Private SGD (Strict) | Clip $C = 0.85$, Noise $\sigma = 1.38$ | $(\epsilon = 2.13, \delta = 10^{-5})$-DP |

---

## 3. Evaluation Factors: Utility, Runtime Cost, and Privacy Leakage

Models are evaluated across three orthogonal dimensions:
1. **Predictive Utility:** Holdout Test Accuracy (%), Cross-Entropy Loss, and Train-Test Generalization Gap ($\text{Acc}_{\text{train}} - \text{Acc}_{\text{test}}$).
2. **Computational Cost:** Wall-clock training time (seconds) and relative runtime overhead multiplier compared to the unregularized baseline.
3. **Membership Privacy Leakage:**
   - **Loss-Threshold MIA AUC** (Yeom et al., 2018)
   - **Confidence-Entropy MIA AUC** (Salem et al., 2019)
   - **Shadow-Model Supervised MIA AUC, Accuracy, Attacker Advantage, and TPR @ 1% FPR** (Shokri et al., 2017; Carlini et al., 2022)
   - **Subgroup Vulnerability Disparity:** MIA AUC on rare/atypical clinical records (`rare_subgroup = 1`) versus majority patient records (`rare_subgroup = 0`).

---

## 4. Limitations, Risks, and Recommendations

- **Memorization of Rare Patients:** In the unregularized `baseline` model, rare clinical phenotypes experience significantly higher membership leakage than majority patients due to larger gradient norms and outlier memorization.
- **Regularization vs. Formal Privacy:** While `dropout_smooth` and `l2_strong` substantially reduce empirical MIA AUC (and even improve test generalization by mitigating overfitting), they do **not** provide a worst-case mathematical bound against adaptive white-box attacks.
- **Deployment Recommendation:** For high-sensitivity clinical registries (oncology, neuro-genetics, psychiatric cohorts), deploy **`dpsgd_moderate`** or **`dpsgd_strict`** combined with API-level confidence rounding (returning only top-1 class label or rounded probabilities).
