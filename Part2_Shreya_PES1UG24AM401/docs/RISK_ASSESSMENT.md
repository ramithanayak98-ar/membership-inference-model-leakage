# Comprehensive Privacy & Security Risk Assessment: Membership Inference on Clinical ML Models

- **Project Title:** Membership Inference and Model Leakage
- **Team Members:**
  - **Ramitha Nayak MV** (`PES1UG24AM392`)
  - **Shreya N Hegde** (`PES1UG24AM401`)
- **Course:** Big Data Systems and Responsible AI Engineering (PES University)

---

## 1. Executive Threat Summary

When a healthcare institution trains a predictive model on sensitive Electronic Health Records (EHR) and genomic biomarker panels, the trained weights $\theta^*$ implicitly encode information about individual training records. In a **Membership Inference Attack (MIA)**, an adversary with black-box query access to the deployed model $f_{\theta^*}(x)$ attempts to determine whether a target patient record $(x, y)$ belonged to the private training cohort $D_{\text{target}}^{\text{in}}$.

**Why Membership Equals Sensitive Attribute Disclosure:**
In clinical deployments, confirming that an individual was part of a specialized hospital training cohort (e.g., an oncology biomarker trial, psychiatric registry, or rare neuro-genetic disorder cohort) directly reveals their confidential clinical status—even if the model is never queried for that attribute directly.

---

## 2. Threat Model & Adversary Capabilities

| Threat Dimension | Specification in Our Evaluation |
| :--- | :--- |
| **Adversary Access Level** | **Black-Box Posterior Access:** Adversary submits feature vector $x$ and observes output probability vector $P(y \mid x)$ (and per-record cross-entropy loss $\ell(f(x), y)$). |
| **Auxiliary Knowledge** | **Shadow Dataset ($D_{\text{shadow}}$):** Adversary possesses a disjoint dataset sampled from the same underlying clinical population distribution to train a Shadow Model $f_{\text{shadow}}$ and calibrate attack decision boundaries. |
| **Attack Vectors Evaluated** | 1. **Loss-Threshold Attack** (Yeom et al., 2018)<br>2. **Confidence & Modified Entropy Attack** (Salem et al., 2019; Song & Mittal, 2021)<br>3. **Supervised Shadow-Model Attack Classifier** (Shokri et al., 2017) |
| **Primary Harm Metrics** | Attack AUC-ROC, Attacker Advantage ($\max(\text{TPR} - \text{FPR})$), **TPR at Low FPR (1% and 5%)** (Carlini et al., 2022), and **Rare Subgroup Disparity ($\Delta\text{AUC}$)**. |

---

## 3. Quantitative Risk Matrix Across Defense Mechanisms

Based on our empirical evaluation on the **CliniLeak** dataset ($N = 2,400$ records, $d = 28$ biomarkers, $C = 6$ phenotypes), we classify each model configuration into a formal Risk Tier:

| Model Configuration | Privacy Budget ($\epsilon$) | Shadow MIA AUC | Attacker Advantage | Rare Subgroup AUC | Utility Cost (Test Acc $\Delta$) | Runtime Cost Multiplier | Overall Risk Classification |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Baseline (No Defense)** | $\infty$ | **0.7390** | 0.4083 | **0.8832** | Baseline (56.17%) | 1.00x (4.74s) | **CRITICAL / UNACCEPTABLE** |
| **2. L2 Regularization (Moderate)** | $\infty$ | **0.6611** | 0.2917 | **0.7927** | +3.17 pp (Gain) | 1.00x (4.76s) | **HIGH RISK** |
| **3. L2 Regularization (Strong)** | $\infty$ | **0.5974** | 0.1967 | **0.6947** | +4.17 pp (Gain) | 1.02x (4.84s) | **MODERATE RISK** |
| **4. Dropout + Confidence Smoothing** | $\infty$ | **0.5839** | 0.1733 | **0.6778** | +3.00 pp (Gain) | 0.82x (3.89s) | **LOW-MODERATE RISK** |
| **5. DP-SGD (Moderate Privacy)** | $\epsilon = 5.30$ | **0.5637** | 0.1100 | **0.5824** | -7.17 pp (Cost) | 1.32x (6.26s) | **LOW RISK (Controlled)** |
| **6. DP-SGD (Strict Privacy)** | $\epsilon = 2.86$ | **0.5445** | 0.0800 | **0.5685** | -8.83 pp (Cost) | 1.33x (6.30s) | **LOW RISK (Regulatory Grade)** |

---

## 4. Key Vulnerability Findings

### Finding 1: Severe Overfitting Drives Membership Leakage in Unprotected Models
The unregularized Baseline model achieves **95.67%** training accuracy versus **56.17%** test accuracy (a **39.50%** generalization gap). Consequently, the mean cross-entropy loss on training members (`0.2486`) is **6.78x lower** than on non-members (`1.6844`). The supervised Shadow Attack exploits this gap to achieve **0.7390 AUC** and **70.08% attack accuracy**.

### Finding 2: Disparate Privacy Vulnerability on Rare Clinical Subpopulations
A critical Responsible AI finding of our study is that **privacy risk is not uniformly distributed**:
- In the unprotected Baseline model, patients in the **Rare/Atypical Clinical Subgroup** (`rare_subgroup = 1`, ~18% of cohort) suffer a **0.8832 Membership Inference AUC**, compared to **0.7043 AUC** for the majority population ($\Delta\text{AUC} = +0.1789$).
- Standard heuristic defenses (`L2 Strong` and `Dropout + Smoothing`) reduce overall MIA AUC to `0.5974` and `0.5839`, **but rare patients still experience high leakage (`0.6947` and `0.6778` AUC)** because heuristic regularizers do not bound individual sample influence.
- Only **DP-SGD** bounds the maximum influence of any single outlier patient via per-sample gradient norm clipping ($C \in \{0.70, 0.95\}$), collapsing the subgroup disparity $\Delta\text{AUC}$ from `+0.1789` down to **`+0.0166`** ($\epsilon = 5.30$) and **`+0.0250`** ($\epsilon = 2.86$).

### Finding 3: The Privacy–Utility–Compute Trilemma
- **Heuristic Regularization (`L2` & `Dropout + Confidence Smoothing`):** Provides a "free lunch" in both utility and speed—actually **improving** holdout test accuracy by `+3.00` to `+4.17` percentage points by preventing overfitting, with zero or negative runtime overhead (`0.82x`–`1.02x`). However, it lacks a formal worst-case privacy guarantee.
- **Formal Differential Privacy (`DP-SGD`):** Provides mathematically provable $(\epsilon, \delta)$-differential privacy and eliminates outlier memorization, but incurs a **7.17 to 8.83 percentage point drop in test accuracy** and a **32%–33% increase in training time** due to per-sample gradient norm computation and Gaussian noise generation.

---

## 5. Regulatory Compliance & Deployment Recommendations

1. **HIPAA Safe Harbor / Expert Determination & GDPR Article 9 / India DPDP Act 2023:**
   Models with MIA AUC $> 0.60$ or Rare Subgroup AUC $> 0.65$ pose an unacceptable re-identification risk and must not be exposed via external prediction APIs.
2. **Tiered Defense-in-Depth Architecture for Production Deployment:**
   - **Training Layer:** Train clinical models with **DP-SGD ($\epsilon \le 5.0, \delta = 10^{-5}$)** combined with early stopping and weight decay to bound worst-case per-patient memorization.
   - **Inference API Layer:** Restrict prediction outputs to **Top-1 Class Labels** or **Temperature-Smoothed / Rounded Confidence Scores** (e.g., 1 decimal place), neutralizing confidence-entropy and shadow-supervised attacks that rely on high-precision softmax probabilities.
   - **Continuous Auditing:** Integrate our automated `run_pipeline.py` MIA benchmark into the CI/CD model registry gate, blocking deployment if any subgroup MIA AUC exceeds `0.58`.
