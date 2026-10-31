# 3-Minute Demo Video Script & In-Class Presentation Guide

- **Project:** Membership Inference and Model Leakage
- **Team Members:**
  - **Ramitha Nayak MV** (`PES1UG24AM392`)
  - **Shreya N Hegde** (`PES1UG24AM401`)
- **Total Duration:** 3 Minutes (180 Seconds)

---

## Segment 1: Problem & Threat Model (0:00 – 0:45) — *Ramitha Nayak MV*

- **Screen to Show:** Open [`README.md`](../README.md) showing the **System & Threat Model Architecture Diagram**.
- **Script:**
  > "Hello everyone, I am Ramitha Nayak MV (`PES1UG24AM392`) and my teammate is Shreya N Hegde (`PES1UG24AM401`). Our project investigates **Membership Inference Attacks and Model Leakage** on sensitive clinical records.
  >
  > When a hospital trains a deep neural network on Electronic Health Records and genomic biomarkers, the model often memorizes training records. In a Membership Inference Attack, an adversary queries the trained model's prediction probabilities to determine whether a specific patient was part of the private training cohort—which directly leaks sensitive diagnoses such as oncology or rare genetic syndromes.
  >
  > To evaluate this, we built the **CliniLeak** pipeline with 2,400 patient records across 28 biomarkers and 6 diagnostic phenotypes, split into four strictly disjoint sets for our Target Model and the attacker's Shadow Model."

---

## Segment 2: Live Pipeline Execution & Attack Suite (0:45 – 1:45) — *Ramitha & Shreya*

- **Screen to Show:** Terminal running `./run_pipeline.sh` (or `python3 run_pipeline.py`) + [`notebooks/membership_inference_results.ipynb`](../notebooks/membership_inference_results.ipynb).
- **Script:**
  > "Here we execute our reproducible pipeline. First, the adversary trains a Shadow MLP on disjoint shadow data to learn how member and non-member output distributions differ.
  >
  > Next, we run three black-box attacks against our Target Model: the **Loss-Threshold Attack**, the **Confidence-Entropy Attack**, and **Shokri et al.'s Supervised Shadow-Model Attack**.
  >
  > On the unprotected **Baseline** model, training accuracy reaches **95.7%** while test accuracy is **56.2%**—a **39.5% generalization gap**. Because member loss is **6.78x lower** than non-member loss, our Shadow Attack achieves a critical **0.7390 AUC** overall, and an alarming **0.8832 AUC** on patients in the **Rare Clinical Subgroup**."

---

## Segment 3: Defense Evaluation, Cost Trade-offs & Risk Assessment (1:45 – 3:00) — *Shreya N Hegde*

- **Screen to Show:** Scroll through the 4 SVG plots in [`notebooks/membership_inference_results.ipynb`](../notebooks/membership_inference_results.ipynb) and [`report/technical_report.pdf`](../report/technical_report.pdf).
- **Script:**
  > "To mitigate this leakage, we evaluated five defenses and measured their exact cost in test accuracy and training time:
  >
  > 1. **Heuristic Regularization ($L_2$ Weight Decay and Inverted Dropout + Confidence Smoothing):** Rather than hurting utility, these regularizers prevent overfitting and actually **improve test accuracy by +3.0 to +4.2 percentage points** with zero or negative runtime overhead (`0.82x` to `1.02x`). They reduce overall MIA AUC from `0.7390` down to `0.5839`. However, our subgroup audit reveals that rare patients still remain vulnerable at **`0.6778` AUC**.
  > 2. **Differentially Private SGD (DP-SGD):** By clipping per-sample gradient norms ($C=0.70$) and injecting calibrated Gaussian noise ($\sigma=2.45$), Strict DP-SGD achieves formal $(\epsilon = 2.86, \delta = 10^{-5})$-Differential Privacy. It reduces overall MIA AUC to **`0.5445`** and collapses rare subgroup disparity $\Delta\text{AUC}$ from `+0.1789` down to **`+0.0250`**. This formal privacy comes at a quantified cost of **8.83 percentage points in test accuracy** and a **1.33x training time overhead** due to per-sample gradient clipping.
  >
  > In our Risk Assessment, we conclude that deploying unregularized clinical models poses a **Critical Risk**, and we recommend a defense-in-depth combination of **DP-SGD** during training and **confidence rounding** at the API layer. Thank you!"
