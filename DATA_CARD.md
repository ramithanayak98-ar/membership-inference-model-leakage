# Data Card: CliniLeak Sensitive Clinical & Genomic EHR Benchmark Dataset

- **Dataset Name:** CliniLeak Sensitive Clinical & Genomic Patient Records Benchmark
- **Version:** 1.0.0
- **Authors / Curators:**
  - **Ramitha Nayak MV** (`PES1UG24AM392`)
  - **Shreya N Hegde** (`PES1UG24AM401`)
- **Course:** Big Data Systems and Responsible AI Engineering (PES University)
- **License:** MIT / Academic Research Use Only

---

## 1. Dataset Summary & Motivation

Machine learning models trained on Electronic Health Records (EHR) and genomic biomarker panels carry severe privacy risks if an adversary can infer whether a specific individual's record was present in the training cohort (**Membership Inference Attack**). Being identified as a member of a clinical hospital cohort (e.g., an oncology or rare neuro-genetic registry) directly leaks sensitive personal health information (PHI) protected under HIPAA, GDPR, and India's DPDP Act 2023.

The **CliniLeak** dataset is a reproducible, privacy-safe synthetic clinical and genomic dataset designed to benchmark:
1. **Membership Inference Vulnerability** across metric-based and shadow-model supervised attacks.
2. **Subgroup Privacy Disparity** between majority patient populations and rare/atypical clinical phenotypes.
3. **Defense Mechanisms** including $L_2$ weight regularization, Dropout with Confidence Smoothing, and Differentially Private Stochastic Gradient Descent (**DP-SGD**).

---

## 2. Dataset Composition & Structure

| Attribute | Specification |
| :--- | :--- |
| **Total Records ($N$)** | $2,400$ patient records (configurable up to $20,000+$ via `--n-samples`) |
| **Feature Dimensionality ($d$)** | $28$ standardized continuous clinical & genomic features |
| **Feature Modalities** | `vitals_biomarker_01..10` (10), `metabolic_panel_01..12` (12), `genomic_locus_01..06` (6) |
| **Target Classes ($C$)** | $6$ mutually exclusive clinical diagnostic phenotypes |
| **Sensitive Subgroup Flag** | `rare_subgroup` $\in \{0, 1\}$ (~18% prevalence representing rare/atypical patients) |
| **Missing Values** | None (Z-score standardized, zero mean, unit variance) |

### Diagnostic Class Distribution

| Class ID | Clinical Phenotype Description | Population Prevalence | Privacy Sensitivity Tier |
| :---: | :--- | :---: | :---: |
| `0` | Normal / Low-Risk Baseline | 24.0% | Low |
| `1` | Type-2 Metabolic Syndrome | 22.0% | Moderate |
| `2` | Chronic Cardiovascular Arrhythmia | 19.0% | Moderate |
| `3` | Autoimmune Inflammatory Disorder | 16.0% | High |
| `4` | Early-Stage Oncological Biomarker Positive | 11.0% | Critical |
| `5` | Rare Neuro-Genetic Syndrome | 8.0% | Critical (Minority Subgroup) |

---

## 3. Experimental Partitioning Protocol

To ensure zero data leakage between the target institution and the adversary's shadow environment, the dataset is partitioned into **four strictly disjoint splits** of equal cardinality:

1. **`target_train` (Target Member Set $D_{\text{target}}^{\text{in}}$):** Used exclusively to train the Target Clinical Model under each defense configuration. Ground-truth membership label $m = 1$.
2. **`target_test` (Target Non-Member Holdout Set $D_{\text{target}}^{\text{out}}$):** Unseen patients from the same underlying population used to evaluate target test accuracy and non-member behavior. Ground-truth membership label $m = 0$.
3. **`shadow_train` (Shadow Member Set $D_{\text{shadow}}^{\text{in}}$):** Disjoint records used by the adversary to train the Shadow Model $f_{\text{shadow}}$.
4. **`shadow_test` (Shadow Non-Member Set $D_{\text{shadow}}^{\text{out}}$):** Disjoint holdout records queried against $f_{\text{shadow}}$ to calibrate attack thresholds and train the supervised Shadow Attack Classifier.

---

## 4. Ethical Considerations & Responsible AI Notes

- **Zero Real Patient Exposure:** All records in CliniLeak are generated via a parametric latent-factor clinical simulation (`src/dataset.py`) with fixed deterministic random seeds (`seed=42`). No real patient PII/PHI is stored or processed.
- **Disparate Impact Auditing:** Rare condition patients (`rare_subgroup = 1`) exhibit higher feature variance and idiosyncratic genomic markers. As demonstrated in our experiments, unregularized models memorize these outlier records at significantly higher rates than majority records, necessitating formal differential privacy (`DP-SGD`).
