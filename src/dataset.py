"""
Dataset module for Membership Inference and Model Leakage Case Study.
Generates and persists a realistic Sensitive Clinical & Genomic Patient Records
dataset (CliniLeak-4K: 4,000 patient records, 32 clinical/genomic biomarkers,
6 diagnostic risk classes) for evaluating target models, shadow models, and
membership inference attacks.

Authors:
  - Ramitha Nayak MV (PES1UG24AM392)
  - Shreya N Hegde (PES1UG24AM401)
"""

import os
import csv
import json
import math
import random
from dataclasses import dataclass
from typing import List


@dataclass
class DataSplit:
    X_target_train: List[List[float]]
    y_target_train: List[int]
    rare_target_train: List[int]
    X_target_test: List[List[float]]
    y_target_test: List[int]
    rare_target_test: List[int]
    X_shadow_train: List[List[float]]
    y_shadow_train: List[int]
    X_shadow_test: List[List[float]]
    y_shadow_test: List[int]
    feature_names: List[str]
    num_classes: int


def generate_clinical_dataset(
    n_samples: int = 4000,
    n_features: int = 32,
    n_classes: int = 6,
    seed: int = 42,
    save_dir: str = "data",
) -> DataSplit:
    """
    Synthesizes a high-dimensional sensitive healthcare dataset representing patient
    electronic health records (EHR), lab panels, and genomic risk markers across
    6 clinical phenotypes:
      0: Normal / Low-Risk Baseline
      1: Type-2 Metabolic Syndrome
      2: Chronic Cardiovascular Arrhythmia
      3: Autoimmune Inflammatory Disorder
      4: Early-Stage Oncological Biomarker Positive
      5: Rare Neuro-Genetic Syndrome (High-Sensitivity Subgroup)
    """
    rng = random.Random(seed)

    # Class prototypes in feature space (realistic overlapping clinical phenotypes)
    prototypes = [
        [rng.gauss(0.0, 0.52) for _ in range(n_features)]
        for _ in range(n_classes)
    ]

    # Realistic clinical prevalence distribution
    class_weights = [0.24, 0.22, 0.19, 0.16, 0.11, 0.08]
    cum_weights = []
    acc = 0.0
    for w in class_weights:
        acc += w
        cum_weights.append(acc)

    def sample_class() -> int:
        r = rng.random()
        for idx, cw in enumerate(cum_weights):
            if r <= cw:
                return idx
        return n_classes - 1

    # Latent correlation matrix (simulates physiological coupling across biomarkers)
    n_latent = 8
    cov_proj = [
        [rng.gauss(0.0, 0.35) for _ in range(n_latent)]
        for _ in range(n_features)
    ]

    X_raw = []
    y_all = []
    rare_all = []

    for _ in range(n_samples):
        c = sample_class()
        is_rare = 1 if (c == n_classes - 1 or rng.random() < 0.14) else 0
        latent = [rng.gauss(0.0, 1.0) for _ in range(n_latent)]
        noise_scale = 1.55 if is_rare else 1.28

        row = []
        proto = prototypes[c]
        for j in range(n_features):
            corr = sum(cov_proj[j][k] * latent[k] for k in range(n_latent))
            val = proto[j] + corr + rng.gauss(0.0, noise_scale)
            # Rare/atypical patients exhibit sharper biomarker spikes (memorization targets)
            if is_rare and j >= n_features - 10:
                val += rng.gauss(0.65 if (j % 2 == 0) else -0.65, 1.15)
            row.append(val)

        # In real clinical cohorts, ~12% of atypical/borderline cases have noisy diagnostic labels
        if is_rare and rng.random() < 0.28:
            c = (c + rng.randint(1, n_classes - 1)) % n_classes

        X_raw.append(row)
        y_all.append(c)
        rare_all.append(is_rare)

    # Z-score standardization across features
    means = [0.0] * n_features
    for row in X_raw:
        for j in range(n_features):
            means[j] += row[j]
    means = [m / n_samples for m in means]

    stds = [0.0] * n_features
    for row in X_raw:
        for j in range(n_features):
            diff = row[j] - means[j]
            stds[j] += diff * diff
    stds = [math.sqrt(s / n_samples) + 1e-8 for s in stds]

    X = [
        [round((row[j] - means[j]) / stds[j], 5) for j in range(n_features)]
        for row in X_raw
    ]

    # Split into 4 disjoint subsets of equal size (q = n_samples // 4)
    q = n_samples // 4
    X_target_train = X[:q]
    y_target_train = y_all[:q]
    rare_target_train = rare_all[:q]

    X_target_test = X[q : 2 * q]
    y_target_test = y_all[q : 2 * q]
    rare_target_test = rare_all[q : 2 * q]

    X_shadow_train = X[2 * q : 3 * q]
    y_shadow_train = y_all[2 * q : 3 * q]

    X_shadow_test = X[3 * q : 4 * q]
    y_shadow_test = y_all[3 * q : 4 * q]

    feature_names = (
        [f"vitals_biomarker_{i+1:02d}" for i in range(10)]
        + [f"metabolic_panel_{i+1:02d}" for i in range(12)]
        + [f"genomic_locus_{i+1:02d}" for i in range(n_features - 22)]
    )

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        csv_path = os.path.join(save_dir, "clinical_patient_records_sample.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["record_id", "split", "is_member", "rare_subgroup", "diagnosis_class"]
                + feature_names[:10]
            )
            # Save representative 400 rows in CSV for easy inspection
            for idx in range(min(200, q)):
                writer.writerow(
                    [f"PAT-T-IN-{idx:04d}", "target_train", 1, rare_target_train[idx], y_target_train[idx]]
                    + X_target_train[idx][:10]
                )
            for idx in range(min(200, q)):
                writer.writerow(
                    [f"PAT-T-OUT-{idx:04d}", "target_test", 0, rare_target_test[idx], y_target_test[idx]]
                    + X_target_test[idx][:10]
                )

        metadata = {
            "dataset_name": "CliniLeak-4K Sensitive Clinical & Genomic EHR Dataset",
            "authors": [
                "Ramitha Nayak MV (PES1UG24AM392)",
                "Shreya N Hegde (PES1UG24AM401)",
            ],
            "total_records": n_samples,
            "num_features": n_features,
            "num_classes": n_classes,
            "class_names": [
                "0: Normal / Low-Risk Baseline",
                "1: Type-2 Metabolic Syndrome",
                "2: Chronic Cardiovascular Arrhythmia",
                "3: Autoimmune Inflammatory Disorder",
                "4: Early-Stage Oncological Biomarker Positive",
                "5: Rare Neuro-Genetic Syndrome",
            ],
            "split_sizes": {
                "target_train_members": len(X_target_train),
                "target_test_non_members": len(X_target_test),
                "shadow_train_members": len(X_shadow_train),
                "shadow_test_non_members": len(X_shadow_test),
            },
            "rare_subpopulation_ratio": round(sum(rare_all) / len(rare_all), 4),
            "seed": seed,
        }
        with open(os.path.join(save_dir, "dataset_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

    return DataSplit(
        X_target_train=X_target_train,
        y_target_train=y_target_train,
        rare_target_train=rare_target_train,
        X_target_test=X_target_test,
        y_target_test=y_target_test,
        rare_target_test=rare_target_test,
        X_shadow_train=X_shadow_train,
        y_shadow_train=y_shadow_train,
        X_shadow_test=X_shadow_test,
        y_shadow_test=y_shadow_test,
        feature_names=feature_names,
        num_classes=n_classes,
    )
