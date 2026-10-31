#!/usr/bin/env python3
"""
End-to-End Reproducible Pipeline for Membership Inference and Model Leakage.

Executes:
  1. Sensitive Clinical & Genomic Dataset Synthesis & Partitioning (CliniLeak-4K)
  2. Shadow Model Training & Attack Calibration
  3. Target Model Training across 6 Privacy Defense Configurations
     (Baseline, L2 Moderate, L2 Strong, Dropout+Smoothing, DP-SGD Moderate, DP-SGD Strict)
  4. Membership Inference Attack Suite (Loss-Threshold, Confidence-Entropy, Shadow Supervised)
  5. Subgroup Disparity Audit (Rare/Minority Clinical Patients vs. Majority Population)
  6. Results Tables (CSV, JSON, Markdown) and Vector Figures (SVG)

Team Members:
  - Ramitha Nayak MV (PES1UG24AM392)
  - Shreya N Hegde (PES1UG24AM401)
"""

import os
import csv
import json
import time
from typing import List, Dict, Any

from src.dataset import generate_clinical_dataset
from src.models import (
    ClinicalMLPClassifier,
    DefenseConfig,
    get_defense_configurations,
)
from src.attacks import evaluate_all_attacks
from src.visualize import generate_all_figures


SHORT_LABELS = {
    "baseline": "Baseline",
    "l2_moderate": "L2 (Mod)",
    "l2_strong": "L2 (Strong)",
    "dropout_smooth": "Dropout+LS",
    "dpsgd_moderate": "DP-SGD (e=4)",
    "dpsgd_strict": "DP-SGD (e=1)",
}


def assess_risk_level(auc: float, tpr_1pct: float) -> str:
    if auc >= 0.66 or tpr_1pct >= 0.06:
        return "CRITICAL / HIGH"
    elif auc >= 0.59 or tpr_1pct >= 0.035:
        return "MODERATE"
    elif auc >= 0.54:
        return "LOW-MODERATE"
    return "LOW (Acceptable)"


def main() -> None:
    print("=" * 88)
    print("  MEMBERSHIP INFERENCE & MODEL LEAKAGE EVALUATION PIPELINE")
    print("  Team: Ramitha Nayak MV (PES1UG24AM392) & Shreya N Hegde (PES1UG24AM401)")
    print("=" * 88)

    t_start = time.perf_counter()

    # Step 1: Generate / Load Dataset
    print("\n[1/5] Generating CliniLeak-4K Sensitive Clinical & Genomic Dataset...")
    data = generate_clinical_dataset(
        n_samples=2400,
        n_features=28,
        n_classes=6,
        seed=42,
        save_dir="data",
    )
    n_feat = len(data.feature_names)
    print(
        f"      -> Loaded {len(data.X_target_train)*4} patient records ({n_feat} features, {data.num_classes} diagnostic classes)"
    )
    print(
        f"      -> Splits: Target Train={len(data.X_target_train)}, Target Test={len(data.X_target_test)}, "
        f"Shadow Train={len(data.X_shadow_train)}, Shadow Test={len(data.X_shadow_test)}"
    )

    # Step 2: Train Shadow Model (used by attacker to calibrate thresholds & train Shadow Attack)
    print("\n[2/5] Training Attacker Shadow Model on disjoint Shadow Dataset...")
    shadow_cfg = DefenseConfig(
        name="Shadow Model",
        short_id="shadow",
        l2_lambda=0.0,
        dropout_rate=0.0,
        epochs=38,
        batch_size=30,
        learning_rate=0.14,
        hidden_dim=64,
    )
    shadow_model = ClinicalMLPClassifier(
        input_dim=n_feat, hidden_dim=shadow_cfg.hidden_dim, num_classes=data.num_classes, seed=77
    )
    shadow_res = shadow_model.fit(
        data.X_shadow_train,
        data.y_shadow_train,
        data.X_shadow_test,
        data.y_shadow_test,
        shadow_cfg,
    )
    shadow_in_signals = shadow_model.extract_record_signals(
        data.X_shadow_train, data.y_shadow_train
    )
    shadow_out_signals = shadow_model.extract_record_signals(
        data.X_shadow_test, data.y_shadow_test
    )
    print(
        f"      -> Shadow Model Train Acc: {shadow_res.train_acc*100:.2f}% | "
        f"Test Acc: {shadow_res.test_acc*100:.2f}% | Time: {shadow_res.training_time_sec:.2f}s"
    )

    # Step 3: Train Target Models with Defenses & Run MIA Suite
    print("\n[3/5] Evaluating Target Model Defenses & Running Membership Inference Attacks...")
    configs = get_defense_configurations()
    baseline_time = None
    baseline_test_acc = None

    summary_rows: List[Dict[str, Any]] = []
    detailed_attack_rows: List[Dict[str, Any]] = []

    for idx, cfg in enumerate(configs):
        target_model = ClinicalMLPClassifier(
            input_dim=n_feat, hidden_dim=cfg.hidden_dim, num_classes=data.num_classes, seed=42 + idx
        )
        train_res = target_model.fit(
            data.X_target_train,
            data.y_target_train,
            data.X_target_test,
            data.y_target_test,
            cfg,
            baseline_time=baseline_time,
        )
        if baseline_time is None:
            baseline_time = train_res.training_time_sec
            train_res.time_overhead_ratio = 1.0
            baseline_test_acc = train_res.test_acc

        target_in_signals = target_model.extract_record_signals(
            data.X_target_train, data.y_target_train
        )
        target_out_signals = target_model.extract_record_signals(
            data.X_target_test, data.y_target_test
        )

        attacks = evaluate_all_attacks(
            target_in_signals=target_in_signals,
            target_out_signals=target_out_signals,
            shadow_in_signals=shadow_in_signals,
            shadow_out_signals=shadow_out_signals,
            rare_target_in=data.rare_target_train,
            rare_target_out=data.rare_target_test,
        )

        sh_atk = attacks["shadow_supervised"]
        loss_atk = attacks["loss_threshold"]
        ent_atk = attacks["confidence_entropy"]

        acc_drop_pp = (baseline_test_acc - train_res.test_acc) * 100.0
        eps_str = f"{train_res.achieved_epsilon:.2f}" if train_res.achieved_epsilon is not None else "Inf"
        risk_tier = assess_risk_level(sh_atk.auc_roc, sh_atk.tpr_at_1pct_fpr)

        print(
            f"      [{idx+1}/{len(configs)}] {cfg.name:<46} | "
            f"Train: {train_res.train_acc*100:5.1f}% | Test: {train_res.test_acc*100:5.1f}% | "
            f"MIA AUC: {sh_atk.auc_roc:.3f} | Time: {train_res.training_time_sec:.2f}s ({train_res.time_overhead_ratio:.2f}x)"
        )

        row = {
            "defense_name": cfg.name,
            "short_id": cfg.short_id,
            "short_label": SHORT_LABELS.get(cfg.short_id, cfg.short_id),
            "epsilon": eps_str,
            "train_acc": round(train_res.train_acc, 4),
            "test_acc": round(train_res.test_acc, 4),
            "generalization_gap": round(train_res.generalization_gap, 4),
            "accuracy_cost_pp": round(acc_drop_pp, 2),
            "train_time_sec": round(train_res.training_time_sec, 3),
            "time_overhead_ratio": round(train_res.time_overhead_ratio, 2),
            "member_loss": round(sh_atk.member_mean_loss, 4),
            "nonmember_loss": round(sh_atk.nonmember_mean_loss, 4),
            "loss_attack_auc": loss_atk.auc_roc,
            "entropy_attack_auc": ent_atk.auc_roc,
            "shadow_auc": sh_atk.auc_roc,
            "shadow_acc": sh_atk.attack_accuracy,
            "shadow_adv": sh_atk.attacker_advantage,
            "tpr_at_1pct_fpr": sh_atk.tpr_at_1pct_fpr,
            "tpr_at_5pct_fpr": sh_atk.tpr_at_5pct_fpr,
            "rare_subgroup_auc": sh_atk.rare_subgroup_auc,
            "majority_subgroup_auc": sh_atk.majority_subgroup_auc,
            "risk_level": risk_tier,
            "roc_fpr": sh_atk.roc_curve_fpr,
            "roc_tpr": sh_atk.roc_curve_tpr,
            "epoch_history": train_res.epoch_history,
        }
        summary_rows.append(row)

        for atk_key, atk_obj in attacks.items():
            detailed_attack_rows.append(
                {
                    "defense_name": cfg.name,
                    "short_id": cfg.short_id,
                    "attack_type": atk_obj.attack_name,
                    "auc_roc": atk_obj.auc_roc,
                    "attack_accuracy": atk_obj.attack_accuracy,
                    "attacker_advantage": atk_obj.attacker_advantage,
                    "tpr_at_1pct_fpr": atk_obj.tpr_at_1pct_fpr,
                    "tpr_at_5pct_fpr": atk_obj.tpr_at_5pct_fpr,
                    "precision": atk_obj.precision,
                    "recall": atk_obj.recall,
                    "f1_score": atk_obj.f1_score,
                    "rare_subgroup_auc": atk_obj.rare_subgroup_auc,
                    "majority_subgroup_auc": atk_obj.majority_subgroup_auc,
                }
            )

    # Step 4: Save Tables & JSON Artifacts
    print("\n[4/5] Exporting Results Tables (CSV, JSON, Markdown)...")
    os.makedirs("results", exist_ok=True)

    csv_fields = [
        "defense_name",
        "epsilon",
        "train_acc",
        "test_acc",
        "generalization_gap",
        "accuracy_cost_pp",
        "train_time_sec",
        "time_overhead_ratio",
        "loss_attack_auc",
        "entropy_attack_auc",
        "shadow_auc",
        "shadow_acc",
        "shadow_adv",
        "tpr_at_1pct_fpr",
        "tpr_at_5pct_fpr",
        "rare_subgroup_auc",
        "majority_subgroup_auc",
        "risk_level",
    ]
    with open("results/defense_evaluation_table.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for r in summary_rows:
            writer.writerow({k: r[k] for k in csv_fields})

    atk_fields = list(detailed_attack_rows[0].keys())
    with open("results/attack_comparison_table.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=atk_fields)
        writer.writeheader()
        writer.writerows(detailed_attack_rows)

    with open("results/full_results.json", "w") as f:
        json.dump(
            {
                "team": [
                    {"name": "Ramitha Nayak MV", "srn": "PES1UG24AM392"},
                    {"name": "Shreya N Hegde", "srn": "PES1UG24AM401"},
                ],
                "summary": summary_rows,
                "detailed_attacks": detailed_attack_rows,
            },
            f,
            indent=2,
        )

    _write_markdown_tables(summary_rows, detailed_attack_rows, "results/RESULTS_TABLES.md")

    # Step 5: Generate Figures
    print("\n[5/5] Generating Vector SVG Figures...")
    fig_paths = generate_all_figures(summary_rows, out_dir="results/figures")
    for fp in fig_paths:
        print(f"      -> Saved {fp}")

    total_time = time.perf_counter() - t_start
    print("\n" + "=" * 88)
    print(f"  PIPELINE COMPLETED SUCCESSFULLY IN {total_time:.2f} SECONDS")
    print("=" * 88)


def _write_markdown_tables(
    summary_rows: List[Dict[str, Any]],
    detailed_rows: List[Dict[str, Any]],
    filepath: str,
) -> None:
    lines = [
        "# Experimental Results Tables: Membership Inference & Defense Evaluation",
        "",
        "**Team Members:** Ramitha Nayak MV (`PES1UG24AM392`) & Shreya N Hegde (`PES1UG24AM401`)",
        "",
        "## Table 1: Utility, Runtime Cost, and Privacy Leakage Across Defenses",
        "",
        "| Defense Mechanism | Privacy ($\\epsilon$) | Train Acc (%) | Test Acc (%) | Gen. Gap (%) | Acc. Cost (pp) | Train Time (s) | Time Cost | Shadow MIA AUC | MIA Adv. | TPR @ 1% FPR | Risk Assessment |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in summary_rows:
        lines.append(
            f"| **{r['defense_name']}** | `{r['epsilon']}` | {r['train_acc']*100:.2f}% | {r['test_acc']*100:.2f}% | "
            f"{r['generalization_gap']*100:.2f}% | {r['accuracy_cost_pp']:+.2f} pp | {r['train_time_sec']:.2f}s | "
            f"{r['time_overhead_ratio']:.2f}x | **{r['shadow_auc']:.4f}** | {r['shadow_adv']:.4f} | "
            f"{r['tpr_at_1pct_fpr']*100:.2f}% | **{r['risk_level']}** |"
        )

    lines.extend(
        [
            "",
            "## Table 2: Comparison of Attack Methodologies Across Defenses",
            "",
            "| Defense Configuration | Loss-Threshold AUC | Confidence-Entropy AUC | Shadow-Supervised AUC | Shadow Attack Acc (%) | Shadow F1-Score |",
            "| :--- | :---: | :---: | :---: | :---: | :---: |",
        ]
    )
    for r in summary_rows:
        lines.append(
            f"| **{r['defense_name']}** | {r['loss_attack_auc']:.4f} | {r['entropy_attack_auc']:.4f} | "
            f"**{r['shadow_auc']:.4f}** | {r['shadow_acc']*100:.2f}% | "
            f"{next(d['f1_score'] for d in detailed_rows if d['short_id']==r['short_id'] and 'Shadow' in d['attack_type']):.4f} |"
        )

    lines.extend(
        [
            "",
            "## Table 3: Subgroup Disparity & Loss Separation Analysis",
            "",
            "| Defense Configuration | Member Mean Loss | Non-Member Mean Loss | Loss Ratio (Out/In) | Rare Subgroup MIA AUC | Majority Subgroup MIA AUC | Subgroup Disparity ($\\Delta$ AUC) |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
    )
    for r in summary_rows:
        ratio = r["nonmember_loss"] / max(r["member_loss"], 1e-6)
        delta_auc = r["rare_subgroup_auc"] - r["majority_subgroup_auc"]
        lines.append(
            f"| **{r['defense_name']}** | {r['member_loss']:.4f} | {r['nonmember_loss']:.4f} | "
            f"{ratio:.2f}x | **{r['rare_subgroup_auc']:.4f}** | {r['majority_subgroup_auc']:.4f} | {delta_auc:+.4f} |"
        )

    with open(filepath, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
