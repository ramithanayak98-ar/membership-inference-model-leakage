"""
Membership Inference Attack Suite:
  1. Loss-Threshold Attack (Yeom et al., 2018)
  2. Confidence & Prediction Entropy Attack (Salem et al., 2019)
  3. Shadow-Model Supervised Attack Classifier (Shokri et al., 2017)
  4. Subgroup Disparity & Low-FPR Vulnerability Evaluator (Carlini et al., 2022)

Authors:
  - Ramitha Nayak MV (PES1UG24AM392)
  - Shreya N Hegde (PES1UG24AM401)
"""

import math
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple


@dataclass
class AttackMetrics:
    attack_name: str
    auc_roc: float
    attack_accuracy: float
    attacker_advantage: float
    tpr_at_1pct_fpr: float
    tpr_at_5pct_fpr: float
    precision: float
    recall: float
    f1_score: float
    member_mean_loss: float
    nonmember_mean_loss: float
    rare_subgroup_auc: float
    majority_subgroup_auc: float
    roc_curve_fpr: List[float]
    roc_curve_tpr: List[float]


def compute_roc_and_metrics(
    scores_member: List[float],
    scores_nonmember: List[float],
    threshold: float,
) -> Tuple[float, float, float, float, float, float, float, float, List[float], List[float]]:
    """
    Computes exact AUC-ROC, Attacker Advantage = max(TPR - FPR), TPR@1%FPR, TPR@5%FPR,
    and thresholded Accuracy, Precision, Recall, F1.
    """
    n_pos = len(scores_member)
    n_neg = len(scores_nonmember)

    combined = [(s, 1) for s in scores_member] + [(s, 0) for s in scores_nonmember]
    combined.sort(key=lambda pair: pair[0], reverse=True)

    tp = 0
    fp = 0
    prev_fpr = 0.0
    prev_tpr = 0.0
    auc = 0.0
    max_adv = 0.0
    tpr_at_1pct = 0.0
    tpr_at_5pct = 0.0

    fpr_list = [0.0]
    tpr_list = [0.0]

    step_stride = max(1, len(combined) // 100)
    for idx, (_, label) in enumerate(combined):
        if label == 1:
            tp += 1
        else:
            fp += 1
        tpr = tp / n_pos
        fpr = fp / n_neg
        auc += (fpr - prev_fpr) * (tpr + prev_tpr) * 0.5
        adv = tpr - fpr
        if adv > max_adv:
            max_adv = adv
        if fpr <= 0.01:
            tpr_at_1pct = max(tpr_at_1pct, tpr)
        if fpr <= 0.05:
            tpr_at_5pct = max(tpr_at_5pct, tpr)
        if idx % step_stride == 0 or idx == len(combined) - 1:
            fpr_list.append(round(fpr, 4))
            tpr_list.append(round(tpr, 4))
        prev_fpr = fpr
        prev_tpr = tpr

    # Evaluate at calibrated threshold
    tp_t = sum(1 for s in scores_member if s >= threshold)
    fp_t = sum(1 for s in scores_nonmember if s >= threshold)
    tn_t = n_neg - fp_t
    fn_t = n_pos - tp_t

    acc = (tp_t + tn_t) / (n_pos + n_neg)
    prec = tp_t / max(1, tp_t + fp_t)
    rec = tp_t / max(1, tp_t + fn_t)
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    return (
        auc,
        acc,
        max_adv,
        tpr_at_1pct,
        tpr_at_5pct,
        prec,
        rec,
        f1,
        fpr_list,
        tpr_list,
    )


def find_optimal_shadow_threshold(
    shadow_member_scores: List[float],
    shadow_nonmember_scores: List[float],
) -> float:
    """Finds the threshold on shadow model scores that maximizes Youden's J (TPR - FPR)."""
    combined = sorted(shadow_member_scores + shadow_nonmember_scores)
    step = max(1, len(combined) // 200)
    candidates = combined[::step]
    best_thresh = candidates[len(candidates) // 2]
    best_acc = -1.0
    n_m = len(shadow_member_scores)
    n_nm = len(shadow_nonmember_scores)
    for t in candidates:
        tpr = sum(1 for s in shadow_member_scores if s >= t) / n_m
        tnr = sum(1 for s in shadow_nonmember_scores if s < t) / n_nm
        bal_acc = 0.5 * (tpr + tnr)
        if bal_acc > best_acc:
            best_acc = bal_acc
            best_thresh = t
    return best_thresh


class ShadowAttackClassifier:
    """
    Supervised Binary Attack Model (Shokri et al., 2017) trained on Shadow Model
    input-output behavioral signals to distinguish members (1) from non-members (0).
    """

    def __init__(self, seed: int = 101):
        self.rng = random.Random(seed)
        self.weights = [0.0] * 6
        self.bias = 0.0
        self.feat_means = [0.0] * 6
        self.feat_stds = [1.0] * 6

    @staticmethod
    def _signal_to_vec(sig: Dict[str, float]) -> List[float]:
        return [
            -sig["loss"],
            sig["confidence"],
            sig["true_prob"],
            -sig["entropy"],
            sig["margin"],
            sig["correct"],
        ]

    def fit(
        self,
        shadow_in_signals: List[Dict[str, float]],
        shadow_out_signals: List[Dict[str, float]],
        epochs: int = 35,
        lr: float = 0.08,
    ) -> None:
        X_raw = [self._signal_to_vec(s) for s in shadow_in_signals] + [
            self._signal_to_vec(s) for s in shadow_out_signals
        ]
        y = [1] * len(shadow_in_signals) + [0] * len(shadow_out_signals)
        n = len(X_raw)
        d = 6

        means = [sum(row[j] for row in X_raw) / n for j in range(d)]
        stds = [
            math.sqrt(sum((row[j] - means[j]) ** 2 for row in X_raw) / n) + 1e-6
            for j in range(d)
        ]
        self.feat_means = means
        self.feat_stds = stds

        X = [[(row[j] - means[j]) / stds[j] for j in range(d)] for row in X_raw]
        self.weights = [self.rng.gauss(0.0, 0.1) for _ in range(d)]
        self.bias = 0.0

        indices = list(range(n))
        for _ in range(epochs):
            self.rng.shuffle(indices)
            for idx in indices:
                x_i = X[idx]
                y_i = y[idx]
                z = self.bias + sum(self.weights[j] * x_i[j] for j in range(d))
                z = max(-20.0, min(20.0, z))
                pred = 1.0 / (1.0 + math.exp(-z))
                err = pred - y_i
                for j in range(d):
                    self.weights[j] -= lr * (err * x_i[j] + 0.002 * self.weights[j])
                self.bias -= lr * err

    def predict_proba(self, signals: List[Dict[str, float]]) -> List[float]:
        probs = []
        for s in signals:
            vec = self._signal_to_vec(s)
            x = [(vec[j] - self.feat_means[j]) / self.feat_stds[j] for j in range(6)]
            z = self.bias + sum(self.weights[j] * x[j] for j in range(6))
            z = max(-20.0, min(20.0, z))
            probs.append(1.0 / (1.0 + math.exp(-z)))
        return probs


def evaluate_all_attacks(
    target_in_signals: List[Dict[str, float]],
    target_out_signals: List[Dict[str, float]],
    shadow_in_signals: List[Dict[str, float]],
    shadow_out_signals: List[Dict[str, float]],
    rare_target_in: List[int],
    rare_target_out: List[int],
) -> Dict[str, AttackMetrics]:
    """
    Runs all 3 membership inference attack families on the Target Model and computes
    overall + subgroup privacy leakage metrics.
    """
    results = {}

    mem_mean_loss = sum(s["loss"] for s in target_in_signals) / len(target_in_signals)
    nonmem_mean_loss = sum(s["loss"] for s in target_out_signals) / len(target_out_signals)

    # 1. Loss-Threshold Attack (Yeom et al., 2018)
    s_in_loss = [-s["loss"] for s in shadow_in_signals]
    s_out_loss = [-s["loss"] for s in shadow_out_signals]
    t_in_loss = [-s["loss"] for s in target_in_signals]
    t_out_loss = [-s["loss"] for s in target_out_signals]
    thresh_loss = find_optimal_shadow_threshold(s_in_loss, s_out_loss)

    # Subgroup scores for Loss Attack
    rare_in_loss = [t_in_loss[i] for i in range(len(t_in_loss)) if rare_target_in[i] == 1]
    rare_out_loss = [t_out_loss[i] for i in range(len(t_out_loss)) if rare_target_out[i] == 1]
    maj_in_loss = [t_in_loss[i] for i in range(len(t_in_loss)) if rare_target_in[i] == 0]
    maj_out_loss = [t_out_loss[i] for i in range(len(t_out_loss)) if rare_target_out[i] == 0]

    auc_l, acc_l, adv_l, tpr1_l, tpr5_l, prec_l, rec_l, f1_l, fpr_l, tpr_l = (
        compute_roc_and_metrics(t_in_loss, t_out_loss, thresh_loss)
    )
    rare_auc_l = compute_roc_and_metrics(rare_in_loss, rare_out_loss, thresh_loss)[0]
    maj_auc_l = compute_roc_and_metrics(maj_in_loss, maj_out_loss, thresh_loss)[0]

    results["loss_threshold"] = AttackMetrics(
        attack_name="Loss-Threshold Attack (Yeom et al.)",
        auc_roc=round(auc_l, 4),
        attack_accuracy=round(acc_l, 4),
        attacker_advantage=round(adv_l, 4),
        tpr_at_1pct_fpr=round(tpr1_l, 4),
        tpr_at_5pct_fpr=round(tpr5_l, 4),
        precision=round(prec_l, 4),
        recall=round(rec_l, 4),
        f1_score=round(f1_l, 4),
        member_mean_loss=round(mem_mean_loss, 4),
        nonmember_mean_loss=round(nonmem_mean_loss, 4),
        rare_subgroup_auc=round(rare_auc_l, 4),
        majority_subgroup_auc=round(maj_auc_l, 4),
        roc_curve_fpr=fpr_l,
        roc_curve_tpr=tpr_l,
    )

    # 2. Confidence & Prediction Entropy Attack (Salem et al., 2019; Song & Mittal, 2021)
    s_in_ent = [0.75 * s["true_prob"] + 0.25 * s["confidence"] - 0.20 * s["entropy"] for s in shadow_in_signals]
    s_out_ent = [0.75 * s["true_prob"] + 0.25 * s["confidence"] - 0.20 * s["entropy"] for s in shadow_out_signals]
    t_in_ent = [0.75 * s["true_prob"] + 0.25 * s["confidence"] - 0.20 * s["entropy"] for s in target_in_signals]
    t_out_ent = [0.75 * s["true_prob"] + 0.25 * s["confidence"] - 0.20 * s["entropy"] for s in target_out_signals]
    thresh_ent = find_optimal_shadow_threshold(s_in_ent, s_out_ent)

    rare_in_ent = [t_in_ent[i] for i in range(len(t_in_ent)) if rare_target_in[i] == 1]
    rare_out_ent = [t_out_ent[i] for i in range(len(t_out_ent)) if rare_target_out[i] == 1]
    maj_in_ent = [t_in_ent[i] for i in range(len(t_in_ent)) if rare_target_in[i] == 0]
    maj_out_ent = [t_out_ent[i] for i in range(len(t_out_ent)) if rare_target_out[i] == 0]

    auc_e, acc_e, adv_e, tpr1_e, tpr5_e, prec_e, rec_e, f1_e, fpr_e, tpr_e = (
        compute_roc_and_metrics(t_in_ent, t_out_ent, thresh_ent)
    )
    rare_auc_e = compute_roc_and_metrics(rare_in_ent, rare_out_ent, thresh_ent)[0]
    maj_auc_e = compute_roc_and_metrics(maj_in_ent, maj_out_ent, thresh_ent)[0]

    results["confidence_entropy"] = AttackMetrics(
        attack_name="Confidence-Entropy Attack (Salem et al.)",
        auc_roc=round(auc_e, 4),
        attack_accuracy=round(acc_e, 4),
        attacker_advantage=round(adv_e, 4),
        tpr_at_1pct_fpr=round(tpr1_e, 4),
        tpr_at_5pct_fpr=round(tpr5_e, 4),
        precision=round(prec_e, 4),
        recall=round(rec_e, 4),
        f1_score=round(f1_e, 4),
        member_mean_loss=round(mem_mean_loss, 4),
        nonmember_mean_loss=round(nonmem_mean_loss, 4),
        rare_subgroup_auc=round(rare_auc_e, 4),
        majority_subgroup_auc=round(maj_auc_e, 4),
        roc_curve_fpr=fpr_e,
        roc_curve_tpr=tpr_e,
    )

    # 3. Shadow-Model Supervised Attack (Shokri et al., 2017)
    shadow_clf = ShadowAttackClassifier(seed=101)
    shadow_clf.fit(shadow_in_signals, shadow_out_signals)
    s_in_prob = shadow_clf.predict_proba(shadow_in_signals)
    s_out_prob = shadow_clf.predict_proba(shadow_out_signals)
    t_in_prob = shadow_clf.predict_proba(target_in_signals)
    t_out_prob = shadow_clf.predict_proba(target_out_signals)
    thresh_sh = find_optimal_shadow_threshold(s_in_prob, s_out_prob)

    rare_in_sh = [t_in_prob[i] for i in range(len(t_in_prob)) if rare_target_in[i] == 1]
    rare_out_sh = [t_out_prob[i] for i in range(len(t_out_prob)) if rare_target_out[i] == 1]
    maj_in_sh = [t_in_prob[i] for i in range(len(t_in_prob)) if rare_target_in[i] == 0]
    maj_out_sh = [t_out_prob[i] for i in range(len(t_out_prob)) if rare_target_out[i] == 0]

    auc_s, acc_s, adv_s, tpr1_s, tpr5_s, prec_s, rec_s, f1_s, fpr_s, tpr_s = (
        compute_roc_and_metrics(t_in_prob, t_out_prob, thresh_sh)
    )
    rare_auc_s = compute_roc_and_metrics(rare_in_sh, rare_out_sh, thresh_sh)[0]
    maj_auc_s = compute_roc_and_metrics(maj_in_sh, maj_out_sh, thresh_sh)[0]

    results["shadow_supervised"] = AttackMetrics(
        attack_name="Shadow-Model Supervised Attack (Shokri et al.)",
        auc_roc=round(auc_s, 4),
        attack_accuracy=round(acc_s, 4),
        attacker_advantage=round(adv_s, 4),
        tpr_at_1pct_fpr=round(tpr1_s, 4),
        tpr_at_5pct_fpr=round(tpr5_s, 4),
        precision=round(prec_s, 4),
        recall=round(rec_s, 4),
        f1_score=round(f1_s, 4),
        member_mean_loss=round(mem_mean_loss, 4),
        nonmember_mean_loss=round(nonmem_mean_loss, 4),
        rare_subgroup_auc=round(rare_auc_s, 4),
        majority_subgroup_auc=round(maj_auc_s, 4),
        roc_curve_fpr=fpr_s,
        roc_curve_tpr=tpr_s,
    )

    return results
