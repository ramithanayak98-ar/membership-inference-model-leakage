"""
Publication-grade SVG chart generator for Membership Inference & Defense Evaluation.
Zero external library dependencies required; outputs crisp vector SVG plots for
README, Jupyter notebooks, and technical reports.

Authors:
  - Ramitha Nayak MV (PES1UG24AM392)
  - Shreya N Hegde (PES1UG24AM401)
"""

import os
from typing import List, Dict, Any


PALETTE = {
    "baseline": "#d62728",        # Crimson Red
    "l2_moderate": "#ff7f0e",     # Amber Orange
    "l2_strong": "#9467bd",       # Royal Purple
    "dropout_smooth": "#1f77b4",  # Steel Blue
    "dpsgd_moderate": "#2ca02c",  # Emerald Green
    "dpsgd_strict": "#008080",    # Deep Teal
}


def generate_all_figures(summary_rows: List[Dict[str, Any]], out_dir: str = "results/figures") -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    paths.append(_plot_roc_curves(summary_rows, os.path.join(out_dir, "fig1_roc_curves.svg")))
    paths.append(_plot_privacy_utility_cost(summary_rows, os.path.join(out_dir, "fig2_privacy_utility_tradeoff.svg")))
    paths.append(_plot_subgroup_disparity(summary_rows, os.path.join(out_dir, "fig3_subgroup_vulnerability.svg")))
    paths.append(_plot_loss_gap(summary_rows, os.path.join(out_dir, "fig4_member_nonmember_loss.svg")))
    return paths


def _plot_roc_curves(rows: List[Dict[str, Any]], filepath: str) -> str:
    w, h = 760, 480
    pad_l, pad_r, pad_t, pad_b = 75, 250, 55, 65
    pw = w - pad_l - pad_r
    ph = h - pad_t - pad_b

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" style="background:#ffffff;font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Helvetica,Arial,sans-serif;">',
        f'<text x="{w/2 - 60}" y="30" text-anchor="middle" font-size="16" font-weight="bold" fill="#1a202c">Membership Inference Attack ROC Curves Across Defenses</text>',
        f'<rect x="{pad_l}" y="{pad_t}" width="{pw}" height="{ph}" fill="#fafafa" stroke="#cbd5e0" stroke-width="1.2"/>',
    ]

    # Grid lines
    for i in range(6):
        val = i * 0.2
        gx = pad_l + val * pw
        gy = pad_t + ph - val * ph
        lines.append(f'<line x1="{gx:.1f}" y1="{pad_t}" x2="{gx:.1f}" y2="{pad_t+ph}" stroke="#e2e8f0" stroke-dasharray="3,3"/>')
        lines.append(f'<line x1="{pad_l}" y1="{gy:.1f}" x2="{pad_l+pw}" y2="{gy:.1f}" stroke="#e2e8f0" stroke-dasharray="3,3"/>')
        lines.append(f'<text x="{gx:.1f}" y="{pad_t+ph+20}" text-anchor="middle" font-size="11" fill="#4a5568">{val:.1f}</text>')
        lines.append(f'<text x="{pad_l-10}" y="{gy+4:.1f}" text-anchor="end" font-size="11" fill="#4a5568">{val:.1f}</text>')

    # Diagonal random baseline
    lines.append(
        f'<line x1="{pad_l}" y1="{pad_t+ph}" x2="{pad_l+pw}" y2="{pad_t}" stroke="#718096" stroke-width="1.5" stroke-dasharray="6,4"/>'
    )

    # Plot each defense ROC curve (Shadow Attack)
    legend_y = pad_t + 20
    for idx, r in enumerate(rows):
        sid = r["short_id"]
        color = PALETTE.get(sid, "#333333")
        fprs = r["roc_fpr"]
        tprs = r["roc_tpr"]
        pts = " ".join(
            f"{pad_l + fx * pw:.1f},{pad_t + ph - ty * ph:.1f}"
            for fx, ty in zip(fprs, tprs)
        )
        lines.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.4" points="{pts}"/>')

        ly = legend_y + idx * 45
        lx = pad_l + pw + 18
        lines.append(f'<line x1="{lx}" y1="{ly}" x2="{lx+24}" y2="{ly}" stroke="{color}" stroke-width="3"/>')
        lines.append(f'<text x="{lx+30}" y="{ly+4}" font-size="11" font-weight="600" fill="#2d3748">{r["short_label"]}</text>')
        lines.append(f'<text x="{lx+30}" y="{ly+18}" font-size="10" fill="#4a5568">AUC = {r["shadow_auc"]:.3f} | Adv = {r["shadow_adv"]:.3f}</text>')

    # Random baseline legend item
    ly = legend_y + len(rows) * 45
    lx = pad_l + pw + 18
    lines.append(f'<line x1="{lx}" y1="{ly}" x2="{lx+24}" y2="{ly}" stroke="#718096" stroke-width="1.5" stroke-dasharray="5,3"/>')
    lines.append(f'<text x="{lx+30}" y="{ly+4}" font-size="11" fill="#718096">Ideal Random Guess (AUC = 0.500)</text>')

    # Axis titles
    lines.append(f'<text x="{pad_l + pw/2}" y="{h-18}" text-anchor="middle" font-size="13" font-weight="600" fill="#2d3748">False Positive Rate (FPR)</text>')
    lines.append(f'<text x="22" y="{pad_t + ph/2}" text-anchor="middle" font-size="13" font-weight="600" fill="#2d3748" transform="rotate(-90, 22, {pad_t + ph/2})">True Positive Rate (TPR)</text>')
    lines.append("</svg>")

    with open(filepath, "w") as f:
        f.write("\n".join(lines))
    return filepath


def _plot_privacy_utility_cost(rows: List[Dict[str, Any]], filepath: str) -> str:
    w, h = 820, 440
    pad_l, pad_r, pad_t, pad_b = 65, 40, 60, 95
    pw = w - pad_l - pad_r
    ph = h - pad_t - pad_b

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" style="background:#ffffff;font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Helvetica,Arial,sans-serif;">',
        f'<text x="{w/2}" y="28" text-anchor="middle" font-size="16" font-weight="bold" fill="#1a202c">Defense Evaluation: Test Accuracy vs. Membership Attack AUC &amp; Runtime Cost</text>',
        f'<rect x="{pad_l}" y="{pad_t}" width="{pw}" height="{ph}" fill="#fafafa" stroke="#cbd5e0" stroke-width="1.2"/>',
    ]

    # Y-axis from 0.40 to 0.90
    y_min, y_max = 0.40, 0.90
    for i in range(6):
        val = y_min + i * 0.10
        gy = pad_t + ph - ((val - y_min) / (y_max - y_min)) * ph
        lines.append(f'<line x1="{pad_l}" y1="{gy:.1f}" x2="{pad_l+pw}" y2="{gy:.1f}" stroke="#e2e8f0" stroke-dasharray="3,3"/>')
        lines.append(f'<text x="{pad_l-8}" y="{gy+4:.1f}" text-anchor="end" font-size="11" fill="#4a5568">{val:.2f}</text>')

    # Ideal privacy line at 0.50
    gy50 = pad_t + ph - ((0.50 - y_min) / (y_max - y_min)) * ph
    lines.append(f'<line x1="{pad_l}" y1="{gy50:.1f}" x2="{pad_l+pw}" y2="{gy50:.1f}" stroke="#2b6cb0" stroke-width="1.2" stroke-dasharray="5,4"/>')
    lines.append(f'<text x="{pad_l+pw-8}" y="{gy50-6:.1f}" text-anchor="end" font-size="10" fill="#2b6cb0">Ideal Privacy Floor (AUC = 0.50)</text>')

    n = len(rows)
    group_w = pw / n
    bar_w = 28

    for idx, r in enumerate(rows):
        cx = pad_l + (idx + 0.5) * group_w
        acc = r["test_acc"]
        auc = r["shadow_auc"]
        time_mult = r["time_overhead_ratio"]

        h_acc = max(4.0, ((acc - y_min) / (y_max - y_min)) * ph)
        h_auc = max(4.0, ((auc - y_min) / (y_max - y_min)) * ph)
        y_acc = pad_t + ph - h_acc
        y_auc = pad_t + ph - h_auc

        # Test Accuracy Bar (Blue)
        lines.append(f'<rect x="{cx - bar_w - 3:.1f}" y="{y_acc:.1f}" width="{bar_w}" height="{h_acc:.1f}" fill="#3182ce" rx="3"/>')
        lines.append(f'<text x="{cx - bar_w/2 - 3:.1f}" y="{y_acc - 6:.1f}" text-anchor="middle" font-size="10" font-weight="bold" fill="#2b6cb0">{acc*100:.1f}%</text>')

        # Attack AUC Bar (Red/Orange)
        lines.append(f'<rect x="{cx + 3:.1f}" y="{y_auc:.1f}" width="{bar_w}" height="{h_auc:.1f}" fill="#e53e3e" rx="3"/>')
        lines.append(f'<text x="{cx + bar_w/2 + 3:.1f}" y="{y_auc - 6:.1f}" text-anchor="middle" font-size="10" font-weight="bold" fill="#c53030">{auc:.3f}</text>')

        # X-axis label + runtime badge
        lines.append(f'<text x="{cx:.1f}" y="{pad_t+ph+22}" text-anchor="middle" font-size="11" font-weight="600" fill="#2d3748">{r["short_label"]}</text>')
        lines.append(f'<text x="{cx:.1f}" y="{pad_t+ph+38}" text-anchor="middle" font-size="10" fill="#718096">Cost: {time_mult:.2f}x time</text>')

    # Legend
    lines.append(f'<rect x="{pad_l+20}" y="40" width="14" height="12" fill="#3182ce" rx="2"/>')
    lines.append(f'<text x="{pad_l+40}" y="50" font-size="11" font-weight="600" fill="#2d3748">Test Set Accuracy (Utility - Higher is Better)</text>')
    lines.append(f'<rect x="{pad_l+320}" y="40" width="14" height="12" fill="#e53e3e" rx="2"/>')
    lines.append(f'<text x="{pad_l+340}" y="50" font-size="11" font-weight="600" fill="#2d3748">Shadow MIA AUC-ROC (Leakage - Closer to 0.50 is Better)</text>')
    lines.append("</svg>")

    with open(filepath, "w") as f:
        f.write("\n".join(lines))
    return filepath


def _plot_subgroup_disparity(rows: List[Dict[str, Any]], filepath: str) -> str:
    w, h = 780, 420
    pad_l, pad_r, pad_t, pad_b = 65, 40, 60, 85
    pw = w - pad_l - pad_r
    ph = h - pad_t - pad_b

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" style="background:#ffffff;font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Helvetica,Arial,sans-serif;">',
        f'<text x="{w/2}" y="28" text-anchor="middle" font-size="16" font-weight="bold" fill="#1a202c">Subgroup Vulnerability: Rare/Atypical Patients vs. Majority Population</text>',
        f'<rect x="{pad_l}" y="{pad_t}" width="{pw}" height="{ph}" fill="#fafafa" stroke="#cbd5e0" stroke-width="1.2"/>',
    ]

    y_min, y_max = 0.45, 0.85
    for i in range(5):
        val = y_min + i * 0.10
        gy = pad_t + ph - ((val - y_min) / (y_max - y_min)) * ph
        lines.append(f'<line x1="{pad_l}" y1="{gy:.1f}" x2="{pad_l+pw}" y2="{gy:.1f}" stroke="#e2e8f0" stroke-dasharray="3,3"/>')
        lines.append(f'<text x="{pad_l-8}" y="{gy+4:.1f}" text-anchor="end" font-size="11" fill="#4a5568">{val:.2f}</text>')

    n = len(rows)
    group_w = pw / n
    bar_w = 26

    for idx, r in enumerate(rows):
        cx = pad_l + (idx + 0.5) * group_w
        rare_auc = r["rare_subgroup_auc"]
        maj_auc = r["majority_subgroup_auc"]

        h_rare = max(4.0, ((rare_auc - y_min) / (y_max - y_min)) * ph)
        h_maj = max(4.0, ((maj_auc - y_min) / (y_max - y_min)) * ph)
        y_rare = pad_t + ph - h_rare
        y_maj = pad_t + ph - h_maj

        lines.append(f'<rect x="{cx - bar_w - 3:.1f}" y="{y_rare:.1f}" width="{bar_w}" height="{h_rare:.1f}" fill="#805ad5" rx="3"/>')
        lines.append(f'<text x="{cx - bar_w/2 - 3:.1f}" y="{y_rare - 6:.1f}" text-anchor="middle" font-size="10" font-weight="bold" fill="#553c9a">{rare_auc:.3f}</text>')

        lines.append(f'<rect x="{cx + 3:.1f}" y="{y_maj:.1f}" width="{bar_w}" height="{h_maj:.1f}" fill="#38b2ac" rx="3"/>')
        lines.append(f'<text x="{cx + bar_w/2 + 3:.1f}" y="{y_maj - 6:.1f}" text-anchor="middle" font-size="10" font-weight="bold" fill="#234e52">{maj_auc:.3f}</text>')

        lines.append(f'<text x="{cx:.1f}" y="{pad_t+ph+22}" text-anchor="middle" font-size="11" font-weight="600" fill="#2d3748">{r["short_label"]}</text>')

    lines.append(f'<rect x="{pad_l+20}" y="40" width="14" height="12" fill="#805ad5" rx="2"/>')
    lines.append(f'<text x="{pad_l+40}" y="50" font-size="11" font-weight="600" fill="#2d3748">Rare/Minority Clinical Subgroup MIA AUC</text>')
    lines.append(f'<rect x="{pad_l+310}" y="40" width="14" height="12" fill="#38b2ac" rx="2"/>')
    lines.append(f'<text x="{pad_l+330}" y="50" font-size="11" font-weight="600" fill="#2d3748">Majority Population MIA AUC</text>')
    lines.append("</svg>")

    with open(filepath, "w") as f:
        f.write("\n".join(lines))
    return filepath


def _plot_loss_gap(rows: List[Dict[str, Any]], filepath: str) -> str:
    w, h = 760, 410
    pad_l, pad_r, pad_t, pad_b = 65, 40, 60, 80
    pw = w - pad_l - pad_r
    ph = h - pad_t - pad_b

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" style="background:#ffffff;font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Helvetica,Arial,sans-serif;">',
        f'<text x="{w/2}" y="28" text-anchor="middle" font-size="16" font-weight="bold" fill="#1a202c">Mean Cross-Entropy Loss: Training Members vs. Holdout Non-Members</text>',
        f'<rect x="{pad_l}" y="{pad_t}" width="{pw}" height="{ph}" fill="#fafafa" stroke="#cbd5e0" stroke-width="1.2"/>',
    ]

    y_min, y_max = 0.0, max(1.6, max(r["nonmember_loss"] for r in rows) * 1.15)
    for i in range(5):
        val = y_min + i * (y_max / 4.0)
        gy = pad_t + ph - (val / y_max) * ph
        lines.append(f'<line x1="{pad_l}" y1="{gy:.1f}" x2="{pad_l+pw}" y2="{gy:.1f}" stroke="#e2e8f0" stroke-dasharray="3,3"/>')
        lines.append(f'<text x="{pad_l-8}" y="{gy+4:.1f}" text-anchor="end" font-size="11" fill="#4a5568">{val:.2f}</text>')

    n = len(rows)
    group_w = pw / n
    bar_w = 26

    for idx, r in enumerate(rows):
        cx = pad_l + (idx + 0.5) * group_w
        m_loss = r["member_loss"]
        nm_loss = r["nonmember_loss"]

        h_m = max(4.0, (m_loss / y_max) * ph)
        h_nm = max(4.0, (nm_loss / y_max) * ph)
        y_m = pad_t + ph - h_m
        y_nm = pad_t + ph - h_nm

        lines.append(f'<rect x="{cx - bar_w - 3:.1f}" y="{y_m:.1f}" width="{bar_w}" height="{h_m:.1f}" fill="#dd6b20" rx="3"/>')
        lines.append(f'<text x="{cx - bar_w/2 - 3:.1f}" y="{y_m - 6:.1f}" text-anchor="middle" font-size="10" font-weight="bold" fill="#9c4221">{m_loss:.2f}</text>')

        lines.append(f'<rect x="{cx + 3:.1f}" y="{y_nm:.1f}" width="{bar_w}" height="{h_nm:.1f}" fill="#4a5568" rx="3"/>')
        lines.append(f'<text x="{cx + bar_w/2 + 3:.1f}" y="{y_nm - 6:.1f}" text-anchor="middle" font-size="10" font-weight="bold" fill="#2d3748">{nm_loss:.2f}</text>')

        lines.append(f'<text x="{cx:.1f}" y="{pad_t+ph+22}" text-anchor="middle" font-size="11" font-weight="600" fill="#2d3748">{r["short_label"]}</text>')

    lines.append(f'<rect x="{pad_l+20}" y="40" width="14" height="12" fill="#dd6b20" rx="2"/>')
    lines.append(f'<text x="{pad_l+40}" y="50" font-size="11" font-weight="600" fill="#2d3748">Training Members Mean Loss (In-Set)</text>')
    lines.append(f'<rect x="{pad_l+300}" y="40" width="14" height="12" fill="#4a5568" rx="2"/>')
    lines.append(f'<text x="{pad_l+320}" y="50" font-size="11" font-weight="600" fill="#2d3748">Holdout Non-Members Mean Loss (Out-Set)</text>')
    lines.append("</svg>")

    with open(filepath, "w") as f:
        f.write("\n".join(lines))
    return filepath
