"""
Neural Network Target & Shadow Models with Privacy Defenses:
  - Baseline (Unregularized MLP)
  - L2 Weight Decay Regularization (Moderate & Strong)
  - Dropout + Confidence Smoothing Regularization
  - Differentially Private SGD (DP-SGD, Abadi et al., 2016) with per-sample
    gradient norm clipping, calibrated Gaussian noise injection, and privacy accounting.

Authors:
  - Ramitha Nayak MV (PES1UG24AM392)
  - Shreya N Hegde (PES1UG24AM401)
"""

import math
import time
import random
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple


@dataclass
class DefenseConfig:
    name: str
    short_id: str
    l2_lambda: float = 0.0
    dropout_rate: float = 0.0
    label_smoothing: float = 0.0
    use_dp_sgd: bool = False
    clip_norm: float = 1.0
    noise_multiplier: float = 0.0
    target_epsilon: Optional[float] = None
    target_delta: float = 1e-5
    epochs: int = 28
    batch_size: int = 50
    learning_rate: float = 0.085
    hidden_dim: int = 40


@dataclass
class TrainingResult:
    config: DefenseConfig
    train_acc: float
    test_acc: float
    generalization_gap: float
    train_loss: float
    test_loss: float
    training_time_sec: float
    time_overhead_ratio: float
    achieved_epsilon: Optional[float]
    epoch_history: List[Dict[str, float]] = field(default_factory=list)


class ClinicalMLPClassifier:
    """
    Two-layer Multi-Layer Perceptron (MLP) with ReLU activation, optional Dropout,
    L2 regularization, Label Smoothing, and per-sample DP-SGD training.
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, seed: int = 42):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.rng = random.Random(seed)

        scale1 = math.sqrt(2.0 / input_dim)
        scale2 = math.sqrt(2.0 / hidden_dim)

        self.W1 = [
            [self.rng.gauss(0.0, scale1) for _ in range(hidden_dim)]
            for _ in range(input_dim)
        ]
        self.b1 = [0.0] * hidden_dim
        self.W2 = [
            [self.rng.gauss(0.0, scale2) for _ in range(num_classes)]
            for _ in range(hidden_dim)
        ]
        self.b2 = [0.0] * num_classes

    @staticmethod
    def _softmax(logits: List[float]) -> List[float]:
        max_l = max(logits)
        exps = [math.exp(l - max_l) for l in logits]
        s = sum(exps) + 1e-12
        return [e / s for e in exps]

    def forward_single(
        self, x: List[float], dropout_rate: float = 0.0, training: bool = False
    ) -> Tuple[List[float], List[float]]:
        """Returns (hidden_activations, class_probabilities)."""
        d_in = self.input_dim
        d_hid = self.hidden_dim
        d_out = self.num_classes
        W1 = self.W1
        b1 = self.b1
        W2 = self.W2
        b2 = self.b2

        h = [0.0] * d_hid
        scale = 1.0 / (1.0 - dropout_rate) if (training and dropout_rate > 0.0) else 1.0
        for k in range(d_hid):
            if training and dropout_rate > 0.0 and self.rng.random() < dropout_rate:
                h[k] = 0.0
                continue
            val = b1[k]
            for i in range(d_in):
                val += x[i] * W1[i][k]
            h[k] = (val * scale) if val > 0.0 else 0.0

        logits = [0.0] * d_out
        for c in range(d_out):
            val = b2[c]
            for k in range(d_hid):
                if h[k] != 0.0:
                    val += h[k] * W2[k][c]
            logits[c] = val

        probs = self._softmax(logits)
        return h, probs

    def fit(
        self,
        X_train: List[List[float]],
        y_train: List[int],
        X_val: List[List[float]],
        y_val: List[int],
        config: DefenseConfig,
        baseline_time: Optional[float] = None,
    ) -> TrainingResult:
        """
        Trains the MLP under either standard mini-batch SGD (with optional L2 / Dropout /
        Label Smoothing) or Differentially Private SGD (DP-SGD) with per-example gradient
        norm clipping and calibrated Gaussian noise.
        """
        t0 = time.perf_counter()
        n = len(X_train)
        d_in = self.input_dim
        d_hid = self.hidden_dim
        d_out = self.num_classes
        batch_size = config.batch_size
        lr0 = config.learning_rate
        l2 = config.l2_lambda
        smooth = config.label_smoothing

        indices = list(range(n))
        history = []

        for epoch in range(config.epochs):
            self.rng.shuffle(indices)
            lr = lr0 / (1.0 + 0.03 * epoch)

            for start in range(0, n, batch_size):
                batch_idx = indices[start : start + batch_size]
                B = len(batch_idx)

                gW1 = [[0.0] * d_hid for _ in range(d_in)]
                gb1 = [0.0] * d_hid
                gW2 = [[0.0] * d_out for _ in range(d_hid)]
                gb2 = [0.0] * d_out

                for idx in batch_idx:
                    x = X_train[idx]
                    y = y_train[idx]
                    h, probs = self.forward_single(
                        x, dropout_rate=config.dropout_rate, training=True
                    )

                    # Output layer gradient (dL / dz2)
                    dz2 = [0.0] * d_out
                    for c in range(d_out):
                        target = (
                            (1.0 - smooth) + smooth / d_out
                            if c == y
                            else smooth / d_out
                        )
                        dz2[c] = probs[c] - target

                    # Hidden layer gradient (dL / dh -> dL / dz1)
                    dz1 = [0.0] * d_hid
                    W2 = self.W2
                    for k in range(d_hid):
                        if h[k] > 0.0:
                            s = 0.0
                            for c in range(d_out):
                                s += dz2[c] * W2[k][c]
                            dz1[k] = s

                    if config.use_dp_sgd:
                        # Compute explicit per-sample parameter gradient norm ||g_i||_2
                        grad_norm_sq = 0.0
                        for c in range(d_out):
                            dz2_c = dz2[c]
                            grad_norm_sq += dz2_c * dz2_c
                            for k in range(d_hid):
                                hk_dz2 = h[k] * dz2_c
                                grad_norm_sq += hk_dz2 * hk_dz2
                        for k in range(d_hid):
                            dz1_k = dz1[k]
                            if dz1_k != 0.0:
                                grad_norm_sq += dz1_k * dz1_k
                                for i in range(d_in):
                                    xi_dz1 = x[i] * dz1_k
                                    grad_norm_sq += xi_dz1 * xi_dz1
                        grad_norm = math.sqrt(grad_norm_sq + 1e-12)
                        clip_factor = min(1.0, config.clip_norm / grad_norm)
                        for c in range(d_out):
                            dz2[c] *= clip_factor
                        for k in range(d_hid):
                            dz1[k] *= clip_factor

                    for c in range(d_out):
                        gb2[c] += dz2[c]
                    for k in range(d_hid):
                        if h[k] > 0.0:
                            hk = h[k]
                            gW2_k = gW2[k]
                            for c in range(d_out):
                                gW2_k[c] += hk * dz2[c]

                    for k in range(d_hid):
                        if dz1[k] != 0.0:
                            gb1[k] += dz1[k]

                    for i in range(d_in):
                        xi = x[i]
                        if xi != 0.0:
                            gW1_i = gW1[i]
                            for k in range(d_hid):
                                if dz1[k] != 0.0:
                                    gW1_i[k] += xi * dz1[k]

                # Inject calibrated Gaussian noise if DP-SGD is enabled
                inv_B = 1.0 / B
                noise_std = (
                    (config.noise_multiplier * config.clip_norm * inv_B)
                    if config.use_dp_sgd
                    else 0.0
                )

                for i in range(d_in):
                    W1_i = self.W1[i]
                    gW1_i = gW1[i]
                    for k in range(d_hid):
                        grad = gW1_i[k] * inv_B + l2 * W1_i[k]
                        if noise_std > 0.0:
                            grad += self.rng.gauss(0.0, noise_std)
                        W1_i[k] -= lr * grad

                for k in range(d_hid):
                    grad = gb1[k] * inv_B
                    if noise_std > 0.0:
                        grad += self.rng.gauss(0.0, noise_std)
                    self.b1[k] -= lr * grad

                for k in range(d_hid):
                    W2_k = self.W2[k]
                    gW2_k = gW2[k]
                    for c in range(d_out):
                        grad = gW2_k[c] * inv_B + l2 * W2_k[c]
                        if noise_std > 0.0:
                            grad += self.rng.gauss(0.0, noise_std)
                        W2_k[c] -= lr * grad

                for c in range(d_out):
                    grad = gb2[c] * inv_B
                    if noise_std > 0.0:
                        grad += self.rng.gauss(0.0, noise_std)
                    self.b2[c] -= lr * grad

            if (epoch + 1) % 4 == 0 or epoch == config.epochs - 1:
                tr_acc, tr_loss = self.evaluate_summary(X_train, y_train)
                te_acc, te_loss = self.evaluate_summary(X_val, y_val)
                history.append(
                    {
                        "epoch": epoch + 1,
                        "train_acc": round(tr_acc, 4),
                        "test_acc": round(te_acc, 4),
                        "train_loss": round(tr_loss, 4),
                        "test_loss": round(te_loss, 4),
                    }
                )

        elapsed = time.perf_counter() - t0
        tr_acc, tr_loss = self.evaluate_summary(X_train, y_train)
        te_acc, te_loss = self.evaluate_summary(X_val, y_val)
        overhead = (elapsed / baseline_time) if baseline_time else 1.0

        achieved_eps = None
        if config.use_dp_sgd:
            achieved_eps = compute_dp_epsilon(
                n_samples=n,
                batch_size=config.batch_size,
                noise_multiplier=config.noise_multiplier,
                epochs=config.epochs,
                delta=config.target_delta,
            )

        return TrainingResult(
            config=config,
            train_acc=tr_acc,
            test_acc=te_acc,
            generalization_gap=tr_acc - te_acc,
            train_loss=tr_loss,
            test_loss=te_loss,
            training_time_sec=elapsed,
            time_overhead_ratio=overhead,
            achieved_epsilon=achieved_eps,
            epoch_history=history,
        )

    def evaluate_summary(
        self, X: List[List[float]], y: List[int]
    ) -> Tuple[float, float]:
        correct = 0
        total_loss = 0.0
        n = len(X)
        for idx in range(n):
            _, probs = self.forward_single(X[idx], training=False)
            pred = max(range(self.num_classes), key=lambda c: probs[c])
            if pred == y[idx]:
                correct += 1
            p_true = max(probs[y[idx]], 1e-12)
            total_loss += -math.log(p_true)
        return correct / n, total_loss / n

    def extract_record_signals(
        self, X: List[List[float]], y: List[int]
    ) -> List[Dict[str, float]]:
        """
        Extracts per-record behavior signals used by Membership Inference Attacks:
          - loss: Cross-entropy loss -log P(y_true | x)
          - confidence: Maximum softmax probability max_c P(y=c | x)
          - true_prob: Softmax probability assigned to ground-truth class y_true
          - entropy: Shannon entropy -sum_c P(c|x) log P(c|x)
          - margin: P(y_true | x) - max_{c != y_true} P(y=c | x)
          - correct: 1.0 if argmax_c P(c|x) == y_true else 0.0
        """
        signals = []
        for idx in range(len(X)):
            _, probs = self.forward_single(X[idx], training=False)
            y_true = y[idx]
            p_true = max(probs[y_true], 1e-12)
            loss = -math.log(p_true)
            conf = max(probs)
            pred = max(range(self.num_classes), key=lambda c: probs[c])
            correct = 1.0 if pred == y_true else 0.0
            entropy = -sum(p * math.log(max(p, 1e-12)) for p in probs)
            max_other = max(
                probs[c] for c in range(self.num_classes) if c != y_true
            )
            margin = p_true - max_other
            signals.append(
                {
                    "loss": loss,
                    "confidence": conf,
                    "true_prob": p_true,
                    "entropy": entropy,
                    "margin": margin,
                    "correct": correct,
                    "true_class": float(y_true),
                }
            )
        return signals


def compute_dp_epsilon(
    n_samples: int,
    batch_size: int,
    noise_multiplier: float,
    epochs: int,
    delta: float = 1e-5,
) -> float:
    """
    Computes the privacy budget epsilon for (epsilon, delta)-Differential Privacy
    using Rényi Differential Privacy (RDP) Gaussian mechanism accounting across
    orders alpha in [2, 128] (Mironov, 2017; Abadi et al., 2016).
    """
    if noise_multiplier <= 0.0:
        return float("inf")
    q = batch_size / float(n_samples)
    steps = int(math.ceil(epochs / q))
    sigma = noise_multiplier

    best_eps = float("inf")
    for alpha in range(2, 129):
        # Subsampled Gaussian RDP bound per step
        rdp_step = (q * q * alpha) / (2.0 * sigma * sigma)
        total_rdp = steps * rdp_step
        eps = total_rdp + math.log(1.0 / delta) / (alpha - 1.0)
        if eps < best_eps:
            best_eps = eps
    return round(best_eps, 2)


def get_defense_configurations() -> List[DefenseConfig]:
    """
    Returns the 6 experimental configurations comparing the unregularized baseline
    against L2 Regularization, Dropout + Label Smoothing, and DP-SGD.
    """
    return [
        DefenseConfig(
            name="Baseline (No Defense)",
            short_id="baseline",
            l2_lambda=0.0,
            dropout_rate=0.0,
            label_smoothing=0.0,
            use_dp_sgd=False,
            epochs=38,
            batch_size=30,
            learning_rate=0.14,
            hidden_dim=64,
        ),
        DefenseConfig(
            name="L2 Regularization (Moderate, lambda=8e-3)",
            short_id="l2_moderate",
            l2_lambda=0.008,
            dropout_rate=0.0,
            label_smoothing=0.0,
            use_dp_sgd=False,
            epochs=38,
            batch_size=30,
            learning_rate=0.12,
            hidden_dim=64,
        ),
        DefenseConfig(
            name="L2 Regularization (Strong, lambda=2.5e-2)",
            short_id="l2_strong",
            l2_lambda=0.025,
            dropout_rate=0.0,
            label_smoothing=0.0,
            use_dp_sgd=False,
            epochs=38,
            batch_size=30,
            learning_rate=0.10,
            hidden_dim=64,
        ),
        DefenseConfig(
            name="Dropout (p=0.35) + Confidence Smoothing",
            short_id="dropout_smooth",
            l2_lambda=0.006,
            dropout_rate=0.35,
            label_smoothing=0.10,
            use_dp_sgd=False,
            epochs=38,
            batch_size=30,
            learning_rate=0.10,
            hidden_dim=64,
        ),
        DefenseConfig(
            name="DP-SGD (Moderate Privacy, eps=5.30, sigma=1.38)",
            short_id="dpsgd_moderate",
            l2_lambda=0.002,
            dropout_rate=0.0,
            label_smoothing=0.0,
            use_dp_sgd=True,
            clip_norm=0.95,
            noise_multiplier=1.38,
            target_epsilon=5.30,
            target_delta=1e-5,
            epochs=38,
            batch_size=30,
            learning_rate=0.080,
            hidden_dim=64,
        ),
        DefenseConfig(
            name="DP-SGD (Strict Privacy, eps=2.76, sigma=2.45)",
            short_id="dpsgd_strict",
            l2_lambda=0.002,
            dropout_rate=0.0,
            label_smoothing=0.0,
            use_dp_sgd=True,
            clip_norm=0.70,
            noise_multiplier=2.45,
            target_epsilon=2.76,
            target_delta=1e-5,
            epochs=38,
            batch_size=30,
            learning_rate=0.068,
            hidden_dim=64,
        ),
    ]
