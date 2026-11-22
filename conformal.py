"""Conformal prediction on direction maps. The estimator is left as it is. Its map is normalized per window, and the
nonconformity of a direction is one minus its normalized score, so a window is scored by how low the true direction
sits on its own map."""
import numpy as np


def norm01(x):
    mn = x.min(-1, keepdims=True); mx = x.max(-1, keepdims=True)
    return (x - mn) / (mx - mn + 1e-12)


def nonconformity(P, true_idx):
    return 1.0 - P[np.arange(len(P)), true_idx]


def conformal_tau(scores_cal, alpha):
    """Threshold at level 1 - alpha with the finite-sample correction: the ceil((1 - alpha)(n + 1))-th smallest of n
    calibration scores. When the calibration windows and a new window are exchangeable, the region of the new window
    holds its true direction with probability at least 1 - alpha."""
    s = np.sort(np.asarray(scores_cal))
    n = len(s)
    if n == 0:
        return np.inf
    if alpha >= 1:
        return -np.inf
    k = int(np.ceil((1 - alpha) * (n + 1)))
    if k > n:
        return np.inf  # every direction is kept
    return s[k - 1]


def region(P, tau):
    """Directions whose nonconformity is at most tau."""
    return 1.0 - P <= tau


def solid_angle_weights(grid):
    w = np.sqrt(np.clip(1 - grid[:, 2].astype(np.float64) ** 2, 0, 1))
    return w / w.sum()


def region_size(P, tau, weights, chunk=256):
    """Share of the grid's solid angle inside the region, in percent."""
    return np.concatenate([region(P[i:i + chunk], tau) @ weights for i in range(0, len(P), chunk)]) * 100


def answer(sizes, max_size):
    """Pass a region on when it is no wider than max_size. Wider ones are withheld."""
    return sizes <= max_size


class AdaptiveCP:
    """Adaptive conformal inference (Gibbs and Candes, 2021). After every window the working level moves by
    gamma (alpha - miss): a miss lowers it, which raises the threshold for the next windows, and a hit raises it a
    little. Over a stream the share of misses is held near alpha even when the conditions drift."""

    def __init__(self, scores_cal, alpha, gamma=0.02):
        self.cal = np.sort(np.asarray(scores_cal)); self.alpha = alpha; self.gamma = gamma; self.level = alpha

    def threshold(self):
        return conformal_tau(self.cal, self.level)

    def update(self, score):
        hit = bool(score <= self.threshold())
        self.level += self.gamma * (self.alpha - (0.0 if hit else 1.0))
        return hit
