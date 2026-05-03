from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import SGDClassifier

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from edge_iiot_robustness import _dense, _fgsm_attack, _pgd_attack, _fit_surrogate


def test_fgsm_and_pgd_respect_bounds() -> None:
    X = np.array(
        [
            [0.0, 0.1, 0.2],
            [0.2, 0.2, 0.3],
            [0.8, 0.7, 0.6],
            [0.9, 0.8, 0.7],
        ],
        dtype=float,
    )
    y = np.array([0, 0, 1, 1], dtype=int)
    surrogate: SGDClassifier = _fit_surrogate(_dense(X), y)
    lower = np.zeros(X.shape[1], dtype=float)
    upper = np.ones(X.shape[1], dtype=float)

    fgsm = _fgsm_attack(X, surrogate, epsilon=0.1, lower=lower, upper=upper)
    pgd = _pgd_attack(X, surrogate, epsilon=0.1, step_size=0.05, steps=3, lower=lower, upper=upper)

    assert fgsm.shape == X.shape
    assert pgd.shape == X.shape
    assert np.all(fgsm >= lower - 1e-12)
    assert np.all(fgsm <= upper + 1e-12)
    assert np.all(pgd >= lower - 1e-12)
    assert np.all(pgd <= upper + 1e-12)
