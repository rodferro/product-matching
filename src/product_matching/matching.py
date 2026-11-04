"""(a) Probability that two listings are the same product.

A logistic regression on the pairwise features, trained on hand-labelled
pairs. Logistic regression is used on purpose: with ~500 labelled pairs a
small, well-regularized model is easier to calibrate and to explain than a
boosted ensemble. Whether to add Platt (sigmoid) calibration on top is decided
by cross-validated Brier score on the training pairs only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import SEED
from .features import FEATURES
from .metrics import best_f1_threshold, brier


def _base_model(c: float) -> object:
    return make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=2000))


@dataclass
class PairMatcher:
    threshold: float = 0.5
    c: float = 1.0
    calibration: str = "none"
    model: object | None = None
    cv_report: dict | None = None

    def fit(self, x: pd.DataFrame, y: np.ndarray) -> PairMatcher:
        """Pick C and calibration by 5-fold CV on the training pairs, then refit on all of them."""
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        candidates = []
        for c in (0.01, 0.1, 1.0, 10.0, 100.0):
            for calibration in ("none", "sigmoid"):
                model = self._make(c, calibration)
                prob = cross_val_predict(model, x[FEATURES], y, cv=cv, method="predict_proba")[:, 1]
                candidates.append((brier(y, prob), c, calibration, prob))
        score, self.c, self.calibration, oof = min(candidates, key=lambda t: t[0])
        self.threshold = best_f1_threshold(y, oof)
        self.cv_report = {
            "brier_by_setting": {f"C={c},cal={cal}": s for s, c, cal, _ in candidates},
            "chosen": {"C": self.c, "calibration": self.calibration, "cv_brier": score},
            "threshold": self.threshold,
        }
        self.model = self._make(self.c, self.calibration).fit(x[FEATURES], y)
        return self

    @staticmethod
    def _make(c: float, calibration: str):
        base = _base_model(c)
        if calibration == "none":
            return base
        return CalibratedClassifierCV(base, method=calibration, cv=3)

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(x[FEATURES])[:, 1]

    def coefficients(self) -> dict[str, float]:
        """Standardized coefficients (only available without the calibration wrapper)."""
        model = self.model
        if isinstance(model, CalibratedClassifierCV):
            coefs = np.mean(
                [cc.estimator[-1].coef_[0] for cc in model.calibrated_classifiers_], axis=0
            )
        else:
            coefs = model[-1].coef_[0]
        return dict(zip(FEATURES, map(float, coefs), strict=True))
