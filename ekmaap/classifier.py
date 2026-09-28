"""Onion condition: sound / sprouted / damaged / rotten.

``HeuristicClassifier`` - fixed thresholds (the demo's starting values). No training; not tuned
                          on real photos; kept only as a fallback before any model exists.
``LearnedClassifier``   - scikit-learn model trained on YOUR labelled onions
                          (``scripts/train_classifier.py``). Reports cross-validated scores with whole
                          photos held out, so onions from one photo never sit in both train and test.

Both return a probability per condition; an onion whose best probability is below the review
threshold goes to the officer instead of being graded silently.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np

from . import CONDITIONS
from .features import NAMES, vector


@dataclass
class Prediction:
    condition: str
    proba: dict
    review: bool
    reasons: list = field(default_factory=list)


class HeuristicClassifier:
    kind = "heuristic"

    def __init__(self, rot_dark_frac=0.08, sprout_green_frac=0.015, damage_solidity=0.93, damage_pale_frac=0.10, review_band=0.2):
        self.t = dict(rot=rot_dark_frac, sprout=sprout_green_frac, sol=damage_solidity, pale=damage_pale_frac, band=review_band)

    def predict(self, f: dict, review_below: float = 0.6) -> Prediction:
        t = self.t
        s = {"rotten": f["dark_frac"] / t["rot"], "sprouted": f["green_frac"] / t["sprout"],
             "damaged": max((1 - f["solidity"]) / max(1e-6, 1 - t["sol"]), f["pale_frac"] / t["pale"])}
        cond = next((c for c in ("rotten", "sprouted", "damaged") if s[c] >= 1), "sound")
        border = [c for c, v in s.items() if 1 - t["band"] < v < 1 + t["band"]]
        reasons = [f"dark {f['dark_frac']:.1%}", f"green {f['green_frac']:.1%}", f"pale {f['pale_frac']:.1%}", f"solidity {f['solidity']:.3f}",
                   "untrained starting rules"]
        if border:
            reasons.append("close to a threshold: " + ", ".join(border))
        proba = {c: (0.7 if c == cond else 0.1) for c in CONDITIONS}
        return Prediction(cond, proba, bool(border), reasons)

    def to_json(self):
        return {"kind": self.kind, **self.t}


class LearnedClassifier:
    kind = "sklearn"

    def __init__(self, model=None, meta=None):
        self.model = model
        self.meta = meta or {}

    @staticmethod
    def _make(seed=0):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        return make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                                                      class_weight="balanced", random_state=seed, n_jobs=-1))

    @classmethod
    def train(cls, X: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int = 0) -> "LearnedClassifier":
        """Fit on all data; report grouped cross-validation (whole photos held out)."""
        from sklearn.metrics import classification_report, confusion_matrix
        from sklearn.model_selection import GroupKFold
        y = np.asarray(y)
        labels = [c for c in CONDITIONS if c in set(y)]
        cv = None
        n_groups = len(set(groups))
        if n_groups >= 3:
            k = min(5, n_groups)
            pred = np.empty(len(y), dtype=object)
            for tr, te in GroupKFold(n_splits=k).split(X, y, groups):
                m = cls._make(seed).fit(X[tr], y[tr])
                pred[te] = m.predict(X[te])
            rep = classification_report(y, pred, labels=labels, output_dict=True, zero_division=0)
            cv = {"scheme": f"GroupKFold k={k} over {n_groups} photos", "accuracy": float((pred == y).mean()),
                  "per_class": {c: {"precision": rep[c]["precision"], "recall": rep[c]["recall"], "n": int(rep[c]["support"])} for c in labels},
                  "confusion": {"labels": labels, "matrix": confusion_matrix(y, pred, labels=labels).tolist()}}
        model = cls._make(seed).fit(X, y)
        counts = {c: int((y == c).sum()) for c in labels}
        return cls(model, {"n": int(len(y)), "n_photos": int(n_groups), "counts": counts, "cv": cv, "features": list(NAMES)})

    def predict(self, f: dict, review_below: float = 0.6) -> Prediction:
        p = self.model.predict_proba(vector(f)[None, :])[0]
        classes = list(self.model.classes_)
        proba = {c: float(p[classes.index(c)]) if c in classes else 0.0 for c in CONDITIONS}
        cond = max(proba, key=proba.get)
        review = proba[cond] < review_below
        reasons = [f"model: {cond} {proba[cond]:.0%}"] + (["low confidence → officer review"] if review else [])
        return Prediction(cond, proba, review, reasons)

    def save(self, path):
        joblib.dump({"kind": self.kind, "model": self.model, "meta": self.meta}, path, compress=3)

    @classmethod
    def load(cls, path) -> "LearnedClassifier":
        obj = joblib.load(path)
        if obj.get("kind") != cls.kind:
            raise ValueError(f"{path} is not a classifier file")
        if obj["meta"].get("features") and obj["meta"]["features"] != list(NAMES):
            raise ValueError("Classifier was trained with a different feature set; retrain it")
        return cls(obj["model"], obj["meta"])

    def to_json(self):
        return {"kind": self.kind, **{k: v for k, v in self.meta.items() if k != "features"}}


def load_classifier(path: str | Path | None):
    return HeuristicClassifier() if path is None else LearnedClassifier.load(path)
