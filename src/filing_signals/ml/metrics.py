"""Evaluation shared by every Phase 3a model.

ROC-AUC and average precision, each with a 95% interval from a bootstrap that
resamples *firms*, not reports: a company appears in several fiscal years, so
resampling reports would treat correlated rows as independent and make the
intervals too narrow. Every model is scored on the same test rows.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

N_BOOT = 2000


def firm_bootstrap(y, score, firms, metric, n_boot: int = N_BOOT, seed: int = 0):
    y, score, firms = np.asarray(y), np.asarray(score), np.asarray(firms)
    uniq, inverse = np.unique(firms, return_inverse=True)
    rows_by_firm = [np.flatnonzero(inverse == i) for i in range(len(uniq))]
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([rows_by_firm[i] for i in pick])
        if y[idx].min() == y[idx].max():
            continue
        stats.append(metric(y[idx], score[idx]))
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def evaluate(y, score, firms) -> dict:
    y = np.asarray(y).astype(int)
    out = {"n": int(len(y)), "positives": int(y.sum())}
    for name, fn in (("auc", roc_auc_score), ("avg_precision", average_precision_score)):
        out[name] = round(float(fn(y, score)), 3)
        lo, hi = firm_bootstrap(y, score, firms, fn)
        out[f"{name}_ci"] = [round(lo, 3), round(hi, 3)]
    out["base_rate"] = round(float(y.mean()), 3)
    return out


def paired_auc_difference(y, score_a, score_b, firms, n_boot: int = N_BOOT, seed: int = 0) -> dict:
    """AUC(a) - AUC(b) on the same rows, with a firm-bootstrap 95% interval.
    The interval excluding 0 is the bar for "a beats b"."""
    y, a, b, firms = map(np.asarray, (y, score_a, score_b, firms))
    diff = roc_auc_score(y, a) - roc_auc_score(y, b)
    uniq, inverse = np.unique(firms, return_inverse=True)
    rows_by_firm = [np.flatnonzero(inverse == i) for i in range(len(uniq))]
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        idx = np.concatenate([rows_by_firm[i] for i in rng.integers(0, len(uniq), len(uniq))])
        if y[idx].min() == y[idx].max():
            continue
        stats.append(roc_auc_score(y[idx], a[idx]) - roc_auc_score(y[idx], b[idx]))
    return {"auc_diff": round(float(diff), 3),
            "ci": [round(float(np.percentile(stats, 2.5)), 3), round(float(np.percentile(stats, 97.5)), 3)]}
