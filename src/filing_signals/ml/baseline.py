"""Phase 3a baselines: the bars the fine-tuned encoder has to clear.

  prior_event       did the same kind of event happen in the year before?
  size+prior        log public float (+ missing flag) and prior_event
  tfidf             TF-IDF of Risk Factors + logistic regression
  tfidf+size+prior  TF-IDF plus the size and prior-event features

The question that matters is whether text beats size+prior: filing language
is a strong proxy for company size, and small companies have far more
auditor changes. `text_vs_size` answers it with a paired firm-bootstrap.

The regularisation strength C is chosen on the validation years; the test
years are touched once, at the end.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from .metrics import evaluate, paired_auc_difference

C_GRID = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]


def load(path: Path) -> dict[str, pd.DataFrame]:
    df = pd.read_parquet(path)
    return {s: df[df.split == s].reset_index(drop=True) for s in ("train", "val", "test")}


def run(dataset: Path, out: Path, log=print) -> dict:
    d = load(dataset)
    results: dict = {}
    test = d["test"]

    results["prior_event"] = evaluate(test.label, test.prior_event.astype(float), test.cik)
    log(f"prior_event: {results['prior_event']}")

    def tabular(df: pd.DataFrame) -> np.ndarray:
        return np.column_stack([df.log_float, (~df.has_float).astype(float), df.prior_event.astype(float)])

    size = LogisticRegression(max_iter=2000).fit(tabular(d["train"]), d["train"].label)
    test["score_size+prior"] = size.predict_proba(tabular(test))[:, 1]
    results["size+prior"] = evaluate(test.label, test["score_size+prior"], test.cik)
    log(f"size+prior: {results['size+prior']}")

    vec = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=5, max_df=0.9,
                          max_features=200_000, dtype=np.float32)
    x = {s: vec.fit_transform(d[s].text) if s == "train" else vec.transform(d[s].text) for s in d}

    # tabular features are standardised on train so one C suits both blocks
    mu, sd = tabular(d["train"]).mean(0), tabular(d["train"]).std(0) + 1e-9
    for name, with_tab in (("tfidf", False), ("tfidf+size+prior", True)):
        feats = {s: hstack([x[s], csr_matrix((tabular(d[s]) - mu) / sd)]).tocsr()
                 if with_tab else x[s] for s in d}
        best = None
        for c in C_GRID:
            model = LogisticRegression(C=c, class_weight="balanced", max_iter=2000)
            model.fit(feats["train"], d["train"].label)
            val_auc = roc_auc_score(d["val"].label, model.predict_proba(feats["val"])[:, 1])
            if best is None or val_auc > best[0]:
                best = (val_auc, c, model)
        val_auc, c, model = best
        score = model.predict_proba(feats["test"])[:, 1]
        results[name] = {"C": c, "val_auc": round(val_auc, 3), **evaluate(test.label, score, test.cik)}
        test[f"score_{name}"] = score
        log(f"{name}: {results[name]}")

    results["by_event"] = {
        model: {ev: round(float(roc_auc_score(test[ev], test[f"score_{model}"])), 3)
                for ev in ("bankruptcy", "restatement", "auditor_change")}
        for model in ("size+prior", "tfidf", "tfidf+size+prior")
    }
    results["text_vs_size"] = paired_auc_difference(
        test.label, test["score_tfidf+size+prior"], test["score_size+prior"], test.cik)
    log(f"tfidf+size+prior minus size+prior: {results['text_vs_size']}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "baseline.json").write_text(json.dumps(results, indent=2))
    test[["accession", "cik", "label", "prior_event", "score_size+prior", "score_tfidf",
          "score_tfidf+size+prior"]].to_parquet(
        out / "baseline_test_scores.parquet")
    return results
