"""Phase 3c: a learned year-over-year change measure.

Measure (paragraph novelty): split this year's section and last year's into
paragraphs, embed them, and for each new paragraph take the cosine distance
to its nearest paragraph from last year. The document's novelty is the
length-weighted mean of those distances: untouched paragraphs score ~0, new
content scores high. Whole-document similarity (Phase 2) cannot tell "one new
risk factor" from "every sentence lightly reworded"; this can.

Fine-tuning: the embedder is taught to ignore cosmetic edits. Positives are
automatically mined pairs from consecutive years that are clearly the same
paragraph lightly edited (high lexical overlap but not identical); other
paragraphs in the batch are negatives (MultipleNegativesRankingLoss). Mining
and training use fiscal years <= 2017 only.

Evaluation: the Phase 2 check, i.e. does the measure rise when the company
completed an acquisition or disposition (8-K Item 2.01) between the two
reports, on fiscal years >= 2020, compared with TF-IDF similarity and the
untuned embedder via paired firm-bootstrap AUC differences.
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import torch

from .metrics import evaluate, paired_auc_difference

BASE = "BAAI/bge-small-en-v1.5"
MIN_PARA_CHARS = 200        # shorter lines are headings and table fragments
MAX_PARAS = 300             # per document, keeps the largest filings tractable
SEED = 20261009

PAIRS_SQL = """
with deal as (
    select sc.accession, sc.section,
           exists (
               select 1 from filings f
               join annual_reports p on p.accession = sc.prior_accession
               where f.cik = sc.cik and f.form = '8-K' and f.items like '%2.01%'
                 and f.filing_date > p.accepted_at::date
                 and f.filing_date < sc.accepted_at::date
           ) as had_deal
    from section_changes sc
)
select sc.cik, sc.section, sc.accession, sc.prior_accession,
       year(sc.fiscal_period_end) as fy, sc.sim_tfidf, d.had_deal,
       cur.text as text, pri.text as prior_text
from section_changes sc
join deal d using (accession, section)
join sections cur on cur.accession = sc.accession and cur.section = sc.section
join sections pri on pri.accession = sc.prior_accession and pri.section = sc.section
"""


def paragraphs(text: str) -> list[str]:
    paras = [p for p in text.split("\n") if len(p) >= MIN_PARA_CHARS]
    return paras[:MAX_PARAS]


def _words(p: str) -> set[str]:
    return set(re.findall(r"[a-z]+", p.lower()))


def novelty(model, cur: list[str], prior: list[str]) -> float:
    if not cur or not prior:
        return float("nan")
    e_cur = model.encode(cur, normalize_embeddings=True, batch_size=64, convert_to_tensor=True)
    e_pri = model.encode(prior, normalize_embeddings=True, batch_size=64, convert_to_tensor=True)
    nearest = (e_cur @ e_pri.T).max(dim=1).values.clamp(max=1.0)
    weights = torch.tensor([len(p) for p in cur], dtype=nearest.dtype, device=nearest.device)
    return float(((1 - nearest) * weights).sum() / weights.sum())


def mine_edit_pairs(df: pd.DataFrame, max_pairs: int = 40_000) -> list[tuple[str, str]]:
    """(this year's paragraph, last year's version of it) where the paragraph
    was lightly edited: word-set Jaccard in [0.6, 0.95)."""
    rng = random.Random(SEED)
    pairs = []
    for row in df.itertuples():
        prior = paragraphs(row.prior_text)
        prior_words = [_words(p) for p in prior]
        for p in paragraphs(row.text):
            w = _words(p)
            best, best_j = None, 0.0
            for q, qw in zip(prior, prior_words):
                j = len(w & qw) / max(1, len(w | qw))
                if j > best_j:
                    best, best_j = q, j
            if best is not None and 0.6 <= best_j < 0.95:
                pairs.append((p, best))
    rng.shuffle(pairs)
    return pairs[:max_pairs]


def finetune(pairs: list[tuple[str, str]], out: Path, log=print):
    from sentence_transformers import SentenceTransformer, losses
    from sentence_transformers.trainer import SentenceTransformerTrainer
    from sentence_transformers.training_args import BatchSamplers, SentenceTransformerTrainingArguments
    from datasets import Dataset

    model = SentenceTransformer(BASE, device="cuda")
    train = Dataset.from_dict({"anchor": [a for a, _ in pairs], "positive": [b for _, b in pairs]})
    args = SentenceTransformerTrainingArguments(
        output_dir=str(out / "embed_ckpt"), num_train_epochs=1, per_device_train_batch_size=64,
        learning_rate=2e-5, warmup_ratio=0.1, fp16=True, seed=SEED, logging_steps=50,
        batch_sampler=BatchSamplers.NO_DUPLICATES, save_strategy="no", report_to=[])
    trainer = SentenceTransformerTrainer(model=model, args=args, train_dataset=train,
                                         loss=losses.MultipleNegativesRankingLoss(model))
    trainer.train()
    path = out / "embed_finetuned"
    model.save(str(path))
    log(f"fine-tuned on {len(pairs):,} edit pairs -> {path}")
    return model


def run(con: duckdb.DuckDBPyConnection, out: Path, log=print) -> dict:
    from sentence_transformers import SentenceTransformer

    torch.manual_seed(SEED)
    df = con.execute(PAIRS_SQL).df()
    train = df[df.fy <= 2017]
    test = df[df.fy >= 2020].reset_index(drop=True)
    log(f"pairs: {len(train)} train (FY<=2017), {len(test)} test (FY>=2020), "
        f"{int(test.had_deal.sum())} test pairs span a deal")

    pairs = mine_edit_pairs(train)
    log(f"mined {len(pairs):,} lightly-edited paragraph pairs from training years")

    base = SentenceTransformer(BASE, device="cuda")
    test["novelty_base"] = [novelty(base, paragraphs(r.text), paragraphs(r.prior_text))
                            for r in test.itertuples()]
    tuned = finetune(pairs, out, log)
    test["novelty_tuned"] = [novelty(tuned, paragraphs(r.text), paragraphs(r.prior_text))
                             for r in test.itertuples()]

    results = {"base_model": BASE, "edit_pairs": len(pairs)}
    for section, g in test.groupby("section"):
        g = g.dropna(subset=["novelty_base", "novelty_tuned"])
        y, firms = g.had_deal.astype(int), g.cik
        tfidf_change = 1 - g.sim_tfidf          # higher = more change, like novelty
        results[section] = {
            "pairs": len(g), "deal_pairs": int(y.sum()),
            "tfidf": evaluate(y, tfidf_change, firms),
            "novelty_base": evaluate(y, g.novelty_base, firms),
            "novelty_tuned": evaluate(y, g.novelty_tuned, firms),
            "tuned_vs_tfidf": paired_auc_difference(y, g.novelty_tuned, tfidf_change, firms),
            "tuned_vs_base": paired_auc_difference(y, g.novelty_tuned, g.novelty_base, firms),
        }
        log(f"{section}: " + json.dumps({k: v["auc"] if isinstance(v, dict) and "auc" in v else v
                                          for k, v in results[section].items()}))
    out.mkdir(parents=True, exist_ok=True)
    (out / "embed_change.json").write_text(json.dumps(results, indent=2))
    test.drop(columns=["text", "prior_text"]).to_parquet(out / "embed_change_test.parquet")
    return results
