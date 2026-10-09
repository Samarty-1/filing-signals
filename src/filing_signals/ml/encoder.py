"""Phase 3a: fine-tune a FinBERT-class encoder on Risk Factors text.

Risk Factors sections run to ~11k tokens at the median; BERT reads 512. Each
document is cut into 512-token chunks that all carry the document's label
(multiple-instance learning, weak labels). Training samples a few chunks per
document per epoch; scoring averages chunk logits over up to EVAL_CHUNKS
chunks. The epoch is chosen on validation AUC, and the test years are scored
once with that checkpoint.

To combine text with size and history, a logistic regression on (encoder
score, log public float, missing-float flag, prior_event) is fitted on the
validation years. That reuses validation data already used to pick the
epoch, which the results state. The key comparison is against the
size+prior and TF-IDF baselines on identical test rows.
"""
from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from transformers import BertConfig, BertForSequenceClassification, get_linear_schedule_with_warmup

from .metrics import evaluate, paired_auc_difference

MODEL = "yiyanghkust/finbert-pretrain"
MAX_LEN = 512
TRAIN_CHUNKS = 8          # sampled per document per epoch
EVAL_CHUNKS = 64          # scored per document (covers the median section ~2x)
EPOCHS = 4
LR = 2e-5
BATCH = 16
SEED = 20261009


def load_tokenizer():
    """FinBERT ships only a WordPiece vocab.txt. transformers v5 can't build a
    tokenizer from that by itself, and BertTokenizerFast(vocab_file=...)
    silently yields a 5-token vocabulary that maps every word to [UNK]. So the
    tokenizer is built with the `tokenizers` library, then checked."""
    from huggingface_hub import hf_hub_download
    from tokenizers import BertWordPieceTokenizer
    from transformers import PreTrainedTokenizerFast

    vocab = hf_hub_download(MODEL, "vocab.txt")
    wordpiece = BertWordPieceTokenizer(vocab, lowercase=True)
    tok = PreTrainedTokenizerFast(tokenizer_object=wordpiece._tokenizer, unk_token="[UNK]",
                                  cls_token="[CLS]", sep_token="[SEP]", pad_token="[PAD]",
                                  mask_token="[MASK]")
    probe = tok.tokenize("substantial doubt about our ability to continue as a going concern")
    if tok.vocab_size < 30_000 or "[UNK]" in probe:
        raise RuntimeError(f"FinBERT tokenizer is broken: vocab {tok.vocab_size}, probe {probe}")
    return tok


def chunk(tokenizer, texts: list[str], max_chunks: int) -> list[list[list[int]]]:
    """Token-id windows of MAX_LEN-2 per document (CLS/SEP added at batch time)."""
    width = MAX_LEN - 2
    out = []
    for text in texts:
        ids = tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
        windows = [ids[i:i + width] for i in range(0, len(ids), width)] or [[]]
        out.append(windows[:max_chunks])
    return out


def collate(tokenizer, windows: list[list[int]], device):
    cls, sep, pad = tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id
    seqs = [[cls] + w + [sep] for w in windows]
    width = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), width), pad, dtype=torch.long)
    mask = torch.zeros((len(seqs), width), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s)
        mask[i, :len(s)] = 1
    return ids.to(device), mask.to(device)


@torch.no_grad()
def score_documents(model, tokenizer, doc_windows, device) -> np.ndarray:
    model.eval()
    flat = [(d, w) for d, ws in enumerate(doc_windows) for w in ws]
    logits = np.zeros(len(flat), dtype=np.float32)
    for i in range(0, len(flat), BATCH * 4):
        batch = flat[i:i + BATCH * 4]
        ids, mask = collate(tokenizer, [w for _, w in batch], device)
        with torch.autocast("cuda", dtype=torch.float16):
            out = model(input_ids=ids, attention_mask=mask).logits.float().squeeze(-1)
        logits[i:i + len(batch)] = out.cpu().numpy()
    doc_ids = np.array([d for d, _ in flat])
    sums = np.bincount(doc_ids, weights=logits, minlength=len(doc_windows))
    counts = np.bincount(doc_ids, minlength=len(doc_windows))
    return sums / np.maximum(counts, 1)


def run(dataset: Path, out: Path, log=print) -> dict:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cuda")

    df = pd.read_parquet(dataset)
    d = {s: df[df.split == s].reset_index(drop=True) for s in ("train", "val", "test")}
    tokenizer = load_tokenizer()
    # the checkpoint's config.json predates `model_type`, so Auto* can't place
    # it; it is a plain BERT-base (BertForMaskedLM) and is loaded as one
    config = BertConfig.from_pretrained(MODEL, num_labels=1)
    model = BertForSequenceClassification.from_pretrained(MODEL, config=config).to(device)

    t0 = time.perf_counter()
    windows = {
        "train": chunk(tokenizer, d["train"].text.tolist(), max_chunks=10_000),
        "val": chunk(tokenizer, d["val"].text.tolist(), EVAL_CHUNKS),
        "test": chunk(tokenizer, d["test"].text.tolist(), EVAL_CHUNKS),
    }
    n_tok = [sum(len(w) for w in ws) for ws in windows["train"]]
    log(f"tokenised in {time.perf_counter() - t0:.0f}s; train tokens/doc median {int(np.median(n_tok)):,}")

    y_train = d["train"].label.astype(np.float32).to_numpy()
    pos_weight = torch.tensor([(1 - y_train.mean()) / y_train.mean()], device=device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    steps_per_epoch = math.ceil(len(windows["train"]) * TRAIN_CHUNKS / BATCH)
    sched = get_linear_schedule_with_warmup(optim, int(0.06 * steps_per_epoch * EPOCHS),
                                            steps_per_epoch * EPOCHS)
    scaler = torch.amp.GradScaler()

    best = {"val_auc": -1.0}
    history = []
    ckpt = out / "encoder_best.pt"
    out.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, EPOCHS + 1):
        model.train()
        examples = []
        for doc, ws in enumerate(windows["train"]):
            for w in random.sample(ws, min(TRAIN_CHUNKS, len(ws))):
                examples.append((w, y_train[doc]))
        random.shuffle(examples)
        t0, running = time.perf_counter(), 0.0
        for step in range(0, len(examples), BATCH):
            batch = examples[step:step + BATCH]
            ids, mask = collate(tokenizer, [w for w, _ in batch], device)
            target = torch.tensor([y for _, y in batch], device=device)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(input_ids=ids, attention_mask=mask).logits.float().squeeze(-1)
            loss = loss_fn(logits, target)
            optim.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optim)
            scaler.update()
            sched.step()
            running += loss.item()
        val_score = score_documents(model, tokenizer, windows["val"], device)
        val_auc = roc_auc_score(d["val"].label, val_score)
        n_batches = math.ceil(len(examples) / BATCH)
        history.append({"epoch": epoch, "train_loss": round(running / n_batches, 4),
                        "val_auc": round(val_auc, 3), "minutes": round((time.perf_counter() - t0) / 60, 1)})
        log(f"epoch {epoch}: {history[-1]}")
        if val_auc > best["val_auc"]:
            best = {"val_auc": val_auc, "epoch": epoch}
            torch.save(model.state_dict(), ckpt)

    model.load_state_dict(torch.load(ckpt, map_location=device))
    val_score = score_documents(model, tokenizer, windows["val"], device)
    test_score = score_documents(model, tokenizer, windows["test"], device)
    test = d["test"]

    # encoder score + size + prior_event, combined on the validation years
    def stacked(df, score):
        return np.column_stack([score, df.log_float, (~df.has_float).astype(float),
                                df.prior_event.astype(float)])
    stack = LogisticRegression(max_iter=1000).fit(stacked(d["val"], val_score), d["val"].label)
    test_combined = stack.predict_proba(stacked(test, test_score))[:, 1]

    baseline = pd.read_parquet(out / "baseline_test_scores.parquet")
    assert (baseline.accession.to_numpy() == test.accession.to_numpy()).all()
    results = {
        "model": MODEL, "epochs_run": EPOCHS, "best_epoch": best["epoch"],
        "history": history,
        "encoder": evaluate(test.label, test_score, test.cik),
        "encoder+size+prior": evaluate(test.label, test_combined, test.cik),
        "encoder_by_event": {ev: round(float(roc_auc_score(test[ev], test_score)), 3)
                             for ev in ("bankruptcy", "restatement", "auditor_change")},
        "vs_tfidf": paired_auc_difference(test.label, test_score, baseline["score_tfidf"], test.cik),
        "combined_vs_tfidf+size+prior": paired_auc_difference(
            test.label, test_combined, baseline["score_tfidf+size+prior"], test.cik),
        "combined_vs_size+prior": paired_auc_difference(
            test.label, test_combined, baseline["score_size+prior"], test.cik),
    }
    (out / "encoder.json").write_text(json.dumps(results, indent=2))
    test.assign(score_encoder=test_score, score_encoder_combined=test_combined)[
        ["accession", "cik", "label", "score_encoder", "score_encoder_combined"]].to_parquet(
        out / "encoder_test_scores.parquet")
    return results
