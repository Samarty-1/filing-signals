"""Phase 3b: QLoRA fine-tuning of a small LLM to read revenue out of MD&A.

Task: given MD&A passages and the fiscal year end, answer the year's total
revenue in US dollars as an integer, or "unknown" when the passage does not
state it. Training targets follow that rule. An example whose figure is not
in the passage is trained towards "unknown", because training a model to
answer it anyway teaches it to invent numbers.

Scoring
  answerable examples:   correct = within 0.5% of the XBRL figure
  unanswerable examples: abstention rate (answers "unknown")
Compared on the same test rows: a regex extractor, the same model zero-shot
with the same prompt, and the fine-tuned model. The zero-shot comparison
isolates what fine-tuning adds.

QLoRA: the base model is loaded in 4-bit NF4 and frozen; LoRA adapters
(r=16) on every attention and MLP projection are trained in bf16, with loss
on the answer tokens only.
"""
from __future__ import annotations

import json
import math
import random
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_PROMPT_TOKENS = 3000
LR = 2e-4
EPOCHS = 1
GRAD_ACCUM = 8
SEED = 20261009
TOLERANCE = 0.005

SYSTEM = "You extract figures from SEC filings. Answer with the number only."


def user_prompt(passage: str, fiscal_year_end: str) -> str:
    return (f"Below are passages from the MD&A section of a 10-K for the fiscal year ended "
            f"{fiscal_year_end}.\n\nWhat was the company's total revenue for that fiscal year, in "
            f"US dollars? Answer with a single integer number of dollars (for example "
            f"1234567000), converting from thousands or millions if the filing uses them. If "
            f"the passages do not state it, answer unknown.\n\nPassages:\n{passage}")


def target(row) -> str:
    return str(int(round(row.revenue_usd))) if row.answerable else "unknown"


def parse_answer(text: str) -> float | None:
    """The model's answer as dollars, or None for unknown / unparseable."""
    text = text.strip().lower()
    if text.startswith("unknown"):
        return None
    m = re.search(r"\d[\d,]*(?:\.\d+)?", text)
    if not m:
        return None
    value = float(m.group(0).replace(",", ""))
    for word, scale in (("billion", 1e9), ("million", 1e6), ("thousand", 1e3)):
        if word in text[m.end():m.end() + 12]:
            value *= scale
    return value


# --- regex baseline ------------------------------------------------------------

_REGEX = re.compile(
    r"(?:total\s+(?:net\s+)?revenues?|net\s+revenues?|total\s+net\s+sales|net\s+sales|revenues?)"
    r"[^$\d]{0,80}\$?\s*(\d[\d,]*(?:\.\d+)?)\s*(billion|million|thousand)?", re.I)
_SCALE_NOTE = re.compile(r"in\s+(thousands|millions|billions)", re.I)


def regex_extract(passage: str) -> float | None:
    """First 'revenue ... <number> [scale]' match; a table-level '(in thousands)'
    note supplies the scale when the number has none."""
    m = _REGEX.search(passage)
    if not m:
        return None
    value = float(m.group(1).replace(",", ""))
    scale = (m.group(2) or "").lower()
    if not scale:
        note = _SCALE_NOTE.search(passage)
        scale = note.group(1).lower().rstrip("s") if note else ""
    return value * {"billion": 1e9, "million": 1e6, "thousand": 1e3}.get(scale, 1)


# --- scoring --------------------------------------------------------------------

def score(df: pd.DataFrame, predictions: list[float | None]) -> dict:
    pred = pd.Series(predictions, index=df.index, dtype="float64")
    ans = df.answerable
    correct = (pred - df.revenue_usd).abs() <= TOLERANCE * df.revenue_usd
    return {
        "answerable_n": int(ans.sum()),
        "accuracy_answerable": round(float(correct[ans].mean()), 3),
        "answered_answerable": round(float(pred[ans].notna().mean()), 3),
        "unanswerable_n": int((~ans).sum()),
        "abstain_unanswerable": round(float(pred[~ans].isna().mean()), 3),
        # among answers given on unanswerable examples, how many still matched
        # (figures can appear in forms the answerability check doesn't cover)
        "accuracy_all": round(float(correct.mean()), 3),
    }


# --- model -------------------------------------------------------------------------
# torch is imported inside each function so the parsing and scoring helpers
# above can be used (and tested) without the GPU stack installed.

def load_model(adapter: Path | None = None):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = "left"
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=quant,
                                                 dtype=torch.bfloat16, device_map={"": 0})
    if adapter is not None:
        model = PeftModel.from_pretrained(model, adapter)
    return tok, model


def encode_prompt(tok, row) -> list[int]:
    # truncate the passage, never the instructions, to fit the token budget
    passage = row.passage
    while True:
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": user_prompt(passage, str(row.fiscal_period_end))}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
        if hasattr(ids, "input_ids"):
            ids = ids["input_ids"]
        if len(ids) <= MAX_PROMPT_TOKENS or len(passage) < 200:
            return list(ids)
        passage = passage[: int(len(passage) * 0.85)]


def generate(tok, model, df: pd.DataFrame, batch: int = 8, log=print) -> list[str]:
    import torch

    model.eval()
    with torch.no_grad():
        return _generate(tok, model, df, batch, log)


def _generate(tok, model, df, batch, log) -> list[str]:
    import torch

    prompts = [encode_prompt(tok, r) for r in df.itertuples()]
    order = np.argsort([len(p) for p in prompts])     # length-sorted batches pad less
    out = [""] * len(prompts)
    t0 = time.perf_counter()
    for b in range(0, len(order), batch):
        idx = order[b:b + batch]
        width = max(len(prompts[i]) for i in idx)
        ids = torch.full((len(idx), width), tok.pad_token_id, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for row, i in enumerate(idx):
            p = prompts[i]
            ids[row, width - len(p):] = torch.tensor(p)
            mask[row, width - len(p):] = 1
        gen = model.generate(input_ids=ids.cuda(), attention_mask=mask.cuda(), max_new_tokens=16,
                             do_sample=False, pad_token_id=tok.pad_token_id)
        for row, i in enumerate(idx):
            out[i] = tok.decode(gen[row, width:], skip_special_tokens=True)
        if (b // batch) % 25 == 0:
            log(f"  generated {b + len(idx)}/{len(order)} ({time.perf_counter() - t0:.0f}s)")
    return out


CKPT_EVERY = 25     # optimizer steps between resumable checkpoints


def _cap_gpu_memory():
    """On Windows the driver silently spills VRAM overflow into system RAM,
    which made training ~13x slower. Capping PyTorch's share makes its
    allocator free cached blocks instead of growing past the card."""
    import torch
    torch.cuda.set_per_process_memory_fraction(0.85)


def train(df: pd.DataFrame, out: Path, log=print) -> Path:
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, set_peft_model_state_dict

    random.seed(SEED)
    torch.manual_seed(SEED)
    _cap_gpu_memory()
    tok, model = load_model()
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"QLoRA: {trainable / 1e6:.1f}M trainable parameters")

    examples = []
    for row in df.itertuples():
        prompt = encode_prompt(tok, row)
        answer = tok(target(row) + tok.eos_token, add_special_tokens=False)["input_ids"]
        # loss on the answer only: prompt positions are masked with -100
        examples.append((prompt + answer, [-100] * len(prompt) + answer))

    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
    total = math.ceil(len(examples) * EPOCHS / GRAD_ACCUM)
    sched = torch.optim.lr_scheduler.LambdaLR(
        optim, lambda s: min(1.0, (s + 1) / max(1, total // 20)) * max(0.0, 1 - s / total))
    # resumable: the shuffle is seeded, so a checkpoint's step count says
    # exactly which examples have been consumed
    ckpt = out / "qlora_train_ckpt.pt"
    start_step = 0
    if ckpt.exists():
        state = torch.load(ckpt, weights_only=False)
        set_peft_model_state_dict(model, state["adapter"])
        optim.load_state_dict(state["optim"])
        sched.load_state_dict(state["sched"])
        start_step = state["step"]
        log(f"  resuming from step {start_step}/{total}")

    model.train()
    step, running, t0 = 0, 0.0, time.perf_counter()
    for epoch in range(EPOCHS):
        random.shuffle(examples)
        for k, (ids, labels) in enumerate(examples, start=1):
            if (k - 1) // GRAD_ACCUM < start_step:      # already trained on
                if k % GRAD_ACCUM == 0:
                    step += 1
                continue
            ids_t = torch.tensor([ids], device="cuda")
            labels_t = torch.tensor([labels], device="cuda")
            loss = model(input_ids=ids_t, labels=labels_t).loss / GRAD_ACCUM
            loss.backward()
            running += loss.item()
            if k % GRAD_ACCUM == 0 or k == len(examples):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                sched.step()
                optim.zero_grad(set_to_none=True)
                step += 1
                if step % 25 == 0:
                    log(f"  step {step}/{total} loss {running / 25:.4f} "
                        f"({(time.perf_counter() - t0) / 60:.1f} min) {_rss()}")
                    running = 0.0
                if step % CKPT_EVERY == 0:
                    from peft import get_peft_model_state_dict
                    torch.save({"step": step, "adapter": get_peft_model_state_dict(model),
                                "optim": optim.state_dict(), "sched": sched.state_dict()}, ckpt)
                    torch.cuda.empty_cache()
    adapter = out / "qlora_revenue_adapter"
    model.save_pretrained(adapter)
    ckpt.unlink(missing_ok=True)
    return adapter


def _rss() -> str:
    try:
        import psutil
        return f"[process RAM {psutil.Process().memory_info().rss / 1e9:.1f} GB]"
    except ImportError:
        return ""


def run(dataset: Path, out: Path, log=print) -> dict:
    import gc

    import torch

    df = pd.read_parquet(dataset)
    out.mkdir(parents=True, exist_ok=True)
    tr, te = df[df.split == "train"], df[df.split == "test"].reset_index(drop=True)
    results = {"model": MODEL, "train_examples": len(tr),
               "train_answerable": round(float(tr.answerable.mean()), 3)}

    results["regex"] = score(te, [regex_extract(p) for p in te.passage])
    log(f"regex: {results['regex']} {_rss()}")

    # each stage is checkpointed, so an interrupted run resumes where it stopped
    zero_path = out / "revenue_zero_shot.parquet"
    if zero_path.exists():
        zero = pd.read_parquet(zero_path).answer.tolist()
        log("zero-shot: reusing saved answers")
    else:
        tok, model = load_model()
        log(f"base model loaded {_rss()}")
        zero = generate(tok, model, te, log=log)
        pd.DataFrame({"accession": te.accession, "answer": zero}).to_parquet(zero_path)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    results["zero_shot"] = score(te, [parse_answer(a) for a in zero])
    log(f"zero-shot: {results['zero_shot']} {_rss()}")

    adapter = out / "qlora_revenue_adapter"
    if (adapter / "adapter_config.json").exists():
        log("qlora: reusing trained adapter")
    else:
        adapter = train(tr, out, log=log)
        gc.collect()
        torch.cuda.empty_cache()
    _cap_gpu_memory()
    tok, model = load_model(adapter)
    tuned = generate(tok, model, te, log=log)
    results["qlora"] = score(te, [parse_answer(a) for a in tuned])
    log(f"qlora: {results['qlora']}")

    te.assign(zero_shot=zero, qlora=tuned)[
        ["accession", "revenue_usd", "answerable", "zero_shot", "qlora"]].to_parquet(
        out / "revenue_test_predictions.parquet")
    (out / "llm_extract.json").write_text(json.dumps(results, indent=2))
    return results
