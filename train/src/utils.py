import torch
import pytrec_eval
from pyserini.search.lucene import LuceneSearcher  # lazy import
from datasets import Dataset as HFDataset
import json, os, random
import math
from typing import Dict, List, Tuple, Optional,Any
from peft import  PeftModel


def build_prompt(tokenizer,system_prompt,user_queries,device="cuda") -> torch.LongTensor:
    """
    prompts: list of chat conversations
            [
                [ {"role": "system", ...}, {"role": "user", ...} ],
                [ {"role": "system", ...}, {"role": "user", ...} ],
                ...
            ]
    """
    batch_prompts = [
    [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content":"[QUERY]: "+ query.strip()+"\n[KEYWORDS]: "},
    ]
    for query in user_queries
    ]
    toks = tokenizer.apply_chat_template(
        batch_prompts,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt",
        return_dict=True,
        padding=True   # important to get [B, T]

    ).to(device)

    return toks




def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def format_path(template: str, **kwargs) -> str:
    return template.format(**kwargs)


def read_jsonl(path: str) -> List[Dict[str, str]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def mean_and_var(vals: List[float], ddof: int = 0) -> Tuple[float, float]:
        n = len(vals)
        if n == 0:
            return float("nan"), float("nan")
        m = sum(vals) / n
        denom = n - ddof
        if denom <= 0:
            return m, float("nan")
        # one-pass stable enough for typical score ranges; can switch to Welford if needed
        sse = sum((x - m) ** 2 for x in vals)
        v = sse / denom
        return m, v
def write_records_with_batch_score_log(
    out_jsonl: str,
    records: List[Dict],
    batch_idx: int,
    iter_idx: int,
    log_path: Optional[str] = None,
    ddof: int = 0,  # 0 = population variance, 1 = sample variance
) -> Tuple[float, float, float, float]:
    """
    Write jsonl records and log mean/variance for the batch.

    Expected record format (keys):
      - "score-q": float
      - "score-t": float
      - "qid", "query", "target" ...

    Returns:
      (mean_score_q, var_score_q, mean_score_t, var_score_t)
    """

    # compute lists safely (ignore missing/None)
    sq_vals = [r.get("score-q") for r in records if r.get("score-q") is not None]
    st_vals = [r.get("score-t") for r in records if r.get("score-t") is not None]

    mean_sq, var_sq = mean_and_var(sq_vals, ddof=ddof)
    mean_st, var_st = mean_and_var(st_vals, ddof=ddof)

    std_sq = math.sqrt(var_sq) if not math.isnan(var_sq) else float("nan")
    std_st = math.sqrt(var_st) if not math.isnan(var_st) else float("nan")

    # write jsonl
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # log line
    log_line = (
        f"[iter={iter_idx} batch={batch_idx}] "
        f"mean(score-q)={mean_sq:.4f} std(score-q)={std_sq:.4f} | "
        f"mean(score-t)={mean_st:.4f} std(score-t)={std_st:.4f} "
        f"(n={len(records)})"
    )

    print(log_line)

    if log_path is not None:
        ensure_dir(os.path.dirname(log_path))
        with open(log_path, "a", encoding="utf-8") as lf:
            lf.write(log_line + "\n")

    return mean_sq, var_sq, mean_st, var_st


class CausalLMPadCollator:
    def __init__(self, tokenizer):
        self.tok = tokenizer
        if self.tok.pad_token_id is None:
            # common for some causal LMs
            self.tok.pad_token = self.tok.eos_token

    def __call__(self, features):
        max_len = max(len(f["input_ids"]) for f in features)

        input_ids, attention_mask, labels = [], [], []
        for f in features:
            l = len(f["input_ids"])
            pad = max_len - l

            input_ids.append(f["input_ids"] + [self.tok.pad_token_id] * pad)
            attention_mask.append(f["attention_mask"] + [0] * pad)
            labels.append(f["labels"] + [-100] * pad)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
    






def _atomic_json_dump(obj, path: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f)
    os.replace(tmp, path)

def _atomic_torch_save(obj, path: str) -> None:
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)

def _to_cpu(x):
    # Move tensors inside nested structures to CPU for safer saving.
    if torch.is_tensor(x):
        return x.detach().cpu()
    if isinstance(x, dict):
        return {k: _to_cpu(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        t = [_to_cpu(v) for v in x]
        return type(x)(t) if isinstance(x, tuple) else t
    return x

def _is_main_process(trainer) -> bool:
    # Works with Trainer+Accelerate
    if hasattr(trainer, "is_world_process_zero") and callable(trainer.is_world_process_zero):
        return trainer.is_world_process_zero()
    if hasattr(trainer, "accelerator"):
        return trainer.accelerator.is_main_process
    return True

def _save_full_checkpoint(
    trainer,
    tokenizer,
    out_dir: str,
    marker_dir: str,
    iter_idx: int,
    batch_idx: int,
    opt_step: int,
    best_eval_delta:float,
):
    os.makedirs(out_dir, exist_ok=True)
    model = trainer.model

    # Unwrap accelerator wrapper if needed
    unwrapped = trainer.accelerator.unwrap_model(model)

    if isinstance(unwrapped, PeftModel):
        unwrapped.save_pretrained(out_dir)          # saves adapter_config.json + weights
    else:
        unwrapped.save_pretrained(out_dir)          # saves config.json + full weights
        # ← this line must NOT be conditional/missing for the non-LoRA path

    tokenizer.save_pretrained(out_dir)

    # 2) Save optimizer/scheduler (+ opt_step) + RNG
    state = {
        "opt_step": int(opt_step),
        "optimizer": _to_cpu(trainer.optimizer.state_dict()),
        "scheduler": _to_cpu(trainer.lr_scheduler.state_dict()),
        "rng": {
            "python": random.getstate(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    _atomic_torch_save(state, os.path.join(out_dir, "trainer_state.pt"))

    # 3) Update resume marker (what was the *last completed* batch)
    last_ckpt_path = os.path.join(marker_dir, "last_ckpt.json")
    os.makedirs(marker_dir, exist_ok=True)
    _atomic_json_dump(
        {
            "ckpt_dir": out_dir,
            "it": int(iter_idx),
            "batch_idx": int(batch_idx),
            "opt_step": int(opt_step),
            "best_eval": None if math.isinf(best_eval_delta) else float(best_eval_delta),
        },
        last_ckpt_path,
    )

def _load_full_checkpoint_state(trainer, ckpt_dir: str):
    st_path = os.path.join(ckpt_dir, "trainer_state.pt")
    
    if not os.path.isfile(st_path):
        return None
    print("yes there is trainer_state")

    st = torch.load(st_path, map_location="cpu")

    # Optimizer/scheduler must exist already
    trainer.optimizer.load_state_dict(st["optimizer"])
    trainer.lr_scheduler.load_state_dict(st["scheduler"])

    # RNG (optional but useful)
    rng = st.get("rng", {})
    try:
        if "python" in rng:
            random.setstate(rng["python"])
        if "torch" in rng:
            torch.set_rng_state(rng["torch"])
        if torch.cuda.is_available() and rng.get("cuda") is not None:
            torch.cuda.set_rng_state_all(rng["cuda"])
    except Exception:
        # If RNG restore fails, training still resumes; it just won't be bitwise identical.
        pass

    return st


def _resume_marker_path(out_dir) -> str:
    return os.path.join(out_dir, "last_ckpt.json")


def _load_last_ckpt(out_dir):
    path = _resume_marker_path(out_dir)
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
def lora_fingerprint(m):
    # works if LoRA params exist
    s = 0.0
    for n, p in m.named_parameters():
        if "lora_" in n:
            s += float(p.detach().abs().sum().cpu())
            break
    return s

def _generation_target(model):
    # unwrap DDP / DataParallel
    if hasattr(model, "module"):
        model = model.module

    # unwrap PEFT
    if hasattr(model, "base_model"):
        bm = model.base_model
        # in PEFT, base_model is often a wrapper that contains the real HF model in `.model`
        if hasattr(bm, "model") and hasattr(bm.model, "generate"):
            return bm.model
        if hasattr(bm, "generate"):
            return bm

    return model

