"""
Model loading and generation.

The GPU model is only instantiated by the pipeline when at least one artifact
must actually be created, so importing this module stays cheap.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import sys
import types
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .config import deep_get, expand, require
from .io_utils import batched, load_partial_ids
from .prompts import PromptBuilder, PromptSpec


# ---------------------------------------------------------------------------
# Reproducibility signature
# ---------------------------------------------------------------------------

def generation_signature(cfg: Mapping[str, Any], prompt: PromptSpec) -> str:
    relevant = {
        "experiment": deep_get(cfg, "experiment.name"),
        "base_model": deep_get(cfg, "model.base_model"),
        "adapter_path": deep_get(cfg, "model.adapter_path"),
        "prompt": prompt.as_dict(),
        "query_decoding": deep_get(cfg, "generation.query_decoding", {}),
        "corpus_decoding": deep_get(cfg, "generation.corpus_decoding", {}),
    }
    blob = json.dumps(relevant, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def resolve_dtype(torch: Any, dtype_name: Optional[str]) -> Any:
    if dtype_name is None or str(dtype_name).lower() in {"none", "auto"}:
        return None
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    key = str(dtype_name).lower()
    if key not in mapping:
        raise ValueError(f"Unsupported model.dtype: {dtype_name}")
    return mapping[key]


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class TextGenerator:
    def __init__(self, cfg: Mapping[str, Any]):
        try:
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "Generation requires torch, transformers, and peft."
            ) from exc

        self.torch = torch
        self.cfg = cfg

        device = str(deep_get(cfg, "model.device", "cuda"))
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but torch.cuda.is_available() is False"
            )

        if torch.cuda.is_available():
            for name, value in (
                ("enable_cudnn_sdp", False),
                ("enable_flash_sdp", True),
                ("enable_math_sdp", True),
                ("enable_mem_efficient_sdp", True),
            ):
                fn = getattr(torch.backends.cuda, name, None)
                if callable(fn):
                    fn(value)

        base_model_name = expand(require(cfg, "model.base_model"))
        adapter_path = expand(require(cfg, "model.adapter_path"))

        adapter_dir = Path(adapter_path)
        is_lora = (adapter_dir / "adapter_config.json").is_file()
        is_full = (adapter_dir / "config.json").is_file() and not is_lora
        if not is_lora and not is_full:
            raise FileNotFoundError(
                f"{adapter_path} contains neither adapter_config.json nor config.json"
            )

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                adapter_path, padding_side="left"
            )
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(
                base_model_name, padding_side="left"
            )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        dtype = resolve_dtype(torch, deep_get(cfg, "model.dtype", "float32"))
        attn_impl = deep_get(cfg, "model.attn_implementation", "sdpa")
        device_map_cfg = deep_get(cfg, "model.device_map", None)

        model_kwargs: Dict[str, Any] = {}
        if dtype is not None:
            model_kwargs["dtype"] = dtype
        if attn_impl:
            model_kwargs["attn_implementation"] = str(attn_impl)
        model_kwargs["device_map"] = (
            device_map_cfg if device_map_cfg is not None else {"": device}
        )

        if is_full:
            print(f"[model] full checkpoint: {adapter_path}")
            self.model = AutoModelForCausalLM.from_pretrained(
                adapter_path, **model_kwargs
            )
            self.base_model = None
        else:
            print(f"[model] base={base_model_name} + LoRA adapter={adapter_path}")
            self.base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name, **model_kwargs
            )
            self.model = PeftModel.from_pretrained(self.base_model, adapter_path)

        self.model.eval()
        self.community_generate = None
        self._setup_grouped_beam(cfg)

    # -- grouped / diverse beam search ------------------------------------

    def _setup_grouped_beam(self, cfg: Mapping[str, Any]) -> None:
        query_decoding = deep_get(cfg, "generation.query_decoding", {}) or {}
        corpus_decoding = deep_get(cfg, "generation.corpus_decoding", {}) or {}

        for name, decoding in (
            ("query_decoding", query_decoding),
            ("corpus_decoding", corpus_decoding),
        ):
            groups = int(decoding.get("num_beam_groups", 1))
            beams = int(decoding.get("num_beams", 1))
            returns = int(decoding.get("num_return_sequences", 1))
            do_sample = bool(decoding.get("do_sample", False))

            if groups > 1 and beams % groups != 0:
                raise ValueError(
                    f"{name}: num_beams must be divisible by num_beam_groups"
                )
            if returns > beams and not do_sample:
                raise ValueError(
                    f"{name}: num_return_sequences cannot exceed num_beams "
                    f"when do_sample=false"
                )
            if groups > 1 and do_sample:
                raise ValueError(f"{name}: num_beam_groups > 1 requires do_sample=false")

        needs_grouped_beam = (
            int(query_decoding.get("num_beam_groups", 1)) > 1
            or int(corpus_decoding.get("num_beam_groups", 1)) > 1
        )
        if not needs_grouped_beam:
            return

        community_tools = deep_get(cfg, "generation.community_tools_path", None)
        if not community_tools:
            raise ValueError(
                "generation.community_tools_path is required when num_beam_groups > 1"
            )

        community_tools = expand(community_tools)
        if community_tools not in sys.path:
            sys.path.insert(0, community_tools)

        try:
            from custom_generate.generate import generate as community_generate
        except ImportError as exc:
            raise RuntimeError(
                f"Could not import custom_generate from {community_tools}"
            ) from exc

        self.community_generate = types.MethodType(community_generate, self.model)

    # -- generation --------------------------------------------------------

    @property
    def input_device(self) -> Any:
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return self.torch.device(str(deep_get(self.cfg, "model.device", "cuda")))

    def generate_batch(
        self,
        prompts: Sequence[str],
        *,
        decoding: Mapping[str, Any],
        max_input_tokens: Optional[int] = None,
    ) -> List[str]:
        enc = self.tokenizer(
            list(prompts),
            padding=True,
            truncation=max_input_tokens is not None,
            max_length=int(max_input_tokens) if max_input_tokens is not None else None,
            return_tensors="pt",
            add_special_tokens=False,
        ).to(self.input_device)

        prompt_len = int(enc["input_ids"].shape[1])
        do_sample = bool(decoding.get("do_sample", False))

        gen_kwargs: Dict[str, Any] = {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "do_sample": do_sample,
            "max_new_tokens": int(decoding.get("max_new_tokens", 64)),
            "repetition_penalty": float(decoding.get("repetition_penalty", 1.0)),
            "num_beams": int(decoding.get("num_beams", 1)),
            "num_return_sequences": int(decoding.get("num_return_sequences", 1)),
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        groups = int(decoding.get("num_beam_groups", 1))
        if groups > 1:
            gen_kwargs["num_beam_groups"] = groups
            gen_kwargs["diversity_penalty"] = float(
                decoding.get("diversity_penalty", 1.0)
            )

        if do_sample:
            gen_kwargs["temperature"] = float(decoding.get("temperature", 1.0))
            gen_kwargs["top_k"] = int(decoding.get("top_k", 500))
            gen_kwargs["top_p"] = float(decoding.get("top_p", 0.9))

        generate_fn = self.model.generate
        if groups > 1:
            if self.community_generate is None:
                raise RuntimeError(
                    "Grouped beam generation was requested but community_generate "
                    "is not loaded"
                )
            generate_fn = self.community_generate

        with self.torch.inference_mode():
            output = generate_fn(**gen_kwargs)

        decoded = self.tokenizer.batch_decode(
            output[:, prompt_len:], skip_special_tokens=True
        )
        decoded = [text.strip() for text in decoded]

        num_return_sequences = int(decoding.get("num_return_sequences", 1))
        if num_return_sequences == 1:
            return decoded

        merged: List[str] = []
        for i in range(len(prompts)):
            group = decoded[i * num_return_sequences : (i + 1) * num_return_sequences]
            merged.append(" ".join(text for text in group if text))
        return merged

    def close(self) -> None:
        del self.model
        if self.base_model is not None:
            del self.base_model
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Resumable generation to JSONL
# ---------------------------------------------------------------------------

def generate_to_jsonl(
    generator: TextGenerator,
    *,
    items: Sequence[Tuple[str, str]],
    output_path: Path,
    batch_size: int,
    prompt_builder: PromptBuilder,
    artifact_kind: str,
    decoding: Mapping[str, Any],
    max_input_tokens: Optional[int] = None,
    resume: bool = True,
    checkpoint_every_batches: int = 1,
) -> None:
    try:
        from tqdm.auto import tqdm
    except ImportError:  # pragma: no cover
        def tqdm(x: Iterable[Any], **_: Any) -> Iterable[Any]:
            return x

    if checkpoint_every_batches <= 0:
        raise ValueError("checkpoint_every_batches must be greater than zero")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Persistent checkpoint file. Not ".tmp": it must survive a process or
    # GPU failure so that generation can resume.
    partial_path = output_path.with_name(output_path.name + ".partial.jsonl")
    expected_ids = {str(record_id) for record_id, _ in items}

    if resume:
        completed_ids = load_partial_ids(partial_path, artifact_kind)
    else:
        completed_ids = set()
        if partial_path.exists():
            partial_path.unlink()

    unexpected_ids = completed_ids - expected_ids
    if unexpected_ids:
        raise RuntimeError(
            f"Partial file contains IDs not present in the current dataset: "
            f"{sorted(unexpected_ids)[:5]}. Delete or rename {partial_path}."
        )

    pending_items = [
        (str(record_id), text)
        for record_id, text in items
        if str(record_id) not in completed_ids
    ]

    print(
        f"[resume] {artifact_kind}: {len(completed_ids):,}/{len(items):,} already "
        f"completed; {len(pending_items):,} remaining"
    )

    if pending_items:
        file_mode = "a" if partial_path.exists() else "w"
        total_batches = (len(pending_items) + batch_size - 1) // batch_size

        with partial_path.open(file_mode, encoding="utf-8") as f:
            for batch_number, batch in enumerate(
                tqdm(
                    batched(pending_items, batch_size),
                    total=total_batches,
                    desc=artifact_kind,
                    dynamic_ncols=True,
                ),
                start=1,
            ):
                ids = [record_id for record_id, _ in batch]
                prompts = [
                    prompt_builder.render(generator.tokenizer, text)
                    for _, text in batch
                ]

                predictions = generator.generate_batch(
                    prompts,
                    decoding=decoding,
                    max_input_tokens=max_input_tokens,
                )

                if len(predictions) != len(ids):
                    raise RuntimeError(
                        f"Generation returned {len(predictions)} predictions for "
                        f"{len(ids)} inputs"
                    )

                for record_id, prediction in zip(ids, predictions):
                    if artifact_kind == "queries":
                        obj = {"_id": record_id, "text": prediction}
                    elif artifact_kind == "corpus":
                        obj = {"_id": record_id, "title": "", "text": prediction}
                    else:
                        raise ValueError(f"Unknown artifact kind: {artifact_kind}")

                    f.write(json.dumps(obj, ensure_ascii=False) + "\n")

                if (
                    batch_number % checkpoint_every_batches == 0
                    or batch_number == total_batches
                ):
                    f.flush()
                    os.fsync(f.fileno())
    else:
        print(f"[resume] All {artifact_kind} records already available")

    # Verify the complete checkpoint before promoting it to the final file.
    completed_ids = load_partial_ids(partial_path, artifact_kind)
    missing_ids = expected_ids - completed_ids
    unexpected_ids = completed_ids - expected_ids

    if missing_ids or unexpected_ids:
        raise RuntimeError(
            f"Generation checkpoint is incomplete. "
            f"Completed={len(completed_ids):,}, expected={len(expected_ids):,}, "
            f"missing examples={sorted(missing_ids)[:5]}, "
            f"unexpected examples={sorted(unexpected_ids)[:5]}"
        )

    # Atomic finalization: indexing and evaluation only ever see a complete file.
    os.replace(partial_path, output_path)
    print(f"[write] Completed {artifact_kind}: {output_path}")
