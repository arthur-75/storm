from src.utils import set_seed, format_path, ensure_dir, write_records_with_batch_score_log, CausalLMPadCollator
from src.utils import _load_full_checkpoint_state, _save_full_checkpoint, _load_last_ckpt
from src.data import PromptSpec, PromptBuilder, QueryDataset,get_dl
from src.keyword import GenerationConfig, RewardConfig, BeamNDCGGenerator
from src.reward import EvalBuilder
import torch.nn.functional as F
from tqdm.auto import tqdm
import math
from typing import List, Dict, Tuple, Optional, Any
from peft import LoraConfig, get_peft_model, PeftModel
import pickle
import torch
import os
import gc

from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)
import json
import random
from dataclasses import dataclass
from pathlib import Path
import shutil


torch.backends.cuda.enable_cudnn_sdp(False)
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)

# ─────────────────────────────────────────────────────────────────────────────
# Data classes (unchanged from original)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BatchRecord:
    iter_idx:           int
    batch_idx:          int
    opt_step:           int
    qids:               list
    queries:            list
    targets:            list
    classic_gen:        list   # now per-sample bool list (was a single bool)

    reward:             list
    scaled_reward:      list
    sig_reward:         list
    sig_scaled_reward:  list
    score_q:            list
    score_t:            list

    pi:                 list
    log_qi:             list
    log_wi:             list
    w_i:                list

    batch_loss:         float
    batch_w_norm:       float
    epsilon_qi:         float


class BatchDebugLogger:
    """Appends one row per sample to a JSONL file."""

    def __init__(self, path: str, flush_every: int = 1):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.flush_every = flush_every
        self._buf: list[dict] = []

    def log(self, record: BatchRecord) -> None:
        n = len(record.qids)
        for i in range(n):
            row = {
                "iter":              record.iter_idx,
                "batch":             record.batch_idx,
                "opt_step":          record.opt_step,
                "qid":               record.qids[i],
                "query":             record.queries[i],
                "target":            record.targets[i],
                "classic_gen":       record.classic_gen[i],   # per-sample
                "reward":            self._to_float(record.reward, i),
                "scaled_reward":     self._to_float(record.scaled_reward, i),
                "sig_reward":        self._to_float(record.sig_reward, i),
                "sig_scaled_reward": self._to_float(record.sig_scaled_reward, i),
                "score_q":           self._to_float(record.score_q, i),
                "score_t":           self._to_float(record.score_t, i),
                "pi":                self._to_float(record.pi, i),
                "log_qi":            self._to_float(record.log_qi, i),
                "log_wi":            self._to_float(record.log_wi, i),
                "w_i":               self._to_float(record.w_i, i),
                "batch_loss":        record.batch_loss,
                "batch_w_norm":      record.batch_w_norm,
                "epsilon_qi":        record.epsilon_qi,
            } 
            self._buf.append(row)

        if len(self._buf) >= self.flush_every:
            self._flush()

    def _flush(self) -> None:
        with open(self.path, "a", encoding="utf-8") as fh:
            for row in self._buf:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._buf.clear()

    def close(self) -> None:
        if self._buf:
            self._flush()

    @staticmethod
    def _to_float(obj, idx: int) -> float:
        try:
            return float(obj[idx])
        except Exception:
            return float("nan")


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class SelfTrainPipeline:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        set_seed(int(cfg.get("seed", 42)))

        self.device = cfg.get("device", "cuda")
        self.base_model = cfg["pipeline"]["base_model_name_or_path"]
        self.num_iterations = int(cfg["pipeline"]["num_iterations"])
        self.run_generation = bool(cfg["pipeline"].get("run_generation", True))
        self.run_training = bool(cfg["pipeline"].get("run_training", True))

        pkl_path = cfg["data"]["pkl_path"]
        with open(pkl_path + "msmarco_queries_qrels.pkl", "rb") as f:
            queries, qrels = pickle.load(f)
        self.queries = queries
        del qrels
        gc.collect()

        g = cfg["generation"]
        self.epsilon_qi = g["epsilon_qi"]
        self.gen_top_k=int(g["top_k"])
        self.gen_top_p=float(g["top_p"])

        self.train_cfg = cfg["training"]
        self.msmarco_v =self.train_cfg["msmarco_v"]
        self.save_steps = self.train_cfg["save_steps"]
        self.token_loss = self.train_cfg["token_loss"]

        r = cfg["reward"]
        retrieval_type = str(r.get("retrieval_type", "bm25")).lower()
        index_path = r.get("index_path", pkl_path + "msmarco_docIndex" + self.msmarco_v)
        if not retrieval_type=="splade": index_path= pkl_path + "msmarco_docIndex" + self.msmarco_v
        msmarco_path= pkl_path + "msmarco_docIndex" + self.msmarco_v
        ret_both= r.get("ret_both",False)


        self.eval_obj = EvalBuilder(
            ce_tsv_path=pkl_path + "dictCE.tsv",
            INDEX_PATH=index_path,#pkl_path + "msmarco_docIndex"+self.msmarco_v,
            lambda_term_freq=r["lambda_term_freq"],
            lambda_llk=r["lambda_llk"],
            freq=r["use_freq"],
            dict_ndcg=pkl_path + "dict_ndcg.json", # dict_ndcg= "beam/data/dict_ndcg.json",
            retrieval_type=retrieval_type,
            splade_query_encoder=r.get("splade_query_encoder", "naver/splade-cocondenser-ensembledistil"),
            splade_batch_size = r.get("splade_batch_size", 16),
            msmarco_path=msmarco_path,
            ret_both=ret_both,
            spalde_query_path=r.get("spalde_query_path", None),
            lambda_sp=r.get("lambda_sp", 0.1),

        )
        print(index_path, retrieval_type)

        p = cfg["prompt"]
        self.prompt_spec = PromptSpec(
            system_prompt=p["system_prompt"],
            user_prefix=p.get("user_prefix", "[QUERY]: "),
            user_suffix=p.get("user_suffix", "\n[KEYWORDS]: "),
        )
        self.prompt_builder = PromptBuilder(self.prompt_spec)

        self.bm25_k = int(r["bm25_k"])
        self.best_beam_sel = bool(r.get("best_beam_sel", False))
        self.lambda_ndcg = r["lambda_ndcg"]

        self.num_beams = int(g["num_beams"])
        self.qi_num_beam = self.num_beams if not self.best_beam_sel else 1
        self.max_new_tokens_gen = int(g["max_new_tokens"])
        self.gen_cfg = GenerationConfig(
            batch_size=int(g["batch_size"]),
            max_new_tokens=int(g["max_new_tokens"]),
            num_beams=self.num_beams,
            repetition_penalty=float(g["repetition_penalty"]),
            do_sample=bool(g.get("do_sample", False)),
            temperature=float(g["temperature"]),
          # 
            
        )

        self.apply_benchmark = r["apply_benchmark"]
        self.reward_cfg = RewardConfig(
            bm25_k=self.bm25_k,
            topk_per_beam=int(r["topk_per_beam"]),
            show_more=bool(r.get("show_more", False)),
            ndcg=str(r.get("ndcg", "soft")),
            _use_likelihood=True,
            best_beam_sel=self.best_beam_sel,
        )

       

        self.w_norm_buffer = 0
        self.loss_buffer = 0

        debug_log_path = format_path(
            self.cfg["logging"]["debug_log_path"],
            exp_name=self.cfg["exp_name"],
        )
        debug_path = os.path.join(
            os.path.dirname(debug_log_path),
            f"{self.cfg['exp_name']}_batch_debug.jsonl",
        )
        self.debug_logger = BatchDebugLogger(path=debug_path, flush_every=1)

        self.eval_cfg = self.cfg.get("eval", {"enabled": False})
        self.eval_enabled = bool(self.eval_cfg.get("enabled", False))

        all_qids = list(self.queries.keys())
        n_hold = int(self.eval_cfg.get("num_queries", 500))
        self.qids = all_qids
        if (
            self.eval_enabled
            and cfg["data"].get("max_queries", None) is None
            and len(all_qids) > n_hold
        ):
            self.qids = all_qids[:-n_hold]
            self.holdout_qids = all_qids[-n_hold:]
        else:
            train_set = set(self.qids)
            heldout_candidates = [q for q in all_qids if q not in train_set]
            self.holdout_qids = (
                heldout_candidates[-n_hold:]
                if len(heldout_candidates) >= n_hold
                else heldout_candidates
            )

        self.holdout_queries = [self.queries[qid] for qid in self.holdout_qids]
        holdout_score_q = {
            qid: float(self.eval_obj.dict_ndcg[qid]) for qid in self.holdout_qids
        }
        self.sq_vals = [holdout_score_q[qid] for qid in self.holdout_qids]
        self.mean_sq = (
            sum(self.sq_vals) / len(self.sq_vals) if self.sq_vals else float("nan")
        )
        self.dl_data_root = cfg["data"].get("dl_data_root", None)
        if self.dl_data_root:
            self.dl19_qu, self.dl19_qrls = get_dl("2019", self.dl_data_root)
            self.dl20_qu, self.dl20_qrls = get_dl("2020", self.dl_data_root)
            self.dlhard_qu, self.dlhard_qrls = get_dl("hard", self.dl_data_root)

        # ── Curriculum: sort training qids easiest → hardest ──────────────
        # Easiest = highest BM25 NDCG (model has least to gain, stable signal).
        # The holdout is already fixed above and is NOT affected by this sort.
        self.curriculum = cfg["data"].get("curriculum", False)
        if  self.curriculum :
            self.qids = sorted(
                self.qids,
                key=lambda qid: float(self.eval_obj.dict_ndcg.get(qid, 0.0)),
                reverse=True,   # descending: highest ndcg first
            )
            ndcg_vals = [float(self.eval_obj.dict_ndcg.get(q, 0.0)) for q in self.qids]
            print(f"Curriculum range: easiest={ndcg_vals[0]:.4f}, hardest={ndcg_vals[-1]:.4f}, median={ndcg_vals[len(ndcg_vals)//2]:.4f}")

        self.best_eval_delta = float("-inf")

    # ──────────────────────────────────────────────────────────────────────
    # Infrastructure helpers (unchanged)
    # ──────────────────────────────────────────────────────────────────────

    def _find_latest_ckpt(self) -> Optional[str]:
        num_batches = math.ceil(len(self.qids) / self.gen_cfg.batch_size)
        for it in range(self.num_iterations, 0, -1):
            for b in range(num_batches, 0, -1):
                _, ckpt_dir = self._batch_paths(it, b)
                if os.path.isdir(ckpt_dir) and (
                    os.path.isfile(os.path.join(ckpt_dir, "adapter_config.json"))
                    or os.path.isfile(os.path.join(ckpt_dir, "config.json"))
                ):
                    return ckpt_dir
        return None

    def _apply_lora_if_needed(self, model):
        lcfg = self.cfg.get("lora", {})
        if not lcfg.get("enabled", True):
            return model       
        lora_config = LoraConfig(
            r=int(lcfg["r"]),
            lora_alpha=int(lcfg["alpha"]),
            lora_dropout=float(lcfg["dropout"]),
            target_modules=list(lcfg["target_modules"]),
            bias="none",
            task_type="CAUSAL_LM",
        )
        return get_peft_model(model, lora_config)

    def _batch_paths(self, iter_idx: int, batch_idx: int):
        jsonl_path = format_path(
            self.cfg["data"]["generation_jsonl"],
            iter=iter_idx,
            batch=batch_idx,
            exp_name=self.cfg["exp_name"],
        )
        ckpt_dir = format_path(
            self.train_cfg["output_dir"],
            iter=iter_idx,
            batch=batch_idx,
            exp_name=self.cfg["exp_name"],
        )
        return jsonl_path, ckpt_dir

    def _load_model_and_tok(self, model_name_or_path: str):
        use_model_parallel = bool(self.cfg.get("model_parallel", True))
        n_gpu = torch.cuda.device_count()

        tok_src = (
            model_name_or_path if os.path.isdir(model_name_or_path) else self.base_model
        )
        tok = AutoTokenizer.from_pretrained(tok_src, padding_side="left")

        # Build model kwargs once — shared by both load paths
        model_kwargs = {
            "attn_implementation": self.cfg.get("attn_implementation", "sdpa"),
        }
        #if bool(self.cfg.get("use_bf16", False)):
        #    model_kwargs["torch_dtype"] = torch.bfloat16

        if use_model_parallel and n_gpu > 1:
            model_kwargs["device_map"] = self.cfg.get("device_map", "balanced")
        else:
            model_kwargs["device_map"] = self.cfg.get("device_map", "auto")

        # Detect checkpoint type
        is_lora_ckpt = os.path.isfile(
            os.path.join(model_name_or_path, "adapter_config.json")
        )
        is_full_ckpt = (
            not is_lora_ckpt
            and os.path.isdir(model_name_or_path)
            and os.path.isfile(os.path.join(model_name_or_path, "config.json"))
            and model_name_or_path != self.base_model   # not the very first run
        )

        if is_full_ckpt:
            # Resume directly from fine-tuned weights
            model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
        else:
            base = AutoModelForCausalLM.from_pretrained(self.base_model, **model_kwargs)
            if is_lora_ckpt:
                model = PeftModel.from_pretrained(base, model_name_or_path, is_trainable=True)
            else:
                model = base   # clean start

        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        return model, tok

    # ──────────────────────────────────────────────────────────────────────
    # BM25 scoring helper (shared by both generation paths)
    # ──────────────────────────────────────────────────────────────────────

    def _compute_t_scores(
        self, targets_text: List[str], qids: List[str]
    ) -> Tuple[List[float], List[float]]:
        """
        Run BM25 once, return (raw, normalised) scores.
 
        raw        : plain BM25 scores, used for JSONL logging.
        normalised : benchmark-adjusted scores used as the reward signal.
                     Equal to raw when apply_benchmark=False.
        """
        raw = self.eval_obj.apply_bm25_reward_hard(targets_text, qids, k=self.bm25_k)
        if self.apply_benchmark:
            normalised = [
                s * (s / (self.eval_obj.dict_ndcg[qid] + 1e-6))
                for qid, s in zip(qids, raw)
            ]
        else:
            normalised = raw
        return raw, normalised

    # ──────────────────────────────────────────────────────────────────────
    # NEW: per-sample routing
    # ──────────────────────────────────────────────────────────────────────

    def _split_batch(
        self,
        qids: List[str],
    ) -> Tuple[List[int], List[int], List[bool]]:
        """
        Flip a Bernoulli(epsilon_qi) coin independently for each sample.

        Returns
        -------
        classic_idx  : indices (into the full batch) routed to classic generation
        beam_idx     : indices routed to beam generation
        is_classic   : per-sample boolean mask (len == len(qids))
        """
        is_classic = [random.random() < self.epsilon_qi for _ in qids]
        classic_idx = [i for i, c in enumerate(is_classic) if c]
        beam_idx    = [i for i, c in enumerate(is_classic) if not c]
        return classic_idx, beam_idx, is_classic

    # ──────────────────────────────────────────────────────────────────────
    # NEW: classic generation on a sub-batch
    # ──────────────────────────────────────────────────────────────────────

    def _classic_generation(
        self,
        model,
        full_enc: Dict,        # tokenised full batch (prompt only, left-padded)
        prompt_len: int,
        indices: List[int],    # which rows of full_enc belong to this sub-batch
       # pad_token_id: int,
    ) -> Tuple[torch.Tensor, List[str]]:
        """
        Run sampling generation on the subset of the batch selected by `indices`.

        We slice the already-tokenised full-batch encoding by row index so that
        prompt_len stays consistent across classic and beam sub-batches.

        Returns
        -------
        full_seq  : (len(indices), total_seq_len) token tensor
        decoded   : decoded target strings (without prompt)
        """
        if not indices:
            return None, []

        sub_input_ids      = full_enc["input_ids"][indices]
        sub_attention_mask = full_enc["attention_mask"][indices]

        model.eval()
        with torch.inference_mode():
            generated = model.generate(
                input_ids=sub_input_ids[:, :prompt_len],
                attention_mask=sub_attention_mask[:, :prompt_len],
                max_new_tokens=self.max_new_tokens_gen,
                do_sample=True,
                use_cache=True,
                temperature=1.0,  ## Q: use variable temperature?
                top_p=0.95,
                top_k=500,
                pad_token_id=self.tok.pad_token_id,
                #pad_token_id=pad_token_id,
            )

        decoded = self.tok.batch_decode(generated[:, prompt_len:], skip_special_tokens=True)
        decoded = [d.strip() for d in decoded]
        return generated, decoded

    # ──────────────────────────────────────────────────────────────────────
    # NEW: beam generation on a sub-batch
    # ──────────────────────────────────────────────────────────────────────

    def _beam_generation(
        self,
        generator: "BeamNDCGGenerator",
        full_enc: Dict,
        prompt_len: int,
        indices: List[int],
        sub_qids: List[str],
    ) -> Tuple[torch.Tensor, List[str]]:
        """
        Run beam generation (with reward hooks) on the subset selected by `indices`.

        We reuse the already-tokenised encoding (sliced by row) so prompt_len
        is identical to the classic sub-batch.

        Returns
        -------
        full_seq  : (len(indices), total_seq_len) token tensor
        decoded   : decoded target strings (without prompt)
        """
        if not indices:
            return None, []

        sub_input_ids      = full_enc["input_ids"][indices]
        sub_attention_mask = full_enc["attention_mask"][indices]

        gen_model = generator._set_reward_hooks(generator.model, qids=sub_qids)

        num_return_sequences = (
            1 if self.reward_cfg.best_beam_sel else self.gen_cfg.num_beams
        )
        gen_model.eval()
        with torch.inference_mode():
            out = gen_model.generate(
                input_ids=sub_input_ids,
                attention_mask=sub_attention_mask,
                do_sample=self.gen_cfg.do_sample,
                temperature=self.gen_cfg.temperature,
                max_new_tokens=self.gen_cfg.max_new_tokens,
                repetition_penalty=self.gen_cfg.repetition_penalty,
                num_beams=self.gen_cfg.num_beams,
                pad_token_id=self.tok.pad_token_id,
                eos_token_id=self.tok.eos_token_id,
                num_return_sequences=num_return_sequences,
                top_k=self.gen_top_k,
                top_p=self.gen_top_p,
                use_cache=True,
            )

        if not self.reward_cfg.best_beam_sel:
            bsz = len(indices)
            B   = self.gen_cfg.num_beams
            sequences = out.view(bsz, B, -1)
            rand_idx  = torch.randint(0, B, (bsz,), device=sequences.device)
            out = sequences[torch.arange(bsz, device=sequences.device), rand_idx]

        decoded = self.tok.batch_decode(out[:, prompt_len:], skip_special_tokens=True)
        decoded = [d.strip() for d in decoded]
        return out, decoded

    # ──────────────────────────────────────────────────────────────────────
    # NEW: reassemble sub-batch results into original batch order
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _reassemble(
        batch_size: int,
        classic_idx: List[int],
        beam_idx:    List[int],
        classic_seq: Optional[torch.Tensor],   # (|C|, L_c)
        beam_seq:    Optional[torch.Tensor],   # (|B|, L_b)
        classic_dec: List[str],
        beam_dec:    List[str],
        pad_token_id,
    ) -> Tuple[torch.Tensor, List[str]]:
        """
        Place sub-batch results back into the original sample order.

        Sequences from the two paths may have different lengths because they
        were generated independently.  We pad both to the same length before
        stacking.  Padding uses 0 (harmless — labels for pad positions will
        be set to IGNORE_INDEX in compute_loss).
        """
        # Build per-sample lists in original order
        seq_map: Dict[int, torch.Tensor] = {}
        dec_map: Dict[int, str]          = {}

        if classic_seq is not None:
            for local_i, global_i in enumerate(classic_idx):
                seq_map[global_i] = classic_seq[local_i]
                dec_map[global_i] = classic_dec[local_i]

        if beam_seq is not None:
            for local_i, global_i in enumerate(beam_idx):
                seq_map[global_i] = beam_seq[local_i]
                dec_map[global_i] = beam_dec[local_i]

        # Pad all sequences to the same length
        max_len = max(t.shape[0] for t in seq_map.values())
        full_seqs = []
        for i in range(batch_size):
            t = seq_map[i]
            pad_len = max_len - t.shape[0]
            if pad_len > 0:
                pad = torch.full((pad_len,), pad_token_id, dtype=t.dtype, device=t.device)
                t = torch.cat([t, pad])
                #t = torch.cat([t, torch.zeros(pad_len, dtype=t.dtype, device=t.device)])
            full_seqs.append(t)

        full_seq_tensor = torch.stack(full_seqs, dim=0)   # (B, max_len)
        decoded_list    = [dec_map[i] for i in range(batch_size)]
        return full_seq_tensor, decoded_list

    # ──────────────────────────────────────────────────────────────────────
    # NEW: generate_dataset_batch — now does per-sample routing
    # ──────────────────────────────────────────────────────────────────────

    def generate_dataset_batch(
        self,
        iter_idx:  int,
        batch_idx: int,
        generator: "BeamNDCGGenerator",
        qids:      List[str],
        qs:        List[str],
    ) -> Tuple[List[str], torch.Tensor, int, List[bool]]:
        """
        1. Tokenise the full batch once  → consistent prompt_len
        2. Per-sample coin flip          → classic_idx, beam_idx
        3. Classic sub-batch generation  (if any)
        4. Beam sub-batch generation     (if any)
        5. Reassemble in original order
        6. Score (BM25) and log

        Returns
        -------
        targets_text  : decoded generated strings, original order
        full_seq_gen  : (B, seq_len) token tensor, original order
        prompt_len    : scalar — same for all samples
        is_classic    : per-sample bool list
        """
        # ── 1. Tokenise full batch ──────────────────────────────────────
        self.out_jsonl = format_path(
            self.cfg["data"]["generation_jsonl"],
            iter=iter_idx, batch=batch_idx, exp_name=self.cfg["exp_name"],
        )
        ensure_dir(os.path.dirname(self.out_jsonl))

        full_enc, prompt_len = generator.get_tokens(qs)

        # ── 2. Per-sample coin flip ────────────────────────────────────
        classic_idx, beam_idx, is_classic = self._split_batch(qids)

        n_classic = len(classic_idx)
        n_beam    = len(beam_idx)
        if batch_idx%50==0:
            print(
                f"[iter={iter_idx} batch={batch_idx}] "
                f"routing: {n_classic} classic / {n_beam} beam  "
                f"(ε={self.epsilon_qi:.2f})"
            )

        #pad_token_id = self.tok.eos_token_id

        # ── 3. Classic sub-batch ───────────────────────────────────────
        classic_seq, classic_dec = self._classic_generation(
            model=generator.model,
            full_enc=full_enc,
            prompt_len=prompt_len,
            indices=classic_idx,
          #  pad_token_id=pad_token_id,
        )

        # ── 4. Beam sub-batch ──────────────────────────────────────────
        sub_qids_beam = [qids[i] for i in beam_idx]
        beam_seq, beam_dec = self._beam_generation(
            generator=generator,
            full_enc=full_enc,
            prompt_len=prompt_len,
            indices=beam_idx,
            sub_qids=sub_qids_beam,
        )

        # ── 5. Reassemble ──────────────────────────────────────────────
        full_seq_gen, targets_text = self._reassemble(
            batch_size=len(qids),
            classic_idx=classic_idx,
            beam_idx=beam_idx,
            classic_seq=classic_seq,
            beam_seq=beam_seq,
            classic_dec=classic_dec,
            beam_dec=beam_dec,
            pad_token_id=self.tok.pad_token_id
        )

        # ── 6. Score & log ─────────────────────────────────────────────
        t_scores_raw, self.t_scores = self._compute_t_scores(targets_text, qids)

        records = [
            {
                "qid":      qid,
                "score-q":  round(self.eval_obj.dict_ndcg[qid] * 100, 2),
                "score-t":  round(st * 100, 2),
                "query":    q,
                "target":   t,
                "classic":  c,
            }
            for qid, q, t, st, c in zip(qids, qs, targets_text, t_scores_raw, is_classic)
        ]
        write_records_with_batch_score_log(
            out_jsonl=self.out_jsonl,
            records=records,
            batch_idx=batch_idx,
            iter_idx=iter_idx,
            log_path=format_path(
                self.cfg["logging"]["debug_log_path"], exp_name=self.cfg["exp_name"]
            ),
        )

        return targets_text, full_seq_gen, prompt_len, is_classic

    # ──────────────────────────────────────────────────────────────────────
    # NEW: compute_loss — pure loss computation, no generation inside
    # ──────────────────────────────────────────────────────────────────────

    def compute_loss(
        self,
        trainer,
        full_seq_gen:  torch.Tensor,   # (B, seq_len)
        targets_text:  List[str],
        t_scores:      List[float],
        prompt_len:    int,
        qids:          List[str],
        queries_text:  List[str],
        is_classic:    List[bool],
    ) -> float:
        """
        Compute the importance-weighted loss for the current batch.

        Identical maths to the original train_batch — the only difference is
        that this function receives already-generated sequences and scores
        rather than generating them internally.

        Updates self.loss_buffer and self.w_norm_buffer for gradient
        accumulation; returns the raw scalar loss for logging.
        """
        IGNORE_INDEX = -100
        model        = trainer.model
        pad_token_id = trainer.processing_class.pad_token_id  

        # ── labels & attention mask ──────────────────────────────────
        labels = full_seq_gen.clone()
        labels[:, :prompt_len] = IGNORE_INDEX
        labels[full_seq_gen == pad_token_id] = IGNORE_INDEX
        attention_mask = (full_seq_gen != pad_token_id).long()

        model.train()
        inputs = {
            "input_ids":      full_seq_gen,
            "attention_mask": attention_mask,
            "labels":         labels,
        }
        inputs = trainer._prepare_inputs(inputs)
        out    = model(**inputs)

        # ── per-token log-prob ───────────────────────────────────────
        logits      = out.logits[:, :-1, :]
        target      = labels[:, 1:]
        safe_target = target.clamp(min=0)
        logProg     = F.log_softmax(logits, dim=-1)
        token_log_p = torch.gather(logProg, dim=-1, index=safe_target.unsqueeze(-1)).squeeze(-1)
        mask        = (target != IGNORE_INDEX)
        token_log_p = token_log_p * mask
        log_softmax = token_log_p.sum(-1)

        if self.token_loss:
            loss = log_softmax / mask.sum(-1)
        else:
            loss = log_softmax

        pi = loss.detach()

        # ── reward ───────────────────────────────────────────────────
        R = (
            (
                torch.tensor(t_scores)
                + self.eval_obj.apply_reward(
                    texts=targets_text,
                    qid=qids,
                    k=self.bm25_k,
                    lks=pi,
                )
            ).to(full_seq_gen.device)
        ) * self.lambda_ndcg

        # ── proposal log-prob: identical formula for all samples ─────
        #
        # The mixture proposal is:
        #   q(y|x) = ε · p_sample(y|x) + (1-ε)/K · p_beam_k(y|x)
        #
        # Using logaddexp for numerical stability. The formula is the same
        # regardless of which path a sample was actually generated from,
        # because that is exactly what makes the importance weight correct.
        log_qi = torch.logaddexp(
            math.log(self.epsilon_qi) + pi,
            torch.full_like(pi, math.log((1 - self.epsilon_qi) / self.qi_num_beam)),
        )

        k = 6; mu = 0.5
        R_scaled = k * (R - mu)
        log_wi   = pi - log_qi + R_scaled - F.softplus(R_scaled)

        batch_loss  = -(log_wi.exp() * loss).sum()
        w_norm      = log_wi.exp().sum()

        # ── backward immediately to free the graph, accumulate floats ──
        # The upstream optimizer step will divide all accumulated gradients
        # by w_norm_buffer, which is equivalent to normalising by the total
        # importance weight across the accumulation window.
        trainer.accelerator.backward(batch_loss)
        self.loss_buffer   += float(batch_loss.detach().cpu())
        self.w_norm_buffer += float(w_norm.detach().cpu())

        model.eval()

        # ── debug logging ────────────────────────────────────────────
        self.debug_logger.log(BatchRecord(
            iter_idx=self.iter_idx,
            batch_idx=self.batch_idx,
            opt_step=None,
            qids=qids,
            queries=queries_text,
            targets=targets_text,
            classic_gen=is_classic,                           # per-sample list
            reward=R.detach().cpu().tolist(),
            scaled_reward=R_scaled.detach().cpu().tolist(),
            sig_reward=torch.sigmoid(R).detach().cpu().tolist(),
            sig_scaled_reward=torch.sigmoid(R_scaled).detach().cpu().tolist(),
            score_q=[round(self.eval_obj.dict_ndcg[qid] * 100, 2) for qid in qids],
            score_t=t_scores,
            pi=pi.cpu().tolist(),
            log_qi=log_qi.detach().cpu().tolist(),
            log_wi=log_wi.detach().cpu().tolist(),
            w_i=log_wi.exp().detach().cpu().tolist(),
            batch_loss=float(batch_loss.detach().cpu()),
            batch_w_norm=float(w_norm.detach().cpu()),
            epsilon_qi=self.epsilon_qi,
        ))

        return float(batch_loss.detach().cpu())

    # ──────────────────────────────────────────────────────────────────────
    # Eval on holdout (unchanged logic, minor refactor)
    # ──────────────────────────────────────────────────────────────────────
    @torch.inference_mode()
    def _eval_on_qids(self, model,queries, qrels, bs: int, max_new: int, rep_pen: float) -> float:
        

        qs =list(queries.values())
        use_qids = list(queries.keys())
        if not use_qids:
            return float("nan")

        st_vals = []
        for start in range(0, len(use_qids), bs):
            batch_qids = use_qids[start:start + bs]
            batch_qs = qs[start:start + bs]
            enc, _ = self.tokenizer(batch_qs)
            out = model.generate(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                max_new_tokens=max_new,
                num_beams=1,
                do_sample=False,
                repetition_penalty=rep_pen,
                eos_token_id=self.tok.eos_token_id,
                pad_token_id=self.tok.pad_token_id,
            )
            preds = self.tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            #t_scores = self.eval_obj.apply_bm25_reward_hard(preds, batch_qids, k=10, this_eval=True)
            t_scores= self.eval_obj.eval_dl(preds,batch_qids,qrels,k=10)
            st_vals.extend(t_scores)

        return (sum(st_vals) / len(st_vals)) if st_vals else float("nan")

    @torch.inference_mode()
    def eval_holdout_greedy_and_log(
        self,
        trainer,
        iter_idx:   int,
        batch_idx:  int,
        train_loss,
    ) -> Dict[str, float]:
        model = trainer.model
        if (not self.eval_enabled) or (len(self.holdout_qids) == 0):
            return {"mean_st": float("nan"), "delta": float("nan")}

        log_path = format_path(
            self.eval_cfg.get("log_path", "runs/selftrain/eval_evolution.log"),
            exp_name=self.cfg["exp_name"],
        )
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        out_jsonl = format_path(
            self.eval_cfg.get("dump_jsonl", None),
            exp_name=self.cfg["exp_name"],
            iter=iter_idx,
            batch=batch_idx,
        )
        os.makedirs(os.path.dirname(out_jsonl), exist_ok=True)

        bs      = int(self.eval_cfg.get("batch_size", 128))
        max_new = int(self.eval_cfg.get("max_new_tokens", 30))
        rep_pen = float(self.eval_cfg.get("repetition_penalty", 1.0))

        self.tok.padding_side = "left"
        model.eval()

        st_vals      = []
        dump_records = []

        for start in range(0, len(self.holdout_qids), bs):
            qids     = self.holdout_qids[start: start + bs]
            qs       = self.holdout_queries[start: start + bs]
            sq_batch = self.sq_vals[start: start + bs]

            enc, prompt_len = self.tokenizer(qs)

            out = model.generate(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                do_sample=False,
                temperature=0,
                max_new_tokens=max_new,
                repetition_penalty=rep_pen,
                pad_token_id=self.tok.pad_token_id,
                eos_token_id=self.tok.eos_token_id,
            )

            preds    = self.tok.batch_decode(out[:, prompt_len:], skip_special_tokens=True)
            use_orginal=False
            if len(self.msmarco_v)>1 : use_orginal=True
            t_scores = self.eval_obj.apply_bm25_reward_hard(preds, qids, k=self.bm25_k,use_orginal=use_orginal,this_eval=True )

            for n_qid, (sq, qid, q, pred) in enumerate(zip(sq_batch, qids, qs, preds)):
                st = t_scores[n_qid]
                st_vals.append(st)
                dump_records.append({
                    "qid":     qid,
                    "score-q": round(sq * 100, 2),
                    "score-t": round(st * 100, 2),
                    "query":   q,
                    "target":  pred,
                })

        mean_st = sum(st_vals) / len(st_vals) if st_vals else float("nan")
        delta   = mean_st - self.mean_sq
        current_lr = trainer.optimizer.param_groups[0]['lr']
        if self.dl_data_root:
            dl19_score = self._eval_on_qids(model,self.dl19_qu, self.dl19_qrls, bs, max_new, rep_pen)
            dl20_score = self._eval_on_qids(model,self.dl20_qu, self.dl20_qrls, bs, max_new, rep_pen)
            dlhard_score = self._eval_on_qids(model,self.dlhard_qu, self.dlhard_qrls, bs, max_new, rep_pen)
        else:dl19_score,dl20_score,dlhard_score=0.0,0.0,0.0
        line = (
                    f"[EVAL] [iter={iter_idx} batch={batch_idx}] "
                    f"heldout_n={len(self.holdout_qids)} "
                    f"mean(score-q)={self.mean_sq:.4f} mean(score-t)={mean_st:.4f} "
                   f"delta={delta:.4f} dl19={dl19_score:.4f} dl20={dl20_score:.4f} dlhard={dlhard_score:.4f} "
                    f"lr={current_lr:.8f} "          # ← add this
                    f"(train_loss={train_loss})"
                )
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

        with open(out_jsonl, "w", encoding="utf-8") as f:
            for rec in dump_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return {"mean_st": float(mean_st), "delta": float(delta)}

    # ──────────────────────────────────────────────────────────────────────
    # Main training loop
    # ──────────────────────────────────────────────────────────────────────

    def run(self) -> None:
        print("now running")

        

        last_ckpt  = format_path(self.train_cfg["last_ckpt"], exp_name=self.cfg["exp_name"])
        latest     = _load_last_ckpt(last_ckpt)
        print("latest", latest)
        start_path = latest["ckpt_dir"] if latest is not None else self.base_model
        print("start_path", start_path)

        model, self.tok = self._load_model_and_tok(start_path)

        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if n_trainable == 0:
            raise RuntimeError(
                "No trainable parameters after loading (LoRA likely loaded in inference mode)."
            )
        
        lora_enabled = self.cfg.get("lora", {}).get("enabled", True)
        if lora_enabled and not isinstance(model, PeftModel):
            model = self._apply_lora_if_needed(model)


       # print("model device map:", model.hf_device_map)

        ckpt_dir = format_path(
            self.train_cfg["output_dir"], iter=0, batch=0, exp_name=self.cfg["exp_name"]
        )
        args = TrainingArguments(
            output_dir=ckpt_dir,
            per_device_train_batch_size=self.gen_cfg.batch_size,
            gradient_accumulation_steps=int(self.train_cfg["gradient_accumulation_steps"]),
            learning_rate=float(self.train_cfg["learning_rate"]),
            num_train_epochs=float(self.train_cfg["num_train_epochs"]),
            logging_steps=int(self.train_cfg.get("logging_steps", 20)),
            save_strategy="no",
            report_to=[],
            remove_unused_columns=False,
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=None,
            data_collator=CausalLMPadCollator(self.tok),
            processing_class=self.tok,
        )
        est_total_steps = self.num_iterations * (
            len(self.qids)
            // (self.gen_cfg.batch_size * self.train_cfg["gradient_accumulation_steps"])
        )
        trainer.create_optimizer_and_scheduler(num_training_steps=est_total_steps)

        #print("trainer device map:", trainer.model.hf_device_map)
        self.eval_obj.tok = self.tok

        generator = BeamNDCGGenerator(
            model=trainer.model,
            tokenizer=self.tok,
            prompt_builder=self.prompt_builder,
            device=self.device,
            gen_cfg=self.gen_cfg,
            reward_cfg=self.reward_cfg,
            eval_obj=self.eval_obj,
        )
        self.tokenizer = generator.get_tokens

        ds = QueryDataset(self.qids, self.queries)
        print("num_processes:", trainer.accelerator.num_processes)
        print("process_index:", trainer.accelerator.process_index)
        print("device:",        trainer.accelerator.device)

        def collate(items: List[Dict]) -> Dict[str, List[str]]:
            return {
                "qids":    [x["qid"]   for x in items],
                "queries": [x["query"] for x in items],
            }

        opt_step = 0
        if latest is not None:
            st = _load_full_checkpoint_state(trainer, start_path)
            if st is not None:
                opt_step = int(st.get("opt_step", 0))
            # restore best delta
            raw = latest.get("best_eval", None)
            if raw is not None:
                self.best_eval_delta = float(raw)

       

        trainer.optimizer.zero_grad(set_to_none=True)

        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"Trainable params: {trainable:,}")
        print(f"Total params: {total:,}")
        print(f"Trainable %: {100 * trainable / total:.4f}%")

        if latest is None:
            resume_it, resume_batch = 1, 1
        else:
            resume_it    = latest["it"]
            resume_batch = latest["batch_idx"] + 1

        for it in range(1, self.num_iterations + 1):
            self.iter_idx = it

            dl = DataLoader(
                ds,
                batch_size=self.gen_cfg.batch_size,
                shuffle=False,
                collate_fn=collate,
                num_workers=4,
                pin_memory=True,
                persistent_workers=True,
            )

            num_batches = len(dl)
            gacc        = int(trainer.args.gradient_accumulation_steps)
            remainder   = num_batches % gacc

            pbar = tqdm(
                enumerate(dl, start=1),
                total=num_batches,
                desc=f"Iter {it}/{self.num_iterations}",
                leave=False,
                dynamic_ncols=True,
            )

            for batch_idx, batch in pbar:
                self.batch_qids   = batch["qids"]
                self.queries_text = batch["queries"]
                self.batch_idx    = batch_idx

                jsonl_path, ckpt_dir = self._batch_paths(it, batch_idx)

                # skip already-completed batches
                if (it < resume_it) or (it == resume_it and batch_idx < resume_batch):
                    pbar.set_postfix(stage="skip", it=it, batch=batch_idx)
                    continue

                # ── GENERATION ────────────────────────────────────────
                if self.run_generation:
                    targets_text, full_seq_gen, prompt_len, is_classic = (
                        self.generate_dataset_batch(
                            iter_idx=it,
                            batch_idx=batch_idx,
                            generator=generator,
                            qids=self.batch_qids,
                            qs=batch["queries"],
                        )
                    )
                    pbar.set_postfix(stage="gen")

                # ── TRAINING ──────────────────────────────────────────
                if self.run_training:
                    #in_last_window = (remainder != 0) and (
                    #    batch_idx > (num_batches - remainder)
                    #)
                    #accum_div  = remainder if in_last_window else gacc   # kept for future use
                    should_step = (batch_idx % gacc == 0) or (batch_idx == num_batches)

                    batch_loss = self.compute_loss(
                        trainer=trainer,
                        full_seq_gen=full_seq_gen,
                        targets_text=targets_text,
                        t_scores=self.t_scores,
                        prompt_len=prompt_len,
                        qids=self.batch_qids,
                        queries_text=self.queries_text,
                        is_classic=is_classic,
                    )
                    pbar.set_postfix(stage="loss", loss=f"{batch_loss:.4f}")
                    del full_seq_gen, targets_text, is_classic

                    self.total_loss = float("nan")
                    if should_step:
                        self.total_loss = self.loss_buffer / self.w_norm_buffer

                        # Gradients were already accumulated via backward() inside
                        # compute_loss.  We now divide every gradient tensor by the
                        # accumulated w_norm so the effective update is:
                        #   Δθ ∝ Σ(w_i · ∇loss_i) / Σ(w_i)
                        for p in trainer.model.parameters():
                            if p.grad is not None:
                                p.grad.div_(self.w_norm_buffer)

                        trainer.optimizer.step()
                        trainer.lr_scheduler.step()
                        trainer.optimizer.zero_grad(set_to_none=True)

                        self.loss_buffer   = 0
                        self.w_norm_buffer = 0
                        opt_step += 1

                        pbar.set_postfix(stage="train", loss=f"{float(self.total_loss):.4f}")

                        do_save = (opt_step % self.save_steps == 0) or (
                            batch_idx == num_batches
                        )
                        if do_save:
                            _, ckpt_dir = self._batch_paths(it, batch_idx)
                            marker_dir  = format_path(
                                self.train_cfg["last_ckpt"], exp_name=self.cfg["exp_name"]
                            )
                            _save_full_checkpoint(
                                trainer=trainer,
                                tokenizer=self.tok,
                                out_dir=ckpt_dir,
                                marker_dir=marker_dir,
                                iter_idx=it,
                                batch_idx=batch_idx,
                                opt_step=opt_step,
                                best_eval_delta=self.best_eval_delta,

                            )
                            print("- Model saved to", ckpt_dir, marker_dir)

                    if self.eval_enabled and should_step:
                        eval_stats=self.eval_holdout_greedy_and_log(
                            trainer=trainer,
                            iter_idx=it,
                            batch_idx=batch_idx,
                            train_loss=self.total_loss,
                        )
                        curr_delta = float(eval_stats.get("delta", float("nan")))
                        if (not math.isnan(curr_delta)) and (curr_delta > self.best_eval_delta) and (opt_step > 50):
                            self.best_eval_delta = curr_delta
                            _, best_ckpt_dir = self._batch_paths(it, batch_idx)
                            marker_dir = format_path(
                                self.train_cfg["last_ckpt"], exp_name=self.cfg["exp_name"]
                            )
                            os.makedirs(marker_dir, exist_ok=True)

                            if not os.path.isdir(best_ckpt_dir):
                                _save_full_checkpoint(
                                    trainer=trainer,
                                    tokenizer=self.tok,
                                    out_dir=best_ckpt_dir,
                                    marker_dir=marker_dir,
                                    iter_idx=it,
                                    batch_idx=batch_idx,
                                    opt_step=opt_step,
                                    best_eval_delta=self.best_eval_delta,
                                )

                            last_ckpt_json = os.path.join(marker_dir, "last_ckpt.json")
                            marker_payload = {}
                            if os.path.isfile(last_ckpt_json):
                                with open(last_ckpt_json, "r", encoding="utf-8") as f:
                                    marker_payload = json.load(f)

                            best_ckpt_export_dir = os.path.join(marker_dir, "best_ckpt")
                            if os.path.isdir(best_ckpt_export_dir):
                                shutil.rmtree(best_ckpt_export_dir)
                            shutil.copytree(best_ckpt_dir, best_ckpt_export_dir)

                            marker_payload.update({
                                "best_ckpt_dir": best_ckpt_dir,
                                "best_ckpt_export_dir": best_ckpt_export_dir,
                                "best_it": int(it),
                                "best_batch_idx": int(batch_idx),
                                "best_opt_step": int(opt_step),
                                "best_delta": float(curr_delta),
                            })

                            with open(last_ckpt_json, "w", encoding="utf-8") as f:
                                json.dump(marker_payload, f)


                else:
                    break

