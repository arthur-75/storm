
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Iterable, Tuple
from torch.utils.data import Dataset, DataLoader
from beir.datasets.data_loader import GenericDataLoader
import os
import torch
from beir import util
import ir_datasets

# -------------------------
# Prompt builder (shared)
# -------------------------
@dataclass
class PromptSpec:
    system_prompt: str
    user_prefix: str = "[QUERY]: "
    user_suffix: str = "\n[KEYWORDS]: "

# =========================================================
# Helpers
# =========================================================
def get_data(dataset: str, data_root: str, split: str = "test", download_if_needed: bool = False):
    """
    Load corpus, queries, qrels for one dataset.
    """
    if download_if_needed:
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
        data_path = util.download_and_unzip(url, data_root)
    else:
        data_path = os.path.join(data_root, dataset)

    if dataset=="msmarco":split="dev"

    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split=split)
    return corpus, queries, qrels

def get_dl(year: str, data_root: str):
    """
    Load DL19/DL20 via BEIR GenericDataLoader — same path and docid format
    as the evaluation script, so docids match the BM25 passage index.
    """
    dataset_name = f"trec-dl-{year}"
    data_path = os.path.join(data_root, dataset_name)
    _, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
    return queries, qrels



class PromptBuilder:
    def __init__(self, prompt: PromptSpec):
        self.prompt = prompt

    def messages_for_query(self, query: str) -> List[Dict[str, str]]:
        return [
            {"role": "system", "content": self.prompt.system_prompt},
            {
                "role": "user",
                "content": f"{self.prompt.user_prefix}{query.strip()}{self.prompt.user_suffix}",
            },
        ]

    def render_prompt_text(self, tokenizer, query: str, add_generation_prompt: bool) -> str:
        msgs = self.messages_for_query(query)
        # We render text (not tokens) to keep training/inference identical.
        return tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
           # padding=True   # important to get [B, T]
        )


# -------------------------
# Data
# -------------------------
class QueryDataset(Dataset):
    def __init__(self, qids: List[str], queries: Dict[str, str]):
        self.qids = qids
        self.queries = queries

    def __len__(self) -> int:
        return len(self.qids)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        qid = self.qids[idx]
        return {"qid": qid, "query": self.queries[qid]}

