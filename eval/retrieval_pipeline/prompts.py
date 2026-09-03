"""
Prompt specification and chat-template rendering.

Language handling:
  * prompt.language_by_dataset is always honoured (explicit wins)
  * for MIRACL datasets without an explicit entry, the language is derived
    from the dataset name unless prompt.auto_language_for_miracl is false
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

from .config import deep_get
from .datasets import MIRACL_LANGUAGE_NAMES, DatasetBundle

DEFAULT_SYSTEM_PROMPT = (
    "From the query generate new semantically related keywords.\n"
    "Output the result strictly as a single comma-separated line."
)


@dataclass
class PromptSpec:
    system_prompt: str
    user_prefix: str = "[QUERY]: "
    user_suffix: str = "\n[KEYWORDS]: "

    def as_dict(self) -> Dict[str, str]:
        return {
            "system": self.system_prompt,
            "user_prefix": self.user_prefix,
            "user_suffix": self.user_suffix,
        }


class PromptBuilder:
    def __init__(self, prompt: PromptSpec):
        self.prompt = prompt

    def messages_for_query(self, query: str) -> list:
        return [
            {"role": "system", "content": self.prompt.system_prompt},
            {
                "role": "user",
                "content": (
                    f"{self.prompt.user_prefix}"
                    f"{query.strip()}"
                    f"{self.prompt.user_suffix}"
                ),
            },
        ]

    def render(self, tokenizer: Any, query: str, add_generation_prompt: bool = True) -> str:
        messages = self.messages_for_query(query)
        kwargs = {
            "tokenize": False,
            "add_generation_prompt": add_generation_prompt,
        }
        try:
            # Qwen3-style tokenizers accept enable_thinking; others do not.
            return tokenizer.apply_chat_template(
                messages, enable_thinking=False, **kwargs
            )
        except TypeError:
            return tokenizer.apply_chat_template(messages, **kwargs)


def resolve_language(cfg: Mapping[str, Any], bundle: DatasetBundle) -> Optional[str]:
    mapping = deep_get(cfg, "prompt.language_by_dataset", {}) or {}
    if isinstance(mapping, Mapping) and bundle.dataset in mapping:
        return str(mapping[bundle.dataset])

    auto = bool(deep_get(cfg, "prompt.auto_language_for_miracl", True))
    if auto and bundle.kind == "miracl" and bundle.language:
        return MIRACL_LANGUAGE_NAMES.get(bundle.language)

    return None


def prompt_for_dataset(cfg: Mapping[str, Any], bundle: DatasetBundle) -> PromptSpec:
    system_prompt = str(deep_get(cfg, "prompt.system", DEFAULT_SYSTEM_PROMPT))

    language = resolve_language(cfg, bundle)
    if language:
        suffix_template = str(
            deep_get(
                cfg,
                "prompt.language_instruction",
                " The keywords must be in {language}.",
            )
        )
        system_prompt += suffix_template.format(language=language)

    return PromptSpec(
        system_prompt=system_prompt,
        user_prefix=str(deep_get(cfg, "prompt.user_prefix", "[QUERY]: ")),
        user_suffix=str(deep_get(cfg, "prompt.user_suffix", "\n[KEYWORDS]: ")),
    )
