"""
Searcher construction.

An index is described by an IndexSpec, which is either a local directory
("path") or a pyserini prebuilt index name ("prebuilt"). MIRACL indexes carry
a language so the correct analyzer is used at search time; SPLADE indexes carry
an encoder name so an impact searcher is created instead of a BM25 one.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from .config import deep_get


@dataclass(frozen=True)
class IndexSpec:
    kind: str  # "path" | "prebuilt"
    value: str
    language: Optional[str] = None
    encoder: Optional[str] = None  # set -> SPLADE impact searcher

    @staticmethod
    def from_path(path: Any, language: Optional[str] = None, encoder: Optional[str] = None) -> "IndexSpec":
        return IndexSpec("path", str(path), language, encoder)

    @staticmethod
    def prebuilt(name: str, language: Optional[str] = None, encoder: Optional[str] = None) -> "IndexSpec":
        return IndexSpec("prebuilt", str(name), language, encoder)

    @property
    def exists(self) -> bool:
        if self.kind == "prebuilt":
            return True  # pyserini downloads/validates it itself
        return Path(self.value).is_dir()

    @property
    def is_impact(self) -> bool:
        return self.encoder is not None

    def describe(self) -> str:
        tag = "prebuilt" if self.kind == "prebuilt" else "local"
        extra = f", lang={self.language}" if self.language else ""
        extra += f", encoder={self.encoder}" if self.encoder else ""
        return f"{self.value} ({tag}{extra})"


class SearcherFactory:
    """Builds and caches searchers so each index is opened only once."""

    def __init__(self, cfg: Mapping[str, Any]):
        self.cfg = cfg
        self._cache: Dict[IndexSpec, Any] = {}

    def get(self, spec: IndexSpec) -> Any:
        if spec not in self._cache:
            self._cache[spec] = self._build(spec)
        return self._cache[spec]

    def _build(self, spec: IndexSpec) -> Any:
        try:
            from pyserini.search.lucene import LuceneImpactSearcher, LuceneSearcher
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("pyserini is required for retrieval") from exc

        if spec.kind == "path" and not Path(spec.value).is_dir():
            raise FileNotFoundError(f"Index directory does not exist: {spec.value}")

        print(f"[search] opening index: {spec.describe()}")

        if spec.is_impact:
            device = str(deep_get(self.cfg, "evaluation.splade_device", "cuda"))
            batch_size = int(deep_get(self.cfg, "evaluation.splade_batch_size", 16))

            if spec.kind == "prebuilt":
                searcher = LuceneImpactSearcher.from_prebuilt_index(
                    spec.value, spec.encoder
                )
                searcher.query_encoder.device = device
                searcher.query_encoder.model.to(device)
            else:
                searcher = LuceneImpactSearcher(
                    spec.value,
                    spec.encoder,
                    device=device,
                    batch_size=batch_size,
                )
            searcher.batch_size = batch_size
            return searcher

        if spec.kind == "prebuilt":
            searcher = LuceneSearcher.from_prebuilt_index(spec.value)
        else:
            searcher = LuceneSearcher(spec.value)

        if spec.language:
            searcher.set_language(spec.language)

        return searcher

    def close(self) -> None:
        self._cache.clear()
