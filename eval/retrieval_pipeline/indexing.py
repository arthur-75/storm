"""
Derived corpora and Lucene index construction.

Only available for datasets that expose a local corpus (BEIR). MIRACL and
BRIGHT are evaluated against their own prebuilt / pre-downloaded indexes.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

from .artifacts import ValidationResult, validate_ids, write_manifest
from .io_utils import atomic_json_dump, load_corpus_jsonl, normalize_text


def index_manifest_path(index_path: Path) -> Path:
    return index_path / "_pipeline_manifest.json"


def validate_index(
    index_path: Path,
    *,
    source_fingerprint: str,
    allow_legacy_without_manifest: bool,
) -> ValidationResult:
    if not index_path.is_dir():
        return ValidationResult(False, "directory is missing")

    # Lucene indexes normally contain at least one segments_* file.
    if not any(index_path.glob("segments_*")):
        return ValidationResult(False, "Lucene segments file is missing")

    manifest_file = index_manifest_path(index_path)
    if not manifest_file.is_file():
        if allow_legacy_without_manifest:
            return ValidationResult(
                True, "Lucene index exists; legacy index has no manifest"
            )
        return ValidationResult(False, "index manifest is missing")

    with manifest_file.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    if manifest.get("source_fingerprint") != source_fingerprint:
        return ValidationResult(False, "index source fingerprint mismatch")

    return ValidationResult(True, "Lucene index and source fingerprint match")


def ensure_keywords_plus_original_corpus(
    *,
    generated_corpus_path: Path,
    original_corpus: Mapping[str, Mapping[str, Any]],
    output_path: Path,
    source_fingerprint: str,
    force: bool = False,
) -> None:
    expected_ids = list(original_corpus)
    validation = validate_ids(
        output_path,
        expected_ids,
        artifact_kind="corpus",
        source_fingerprint=source_fingerprint,
        allow_legacy_without_manifest=False,
    )
    if validation.valid and not force:
        print(f"[reuse] combined corpus: {output_path} ({validation.reason})")
        return

    generated = load_corpus_jsonl(generated_corpus_path)
    if set(generated) != set(original_corpus):
        raise RuntimeError("Generated corpus IDs do not match the original corpus IDs")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_name(output_path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for doc_id in original_corpus:
            keyword_text = generated[doc_id]["text"]
            row = original_corpus[doc_id]
            original_text = normalize_text(
                f"{row.get('title', '')} {row.get('text', '')}"
            )
            obj = {
                "_id": doc_id,
                "title": "",
                "text": normalize_text(f"{keyword_text} {original_text}"),
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    os.replace(tmp, output_path)

    write_manifest(
        output_path,
        artifact_type="keywords_plus_original_corpus",
        source_fingerprint=source_fingerprint,
        count=len(original_corpus),
    )
    print(f"[write] combined corpus: {output_path}")


def build_lucene_index(
    *,
    corpus: Mapping[str, Mapping[str, Any]],
    index_path: Path,
    source_fingerprint: str,
    mode: str,
    threads: int,
    allow_legacy_without_manifest: bool,
    language: str | None = None,
) -> None:
    validation = validate_index(
        index_path,
        source_fingerprint=source_fingerprint,
        allow_legacy_without_manifest=allow_legacy_without_manifest,
    )

    if mode == "skip":
        if not validation.valid:
            raise RuntimeError(
                f"Index is required but invalid: {index_path}: {validation.reason}"
            )
        print(f"[reuse] index: {index_path} ({validation.reason})")
        return

    if mode == "auto" and validation.valid:
        print(f"[reuse] index: {index_path} ({validation.reason})")
        return

    print(f"[build] index: {index_path} ({validation.reason})")

    collection_dir = index_path.with_name(index_path.name + ".collection.tmp")
    temp_index = index_path.with_name(index_path.name + ".index.tmp")

    shutil.rmtree(collection_dir, ignore_errors=True)
    shutil.rmtree(temp_index, ignore_errors=True)
    collection_dir.mkdir(parents=True, exist_ok=True)

    collection_file = collection_dir / "pyserini.jsonl"
    with collection_file.open("w", encoding="utf-8") as f:
        for doc_id, row in corpus.items():
            contents = normalize_text(f"{row.get('title', '')} {row.get('text', '')}")
            f.write(
                json.dumps({"id": str(doc_id), "contents": contents}, ensure_ascii=False)
                + "\n"
            )

    command = [
        sys.executable,
        "-m",
        "pyserini.index.lucene",
        "--collection",
        "JsonCollection",
        "--input",
        str(collection_dir),
        "--index",
        str(temp_index),
        "--generator",
        "DefaultLuceneDocumentGenerator",
        "--threads",
        str(max(1, threads)),
        "--storePositions",
        "--storeDocvectors",
        "--storeRaw",
    ]
    if language:
        # Non-English analyzers (MIRACL-style corpora indexed locally).
        command += ["--language", str(language)]

    subprocess.run(command, check=True)

    if index_path.exists():
        shutil.rmtree(index_path)
    os.replace(temp_index, index_path)

    atomic_json_dump(
        {"source_fingerprint": source_fingerprint, "count": len(corpus)},
        index_manifest_path(index_path),
    )
    shutil.rmtree(collection_dir, ignore_errors=True)
    print(f"[ready] index: {index_path}")
