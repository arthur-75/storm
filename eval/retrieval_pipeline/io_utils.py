"""
Reading and writing data: JSONL artifacts, atomic writes, resumable partial
files and corpus shard merging.
"""

from __future__ import annotations

import gzip
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Sequence, Set, Tuple


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def normalize_text(text: Any) -> str:
    return " ".join(str(text or "").split())


def atomic_json_dump(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def load_json(path: Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def batched(
    items: Sequence[Tuple[str, str]],
    batch_size: int,
) -> Iterator[Sequence[Tuple[str, str]]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


# ---------------------------------------------------------------------------
# JSONL formats
# ---------------------------------------------------------------------------

def parse_query_record(obj: Mapping[str, Any], line_no: int, path: Path) -> Tuple[str, str]:
    # Canonical format: {"_id": "...", "text": "..."}
    if "_id" in obj and "text" in obj:
        return str(obj["_id"]), str(obj["text"])
    # Also accept: {"qid": "...", "text": "..."}
    if "qid" in obj and "text" in obj:
        return str(obj["qid"]), str(obj["text"])
    # Backward compatibility with the original script: {"qid-value": "text"}
    if len(obj) == 1:
        key, value = next(iter(obj.items()))
        return str(key), str(value)
    raise ValueError(f"Unsupported query JSONL record at {path}:{line_no}: {obj}")


def load_query_jsonl(path: Path) -> Dict[str, str]:
    records: Dict[str, str] = {}
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(f"Expected object at {path}:{line_no}")
            qid, text = parse_query_record(obj, line_no, Path(path))
            if qid in records:
                raise ValueError(f"Duplicate query id {qid!r} in {path}")
            records[qid] = text
    return records


def iter_jsonl_records(path: Path) -> Iterator[Dict[str, Any]]:
    """Yield records from a .jsonl or .jsonl.gz file."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:  # type: ignore[operator]
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(f"Expected object at {path}:{line_no}")
            yield obj


def corpus_files(path: Path) -> List[Path]:
    """
    Accept either a single JSONL file or a directory of shards
    (docs-0.jsonl.gz, docs-1.jsonl.gz, ...), sorted for determinism.
    """
    path = Path(path)
    if path.is_file():
        return [path]
    if not path.is_dir():
        return []

    files: List[Path] = []
    for pattern in ("*.jsonl", "*.jsonl.gz", "*.json.gz"):
        files.extend(path.rglob(pattern))
    return sorted(files)


def load_corpus_any(path: Path) -> Dict[str, Dict[str, str]]:
    """
    Load a corpus from a file or a directory of shards, accepting both the
    pipeline format ({"_id", "title", "text"}) and the raw MIRACL/HuggingFace
    format ({"docid", "title", "text"}).
    """
    files = corpus_files(path)
    if not files:
        raise FileNotFoundError(f"No corpus JSONL files found under: {path}")

    records: Dict[str, Dict[str, str]] = {}
    for file_path in files:
        for obj in iter_jsonl_records(file_path):
            doc_id = obj.get("_id", obj.get("docid", obj.get("id")))
            if doc_id is None or "text" not in obj:
                raise ValueError(
                    f"Corpus record needs an id (_id/docid/id) and 'text': {file_path}"
                )
            doc_id = str(doc_id)
            if doc_id in records:
                raise ValueError(f"Duplicate document id {doc_id!r} in {file_path}")
            records[doc_id] = {
                "title": str(obj.get("title", "")),
                "text": str(obj["text"]),
            }

    print(f"[data] loaded {len(records):,} documents from {len(files)} file(s)")
    return records


def load_corpus_jsonl(path: Path) -> Dict[str, Dict[str, str]]:
    records: Dict[str, Dict[str, str]] = {}
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict) or "_id" not in obj or "text" not in obj:
                raise ValueError(
                    f"Corpus record must contain '_id' and 'text' at {path}:{line_no}"
                )
            doc_id = str(obj["_id"])
            if doc_id in records:
                raise ValueError(f"Duplicate document id {doc_id!r} in {path}")
            records[doc_id] = {
                "title": str(obj.get("title", "")),
                "text": str(obj["text"]),
            }
    return records


def save_query_jsonl(records: Sequence[Tuple[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for qid, text in records:
            f.write(
                json.dumps({"_id": str(qid), "text": str(text)}, ensure_ascii=False)
                + "\n"
            )
    os.replace(tmp, path)


def save_corpus_jsonl(records: Sequence[Tuple[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for doc_id, text in records:
            f.write(
                json.dumps(
                    {"_id": str(doc_id), "title": "", "text": str(text)},
                    ensure_ascii=False,
                )
                + "\n"
            )
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Resumable generation checkpoints
# ---------------------------------------------------------------------------

def load_partial_ids(partial_path: Path, artifact_kind: str) -> Set[str]:
    """
    Read IDs already generated in a partial JSONL file.

    If the process stopped while writing the final line, truncate that
    incomplete line and preserve all previous valid records.
    """
    completed_ids: Set[str] = set()

    if not partial_path.is_file():
        return completed_ids

    valid_end = 0

    with partial_path.open("rb") as f:
        while True:
            line = f.readline()
            if not line:
                break

            try:
                obj = json.loads(line.decode("utf-8"))

                if artifact_kind == "queries":
                    record_id, _ = parse_query_record(obj, line_no=0, path=partial_path)
                elif artifact_kind == "corpus":
                    if (
                        not isinstance(obj, dict)
                        or "_id" not in obj
                        or "text" not in obj
                    ):
                        raise ValueError("Invalid corpus partial record")
                    record_id = str(obj["_id"])
                else:
                    raise ValueError(f"Unknown artifact kind: {artifact_kind}")

                if record_id in completed_ids:
                    raise ValueError(f"Duplicate ID in partial file: {record_id}")

                completed_ids.add(record_id)
                valid_end = f.tell()

            except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
                print(
                    f"[resume] Removing incomplete/corrupt final record "
                    f"from {partial_path}"
                )
                break

    file_size = partial_path.stat().st_size
    if valid_end < file_size:
        with partial_path.open("r+b") as f:
            f.truncate(valid_end)

    return completed_ids


# ---------------------------------------------------------------------------
# Sharded corpus generation
# ---------------------------------------------------------------------------

def select_contiguous_shard(
    items: Sequence[Tuple[str, str]],
    shard_id: int,
    num_shards: int,
) -> List[Tuple[str, str]]:
    """
    Split items into deterministic contiguous parts.

    For two shards:
      shard 0 gets approximately the first half
      shard 1 gets approximately the second half
    """
    if num_shards < 1:
        raise ValueError("num_shards must be >= 1")
    if shard_id < 0 or shard_id >= num_shards:
        raise ValueError(
            f"shard_id must be between 0 and {num_shards - 1}, got {shard_id}"
        )

    start = len(items) * shard_id // num_shards
    end = len(items) * (shard_id + 1) // num_shards
    return list(items[start:end])


def shard_artifact_path(output_path: Path, shard_id: int, num_shards: int) -> Path:
    """
    corpus.jsonl becomes corpus.part-00000-of-00002.jsonl
    """
    return output_path.with_name(
        f"{output_path.stem}"
        f".part-{shard_id:05d}-of-{num_shards:05d}"
        f"{output_path.suffix}"
    )


def merge_corpus_shards(
    *,
    final_path: Path,
    num_shards: int,
    expected_ids: Sequence[str],
) -> None:
    """
    Merge completed corpus shard files into the canonical corpus.jsonl.

    The shard files remain on disk after merging so that they can be reused.
    """
    expected_id_set = {str(x) for x in expected_ids}
    seen_ids: Set[str] = set()

    shard_paths = [
        shard_artifact_path(final_path, shard_id=i, num_shards=num_shards)
        for i in range(num_shards)
    ]

    missing_files = [str(p) for p in shard_paths if not p.is_file()]
    if missing_files:
        raise FileNotFoundError(
            "Some corpus shards are not complete:\n" + "\n".join(missing_files)
        )

    final_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = final_path.with_name(final_path.name + ".merge.tmp")

    with temporary_path.open("w", encoding="utf-8") as output_file:
        for shard_path in shard_paths:
            with shard_path.open("r", encoding="utf-8") as shard_file:
                for line_number, line in enumerate(shard_file, start=1):
                    if not line.strip():
                        continue

                    obj = json.loads(line)
                    if (
                        not isinstance(obj, dict)
                        or "_id" not in obj
                        or "text" not in obj
                    ):
                        raise ValueError(
                            f"Invalid corpus record at {shard_path}:{line_number}"
                        )

                    document_id = str(obj["_id"])
                    if document_id in seen_ids:
                        raise ValueError(
                            f"Duplicate document ID across shards: {document_id}"
                        )

                    seen_ids.add(document_id)
                    output_file.write(json.dumps(obj, ensure_ascii=False) + "\n")

        output_file.flush()
        os.fsync(output_file.fileno())

    missing_ids = expected_id_set - seen_ids
    unexpected_ids = seen_ids - expected_id_set

    if missing_ids or unexpected_ids:
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Corpus shard merge failed. "
            f"Expected={len(expected_id_set):,}, found={len(seen_ids):,}, "
            f"missing examples={sorted(missing_ids)[:5]}, "
            f"unexpected examples={sorted(unexpected_ids)[:5]}"
        )

    os.replace(temporary_path, final_path)
    print(f"[merge] Merged {num_shards} corpus shards into {final_path}")