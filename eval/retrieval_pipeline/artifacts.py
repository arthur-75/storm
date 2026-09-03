"""
Artifact manifests and reuse validation.

An artifact (generated queries / generated corpus / Lucene index) is reusable
only when its IDs match the source dataset AND its manifest fingerprint matches
the current source fingerprint.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from .io_utils import atomic_json_dump, load_corpus_jsonl, load_query_jsonl


@dataclass
class ValidationResult:
    valid: bool
    reason: str
    count: int = 0


def manifest_path(artifact_path: Path) -> Path:
    return artifact_path.with_name(artifact_path.name + ".manifest.json")


def read_manifest(artifact_path: Path) -> Optional[Dict[str, Any]]:
    path = manifest_path(artifact_path)
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else None


def write_manifest(
    artifact_path: Path,
    *,
    artifact_type: str,
    source_fingerprint: str,
    count: int,
    generation_signature: Optional[str] = None,
) -> None:
    payload: Dict[str, Any] = {
        "artifact_type": artifact_type,
        "source_fingerprint": source_fingerprint,
        "count": count,
    }
    if generation_signature is not None:
        payload["generation_signature"] = generation_signature
    atomic_json_dump(payload, manifest_path(artifact_path))


def validate_ids(
    artifact_path: Path,
    expected_ids: Sequence[str],
    *,
    artifact_kind: str,
    source_fingerprint: str,
    allow_legacy_without_manifest: bool,
) -> ValidationResult:
    if not artifact_path.is_file():
        return ValidationResult(False, "file is missing")

    try:
        if artifact_kind == "queries":
            records = load_query_jsonl(artifact_path)
        elif artifact_kind == "corpus":
            records = load_corpus_jsonl(artifact_path)
        else:
            raise ValueError(f"Unknown artifact kind: {artifact_kind}")
    except Exception as exc:
        return ValidationResult(False, f"cannot parse artifact: {exc}")

    actual_ids = set(records)
    expected_set = set(expected_ids)

    if len(records) != len(expected_ids):
        return ValidationResult(
            False,
            f"count mismatch: artifact={len(records)}, source={len(expected_ids)}",
            len(records),
        )

    if actual_ids != expected_set:
        missing = sorted(expected_set - actual_ids)[:5]
        unexpected = sorted(actual_ids - expected_set)[:5]
        return ValidationResult(
            False,
            f"ID mismatch; missing={missing}, unexpected={unexpected}",
            len(records),
        )

    manifest = read_manifest(artifact_path)
    if manifest is None:
        if allow_legacy_without_manifest:
            return ValidationResult(
                True,
                "exact count and ID set match; legacy artifact has no manifest",
                len(records),
            )
        return ValidationResult(False, "manifest is missing", len(records))

    if manifest.get("source_fingerprint") != source_fingerprint:
        return ValidationResult(
            False, "source corpus/query fingerprint mismatch", len(records)
        )

    if int(manifest.get("count", -1)) != len(records):
        return ValidationResult(False, "manifest count mismatch", len(records))

    return ValidationResult(
        True, "count, IDs, and source fingerprint match", len(records)
    )


def artifact_action(mode: str, validation: ValidationResult) -> str:
    if mode == "force":
        return "generate"
    if mode == "auto":
        return "reuse" if validation.valid else "generate"
    return "reuse" if validation.valid else "unavailable"
