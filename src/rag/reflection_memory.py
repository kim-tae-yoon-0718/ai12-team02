"""Read-only retrieval of developer-approved semantic records (memory A)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


class ReflectionMemoryError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _repo_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[2] / path
    return path.resolve()


class ReflectionMemory:
    """Retrieve similar approved records without interpreting question text."""

    def __init__(self, cfg: dict[str, Any], client: Any):
        self.client = client
        self.model = str(cfg.get("reflection_memory_embedding_model") or "")
        self.top_k = int(cfg.get("reflection_memory_top_k", 3))
        self.min_similarity = float(cfg.get("reflection_memory_min_similarity", 0.58))
        if not self.model:
            raise ReflectionMemoryError("reflection memory embedding model is required")
        if self.top_k < 1 or self.top_k > 10:
            raise ReflectionMemoryError("reflection_memory_top_k must be 1..10")
        if not -1.0 <= self.min_similarity <= 1.0:
            raise ReflectionMemoryError("invalid reflection memory similarity threshold")

        self.records_path = _repo_path(str(cfg.get("reflection_memory_records") or ""))
        self.vectors_path = _repo_path(str(cfg.get("reflection_memory_vectors") or ""))
        self.manifest_path = _repo_path(str(cfg.get("reflection_memory_manifest") or ""))
        for path in (self.records_path, self.vectors_path, self.manifest_path):
            if not path.is_file():
                raise ReflectionMemoryError(f"reflection memory asset missing: {path}")

        self.records = [
            json.loads(line)
            for line in self.records_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not self.records:
            raise ReflectionMemoryError("approved memory A is empty")
        if any(row.get("promotion_status") != "approved" for row in self.records):
            raise ReflectionMemoryError("unapproved record found in memory A")
        ids = [row.get("record_id") for row in self.records]
        if len(ids) != len(set(ids)) or any(not value for value in ids):
            raise ReflectionMemoryError("memory A record ids are missing or duplicated")

        self.vectors = np.load(self.vectors_path, allow_pickle=False)
        if self.vectors.ndim != 2 or self.vectors.shape[0] != len(self.records):
            raise ReflectionMemoryError("memory A vector rows do not match records")
        norms = np.linalg.norm(self.vectors.astype(np.float32), axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ReflectionMemoryError("memory A contains a zero vector")
        self.vectors = self.vectors.astype(np.float32) / norms

        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        expected = {
            "records_sha256": _sha256(self.records_path),
            "vectors_sha256": _sha256(self.vectors_path),
            "embedding_model": self.model,
            "record_count": len(self.records),
            "vector_dimension": int(self.vectors.shape[1]),
        }
        for key, value in expected.items():
            if manifest.get(key) != value:
                raise ReflectionMemoryError(
                    f"reflection memory manifest mismatch: {key}")
        self.manifest = manifest

    def retrieve(self, question: str, usage: Any) -> list[dict[str, Any]]:
        response = self.client.embeddings.create(model=self.model, input=[question])
        tokens = int(getattr(getattr(response, "usage", None), "total_tokens", 0) or 0)
        usage.add_embedding(tokens)
        vector = np.asarray(response.data[0].embedding, dtype=np.float32)
        if vector.ndim != 1 or vector.shape[0] != self.vectors.shape[1]:
            raise ReflectionMemoryError("query embedding dimension mismatch")
        norm = float(np.linalg.norm(vector))
        if norm == 0:
            raise ReflectionMemoryError("query embedding is zero")
        scores = self.vectors @ (vector / norm)
        indices = np.argsort(-scores)[: self.top_k]
        selected: list[dict[str, Any]] = []
        for index in indices:
            similarity = float(scores[index])
            if similarity < self.min_similarity:
                continue
            row = self.records[int(index)]
            selected.append({
                "record_id": row["record_id"],
                "question": row["question"],
                "approved_plan": row["approved_plan"],
                "verified_strategy": row.get("verified_strategy", []),
                "review_reason": row.get("review_reason", ""),
                "reviewed_by": row.get("reviewed_by"),
                "reviewed_at": row.get("reviewed_at"),
                "removable_tag": row.get("removable_tag"),
                "similarity": round(similarity, 6),
            })
        return selected
