from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def build_cache_key(
    *,
    evaluation_id: str,
    judge_name: str,
    question: str,
    answer: str,
    contexts: list[dict[str, Any]],
    corpus_version: str,
    extraction_table_version: str,
) -> str:
    """
    캐시 키에 코퍼스 버전과 추출 테이블 버전을 반드시 포함합니다.
    이를 통해 데이터셋/추출 결과가 바뀌었는데 이전 Judge 결과가 재사용되는 문제를 방지합니다.
    """
    payload = {
        "evaluation_id": evaluation_id,
        "judge_name": judge_name,
        "question": question,
        "answer": answer,
        "contexts": contexts,
        "corpus_version": corpus_version,
        "extraction_table_version": extraction_table_version,
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class FileCache:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)

    def get(self, key: str) -> dict[str, Any] | None:
        path = self.directory / f"{key}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def set(self, key: str, value: dict[str, Any]) -> None:
        path = self.directory / f"{key}.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(value, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp.replace(path)
