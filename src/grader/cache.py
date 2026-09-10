from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


# ★캐시 키 형식 버전. 올리면 예전 형식으로 만들어진 파일은 새 코드에서
#   절대 다시 쓰이지 않는다(키가 통째로 달라진다). 심판 설정이 키에 없던 시절의
#   결과가 새 설정 실행에 재사용되는 것을 막는 안전장치다.
CACHE_KEY_SCHEMA = "v2"


def build_cache_key(
    *,
    evaluation_id: str,
    judge_name: str,
    question: str,
    answer: str,
    contexts: list[dict[str, Any]],
    corpus_version: str,
    extraction_table_version: str,
    judge_model: str,
    judge_family: str,
    judge_provider: str,
    judge_temperature: float,
    judge_tier: str,
    prompt_version: str,
    prompt_sha256: str,
    scorer_version: str,
    evalset_version: str,
) -> str:
    """심판 결과 캐시 키.

    ★예전 키에는 질문·답변·컨텍스트·코퍼스·추출표만 들어 있었다. 그래서
      **심판 모델·계열·provider·temperature·등급·채점 프롬프트(이름·버전·내용)·
      채점기 버전·평가셋 버전이 바뀌어도 예전 판정을 그대로 다시 썼다.**
      "심판을 바꿨는데 점수가 안 변한다"가 그 증상이다.

    prompt_sha256 을 넣는 이유: 프롬프트 파일 내용만 바뀌고 버전 문자열(v1)이
    그대로인 경우가 흔하다. 버전만 보면 그 변경을 놓친다.

    모델 계열은 **정규화한 값**을 넣는다 — 'gpt' 와 'openai' 가 다른 키를 만들면
    같은 심판이 캐시를 두 벌 갖게 된다.
    """
    from .readiness import normalize_family

    payload = {
        "schema": CACHE_KEY_SCHEMA,
        "evaluation_id": evaluation_id,
        "judge_name": judge_name,
        "question": question,
        "answer": answer,
        "contexts": contexts,
        "corpus_version": corpus_version,
        "extraction_table_version": extraction_table_version,
        # 심판 설정
        "judge_model": judge_model,
        "judge_family": normalize_family(judge_family) or str(judge_family),
        "judge_provider": judge_provider,
        "judge_temperature": judge_temperature,
        "judge_tier": judge_tier,
        # 채점 프롬프트
        "prompt_version": prompt_version,
        "prompt_sha256": prompt_sha256,
        # 채점 기준 자체의 버전
        "scorer_version": scorer_version,
        "evalset_version": evalset_version,
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
