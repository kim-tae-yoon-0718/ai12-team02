from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from pydantic import ValidationError

from .models import ALLOWED_ANSWER_TYPES, EvaluationItem, ModelResponse, as_id_list


def load_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{line_no} JSON 파싱 실패: {exc}"
                ) from exc
    return records


def validate_evaluation_set(path: str | Path) -> list[EvaluationItem]:
    result: list[EvaluationItem] = []
    for idx, record in enumerate(load_jsonl(path), start=1):
        try:
            result.append(EvaluationItem.model_validate(record))
        except ValidationError as exc:
            raise ValueError(
                f"{path} line#{idx} 평가셋 스키마 오류:\n{exc}"
            ) from exc
    return result


def validate_model_responses(path: str | Path) -> list[ModelResponse]:
    result: list[ModelResponse] = []
    for idx, record in enumerate(load_jsonl(path), start=1):
        try:
            result.append(ModelResponse.model_validate(record))
        except ValidationError as exc:
            raise ValueError(
                f"{path} line#{idx} 모델 응답 스키마 오류:\n{exc}"
            ) from exc
    return result


def index_by_id(items: Iterable) -> dict[str, object]:
    indexed: dict[str, object] = {}
    for item in items:
        item_id = item.id
        if item_id in indexed:
            raise ValueError(f"중복 id 발견: {item_id}")
        indexed[item_id] = item
    return indexed


# ------------------------------------------------------------------ CI 2층: 평가셋 무결성 (2-17)

def check_evalset_integrity(
    items: Iterable[EvaluationItem],
    corpus_doc_ids: set[str] | None = None,
    quota: dict[str, int] | None = None,
    quota_tolerance: float = 0.2,
) -> list[str]:
    """CI가 평가셋에 대해 매번 확인할 것(2-17).

    ★ 이 검사들은 모델 성능과 무관하고 값이 싸다. 성능 평가보다 먼저 돌린다 —
      평가셋이 깨진 채로 성능을 재면 그 숫자는 아무 뜻이 없다.
    """
    items = list(items)
    problems: list[str] = []

    # (1) 중복 id
    seen: dict[str, int] = {}
    for it in items:
        seen[it.id] = seen.get(it.id, 0) + 1
    for k, v in seen.items():
        if v > 1:
            problems.append(f"[중복] id 중복 {k} x{v}")

    # (2) 필드 간 정합성 — 스키마 v0.2(임현진, 2026-08-28). 타입은 pydantic이 이미 검사했다.
    #     document_unspecified/time_dependent/conversational 은 각각 unspecified_type/
    #     reference_time/scenario_type 존재로 파생하는 computed_field 라 "둘 중 하나만
    #     채워진 모순" 상태 자체가 불가능해졌다 — v0.1 시절 정합성 검사는 죽은 코드라 제거.
    for it in items:
        allowed = ALLOWED_ANSWER_TYPES.get(it.task_type, set())
        if it.answer_type not in allowed:
            problems.append(
                f"[정합성] {it.id}: task_type={it.task_type} 에 허용되지 않는 "
                f"answer_type={it.answer_type} (허용: {sorted(allowed)})")
        if it.answer_type == "list" and not isinstance(it.answer_normalized or it.answer_raw, list):
            problems.append(f"[정합성] {it.id}: answer_type=list 인데 정답이 배열이 아님(3-4-4 항목 단위 채점 불가)")
        if it.answer_type == "summary" and not it.checkpoints:
            problems.append(f"[정합성] {it.id}: answer_type=summary 인데 answer_raw 에 체크포인트 배열 없음(2-8-2)")
        if it.answer_type == "unanswerable" and not (isinstance(it.answer_raw, str) and it.answer_raw):
            problems.append(f"[정합성] {it.id}: answer_type=unanswerable 인데 answer_raw 에 사유 문자열 없음(2-3, null 금지)")
        if it.unspecified_type is not None and not as_id_list(it.intermediate_answer):
            problems.append(f"[정합성] {it.id}: unspecified_type 인데 intermediate_answer 없음(3-2-2 분리 채점 불가)")

    # (3) 참조 무결성 — 근거/정답이 가리키는 문서가 코퍼스에 실재하는가
    if corpus_doc_ids is not None:
        for it in items:
            if it.location is not None and it.location.document not in corpus_doc_ids:
                problems.append(f"[참조] {it.id}: 코퍼스에 없는 document {it.location.document}")
            for d in as_id_list(it.document_id):
                if d not in corpus_doc_ids:
                    problems.append(f"[참조] {it.id}: document_id 미존재 {d}")
            for d in as_id_list(it.intermediate_answer):
                if d not in corpus_doc_ids:
                    problems.append(f"[참조] {it.id}: intermediate_answer 미존재 문서 {d}")
            # v0.2: 선별형 정답 문서 집합은 answer_raw 배열
            if it.answer_type == "document_set":
                for d in as_id_list(it.answer_raw):
                    if d not in corpus_doc_ids:
                        problems.append(f"[참조] {it.id}: answer_raw 문서 미존재 {d}")

    # (4) task_type 할당량 이탈
    if quota:
        counts = {t: 0 for t in quota}
        for it in items:
            if it.task_type in counts:
                counts[it.task_type] += 1
        for t, target in quota.items():
            got = counts.get(t, 0)
            if target and abs(got - target) > max(1, target * quota_tolerance):
                problems.append(f"[할당량] task_type={t} 설계 {target} vs 실제 {got}")

    return problems


# ------------------------------------------------------------------ CI 1층: 데이터 정상성 (1-19-1)

def check_data_sanity(manifest: dict, thresholds: dict | None = None) -> list[str]:
    """데이터 계약 검사(1-19-1, 박예진 소관 산출물 기준).

    ★ 모델 성능 평가가 아니라 데이터 계약 검사다. 훨씬 싸고 훨씬 빨리 잡는다.

    [대기 ← 박예진] 기준선(정상 로드율·결측률)은 1-11·1-13·1-12-2 실측값으로 채운다.
    thresholds 가 비어 있으면(팀 실측 전) 검사를 건너뛰지 않고, "기준 없음"으로 통과시킨다 —
    임의의 숫자를 기준으로 쓰지 않는다.
    """
    th = thresholds or {}
    problems: list[str] = []

    exp = th.get("expected_docs")
    if exp is not None and manifest.get("doc_count") != exp:
        problems.append(f"문서 수 {manifest.get('doc_count')} ≠ 예상 {exp}")
    lr = manifest.get("load_rate")
    if lr is not None and "min_load_rate" in th and lr < th["min_load_rate"]:
        problems.append(f"정상 로드율 {lr} < 기준 {th['min_load_rate']}")
    if manifest.get("duplicate_doc_ids"):
        problems.append(f"문서 ID 중복 {manifest['duplicate_doc_ids']}")
    for k in ("corpus_version", "extraction_table_version"):
        if not manifest.get(k):
            problems.append(f"버전 꼬리표 없음: {k} — 섞임 검출 불가")
    miss = manifest.get("critical_column_missing_rate")
    if miss is not None and "max_critical_missing_rate" in th and miss > th["max_critical_missing_rate"]:
        problems.append(f"critical 컬럼 결측률 {miss} > 기준 {th['max_critical_missing_rate']}")

    # 2026-08-27 박예진 1-11 확정 — 4종 CI 데이터 게이트("논쟁 없음"으로 확정된 항목).
    # manifest에 해당 키가 없으면(아직 그 산출물을 안 만든 단계) 조용히 건너뛴다 — 있는데
    # 기준을 넘을 때만 문제로 잡는다.
    raw_count = manifest.get("raw_doc_count")
    md_count = manifest.get("md_doc_count")
    if raw_count is not None and md_count is not None and "max_raw_md_count_diff" in th:
        diff = abs(raw_count - md_count)
        if diff > th["max_raw_md_count_diff"]:
            problems.append(
                f"raw/md 문서 수 불일치: raw={raw_count} md={md_count} (허용 diff {th['max_raw_md_count_diff']})"
            )
    fn_mismatches = manifest.get("filename_normalization_mismatches")
    if fn_mismatches is not None and "max_filename_mismatches" in th and fn_mismatches > th["max_filename_mismatches"]:
        problems.append(f"파일명 정규화 불일치 {fn_mismatches}건 > 기준 {th['max_filename_mismatches']}")
    csv_fail = manifest.get("csv_decode_failure_rows")
    if csv_fail is not None and "max_csv_decode_failures" in th and csv_fail > th["max_csv_decode_failures"]:
        problems.append(f"CSV 디코딩 실패 행 {csv_fail}건 > 기준 {th['max_csv_decode_failures']}")

    # 2026-08-27 박예진 파싱 품질 실측 공유분 — 표 보존율/인코딩 깨짐.
    # (참고: 원본 대비 손실률은 팀 결정으로 이 기준선에 넣지 않는다 — CI/평가 지표 대상 아님)
    table_rate = manifest.get("table_preservation_rate")
    if table_rate is not None and "min_table_preservation_rate" in th and table_rate < th["min_table_preservation_rate"]:
        problems.append(f"표 보존율 {table_rate} < 기준 {th['min_table_preservation_rate']}")
    enc_fail = manifest.get("encoding_corruption_count")
    if enc_fail is not None and "max_encoding_corruption" in th and enc_fail > th["max_encoding_corruption"]:
        problems.append(f"인코딩 깨짐 {enc_fail}건 > 기준 {th['max_encoding_corruption']}")

    return problems


def check_data_warnings(manifest: dict, warn_thresholds: dict | None = None) -> list[str]:
    """2026-08-27 박예진 공유 — 게이트가 아니라 경고. 실험을 막지 않고 사람이 한 번 보고
    넘기는 용도다.

    ★ 표 0개 문서: 실측 100건 전부 표가 최소 3개였다. 앞으로 새 공고를 계속 받는데
    (1-9 확정) 스캔된 한글 파일이 섞여 들어오면 kordoc의 OCR 경로가 없어 글자가 거의
    없는 md가 만들어진다 — 그런데 이게 1-11 게이트 4종(로드율/raw=md/파일명/CSV)을
    전부 통과한다("500자 미만은 로드 실패로 안 센다"는 규정 때문에 더욱 그렇다).
    표 0개는 이 구멍을 잡는 유일한 값싼 신호다.

    ★ 왜 게이트가 아니라 경고인가: 표가 원래 없는 정상 RFP가 있을 가능성이 미확인이다.
    게이트로 두면 그런 문서 하나에 전체 실행이 멈춘다 — 경고면 오탐 비용이 거의 0이다.

    ★ 한계(1-19-1에 기록): 문서 일부만 스캔인 경우는 못 잡는다 — 나머지 부분에 표가
    있으면 통과한다.
    """
    th = warn_thresholds or {}
    warnings: list[str] = []
    zero_table = manifest.get("zero_table_doc_count")
    if zero_table is not None and "max_zero_table_docs" in th and zero_table > th["max_zero_table_docs"]:
        warnings.append(
            f"[경고] 표 0개인 문서 {zero_table}건 — 스캔된 한글 파일(OCR 미실행) 가능성. "
            "게이트 아님, 실행을 막지 않는다. 사람이 직접 열어 확인할 것 "
            "(한계: 문서 일부만 스캔인 경우는 못 잡음 — 1-19-1)"
        )
    return warnings
