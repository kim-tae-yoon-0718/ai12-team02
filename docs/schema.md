# 평가셋 스키마 v0.2 (임현진, 2026-08-28)

한 문항 = JSONL 한 줄. `src/grader/models.py` 의 `EvaluationItem` 이 계약.
15개 필드 (공통 필수 5 + 조건부 10). `null` 조건부 필드는 **키 생략**.

## 필드

| 필드 | 타입 | 언제 |
| --- | --- | --- |
| `id` | str | 항상 (접두어 권장: `SEL-`/`EXT-`/`QA-`/`PRAC-`) |
| `question` | str | 항상 |
| `task_type` | `selection` \| `extraction` \| `qa` | 항상 |
| `answer_type` | `document_set` \| `value` \| `list` \| `summary` \| `comparison` \| `unanswerable` | 항상 |
| `answer_raw` | any | 항상 — 아래 참고 |
| `document_id` | str (배열도 허용) | extraction/qa 이고 `unspecified_type` 없을 때 |
| `unspecified_type` | `abbreviation` \| `org_only` \| `time_reference` \| `ambiguous_match` | 2-2-2 문서 미특정 |
| `intermediate_answer` | str/배열 | `unspecified_type` 있을 때 (필수) |
| `scenario_type` | `workflow_chain` \| `anaphora` \| `condition_add` \| `selection_to_extraction` | 2-5 후속 질문 |
| `active_document_id` | str | `scenario_type` 있을 때 |
| `reference_time` | str(date) | 2-6-2 시간 의존 — 상수 `"2024-06-01"` |
| `field_tag` | `critical` \| `major` \| `minor` | 오답 심각도 + 3-3-2 가중 채점 입력 |
| `answer_source` | `table` \| `verified` \| `metadata` | selection 필수 |
| `answer_normalized` | any | 정규화 대상 값(금액/날짜/기관명)일 때 |
| `location` | `{document, section, ref_no}` | 답이 문서 특정 위치에 근거할 때 (2-9 확정 단위) |

## `answer_raw` 는 `answer_type` 에 따라

| answer_type | answer_raw | 예 |
| --- | --- | --- |
| `document_set` | 문서 ID 배열 | `["DOC-014","DOC-036"]` |
| `value` | 값 | `"222,180,200원"` / `"지역 제한 없음"` |
| `list` | 항목 배열 | `["사업자등록증","제안서"]` |
| `summary` | 체크포인트 배열 (별도 `checkpoints` 필드 아님) | `["과업 범위","사업기간"]` |
| `comparison` | 비교표 | — |
| `unanswerable` | 기권 사유 문자열 (별도 필드 아님, `null` 금지) | `"그런 사업을 찾을 수 없습니다"` |

## task_type ↔ answer_type 허용 조합 (`models.ALLOWED_ANSWER_TYPES`, 2층에서 강제)

- `selection` → `document_set`
- `extraction` → `value`, `list`
- `qa` → `value`, `summary`, `comparison`, `unanswerable`

## v0.1 → v0.2 제거된 필드

- `document_unspecified` / `time_dependent` / `conversational` → 각각 `unspecified_type` /
  `reference_time` / `scenario_type` 존재 여부로 **파생**(`EvaluationItem` computed_field).
- `difficulty` → 폐기 (2-4).
- `schema_version` → 문항이 아니라 파일 단위 `VERSION.txt`.
- `checkpoints` / `unanswerable_reason` → `answer_raw` 로 흡수. `EvaluationItem.checkpoints` /
  `.unanswerable_reason` 는 읽기 전용 뷰(입력 필드 아님, `extra=forbid`).
- `completeness_rule` → `list` 는 항상 exact-all(코드 고정).

## 팀 결정 (2026-08-31) — C / D

- **C. field_absent**: 새 필드 신설 안 함. 답없음 문제로 쓰면 `answer_type=unanswerable` +
  `answer_raw` 부재 문구. 도메인상 유의미한 답이면 `extraction·value` + `answer_raw` 확정 문구
  ("지역제한 없음"). `conflict` 셀은 정답으로 쓰지 않고 문항에서 제외(v1 미포함).
- **D. 복수 정답**: 라벨 배열 구조 신설 안 함. 의미가 다른 값마다 별도 문항으로 분리
  (과업수행기간 문항 1개 + 사업개요 사업기간 문항 1개), 각각 독립 채점.

## practice 세트 주석 키

`practice_items.jsonl` 은 `_source_note` 같은 `_` 시작 키를 가질 수 있다. 로드 시 조용히
제거된다(`models` before-validator). 최종셋 검사(`--strict`)에서는 `_` 키가 있으면 FAIL.

## 시스템 응답 (`ModelResponse`)

`id`, `answer`, `structured_answer`, `contexts[]`, `retrieved[]`, `reranked[]`,
`citations[]`(Location), `selected_document_ids[]`, `abstained`, `unanswerable_reason`,
`route`, `failure`, `latency_ms`, `cost_usd`.
