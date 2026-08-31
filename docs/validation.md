# 검증 (`src/grader/validation/`)

평가셋·데이터 **자체**의 검증. 모델 성능과 무관하고 값이 싸다 → 성능 평가보다 먼저 돈다.
평가셋이 깨진 채로 성능을 재면 그 숫자는 아무 뜻이 없다.

| 모듈 | 함수 | 무엇 |
| --- | --- | --- |
| `schema.py` | `load_jsonl` · `validate_evaluation_set(strict_meta=)` · `validate_model_responses` · `index_by_id` | JSONL 로딩 + pydantic 스키마 검증. `strict_meta` → 최종셋에서 `_` 주석 키 FAIL |
| `answer.py` | `check_answer_types` | task_type↔answer_type 허용 조합 + list/summary/unanswerable/unspecified 정합성 |
| `location.py` | `check_locations(require_ref_no=)` | `location` 3필드가 실제 값인지 (자리표시자·빈값 잡음 — ref_no 단위 채점 전제) |
| `provenance.py` | `check_provenance` · `check_reproducible` | 6-자산 기록 완전성 / git_dirty·UNKNOWN 축 |
| `assets.py` | `check_references` · `check_quota` · `check_data_sanity` · `check_data_warnings` | 코퍼스 문서 참조 무결성 + 수집중복 제외(1-9-1) + 할당량 + CI 1층 데이터 정상성 |
| `__init__.py` | `check_evalset_integrity` | CI 2층 오케스트레이터 (중복 id + answer + references + quota) |

## `check_evalset_integrity` (CI 2층, 2-17)

```python
check_evalset_integrity(items, corpus_doc_ids=None, quota=None,
                        retrieval_excluded_ids=None) -> list[str]  # 빈 리스트면 통과
```

- **중복 id**
- **answer_type 정합성** (`answer.check_answer_types`)
- **참조 무결성** (`assets.check_references`) — 정답 근거 문서가 코퍼스에 실재 +
  검색 대상(`retrieval_excluded_ids` = 수집 중복 `RFP-000006`/`RFP-000017` 제외, 검색대상 98건)
- **할당량** (`assets.check_quota`) — `gate.task_quota` 있을 때만

## CLI

```bash
python -m grader.cli validate --evaluation-set data/evalsets/final/final.jsonl --strict
# run 실행 시 2층으로 자동 포함. --corpus-doc-ids / --excluded-doc-ids (JSON 배열) 로 참조 검사 활성화
```

## 아직 조율 필요

- `location` 좌표 형식: 내 `{document, section, ref_no}` vs 박예진 청크의
  `section_path` 배열 + `"제18조(평가배점) · 표 7 (2/3)"`. 임현진 2-9(평가셋 근거 좌표)와
  정합돼야 좌표 채점(3-4-3)이 성립.
