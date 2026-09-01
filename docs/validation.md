# 검증

평가셋·데이터 **자체**의 검증. 모델 성능과 무관하고 값이 싸다 → 성능 평가보다 먼저 돈다.
평가셋이 깨진 채로 성능을 재면 그 숫자는 아무 뜻이 없다.

## 역할 분담 (2026-09-01 단일화)

| 검사 | 위치 | 담당 |
| --- | --- | --- |
| **평가셋 계약** — 스키마·조합·중복 id·할당량·참조 무결성·좌표·유출 (2-17) | `src/checks/check_evalset.py` | 임현진 (단일 출처) |
| 평가셋 → `EvaluationItem` 파싱 (채점 입력 계약) | `grader/validation/schema.py` | 김하루 |
| 6-자산 provenance 기록 완전성·재현성 (채점 **결과** 검증) | `grader/validation/provenance.py` | 김하루 |
| CI 1층 데이터 정상성 (data_manifest, 박예진 산출물) | `grader/validation/assets.py` | 김하루 |

`grader/validation/` 는 더 이상 평가셋 규칙을 갖지 않는다. `check_evalset_integrity()` 는
`checks.check_evalset.run_all` 을 부르는 얇은 어댑터일 뿐이다(EvaluationItem/dict 둘 다 받음).

> HJ 의 실제 `check_evalset.py` 가 dev 에 머지되면 벤더링본(`src/checks/check_evalset.py`)을
> 그 버전으로 재조정한다. 현재 벤더링본은 origin/HJ 기준 + C4 버그 수정 + C5/C6 구현 +
> 좌표 검사(2-9) 추가 + `run_all()`/`main()` 추가.

## `checks.check_evalset` (CI 2층, 2-17)

```python
from checks.check_evalset import run_all
run_all(records,                       # raw dict 리스트 — pydantic 파싱 전
        doc_ids_path=None, excluded_ids=None, practice_path=None,
        evalset_version_path=None, corpus_version_path=None,
        strict=False, final_set=False,
        leak_repo_root=None, leak_exclude=()) -> list[str]   # 빈 리스트면 통과
```

값싼 것부터: **C1 스키마 → C2 중복 id → C3 할당량 → C4 참조 무결성 → 1-9-1 → C5 버전 → C6 유출**

- **C1** `check_schema` — 필수 5필드 · FIELD_SPEC 타입/enum · 폐기 필드(v0.2) · task↔answer 조합 ·
  필드 의존성 · selection 은 answer_source 필수 · 명시적 null 금지. **`location` 이 배열이면
  원소마다 3키 검증** (비교형 2-8-4)
- **C2** `check_dup_ids` / **C3** `check_quota` (기본 선별25:추출15:QA10, `strict` 면 정확 대조)
- **C4** `check_ref_intg` — `location.document`(배열 포함)가 corpus_doc_ids.json 에 실재. 파일 없으면 SKIP
- **1-9-1** `check_excluded_as_gold` *(grader 부가분)* — 수집 중복 문서(RFP-000006/17)를 정답 근거로 쓰면 금지
- **C5** `check_version` — evalset VERSION.txt 의 `corpus:` 와 corpus VERSION.txt 대조. `[대기]` → SKIP
- **C6** `check_leak` — `final_set=True` 일 때: PRAC- 접두어 · practice 세트 id/문서 교집합
- **【25】** `scan_tracked_files` *(grader 부가분)* — 문항 텍스트가 프롬프트·코드에 유출 (final / `--leak-check`)

> **HJ 원본에 없는 것** (grader 부가분): `check_excluded_as_gold`(1-9-1), `scan_tracked_files`(【25】),
> `load_jsonl` 의 멀티라인 JSON 허용. HJ 머지 시 이 셋만 재조정.

## CLI

```bash
python -m checks.check_evalset final.jsonl --doc-ids data/gold/corpus_doc_ids.json \
  --evalset-version /srv/rfp/evalset/v1/VERSION.txt --final-set --strict --leak-scan-root .
```

## 좌표 어댑터 (`models.Location.from_chunk`)

**임현진 09-01 확정** — `location = {document, section, ref_no}`:
- `section` ← 청킹 산출물 `section_path` 의 **리프 요소** (`{title:...}` dict / 문자열 둘 다 처리)
- `ref_no` ← extraction_v3 기계 표기 **`"{block_type} {block_index}"`** — `table`/`paragraph`/`heading`
  (예: `"paragraph 3"`, `"table 12"`, `"heading 0"`). 박예진 청크의 `"text"` 는 `"paragraph"` 로 정규화.
  **사람 표기 "문단 N"·`(part/of)` 폐기.**
- **예외** `answer_source=metadata` (본문 블록 없음, CSV 답변): `section="CSV"`, `ref_no="CSV: {컬럼명}"` 고정.
  이 문항은 `grade_citation`/`grade_retrieval` 좌표 채점에서 **제외**(applicable=False).
- 비교형(2-8-4): `location` 은 **문서별 객체 배열**. `EvaluationItem.gold_locations()` 가 항상 리스트로 정규화.
  `grade_citation` 은 정답 좌표 하나하나가 citation 과 맞는 비율로 점수.

`ContextChunk` / `RetrievedItem` 은 `location` 이 없고 `block_type`+`block_index`(또는 옛 `location_label`)가
있으면 자동 합성한다.

> ⚠️ **박예진 청크(chunks_v2)는 아직 `block_index` 를 안 실어 보낸다** (`table_idx`·`location_label "표 1"` 만).
> evalset ref_no(`"table 12"`)와 청크 ref_no 가 **번호 체계가 달라** `precision=ref_no` citation 매칭이
> 성립하지 않는다. 박예진↔이태민 확인 필요 — 그전까지는 `config.retrieval.precision: section` 권장.

## 박예진 산출물 → grader 입력 (`scripts/`)

```bash
RAG_ROOT=/srv/rfp python scripts/build_doc_ids.py --out-dir data/gold
RAG_ROOT=/srv/rfp python scripts/build_data_manifest.py --out data/gold/data_manifest.json
```
- `corpus_doc_ids.json` (active 100) · `excluded_doc_ids.json` (RFP-000006/17, 검색대상 98)
- `data_manifest.json` — `corpus_v2/manifest.json` + `registry_metadata.json` + `chunks_v1/stats.json`
