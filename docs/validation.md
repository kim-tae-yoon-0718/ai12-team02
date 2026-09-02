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
        evalset_version_path=None, corpus_version_path=None, chunking_version_path=None,
        strict=False, final_set=False,
        leak_repo_root=None, leak_exclude=()) -> list[str]   # 빈 리스트면 통과
```

값싼 것부터: **C1 스키마 → C2 중복 id → C3 할당량 → C4 참조 무결성 → 1-9-1 → C5 버전 → C6 유출**

- **C1** `check_schema` — 필수 5필드 · FIELD_SPEC 타입/enum · 폐기 필드(v0.2) · task↔answer 조합 ·
  필드 의존성 · selection 은 answer_source 필수 · 명시적 null 금지. `location`(배열이면 원소마다):
  `{document, section, ref_no, line}` 4키 필수 — 단, `answer_source=metadata` 는 `line` 면제 (임현진 09-02)
- **C2** `check_dup_ids` / **C3** `check_quota` (기본 선별25:추출15:QA10, `strict` 면 정확 대조)
- **C4** `check_ref_intg` — `location.document`(배열 포함)가 corpus_doc_ids.json 에 실재. 파일 없으면 SKIP
- **1-9-1** `check_excluded_as_gold` *(grader 부가분)* — 수집 중복 문서(RFP-000006/17)를 정답 근거로 쓰면 금지
- **C5** `check_version` — evalset VERSION.txt 의 `corpus:` / `chunking:` 를 각각 corpus·chunks VERSION.txt 와 대조. `[대기]` → SKIP
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

**박예진·임현진 최종 확정 (2026-09-02)** — 평가셋 `location` 과 청크가 **같은 규약**:

| 필드 | 값 |
| --- | --- |
| `document` | `document_id` |
| `section` | `section_path` 의 리프(마지막) 요소 |
| `ref_no` | `location_label` 을 **그대로** (예: `"4. 제안 요청내용 · 문단 1-57"`, `"2. 사업개요 · 표 1"`) |
| `line` | 평가셋 = 단일값 / 청크 = `md_line_start` (`line_end` 는 매칭용 범위 = `md_line_end`) |

`match_location(precision="ref_no")` 대조 순서:
1. 문서 + 절(section) 먼저 일치해야 함
2. 양쪽에 `line` 이 있으면 → **청크 `[line, line_end]` 범위에 평가셋 `line` 이 드는지** (가장 정밀, 같은 라벨이 문서에 여럿일 때 명확)
3. `line` 없으면 → `ref_no` 문자열 일치

- **예외** `answer_source=metadata` (CSV 답변): `section="CSV"`, `ref_no="CSV: {컬럼명}"`, `line` 없음 → `grade_citation`/`grade_retrieval` 좌표 채점 **제외**
- **비교형(2-8-4)**: `location` 은 문서×필드 배열 `[{document, field, section, ref_no, line}, …]`. `grade_citation` 이 정답 좌표 하나하나를 citations 배열과 대조하고, "근거 누락"(`n_missing_evidence`) vs "잘못된 근거"(`n_wrong_citations`) 를 분리

`ContextChunk` / `RetrievedItem` 은 `location` 이 없고 청크 필드(`section_path`, `location_label`, `md_line_start/end`)가 있으면 `Location.from_chunk` 로 자동 합성한다.

## 박예진 산출물 → grader 입력 (`scripts/`)

```bash
RAG_ROOT=/srv/rfp python scripts/build_doc_ids.py --out-dir data/gold
RAG_ROOT=/srv/rfp python scripts/build_data_manifest.py --out data/gold/data_manifest.json
```
- `corpus_doc_ids.json` (active 100) · `excluded_doc_ids.json` (RFP-000006/17, 검색대상 98)
- `data_manifest.json` — `corpus_v2/manifest.json` + `registry_metadata.json` + `chunks_v3/stats.json` (자동 최신 감지)
