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
run_all(records,                       # raw dict (JSONL 한 줄) 리스트 — pydantic 파싱 전
        corpus_doc_ids=None, excluded_doc_ids=None,
        quota=None, quota_strict=False,
        practice_path=None, version_txt_path=None,
        leak_repo_root=None, leak_exclude=()) -> list[str]   # 빈 리스트면 통과
```

값싼 것부터: **C1 스키마 → C2 중복 id → C3 할당량 → 좌표 → C4 참조 무결성 → C5 버전 → C6 유출**

- **C1** `check_schema` — 필수 5필드 · FIELD_SPEC 타입/enum · 폐기 필드(v0.2) · task↔answer 조합 ·
  필드 의존성 · selection 은 answer_source 필수 · 명시적 null 금지 · 정답 형태(list/unanswerable)
- **C2** `check_dup_ids`
- **C3** `check_quota` — 기본 선별25:추출15:QA10. `strict=False` 면 미지의 task_type 만 경고
- **좌표** `check_location_coords` — 3필드 실제 값 여부 + part/of `(N/M)` 경고 (2-9)
- **C4** `check_ref_intg` — 정답 근거 문서가 코퍼스 실재 + 검색 대상
  (`excluded_doc_ids` = 수집 중복 `RFP-000006`/`RFP-000017`, 검색대상 98건). ID 없으면 SKIP
- **C5** `check_version` — VERSION.txt corpus 값 대조. `[대기]`/파일 없음 → SKIP
- **C6** `check_leak` — practice id 접두어(PRAC-)·practice 세트 교집합·practice 전용 문서(2-13)·
  문항 텍스트가 추적 파일에 유출(`--leak-check` / final 모드)

## CLI

```bash
# 독립 실행
python -m checks.check_evalset data/evalsets/final/final.jsonl \
  --doc-ids data/gold/corpus_doc_ids.json --excluded-doc-ids data/gold/excluded_doc_ids.json \
  --practice data/evalsets/practice/practice_items.jsonl --strict --leak-scan-root .

# grader run 실행 시 2층으로 자동 포함
python -m grader.cli run --evaluation-set ... --mode ci \
  --corpus-doc-ids data/gold/corpus_doc_ids.json --excluded-doc-ids data/gold/excluded_doc_ids.json
```

## 박예진 청크 좌표 어댑터 (`models.Location.from_chunk`, 구현됨)

이태민 검색 출력이 박예진 청크 스키마(`section_path` 배열 + `location_label`
`"제18조(평가배점) · 표 7 (2/3)"`)로 오면 `{document, section, ref_no}` 로 변환한다.
`RetrievedItem` / `ContextChunk` 는 `location` 이 없고 `section_path`/`location_label` 이
있으면 자동 합성한다.

- `section` ← `section_path[-1]` (없으면 `location_label` 의 ` · ` 앞부분)
- `ref_no` ← ` · ` 뒤에서 `(part/of)` 제거 → `"표 7"` / `"문단 1-4"`

**팀 확정 (2026-08-31)**: 좌표 단위는 `ref_no = "표 7"`, **part/of 미포함**. `check_location_coords`
는 옛 형식 `표 7 (2/3)` 가 남아 있으면 경고하고(정리 유도), 채점(`Location.key("ref_no")`)은
자동으로 떼고 매칭한다.

## 박예진 산출물 → grader 입력 (`scripts/`)

```bash
RAG_ROOT=/srv/rfp python scripts/build_doc_ids.py --out-dir data/gold
RAG_ROOT=/srv/rfp python scripts/build_data_manifest.py --out data/gold/data_manifest.json
```
- `corpus_doc_ids.json` (active 100) · `excluded_doc_ids.json` (RFP-000006/17, 검색대상 98)
- `data_manifest.json` — `corpus_v2/manifest.json` + `registry_metadata.json` + `chunks_v1/stats.json`
