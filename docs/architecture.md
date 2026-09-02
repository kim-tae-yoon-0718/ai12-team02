# 아키텍처

**체크리스트3 — 평가 인프라.** 평가셋(임현진)을 기준으로 시스템 출력(이태민)을 채점한다.
담당 김하루, 브랜치 `HR`.

## 데이터 흐름

```
평가셋 문항 (EvaluationItem)  ─┐
시스템 응답 (ModelResponse)   ─┼─▶  runner.run_one  ─▶  EvaluationResult (문항별)
설정 (config/grader.yaml)  ─┘                           │
                                                          ▼
                                      diagnostics.full_report  ─▶  report.json (집계)
```

## CI 5층 (`runner.execute`, `grader run --mode ...`)

값싼 검사를 앞에, 비싼 평가를 뒤에. 앞에서 걸리면 뒤는 안 돈다.

| 층 | 검사 | 모드 | 실패 시 |
| --- | --- | --- | --- |
| 1 | 데이터 정상성 (`validation.check_data_sanity`) | `--data-manifest` 있을 때만 | exit 1 |
| 2 | 평가셋 계약 검사 (`checks.check_evalset.run_all`, 2-17 · 임현진 단일 출처) | 항상 | exit 1 |
| 3 | 추출 테이블 감사 (`extraction.grade_extraction_audit`, 3-2-1) | `--extraction-audit` 있을 때 | `enforce_gates` 시 exit 1 |
| 4 | 소규모 성능 평가 (층화 부분집합) | `ci` / `development` | 게이트 위반 시 exit 1 |
| 5 | 전체 평가 | `final` (`--allow-final` + 실제 심판 + `--extraction-audit` 필수) | 게이트 위반 시 exit 1 |

- `mode=ci`: `--practice-set` **필수**. 없으면 하드 실패(exit 1) — 최종셋 폴백 없음(임현진 2-17).
- `mode=final`: 누수 방지. `--allow-final`, 실제 심판(StubJudge 불가), 독립 원문 표본 대조(`--extraction-audit`) 모두 필요.

### 강제 주입 (3-1 / 3-2-2 — `force_inject.py`)

특정 단계를 "완벽했다고 가정"하고 채점 → 그 단계만 고치면 성능이 얼마나 오르는지의 **상한**.

- `--force-context --chunks <jsonl>` : 정답 근거 청크를 context·citation 으로 주입 (3-1)
- `--force-doc` : 정답 문서 ID 를 `selected_document_ids` 로 주입 (3-2-2)

★ `report.manifest.forced` 에 표시된다. **일반 실행과 비교 금지** — 상한으로만 읽는다.
★ 좌표 매칭은 `section` 단위 (박예진 청크에 `block_index` 없어 `ref_no` 매칭 불가).

## 패키지 지도 (`src/grader/`)

| 모듈 | 역할 | 문서 |
| --- | --- | --- |
| `models.py` | 입력/출력 계약 (pydantic) | [schema.md](schema.md) |
| `config.py` | `config/*.yaml` 로더 (+ 오버레이 병합) | — |
| `normalize.py` | 3-7 금액·날짜·유니코드 정규화, `match_short`/`match_location` | — |
| `task_scoring.py` | 3-3~3-4-5 내용 채점 + 형식 계약 + 기권 분리 | [scoring.md](scoring.md) |
| `retrieval.py` | 3-2 검색 평가 + 3-4-3 출처 좌표 채점 | [scoring.md](scoring.md) |
| `extraction.py` | 3-2-1 추출 테이블 감사 / 3-2-2 문서 특정 | [scoring.md](scoring.md) |
| `judge.py` `prompts.py` `providers.py` | 3-8/3-9 LLM 심판 하네스 | — |
| `force_inject.py` | 3-1/3-2-2 강제 주입 (검색·문서특정 완벽 가정 상한) | 위 §강제 주입 |
| `validation/` | 평가셋 파서 + provenance + 1층 데이터 정상성 | [validation.md](validation.md) |
| `checks/check_evalset.py` | 평가셋 계약 검사 (2-17, 임현진 단일 출처, 벤더링) | [validation.md](validation.md) |
| `diagnostics/` | 집계 리포트 + 진단표 + 회귀 + 오염 방지 | — |
| `versioning.py` | `VERSION.txt` (평가셋·코퍼스 버전) 읽기 | [provenance.md](provenance.md) |
| `runner.py` | 5층 실행기 (진입점) | — |
| `cli.py` | `validate` / `run` / `diagnose` / `regression` | — |

## 실제 데이터 연결 (`$RAG_ROOT=/srv/rfp`)

박예진 공식 산출물(`corpus_v2` / `document_registry_v2` / `chunks_v1`)이 있으면:

```bash
# 박예진 산출물 → grader 입력 재생성 (data/gold/ 는 NDA, 커밋 안 함 — 매번 만든다)
#   100 active / 2 excluded(RFP-000006·17) / 98 eligible
RAG_ROOT=/srv/rfp make data

# 1층 데이터 정상성 스모크
RAG_ROOT=/srv/rfp make check

# 실제 평가셋이 오면 2층 참조 무결성까지:
RAG_ROOT=/srv/rfp python -m grader.cli run --mode checks \
    --evaluation-set data/evalsets/final/final.jsonl \
    --data-manifest data/gold/data_manifest.json \
    --corpus-doc-ids data/gold/corpus_doc_ids.json \
    --excluded-doc-ids data/gold/excluded_doc_ids.json
```

- 6-자산 provenance 는 `config/base.yaml §①` > `$RAG_ROOT/.../VERSION.txt` 에서 **자동** 채워진다
  (현재 실측: `corpus=v2 preprocess=v2 table=v2`). `--corpus` 등은 실험 override 용.
- 이태민 검색 출력이 박예진 청크 스키마(`section_path` + `location_label`)로 오면
  `Location.from_chunk` 가 `{document, section, ref_no}` 로 변환 — 분할 표 `표 7 (2/3)` 는 `표 7` 과 매칭.

## 실행

```bash
pip install -e ".[dev]"
python -m grader.cli --help
pytest -q          # 180 tests
```
