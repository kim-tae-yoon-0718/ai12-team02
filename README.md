# RAG Grader Pipeline

**입찰공고 RAG 시스템을 평가셋(Ground Truth) 기준으로 자동 채점하는 파이프라인.**

> 체크리스트 3 — 평가 인프라 · 지표 · 담당 **김하루** · 브랜치 `HR`

이 파이프라인은 팀이 만든 세 가지 산출물을 입력으로 받는다.

| 입력 | 만드는 사람 | 무엇 |
| --- | --- | --- |
| **평가셋** (`*.jsonl`) | 임현진 (스키마 v0.2) | 문항 + 정답 + 정답 근거 위치 |
| **모델 응답** (`responses.jsonl`) | 이태민 | 시스템 답변 + 검색 결과 + 출처 |
| **데이터 산출물** (`/srv/rfp`) | 박예진 | 코퍼스·청크·추출 테이블 + 버전 |

그리고 하나의 숫자가 아니라 **"어디가 잘못됐는지 진단 가능한" 층별·부품별 점수**를 낸다.

핵심 원칙 — *채점기는 모든 실험 판단의 저울이다. 저울이 엉터리면 실험 전체가 방향을 잃는다.*
그래서 이 저장소에는 **채점기 자신을 검증하는 코드**(`tests/`, `scripts/demo_scoring.py`)가 반드시 포함된다.

---

## 1. 빠른 시작

```bash
make install                 # pip install -e ".[dev]"
make test                    # pytest -q   (현재 180 tests)
make lint                    # ruff check src tests scripts

# 평가셋만 검사 (모델 호출 없음)
python -m grader.cli validate --evaluation-set tests/fixtures/evaluation_set.jsonl

# 값싼 검사 층만 (1~3층)
python -m grader.cli run --evaluation-set tests/fixtures/evaluation_set.jsonl --mode checks

# 개발 채점 (모델 응답 필요)
python -m grader.cli run \
  --evaluation-set tests/fixtures/evaluation_set.jsonl \
  --responses tests/fixtures/model_responses.jsonl --mode development

# 채점 결과를 사람이 읽는 형태로
python scripts/show_results.py \
  --evaluation-set tests/fixtures/evaluation_set.jsonl \
  --per-item artifacts/scores/per_item.jsonl --only FAIL

# 채점기가 정답/오답/환각을 제대로 가르는지 데모 (실 practice 8문항, /srv/rfp 필요)
python scripts/demo_scoring.py
```

`GRADER_PROVIDER=mock`(기본) / `openai_compatible`(+ `GRADER_API_KEY` / `GRADER_MODEL`).
`config/grader.yaml` 의 `use_stub_judge: true` 동안은 provider 무관하게 **StubJudge**(배관 점검용, `tier=final` 불가).

---

## 2. 핵심 개념

### 2-1. 두 개의 입력 계약

**평가셋 문항** (`grader.models.EvaluationItem`, 스키마 v0.2) — [`docs/schema.md`](docs/schema.md)

| 필드 | 무엇 | 채점기가 어떻게 쓰나 |
| --- | --- | --- |
| `task_type` | `selection` / `extraction` / `qa` | **점수를 종류별로 나눠 볼 때만** 씀 (채점 자체엔 안 씀) |
| `answer_type` | `document_set` / `value` / `list` / `summary` / `comparison` / `unanswerable` | **채점기 선택** — 이걸로 `grade_content` 가 분기 |
| `answer_raw` / `answer_normalized` | 정답 (표기 그대로 / 정규화형) | 채점 기준값 |
| `location` | 정답 근거 좌표 `{document, section, ref_no}` — 비교형은 **배열** `[{document, field, section, ref_no}, …]` | 검색·출처 좌표 채점 (3-2, 3-4-3) |
| `field_tag` | `critical` / `major` / `minor` (오류 심각도) | 등급 가중·분리 집계·게이트 (3-3-2) |
| `answer_source` | `table` / `verified` / `metadata` | `metadata`(CSV 답변)면 좌표 채점 제외 |
| `unspecified_type` · `scenario_type` · `active_document_id` · `reference_time` | 문항 조건 | 파생 플래그·검증 |

**모델 응답** (`grader.models.ModelResponse`) — 전체 필드표는 [`docs/response_contract.md`](docs/response_contract.md)

| 필드 | 무엇 |
| --- | --- |
| `answer` / `structured_answer` | 최종 답변 (글자 / 목록·비교표 구조) |
| `contexts` / `retrieved` / `reranked` | 실제 LLM 입력 청크 / 검색 후보 풀 / 재정렬 후 |
| `citations` | 답변이 근거로 든 출처 좌표 (배열) |
| `selected_document_ids` / `active_document_id` | 선별형 결과 / 대화형 상태 |
| `abstained` | 답 거부 여부 |
| `route` | 내부 실행 경로 — **점수에 안 씀, 결과에 기록만.** `task_type ≠ route` 여도 감점 없음 |
| `failure` / `latency_ms` / `cost_usd` | 오류 사유 / 응답시간 / 비용 |

### 2-2. 5층 실행 구조 (`grader.runner`) — [`docs/architecture.md`](docs/architecture.md)

값싼 검사를 앞에, 비싼 평가를 뒤에. 앞에서 걸리면 뒤는 안 돈다.

| 층 | 검사 | 담당 | 모드 |
| --- | --- | --- | --- |
| **1** 데이터 정상성 | `data_manifest.json` 문서 수·로드율·버전 꼬리표 | 박예진 (1-19-1) | `--data-manifest` 있을 때 |
| **2** 평가셋 계약 검사 | 스키마·조합·중복·할당량·참조무결성·좌표·유출 | 임현진 (2-17) → `src/checks/check_evalset.py` | 항상 |
| **3** 추출 테이블 정확도 | 원문 표본 대조 (순환 방지) | 박예진 표본 (3-2-1) | `--extraction-audit` 있을 때 |
| **4** 소규모 성능 평가 | 층화 부분집합 채점 | 김하루 | `ci` / `development` |
| **5** 전체 평가 | 최종 비교용 | 김하루 | `final` (`--allow-final` + 실제 심판 + `--extraction-audit` 필수) |

- `mode=ci` — `--practice-set` **필수**. 없으면 하드 실패 (최종셋 폴백 없음, 임현진 2-17)
- `mode=final` — **누수 방지.** 최종셋 점수를 매 커밋마다 보면 결국 그걸 보며 고치게 된다 → `--allow-final` + 실행자 기록 + StubJudge 금지
- **강제 주입** (`--force-context` / `--force-doc`) — 검색·문서특정이 완벽했다고 가정한 **성능 상한**. `report.manifest.forced` 에 표시, 일반 실행과 비교 금지

### 2-3. 채점이 실제로 어떻게 동작하는가

한 문항의 채점 흐름 (`grader.task_scoring.score_item`):

```
① 기권 판정 (grade_abstention)
   answer_type=unanswerable 이면 → "제대로 기권했나"가 곧 content 점수
   그 외 → ②로

② 내용 채점 (grade_content — answer_type 으로 분기)
   value        → grade_short_answer   정확 일치 (금액·날짜·유니코드 정규화, 정답+덧붙임 거부)
   list         → grade_list           exact-all — 빠뜨림도 덧붙임(환각 항목)도 있으면 실패
   summary      → grade_summary_checkpoint  체크포인트 배열 중 몇 개 언급했나
   comparison   → grade_comparison     (문서×필드) 칸 단위 값 일치율, 틀린 칸 지목
   document_set → grade_selection      recall 가중 + "완전 재현 비율"

③ 형식 계약 (check_format)
   목록형인데 structured_answer 가 배열 아님 / 비교형인데 표 아님 → 형식 위반
   내용이 맞아도 output contract 어기면 최종 FAIL (format_status 는 내용과 독립 축)

④ 상태 조합 (combine_status)
   list/value/기권 = 이진 → 즉시 PASS/FAIL
   document_set/summary/comparison = 부분점수 → baseline 실측 전까지 "PENDING_THRESHOLD"
```

문항과 별도로 매 실행마다:

- **검색 채점** (`grader.retrieval.grade_retrieval`) — retrieval_k / reranker_k / context_k 세 단계 각각. 재현율 실패(후보 풀에도 없음) vs 순위 실패(있는데 상위권 못 듦) 구분
- **출처 좌표 채점** (`grader.retrieval.grade_citation`, 3-4-3) — 정답 위치 vs 인용 좌표. "근거 누락"(정답 위치인데 인용 안 함)과 "잘못된 근거"(인용했는데 정답 위치 아님)를 분리
- **LLM 심판** (`grader.judge`) — `judge_faithfulness` 만. temperature=0 / 생성모델과 다른 계열 / 사람 대조 검증 통과 → 3종을 코드로 강제

집계 단계 (`grader.diagnostics`):

- **3-1-0 진단표** — 여러 지표를 나란히 놓고 조합으로 범인 특정 (`검색 재현율 ↓ + 충실성 ↑` = 검색 탓, `추출 정확도 ↓ + 선별형 점수 ↑` = 순환 경보, …)
- **3-3-2 등급 반영** — ②분리 집계(critical/major/minor 각각) + ①가중 평균(5:3:2) + ③게이트를 항상 나란히
- **3-13 / 3-17** — 변동 폭 측정 → 회귀 판정선 → 점수 변화의 원인(시스템 / 코퍼스 / 평가셋) 분리

---

## 3. 전체 파일 구조

```text
config/
  grader.yaml         채점 설정 한 곳 (judge / retrieval / grading / gate / provenance)
  ci.yaml local.yaml  grader.yaml 위에 바뀐 줄만 덮는 오버레이
  base.yaml           팀 공통 모델 설정 (origin/dev 와 동일본 — 청킹·임베딩·검색·생성)

prompts/
  judge_faithfulness.v1.md   충실성(근거 기반 여부) 심판 프롬프트 원문
  judge_checkpoint.v1.md     요약형 체크포인트 언급 여부 매처
  judge_list_item.v1.md      목록형 항목 언급 여부 매처

src/checks/
  check_evalset.py    평가셋 계약 검사 (2-17) — 임현진 단일 출처를 벤더링 + grader 부가분

src/grader/                  ── 채점기 본체 (총 ~4,600 LOC) ──
  models.py           입력/출력 계약 (pydantic)          ← 팀원은 여기부터
  config.py           config/*.yaml 로더 + 오버레이 병합
  normalize.py        3-7 금액·날짜·유니코드 정규화, 단답/좌표 매칭
  cache.py            심판 응답 파일 캐시 (키에 코퍼스·추출표 버전 포함)
  providers.py        심판 백엔드 (MockJudge / OpenAICompatible)
  prompts.py          프롬프트 파일 로딩 + {{변수}} 치환 + 버전

  task_scoring.py     answer_type 별 채점기 본체 + 형식/기권/상태 조합
  retrieval.py        3-2 검색 평가 + 3-4-3 출처 좌표 채점
  extraction.py       3-2-1 추출 테이블 정확도 + 3-2-2 문서 특정 + 순환 경보
  judge.py            3-8 LLM 심판 설계 / 3-9 검증 / 3-11 심판 등급
  force_inject.py     3-1 / 3-2-2 강제 주입 (검색·문서특정 완벽 가정 상한)

  validation/
    schema.py         평가셋·응답 JSONL → pydantic 파싱 (채점 입력 계약)
    provenance.py     6-자산 provenance 기록 완전성·재현성 (채점 결과 검증)
    assets.py         CI 1층 데이터 정상성 (data_manifest)
    __init__.py       check_evalset_integrity() — 2층 어댑터

  diagnostics/
    statistics.py     집계 프리미티브 (칸 크기 경고 / severity / 기권 / 형식 / 통합점수)
    report.py         3-12 최종 리포트 + 3-1-0 진단표 렌더링
    retrieval.py citation.py extraction.py   각 축 집계 (per-item 채점은 상위 모듈)
    regression.py     3-13 변동 폭 / 3-13-1 회귀 판정 / 3-17 원인 분리
    contamination.py  4-9 실험 오염 방지 (채점기 유리하게 만든 변경 기록 강제)

  versioning.py       평가 자산 버전의 출처 (VERSION.txt 파싱, RAG_ROOT 조립)
  runner.py           5층 실행기 (execute) + 문항 채점 (run_one) + 집계 (grade_all)
  cli.py              validate / run / diagnose / regression 서브커맨드

scripts/
  build_doc_ids.py         /srv/rfp 등록부 → corpus_doc_ids.json / excluded_doc_ids.json
  build_data_manifest.py   /srv/rfp corpus·chunks·registry → data_manifest.json (v3 자동 감지)
  show_results.py          per_item.jsonl → 질문/정답/시스템답/판정/이유 (3-14)
  demo_scoring.py          실 practice 8문항 × (정답/오답/환각) 채점 데모
  run_*.py                 cli 서브커맨드 얇은 래퍼

docs/       schema · scoring · validation · provenance · architecture · response_contract
data/       로컬 데이터 (NDA — data/gold/*.json 만 커밋, 나머지 .gitkeep)
artifacts/  실행 산출물 자리 (구조만 커밋)
tests/      unit/ · integration/ · regression/ · checks/ · fixtures/   (180 tests)
```

---

## 4. 모듈 상세 — 각 파일이 무엇을 하고 왜 존재하는가

### 4-1. 계약 · 기반

| 파일 | 핵심 정의 / 함수 | 역할 |
| --- | --- | --- |
| **`models.py`** (476) | `EvaluationItem` · `ModelResponse` · `Location` · `ContextChunk` · `RetrievedItem` · `TaskScore` · `FormatStatus` · `AbstentionResult` · `Provenance` · `EvaluationResult` | 모든 입출력의 pydantic 계약. `Location.from_chunk()` 가 박예진 청크 스키마(`section_path`, `block_type`, `block_index`, `md_line_*`)를 `{document, section, ref_no}` 좌표로 변환. `EvaluationItem.location` 은 단일 객체 또는 배열(비교형), `gold_locations()` 가 항상 리스트로 정규화 |
| **`normalize.py`** (256) | `normalize_text` · `parse_amount` / `extract_amounts` · `parse_date` · `match_short` · `match_location` | 3-7 함정 처리. `match_short` — 정규화 후 정확 일치를 **먼저** 보고(정답 자체가 서술형이어도 통과), 그다음 금액/날짜를 문장에서 뽑되 **뒤에 군더더기가 길면 거부**(정답+환각 덧붙임 방지). `match_location` — `ref_no` 정밀도에서 양쪽에 `line` 있으면 청크 line 범위 포함으로 대조(`block_index` 가 source_type 별이라 모호) |
| **`config.py`** (232) | `GraderConfig` 와 7개 하위 dataclass · `load_config` · `read_base_yaml_assets` | `config/grader.yaml` 로드 + `--overlay` 병합(바뀐 줄만). `config/base.yaml §① 6칸` 을 grader 설정보다 우선(6-자산 단일 출처) |
| **`cache.py`** (56) | `build_cache_key` · `FileCache` | 심판 응답 재사용. **캐시 키에 코퍼스·추출 테이블 버전 포함** — 데이터가 바뀌었는데 옛 응답이 나오면 갱신 효과가 0으로 보이는 사고 방지 (3-11) |
| **`providers.py`** (86) | `MockJudgeProvider` · `OpenAICompatibleProvider` · `build_provider` | 심판 백엔드 추상화. mock 은 결정론적, openai_compatible 은 `GRADER_API_KEY` |
| **`prompts.py`** (45) | `PromptRepository.render` / `version_of` | `prompts/<name>.<version>.md` 로딩 + `{{변수}}` 치환. 프롬프트 버전이 provenance 에 기록됨 |
| **`versioning.py`** (86) | `read_versions` · `read_schema_version` | 자산별 `VERSION.txt`(포맷이 조금씩 다름)를 파싱해 6-자산 이름으로 정규화. 절대 경로 하드코딩 금지 — `RAG_ROOT` + 코드 조립 |

### 4-2. 채점 로직

| 파일 | 핵심 함수 | 역할 |
| --- | --- | --- |
| **`task_scoring.py`** (343) | `grade_short_answer` · `grade_list` · `grade_summary_checkpoint` · `grade_comparison` · `grade_selection` · `grade_abstention` · `check_format` · `grade_content` · `combine_status` · `score_item` | 채점기 본체. `grade_content` 가 `answer_type` 으로 분기. `grade_list` — exact-all: 빠뜨림도 덧붙임(환각 항목)도 있으면 0점, 둘은 `detail` 에 따로 셈 (3-4-4 "다 넣고 환각" > "하나 빠뜨림" 역전 방지). `grade_comparison` — `[{항목, RFP-ID: 값}]` 실제 형식 파싱, 틀린 칸을 `{document_id, field, gold, pred}` 로 지목. 부분점수 태스크는 `PENDING_THRESHOLD` — baseline 실측 전 임의 경계 금지 |
| **`retrieval.py`** (266) | `grade_retrieval` · `grade_citation` · `aggregate_retrieval` · `aggregate_citation` | 3-2 검색 평가 — 3단계(retrieval_k/reranker_k/context_k) 각각 recall/precision/MRR, `failure_kind` 로 재현율 실패 vs 순위 실패 판정. 3-4-3 좌표 채점 — 정답 위치가 여러 개(비교형)면 매칭 비율, `n_missing_evidence` / `n_wrong_citations` 분리, `answer_source=metadata` 는 제외 |
| **`extraction.py`** (205) | `grade_extraction_audit` · `grade_doc_selection` · `circularity_flag` · `aggregate_doc_selection` | 3-2-1 추출 테이블을 **평가셋과 독립적으로**(원문 표본 대조) 채점 — 선별형 정답을 그 테이블로 만들었으면 테이블이 틀려도 만점 나는 순환을 잡는 유일한 장치. `circularity_flag` = "추출 정확도 ↓ + 선별형 점수 ↑" 경보 |
| **`judge.py`** (184) | `Judge.assert_ready` · `Judge.score` · `load_verification` · `parse_verdict` · `tier_correlation` · `StubJudge` · `make_item_matcher` | 3-8 심판. `assert_ready()` — 사람 대조 기록 없음 / 일치도 하한 미달 / 블라인드 아님 → `tier=final` 거부. `StubJudge` 는 배관 점검용(문자열 포함 판정), final 불가 |
| **`force_inject.py`** (98) | `load_chunk_index` · `apply_forced_context` · `apply_forced_doc` | 정답 근거 청크를 context·citation 으로 주입 / 정답 문서를 selected_document_ids 로 주입. "그 단계만 고치면 얼마나 오르나" 상한 |

### 4-3. 평가셋·데이터 검증

| 파일 | 핵심 함수 | 역할 |
| --- | --- | --- |
| **`src/checks/check_evalset.py`** (430) | `check_schema` (C1) · `check_dup_ids` (C2) · `check_quota` (C3) · `check_ref_intg` (C4) · `check_version` (C5) · `check_leak` (C6) · `check_excluded_as_gold` (1-9-1) · `scan_tracked_files` (【25】) · `run_all` | **평가셋 계약 검사 단일 출처.** 핵심 C1~C6 은 임현진 `origin/HJ` 를 벤더링(HJ 머지 시 재조정). grader 부가분 3개 — 멀티라인 JSON 로더, 수집중복 문서를 정답 근거로 쓰면 금지, 문항 텍스트가 프롬프트·코드에 유출됐는지 |
| **`validation/schema.py`** (56) | `load_jsonl` · `validate_evaluation_set` · `validate_model_responses` · `index_by_id` | JSONL → `EvaluationItem` / `ModelResponse` pydantic 파싱. `strict_meta` → 최종셋에서 `_` 주석 키 FAIL. 로더는 `check_evalset` 것을 재사용(멀티라인 JSON 허용) |
| **`validation/provenance.py`** (39) | `check_provenance` · `check_reproducible` | 결과에 6-자산이 정확한 이름으로 모두 있는지 / `git_dirty` false 이고 `UNKNOWN` 축 없는지 (최종 실험 게이트) |
| **`validation/assets.py`** (74) | `check_data_sanity` · `check_data_warnings` | CI 1층 — data_manifest 의 문서 수·로드율·버전 꼬리표·raw↔md 수 일치 등. 표 0개 문서는 게이트가 아니라 경고(스캔 한글 파일 신호) |
| **`validation/__init__.py`** (77) | `check_evalset_integrity` | 2층 어댑터 — `EvaluationItem` in-memory 호출(테스트)용. runner/cli 는 `check_evalset.run_all` 직접 호출 |

### 4-4. 집계·진단

| 파일 | 핵심 함수 | 역할 |
| --- | --- | --- |
| **`diagnostics/statistics.py`** (140) | `cell_report` · `severity_report` · `severity_weighted_score` · `abstention_report` · `format_report` · `integrated_score` | 집계 프리미티브. `cell_report` — 축을 곱할수록 칸당 문항 급감 → 칸 크기 자동 경고. `severity_weighted_score` / `integrated_score` — 가중치(`gate.field_tag_weight` / `gate.task_weight`)가 비면 **감으로 안 채우고 계산 자체를 안 함** |
| **`diagnostics/report.py`** (213) | `main_metrics` · `full_report` · `classify` · `diagnose` · `render_table` | 3-5 태스크별 주 지표(`out["primary"]`) + 2-16 통합 점수. 3-1-0 진단표 — 지표 조합 → 범인 특정 규칙. `cli diagnose` 진입점 |
| **`diagnostics/regression.py`** (266) | 변동 폭 측정 · 회귀 판정(n-sigma) · `attribute_change` | 3-13 → 3-13-1 → 3-17. 변동 폭이 참고값이 아니라 **CI 통과·실패를 가르는 판정선**. `cli regression` 진입점. `n_sigma`·게이트 목록은 baseline 실측 후 확정 |
| **`diagnostics/contamination.py`** (104) | 실험 변경 기록 강제 | 4-9 — 심판을 바꿔서 / 평가셋 표현에 맞춰 프롬프트를 고쳐 점수를 올린 것을 개선으로 인정하지 않음. 기록 없으면 "무엇이 개선됐다는지"조차 알 수 없음 |
| `diagnostics/retrieval.py citation.py extraction.py` | `aggregate_*` 재노출 | per-item 채점은 상위 모듈, 여기는 집계만 (얇은 재노출) |

### 4-5. 실행

| 파일 | 핵심 함수 | 역할 |
| --- | --- | --- |
| **`runner.py`** (563) | `GraderRunner.run_one` · `execute` · `grade_all` · `layer_eval` · `stratified_subset` · `build_judge` | 5층 실행기. `run_one` — 문항 하나에 task_scoring + retrieval 진단 + citation + LLM judge 를 합쳐 `EvaluationResult`. `execute` — 1→5층 순서로, 앞에서 FAIL 이면 즉시 종료. `stratified_subset` — CI/dev 부분집합을 `task_type × field_tag` 로 층화 |
| **`cli.py`** (204) | `build_parser` · `main` | `validate`(전체 계약 검사) / `run`(층 실행) / `diagnose` / `regression`. `--force-context` / `--force-doc` / `--chunks`, `--corpus-doc-ids` / `--excluded-doc-ids`, `--allow-final` / `--runner` 등 |

---

## 5. 설정 — `config/grader.yaml`

| 블록 | 무엇 |
| --- | --- |
| `judge` | 심판 프롬프트 파일 목록, `whole_item_metrics`(문항 전체 채점은 `judge_faithfulness` 만), `model` / `family` / `tier`, `verification_path`(3-9 사람 대조 기록 — 없으면 `tier=final` 거부), `min_agreement` |
| `retrieval` | `retrieval_k` / `reranker_k` / `context_k`(세 단계 분리), `precision`(현재 `section` — evalset `line` 도착 시 `ref_no` 로), `eval_k`(다단계 k 분석) |
| `grading` | `docset_partial_credit` / `miss_weight`(선별형 recall 가중), `require_table_format`, `grade_citations` |
| `gate` | `severity_gate.critical`(치명 필드 정답률 하한 — 임시값), `column_severity`(추출 12필드 등급), `task_quota` / `task_weight`(비어있음 — 1-2 숫자 대기), `field_tag_weight`(critical:major:minor = 5:3:2), `data_thresholds` / `data_warn_thresholds`(1층 게이트/경고) |
| `provenance` | 6-자산 기본값 (`corpus` / `preprocess` / `table` / `index` / `evalset` / `scorer`) — CLI 인자 > `base.yaml §①` > VERSION.txt > 여기 순 |

`config/ci.yaml` / `config/local.yaml` 은 `--overlay` 로 바뀐 줄만 덮는다.

---

## 6. 산출물

| 파일 | 내용 |
| --- | --- |
| `artifacts/scores/report.json` | `manifest`(실행 조건·6-자산 provenance·git 상태·`forced` 표시) + `layers`(층별 PASS/FAIL/SKIP + problems) + `summary`(3-1-0 진단표: 태스크별 주 지표·severity·citation·통합 점수·순환 경보) |
| `artifacts/scores/per_item.jsonl` | 문항별 — `task_score`(점수 + `detail`: why / missing / extra / wrong_cells), `format_status`, `retrieval`(3단계), `citation`, `abstention`, `final_status`, `provenance` |

`scripts/show_results.py` 로 `per_item.jsonl` 을 질문/정답/시스템답/판정/이유 형태로 읽는다 (3-14 수동 확인).

---

## 7. 팀 인터페이스 & 현재 상태

### 확정되어 코드에 반영됨

- 스키마 v0.2 (임현진 `check_evalset.py` FIELD_SPEC 와 정합)
- `location = {document, section, ref_no}`, 비교형은 `[{document, field, section, ref_no}, …]` 배열
- `ref_no` = `location_label` 그대로 (`"4. 제안 요청내용 · 문단 1-57"`) + `line`(단일값/청크는 md_line_start). 평가셋·청크 동일 규약 (박예진·임현진 09-02 최종)
- `answer_source=metadata` → `section="CSV"`, `ref_no="CSV: {컬럼명}"`, 좌표 채점 제외
- 평가셋 계약 검사 = `src/checks/check_evalset.py` 단일 출처
- 6-자산 provenance = `base.yaml §①` 단일 출처, VERSION.txt 자동 조회
- `route` 는 점수에 안 씀 (`task_type ≠ route` 무감점)

### 대기 중 (코드는 준비됨, 팀 산출물 도착 시 활성)

| 대기 | 담당 | 도착하면 |
| --- | --- | --- |
| 3-4 threshold (선별·요약·비교 PASS/FAIL 경계) | baseline (이태민 + 박예진 4-1) | `PENDING_THRESHOLD` 해소 |
| 3-9 사람 채점 표본 | 전원 교차 | `verification_path` 채우면 `tier=final` 가능 |
| 3-13 변동 폭 실측 | baseline | 회귀 판정선(n-sigma) 확정 |
| 3층 추출 원문대조 JSONL | 김태윤 | `--extraction-audit` 연결 |
| `task_weight` / `task_quota` 숫자 | 임현진 / 1-2 | 통합 점수·할당량 검사 활성 |

---

## 8. 문서

| 문서 | 내용 |
| --- | --- |
| [`docs/schema.md`](docs/schema.md) | 평가셋 문항 입력 계약 (v0.2 15필드) |
| [`docs/scoring.md`](docs/scoring.md) | 채점 정책 — 내용/형식/기권/가중치, 좌표 채점 |
| [`docs/validation.md`](docs/validation.md) | 평가셋 계약 검사 (C1~C6 + 부가분), 좌표 어댑터 |
| [`docs/provenance.md`](docs/provenance.md) | 6-자산 provenance / 재현성 |
| [`docs/architecture.md`](docs/architecture.md) | 5층 실행 흐름 + 패키지 지도 + 강제 주입 |
| [`docs/response_contract.md`](docs/response_contract.md) | 모델 응답 필드 목록 (이태민 → 채점기) |
