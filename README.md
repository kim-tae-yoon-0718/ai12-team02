# RAG Grader Pipeline

입찰공고 RAG 시스템의 평가셋(Ground Truth)을 기준으로 실제로 채점을 수행하는 파이프라인.
확정 평가셋 스키마(v0.1)를 입력 계약으로 삼고, 팀이 확정한 채점 정책(태스크 분류/목록형
완전일치/형식 계약/버전 provenance 등)을 코드로 강제한다.

## 현재 구현 범위

- 확정 평가셋 스키마(**v0.2**, 15개 필드) 검증 + 2-17 무결성 검사(중복 id·참조 무결성·
  정합성·task_type↔answer_type 허용조합·할당량)
- `task_type = selection/extraction/qa` 3분류. `answer_type = document_set/value/list/
  summary/comparison/unanswerable`
- 목록형(`list`) 최종 판정은 **exact-all 고정**(코드에 고정, 스키마 필드 아님).
  Coverage/Missing/Extra 는 진단용
- **형식 계약(3-3)**: `format_status`를 내용과 독립된 축으로 저장 — 내용이 맞아도 output
  contract 위반이면 최종 PASS 아님(`PASS` / `FAIL-content` / `FAIL-format` /
  `FAIL-format+content`). 부분점수 태스크(document_set/summary/comparison)는 baseline 실측
  전까지 임의 경계를 만들지 않고 `PENDING_THRESHOLD`로 남김
- 기권 분리 추적(v0.2: `answer_type=unanswerable`, 사유는 `answer_raw`): 정상 기권 /
  환각(기권해야 하는데 답함) / 과잉 거절(답이 있는데 기권) 구분. `field_tag=critical`
  문항의 기권(`critical_abstain`)은 과잉 거절과 **별도 축**으로 집계하며, severity
  가중치(critical:major:minor = 5:3:2)와도 서로 대체하지 않는다
- **검색 진단(3-2)**: `retrieval_k` / `reranker_k` / `context_k` 세 단계를 각각 Recall@k·
  Precision@k·MRR로 재고, `recall_failure`(후보에도 없음) vs `rank_failure`(후보엔 있으나
  context까지 못 옴)를 구분. ★최종 k는 미확정 — 세 값을 하나로 합치지 않음.
  [2026-08-27] 이태민·김하루가 `top_k=5(잠정)` baseline에 잠정 합의, 채점은 k=3/5 등
  여러 단계로 별도 분석하기로 함 — 다만 이 값이 세 단계 중 정확히 무엇인지는 아직
  불명확해 `configs/grader.yaml`의 실제 기본값은 그대로 두었다(자세한 내용은
  `configs/grader.yaml`의 `retrieval` 블록 주석 참고)
- **추출 테이블 평가(3-2-1)**: 값 있음/항목 없음/추출 실패 3상태 혼동표, absent↔failed 혼동률,
  D7 순환 경보(`answer_source=table`인데 추출 정확도가 낮으면 경고)
- **출처 좌표 채점(3-4-3, 2026-08-27 5차 신규)**: `retrieval.grade_citation()` —
  정답 `location`과 응답 `citations`가 실제로 같은 좌표(2-9 확정 단위)를 가리키는지
  채점한다. 이전에는 citation 유무만 보고("있다") 그마저 형식 FAIL로 안 썼는데, "맞다"를
  재는 로직 자체가 없었다. citation 미표기는 여전히 `check_format`을 FAIL로 만들지
  않지만(내용/형식 분리 원칙 유지), `citation_accuracy`(좌표 일치율)와
  `no_citation_rate`(미표기율)를 분리해서 진단에 낸다 — "안 붙임"과 "틀리게 붙임"은
  다른 문제라 하나로 합치지 않는다
- **LLM Judge(3-8/3-9)**: `judge_faithfulness.v1.md`(전체 1회) / `judge_checkpoint.v1.md`·
  `judge_list_item.v1.md`(항목 단위 매처) — 팀 확정 프롬프트 원문을 그대로 연결. temperature=0
  강제, 생성 모델과 다른 계열 요구, 사람 대조 기록 없으면 `tier=final` 실행 거부
  (`Judge.assert_ready`)
- **CI 5층 실행기(3-6-1)**: 1층 데이터 정상성 → 2층 평가셋 무결성 → 3층 추출 표본 대조 →
  4층 CI/dev 층화 부분집합 → 5층 전체(`--allow-final` 명시 시에만).
  [임현진 2-17 확정] CI(`mode=ci`)에 상시 노출되는 것은 practice 세트
  (검수 탈락분, `practice_items.jsonl`)뿐이어야 하고 최종 50문항은 상시 노출하지 않는다 —
  `--practice-set`으로 그 파일을 주면 거기서만 채점 대상을 가져오고, **안 주면 `mode=ci`는
  하드 실패한다(exit 1)**. 최종셋으로의 폴백은 없다 — 유출은 경고로 뭉갤 문제가 아니다
- **통합 점수(2-16 확정)**: 태스크별 3개 점수 + `gate.task_weight`(1-2 업무빈도·위험
  기준) 가중 평균을 항상 나란히 낸다. 가중치 숫자가 아직 없으면 `integrated_score`는
  `null`로 남기고 이유를 `integrated_note`에 적는다 — 감으로 채우지 않는다
- **데이터 경고(1층, 게이트 아님)**: `gate.data_warn_thresholds`로 "실행을 막지 않고
  사람이 보고 넘기는" 신호를 별도로 관리한다. 첫 항목은 표 0개 문서 —
  스캔된 한글 파일이 섞여 들어오면 OCR이 안 돌아 글자가 거의 없는 md가 1-11 게이트
  4종을 전부 통과하는데, 표 0개(실측 100건 전부 표 최소 3개)가 이를 잡는다. 게이트로
  두면 문서 하나에 실행 전체가 멈추므로 의도적으로 경고로만 둠(한계: 부분 스캔은
  못 잡음 — 1-19-1)
- **회귀 판정(3-13/3-13-1/3-17)**: 변동 폭 측정 → n_sigma 게이트/경고 분리 → 원인 분리
  (코퍼스/평가셋/채점기/시스템 중 무엇이 바뀌었는지)
- **오염 방지(4-9)**: Judge/프롬프트 변경만으로 지표가 오른 경우를 "시스템 개선"으로 자동
  인정하지 않음 — 실험은 `target_metrics`/`protected_metrics`/`expected_direction`을
  선언해야 함
- **팀 확정 6-자산 Version provenance**: 평가 결과 하나가 어떤 자산 조합에서 나왔는지
  **정확히 6개 필드**로 못박는다. 필드명·값 규약(`v1`/`v2` …)은 팀 실험 인프라
  `base.yaml §① "재료 버전 6칸 (필드명 고정 — 절대 개명 금지)"`과 **정확히 일치** —
  `corpus`(①원문 코퍼스) / `preprocess`(②전처리) / `table`(③추출 테이블) /
  `index`(④검색 인덱스, 4번) / `evalset`(⑤평가셋) / `scorer`(⑥채점기, 김하루가 올림).
  축 목록·순서·필드명은 `models.PROVENANCE_ASSETS` 한 곳에서만 정의. 값이 없으면
  필드를 빼지 않고 `"UNKNOWN"`(★누락 ≠ 미상). `EvaluationResult.provenance`는
  기본값이 빈 `Provenance`라 절대 비지 않으며, `report.json`의 `manifest.provenance`와
  `per_item.jsonl`의 각 문항 `provenance`가 같은 출처(`runner.provenance()`)를 쓴다.
  - `--corpus/--preprocess/--table/--index/--evalset/--scorer` 로 주입. `evalset`·`corpus`는
    미지정 시 `$RAG_ROOT/evalset/v1/VERSION.txt`(팀 확정 경로, `key: value` 포맷)에서 자동 조회.
  - 심판 프롬프트 세부 버전은 6칸 밖 — `manifest.judge_prompt_versions`에 별도 기록
    (scorer 축이 움직였을 때 코드 변경인지 프롬프트 변경인지 가르는 재료).
  - `manifest`에 `git_commit`/`git_dirty` 자동 기록(규약 §2-4). `git_dirty=true`면 재현 불가.
  - `regression.attribute_change`(3-17)가 이 6축을 그대로 diff 한다.
- 캐시 키에 코퍼스/추출 테이블 버전 포함 — 코퍼스가 갱신됐는데 옛 Judge 응답이 재사용되는
  것을 방지

## 파일 지도

```text
rag_grader_pipeline_v0/
├── pyproject.toml
├── configs/grader.yaml          실행 설정(judge/retrieval/grading/gate/provenance)
├── data/
│   ├── evaluation/               평가셋·모델 응답 JSONL (현재는 배관 점검용 샘플)
│   └── prompts/                  judge_*.v1.md — 팀 확정 프롬프트 원문(수정 없이 연결)
├── src/grader/
│   ├── models.py                 pydantic 입력/출력 계약 (임현진 확정 스키마 기준)
│   ├── normalize.py               3-7 금액·날짜·유니코드 정규화
│   ├── retrieval.py               3-2 검색 평가 (retrieval_k/reranker_k/context_k)
│   ├── extraction.py              3-2-1 추출 테이블 평가 / 3-2-2 문서 특정 / D7 순환 경보
│   ├── task_scoring.py            3-3~3-4-5 채점기 본체 + 형식 계약 + 기권 분리
│   ├── judge.py                   3-8/3-9 LLM judge 하네스 (assert_ready 3종 가드)
│   ├── diagnostics.py             3-1-0 팀 공용 진단표 + 3-12 집계 + 3-5 주 지표
│   ├── regression.py              3-13 변동 폭 / 3-13-1 회귀 판정 / 3-17 원인 분리
│   ├── contamination.py           4-9 실험 오염 방지
│   ├── runner.py                  3-6-1 CI 5층 실행기
│   ├── prompts.py                 {{변수}} 치환 프롬프트 로더
│   ├── providers.py               Judge provider(mock/openai_compatible) 교체 구조
│   ├── validation.py              스키마 로딩 + 2-17 무결성 + 1층 데이터 정상성
│   ├── config.py                  configs/grader.yaml 로더
│   └── cli.py                     validate / run / diagnose / regression
└── tests/                         단위 테스트 + 층 실행 스모크 테스트
```

## 설치

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## 환경변수

```bash
cp .env.example .env
```

`GRADER_PROVIDER=mock`(기본값)은 개발용 목업이다. 실제 모델을 붙이려면
`GRADER_PROVIDER=openai_compatible`과 `GRADER_API_KEY`/`GRADER_MODEL`을 설정한다.
`configs/grader.yaml`의 `use_stub_judge: true`인 동안은 provider 설정과 무관하게
StubJudge(문자열 포함 판정)가 쓰인다 — 이것은 심판이 아니라 배관 점검용이며 `tier=final`
에는 쓸 수 없다.

## 평가셋 형식

한 문항당 JSON object 하나를 JSONL에 넣는다. 필드명은 임현진 확정 **스키마 v0.2**
(2026-08-28, 15개 필드)를 그대로 쓴다.

- `answer_type` 값: `document_set` / `value` / `list` / `summary` / `comparison` /
  `unanswerable`. task_type 별 허용 조합 —
  selection→`document_set` · extraction→`value|list` · qa→`value|summary|comparison|unanswerable`.
- 선별형 정답 문서 집합은 `answer_raw` 배열에 담는다(`document_id` 아님).
- 요약형 체크포인트 배열도 `answer_raw` 에, 기권 사유 문자열도 `answer_raw` 에 담는다
  (별도 `checkpoints`/`unanswerable_reason` 필드 없음, `null` 금지).
- v0.1의 `document_unspecified`/`time_dependent`/`conversational`은 각각
  `unspecified_type`/`reference_time`/`scenario_type` 존재 여부로 코드에서 자동
  파생된다(`models.EvaluationItem` computed_field). `difficulty`는 폐기,
  `schema_version`은 문항이 아니라 파일 단위(`VERSION.txt`).

```json
{
  "id": "EXT-007",
  "question": "이 사업 사업 금액이 얼마야",
  "task_type": "extraction",
  "answer_type": "value",
  "document_id": "DOC-091",
  "field_tag": "critical",
  "answer_source": "verified",
  "answer_raw": "222,180,200원",
  "answer_normalized": 222180200,
  "location": { "document": "DOC-091", "section": "Ⅲ. 사업 개요", "ref_no": "표 5" }
}
```

## 실행

```bash
# 스키마만 검사
python -m grader.cli validate --evaluation-set data/evaluation/evaluation_set.jsonl

# 값싼 검사만 (1~3층, 모델 호출 없음)
python -m grader.cli run --evaluation-set data/evaluation/evaluation_set.jsonl --mode checks

# CI용 층화 부분집합 (4층) — ★practice 세트 필수. 없으면 exit 1 (최종 50문항 폴백 없음, 2-17)
python -m grader.cli run \
  --evaluation-set data/evaluation/evaluation_set.jsonl \
  --responses data/evaluation/model_responses.jsonl \
  --practice-set data/evaluation/practice_items.jsonl \
  --mode ci

# 개발용
python -m grader.cli run \
  --evaluation-set data/evaluation/evaluation_set.jsonl \
  --responses data/evaluation/model_responses.jsonl \
  --mode development

# 최종 전체 평가 — ★누수 방지 가드. 명시적으로만
python -m grader.cli run \
  --evaluation-set data/evaluation/evaluation_set.jsonl \
  --responses data/evaluation/model_responses.jsonl \
  --mode final --allow-final --runner 김하루

# 팀 공용 진단표
python -m grader.cli diagnose --table
python -m grader.cli diagnose --report artifacts/report.json

# 변동 폭 → 회귀 판정 (★CI를 켜기 전에 반드시)
python -m grader.cli regression --runs artifacts/r1.json artifacts/r2.json artifacts/r3.json --out artifacts/variance.json
python -m grader.cli regression --variance artifacts/variance.json --baseline artifacts/prev.json --current artifacts/now.json
```

산출물: `artifacts/report.json`(층별 결과 + 집계 + 실행 조건), `artifacts/per_item.jsonl`
(문항별 결과). 집계만 남기면 "왜 떨어졌는지" 분석할 재료가 없으므로 문항별 결과를 항상 남긴다.

## 응답(모델 출력) 형식

```json
{
  "id": "Q0001",
  "answer": "5억원",
  "contexts": [{"document_id": "DOC001", "text": "예산은 5억원이다.",
               "location": {"document": "DOC001", "section": "3장 2절", "ref_no": "표3"}}],
  "retrieved": [{"document_id": "DOC001", "location": {...}, "score": 0.91}],
  "reranked": [{"document_id": "DOC001", "location": {...}, "score": 0.98}],
  "selected_document_ids": [],
  "structured_answer": null,
  "abstained": false
}
```

`retrieved`/`reranked`/`contexts`는 각각 retrieval_k/reranker_k/context_k 단계에 대응한다.
아직 이태민의 실제 검색 결과 JSON 구조가 들어오지 않았으므로, 필드명은 확정이 아니라
채점기가 기대하는 계약이다 — 실제 구조가 오면 이 계약에 맞춰 변환하거나(권장), 계약 자체를
갱신한다.

## 설계 원칙

- 스키마 필드명은 팀 확정본을 그대로 유지한다. 확정되지 않은 정책은 `configs/grader.yaml`
  또는 provider에 주입하고, 코드에 임의로 하드코딩하지 않는다.
- 채점 프롬프트는 파일로 분리하고 버전을 붙인다(`data/prompts/<name>.<version>.md`,【23】).
  프롬프트가 바뀌면 이전 점수와의 비교가 무효가 될 수 있다.
- Judge는 faithfulness(충실성)와 항목 단위 매칭(체크포인트/목록 항목)만 담당한다.
  correctness/citation/abstention은 규칙 기반(task_scoring/retrieval/extraction)으로 낸다
  — LLM 판정은 필요한 곳에만 최소로 쓴다(【22】).
- 코퍼스/추출 테이블 버전이 달라지면 캐시가 자동으로 달라진다.
- Judge/프롬프트 변경만으로 오른 점수는 "시스템 개선"으로 자동 인정하지 않는다(4-9).

## 아직 비어 있는 자리 ([대기])

| 자리 | 대기 대상 |
|---|---|
| `data/evaluation/*.jsonl` 실제 50문항 | [임현진 2-13 확정] 최종 평가셋은 임현진이 전체를 보유하고, 팀에는 유형별 예시 2~3개만 공개(최종 평가 후 전체 공개). 이 저장소가 실제로 받아야 할 것은 예시 2~3개(스키마 확인용)와 `practice_items.jsonl`(CI용) |
| `data/evaluation/practice_items.jsonl` | [임현진 2-17 확정] CI에 상시 노출할 practice 세트(검수 탈락분). 임현진이 직접 채우는 중. ★없으면 `mode=ci`는 하드 실패(exit 1) — 최종셋 폴백 없음 |
| `gate.column_severity` | 박예진 1-12-1 필드별 critical/major/minor 배정 |
| ~~`gate.data_thresholds`~~ ✅ 2026-08-27 반영(3차) | 박예진 1-11 확정 게이트 4종(로드율/raw=md 개수/파일명 정규화/CSV 디코딩) + 표 보존율 100%·인코딩 깨짐 0건까지 `configs/grader.yaml`과 `validation.check_data_sanity`에 반영함. ★"원본 대비 손실률"은 팀 결정으로 **영구히** 측정하지 않는다(위 재확인 섹션 참고 — "제외"가 아니라 "확정") |
| ~~`gate.data_warn_thresholds`(신규)~~ ✅ 2026-08-27 반영(3차) | 박예진 확정 — 표 0개 문서(스캔된 한글 파일 OCR 미실행 신호)를 **게이트가 아니라 경고**로 추가함. `validation.check_data_warnings()` + `configs/grader.yaml`의 `gate.data_warn_thresholds`. 한계: 문서 일부만 스캔인 경우는 못 잡음(1-19-1에 기록) |
| `gate.doc_org` | 박예진 1-9-1 document_id → 발주기관 |
| `gate.task_quota` — 값은 확정, 적용은 보류 | 임현진 2-12 확정: selection 25 / extraction 15 / qa 10(총 50). 2-13에 따라 이 저장소가 최종 50문항 파일 자체를 갖지 않을 가능성이 높아, 이 값은 임현진 쪽 평가 실행 환경에서 쓰일 것으로 보고 코드(`configs/grader.yaml` 주석)에만 남겨 둠 |
| ~~`gate.task_weight`(2-16 통합 점수)~~ ✅ 산출 방식 반영, 값은 대기 | [2026-08-27 임현진 2-16 확정] 통합 점수를 태스크별 3개 점수와 항상 나란히 낸다는 방식 자체는 `diagnostics.integrated_score()`로 구현함. 가중치 숫자(1-2 업무빈도·위험)만 아직 없어 `gate.task_weight: {}`로 비워 둠 |
| `retrieval.retrieval_k/reranker_k/context_k` 확정값 | 2026-08-27 이태민·김하루 `top_k=5(잠정)` baseline 잠정 합의(3단계 중 어느 것인지는 미정) — 이태민의 실제 검색/재정렬 구조가 오면 최종 확정 |
| 응답 JSON의 `retrieved`/`reranked` 실제 스키마 | 이태민 4-9-3 |
| `judge.verification_path`(3-9 사람 대조 기록) | 팀 대조 실시 후 |
| `gate.severity_gate` / 부분점수 태스크 threshold | baseline 실측 후 |
| regression `n_sigma` | [2026-08-27 임현진 2-17] 1문항=몇%p은 확정(선별 4%p·추출 약6.7%p·QA 10%p, `regression.pp_per_item()`으로 이미 계산 가능) — 다만 이걸 바탕으로 한 실제 n_sigma·게이트 확정은 baseline 실측 후 |
| 비교형(2-8-4) 정답의 최종 스키마 표현 | 임현진 확정 — 현재는 `"document_id|field|value"` 문자열 목록으로 잠정 표현(`task_scoring._parse_comparison_gold` 참고) |
| 시간 의존 문항의 "마감일 미상" 표기 관례 | [2026-08-27 박예진 공유] 2024-06-01 기준 미경과 67 / 경과 25 / 미상 8건. `reference_time`이 빈 문자열이 아니면(예: `"unknown"`) 2-17 무결성 검사를 이미 통과하므로 코드 변경은 필요 없지만, 8건에 실제로 어떤 문자열을 쓸지는 임현진과 확인 필요 |

빈 자리는 추측으로 채우지 않았다. 실측이 오면 그 값을 덮어쓰게 되기 때문이다.

## 스키마 편차 — 해소됨 (2026-08-28 v0.2 통일)

예전엔 이 코드가 확정 15개 필드 밖에 `checkpoints` / `unanswerable_reason` 두 필드를
따로 들고 있었다. **v0.2 팀 규약대로 둘 다 별도 스키마 필드에서 제거**했다:

- 요약형(`answer_type=summary`)의 체크포인트 배열 → `answer_raw` 에 담는다.
- 기권(`answer_type=unanswerable`)의 사유 문자열 → `answer_raw` 에 담는다(`null` 금지).

채점기 호환을 위해 `EvaluationItem.checkpoints` / `.unanswerable_reason` 는 **읽기
전용 프로퍼티(= `answer_raw` 뷰)** 로만 남겼다 — 입력 필드가 아니므로
`model_validate({"checkpoints": ...})` 는 `extra=forbid` 로 거부된다. 채점 기능
자체(`grade_summary_checkpoint`, `grade_abstention`)는 그대로다.

## 2026-08-28 v0.2 스키마 반영 (필드 20개 → 15개)

- **제거된 5개 필드**: `document_unspecified`·`time_dependent`·`conversational`·
  `difficulty`·`schema_version`(문항 레벨). 이 저장소가 이미 세 불리언 필드를
  실제 게이트/정합성 로직에서 참조하고 있어서 단순 삭제로 끝나지 않았다 —
  아래처럼 코드를 맞췄다.
  - `document_unspecified`/`time_dependent`/`conversational`은 `models.EvaluationItem`의
    `computed_field`로 남겨 각각 `intermediate_answer`/`reference_time`/
    `active_document_id` 존재 여부로 파생한다. `extraction.grade_doc_selection`,
    `validation.check_evalset_integrity`는 그대로 `item.document_unspecified` 등을
    읽으므로 호출부는 수정하지 않았다. ★이 파생은 "존재하면 True"라는 근사이고
    v0.1 시절 완전한 논리적 동치가 보장돼 있던 건 아니므로, v0.2 첫 실 데이터로
    한 번 대조 확인 권장.
  - 위 파생의 결과로 `check_evalset_integrity`에 있던 3개 정합성 검사(document_
    unspecified인데 intermediate_answer 없음 등)는 논리적으로 항상 거짓이 되어
    죽은 코드가 됐다 — 제거했다.
  - `difficulty`는 파생하지 않고 완전히 제거했다(`diagnostics.full_report`의
    `by_difficulty`, `runner._result_row`의 `difficulty` 키 삭제).
  - `schema_version`(문항 레벨)은 신규 모듈 `grader.versioning.read_schema_version()`이
    저장소 루트 `VERSION.txt`를 읽어 대신한다. `runner.GraderRunner`가 초기화 시
    한 번 읽어 `EvaluationResult.schema_version`에 채운다. ★`VERSION.txt`의
    정확한 경로/포맷은 이번 요청 범위에 명시되지 않아 임시로 "루트에 문자열
    한 줄"로 잡았다 — 확인 필요.
  - `configs/grader.yaml` 최상단의 `schema_version`(`GraderConfig.schema_version`)은
    위와 별개의, 현재 코드 어디서도 소비되지 않는 값이다. 이번 변경에서는 건드리지
    않았다 — `VERSION.txt`와 합칠지는 팀 확인 필요.
- **field_tag 재정의 재확인 (미해결)**: field_tag의 의미를 "부분점수 인정 여부"에서
  "오답 심각도 표시(critical/major/minor) + 등급 반영의 가중 입력"으로 바꾸는 안이
  나왔는데, `diagnostics.py`에는 이미 "①가중 평균은 근거 없어 채택하지 않음(②분리
  집계+③게이트로 확정)"이라는 명시적 결정이 있다. field_tag를 가중 입력으로 쓰는
  안이 이 결정을 뒤집는 것인지, 아니면 별도 축을 추가하는 것인지 재확인 전까지는
  등급 반영 코드를 변경하지 않았다.

## 2026-08-27 팀 공유분 재확인 (2차)

- **추출 실패 3상태**: 박예진이 공유한 "값 있음/항목 없음/추출 실패" 3상태 관리는
  `extraction.EXTRACT_STATES = ("value", "absent", "failed")`로 이미 반영돼 있다 —
  이번에 코드를 다시 확인했고 별도 수정은 없었다.
- **표 개수 [정정 완료, 2026-08-27 3차→4차]**: 기존 "8,051개(문서당 약 85개)"는
  근거 불명확 사유로 **폐기**됐다(김태윤 확인). 정정된 실측값:
  바깥 표만 **7,916개** / `<table>` 태그 전부(중첩 포함) **8,550개**(여는 태그=닫는
  태그, 깨진 곳 없음) / 표 안에 든 표(nested) **634개** / 문서당 최소 3·중앙값
  77·최대 218 / **표 0개 문서 0건**(→ 위 `gate.data_warn_thresholds.max_zero_table_docs`
  경고가 실측 전수(0건)와 일치함을 재확인). 여전히 CI 게이트로는 걸지 않는다 —
  코퍼스가 계속 갱신되는 축(3-17)이라 표 개수는 시간이 지나면 자연히 늘어나고,
  고정된 개수와 정확히 같아야 한다는 게이트를 걸면 다음 코퍼스 갱신에서 바로
  깨진다. 게이트로 쓰려면 "허용 범위(±n%)"가 먼저 정해져야 한다.
- **원본 대비 손실률**: [2026-08-27 확정] **측정하지 않는 것으로 확정됐다** — "아직
  안 왔다"가 아니라 "앞으로도 안 온다"는 뜻이다. 이유: HWP를 읽는 도구가 kordoc
  하나뿐이라, 같은 도구로 뽑은 값을 같은 도구 결과와 자동으로 대조하면 항상
  일치한다 — 자동 대조가 성립하지 않는 구조적 문제다. **3-1/3-6-1이 "파싱 손실
  진단 기준선" 항목을 갖고 있는데, 이 숫자는 오지 않는다** — 계속 대기 상태로
  두지 말고 이 사실 자체를 관련 항목에 반영해야 한다(아래 참고). 대신 쓸 수 있는
  것: 표 보존율 100% · 인코딩 깨짐 0건 · 변환 100/100 · 표 개수 실측치, 그리고
  간접 지표(문서당 5.1만~27.8만 자, 장절 인식률 95/100, 문서 끝이 실제 RFP
  말미인지). ★이걸 팀에 전달할 때는 "측정 불가"가 아니라 **"3주 프로젝트 범위를
  넘어서 하지 않기로 함"**으로 적을 것 — 육안 대조 자체는 가능하다(1-12에서 실제로
  원본 PDF를 열어 값 뒤섞임을 찾은 적 있음). 이 구분이 있어야 나중에(1-21) "안 한
  것"으로 정확히 회수된다.

## 2026-08-27 체크리스트1(2부) 공유분 확인 (4차)

- **표 개수 정정**: 위 "2차" 섹션에 반영 완료(7,916/8,550/634/중앙값 77/표0개 0건).
- **코퍼스 정본(正本) 정의 변경 [확인 필요, 코드 영향 아직 없음]**: 전달 내용상
  코퍼스 정본이 `interim/md`에서 `processed/corpus_vN`으로 바뀐다 — md는 이제
  "중간 산출물"로 재정의되고, `VERSION.txt`도 `interim/md`가 아니라 `processed/`
  아래 코퍼스 버전별로 생긴다. 이 저장소는 `corpus_version`을 문자열 인자로만
  받고(`GraderRunner(corpus_version=...)`) 특정 경로 구조를 가정하지 않으므로 지금
  당장 고칠 코드는 없다. 다만 나중에 `processed/corpus_vN/VERSION.txt`를 자동으로
  읽어 `corpus_version`을 채우는 기능을 추가한다면 이 경로 규칙을 따라야 한다.
- **청크 메타데이터 스키마(이태민)**: `{"doc_id", "chapter", "type": "table|text",
  "table_idx", "part", "of", "text"}` 확정 전달받음. `ModelResponse.contexts`/
  `retrieved`/`reranked`의 실제 스키마는 위 "아직 비어 있는 자리" 표의 4-9-3
  항목대로 여전히 [대기]다 — 지금 이 구조로 바로 고정하지 않는다(이태민 쪽 최종
  확정을 한 번 더 기다림, 특히 `table_idx`가 표 정정값(7,916 vs 8,550 vs 634)
  중 어느 집합을 가리키는지 확인 필요).
- **📤 넘길 것/📥 받을 것 현황판**: 프로세스 추적용 표라 이 저장소의 코드/설정
  변경 대상은 아니다 — 팀 커뮤니케이션 참고용으로만 확인함.
