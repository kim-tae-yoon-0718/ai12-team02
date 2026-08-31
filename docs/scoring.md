# 채점 규칙

## 내용 채점 (`task_scoring.py`)

| answer_type | 함수 | 판정 |
| --- | --- | --- |
| `value` | `grade_short_answer` | 이진 (정규화 후 정확 일치). 금액/날짜는 `normalize.match_short` |
| `list` | `grade_list` | **exact-all 고정** — 5개 중 4개는 오답. Coverage/Missing/Extra 는 진단용 |
| `document_set` | `grade_selection` | 부분점수 (recall 가중, `miss_weight`). 정답은 `answer_raw` 배열 |
| `summary` | `grade_summary_checkpoint` | 체크포인트 커버리지 (`answer_raw` 배열, `judge_checkpoint` 매처) |
| `comparison` | `grade_comparison` | 셀 정확도 (지정 항목 값 일치까지만, 우열 판단 안 함) |
| `unanswerable` | (기권 채점으로 분기) | `grade_abstention` |

## 형식 계약 (3-3) — 내용과 독립 축

`check_format` 이 output contract 위반을 잡는다. **내용이 맞아도 형식 위반이면 최종 PASS 아님.**

- `final_status` ∈ `PASS` / `FAIL-content` / `FAIL-format` / `FAIL-format+content` / `PENDING_THRESHOLD`
- `list` → `structured_answer` 가 배열이어야 / `comparison` → 표 형태 / `document_set` → `selected_document_ids` 배열
- 이진 태스크(`list`/`value`/기권)만 즉시 PASS/FAIL. 부분점수 태스크(`document_set`/`summary`/`comparison`)는
  baseline 실측 전까지 `PENDING_THRESHOLD` — 임의 경계 안 만든다.

## 기권 (`grade_abstention`, 2-3 → 3-3)

`should_abstain = (answer_type == "unanswerable")`. 응답의 `abstained` 와 대조:

| | 기권함 | 안 함 |
| --- | --- | --- |
| **기권해야 함** | `ok` | `hallucination` (답을 지어냄) |
| **안 해도 됨** | `field_tag=critical` → `critical_abstain` / 아니면 `over_refusal` | `ok` |

`critical_abstain`(치명 문항의 "못 찾으면 기권")은 정상 동작 후보(1-5) — 과잉 거절과 **별도 축**.

## field_tag severity 가중치 (3-3-2, 팀 확정)

`critical : major : minor = 5 : 3 : 2` (`configs/default.yaml` `gate.field_tag_weight`).

3-3-2 는 세 가지를 **항상 나란히** 낸다 (하나가 다른 하나를 대체하지 않음):
1. **①가중 평균** (`severity_weighted_score`) — 문항별 점수 × field_tag 가중치
2. **②분리 집계** (`severity_report.by_field_tag`) — 등급별 점수를 따로
3. **③게이트** (`severity_report.gate`) — critical 절대 기준선

★ `critical_abstain`(기권 축)과 severity 가중치는 **서로 대체하지 않는 별도 평가 축**.

## 검색 평가 (`retrieval.py`, 3-2)

- `retrieval_k` / `reranker_k` / `context_k` 세 단계를 각각 Recall@k·Precision@k·MRR로.
- `recall_failure`(후보 풀에도 없음) vs `rank_failure`(후보엔 있으나 context 못 옴) 구분.
- **좌표 단위 = `ref_no`** (`DEFAULT_PRECISION`, 2-9 확정 = document+section+ref_no).
  검색 recall 과 citation 정확도가 같은 단위를 쓴다.
- **다단계 k** (`config.retrieval.eval_k = [3, 5]`): 이태민 `top_k=5` 결과를 k=3/5 로 잘라
  `retrieval_k` 단계에 `by_k` (`recall@k`/`precision@k`/`mrr@k`)를 붙인다. 이태민 협의 확정.

## 박예진 청크 좌표 어댑터

이태민 검색 출력이 박예진 청크 스키마(`section_path`·`location_label`)로 오면
`Location.from_chunk` 가 `{document, section, ref_no}` 로 변환. 분할 표 `표 7 (2/3)` → `표 7`.
`RetrievedItem`/`ContextChunk` 가 `location` 없고 청크 필드가 있으면 자동 합성.

## 출처 좌표 채점 (`grade_citation`, 3-4-3)

정답 `location` 과 응답 `citations` 의 좌표가 실제로 일치하는지. citation 미표기는
`check_format` FAIL 로 만들지 **않는다** — `citation_accuracy`(좌표 일치율)와
`no_citation_rate`(미표기율)를 분리해서 진단. "안 붙임"과 "틀리게 붙임"은 다른 문제.

## 추출 테이블 감사 (`grade_extraction_audit`, 3-2-1)

★ 순환 문제: `answer_source=table` 문항의 정답을 그 테이블로 만들었으면 테이블이 틀려도
만점. → 테이블 정확도는 **평가셋과 독립적으로** 잰다 (원문 표본 대조, `--extraction-audit`).
`mode=final` 은 이 파일 없이 실행 불가. `circularity_flag`(추출↓ + 선별↑ 경보)는 항상.

감사 행 상태 어휘는 현재 `("value","absent","failed")`. B-2 확정 5상태
(`value_present`/`field_absent`/`external_reference`/`not_disclosed`/`conflict` + `extraction_failed`)
반영은 태윤 원문대조 JSONL 핸드오프 후. ※ 문항 스키마와는 별개(위 C 결정 참고).
