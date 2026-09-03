# 모델 결과 응답 계약 (이태민 → 채점기)

`grader.models.ModelResponse` — 문항 하나에 대한 4번(이태민) 시스템 출력.
`responses.jsonl` 로 줄마다 한 객체, `grader run --responses` 로 연결.

세 사람이 같은 계약을 쓰면, 평가셋(현진)의 `task_type=qa/selection/extraction` 분류는
그대로 두고 모델은 내부적으로 추출·선별·청크 검색 경로를 자유롭게 써도 된다.

## 필드

| 필드 | 타입 | 필수 | 무엇 / 채점기가 어떻게 쓰나 |
| --- | --- | --- | --- |
| `id` | str | ✅ | 평가셋 문항 id 와 1:1 |
| `answer` | str | ✅ | 최종 답변 텍스트. `answer_type=value/summary/comparison/unanswerable` 채점 입력 |
| `structured_answer` | list \| dict \| null | 조건부 | `list` → `["항목1","항목2",…]` / `comparison` → `{ "RFP-000038": {"예산":"…","사업기간":"…"}, … }`. 없으면 목록·비교형 형식 위반(4-11-1) |
| `contexts` | 청크[] | 검색 채점 시 | **실제 LLM 입력에 들어간 청크**(context_k). retrieval `context_k` 단계 + citation 채점 |
| `retrieved` | 청크[] | 검색 채점 시 | retrieval_k 후보 풀 전체 — 재현율 실패 판정 |
| `reranked` | 청크[] | 선택 | reranker_k 이후 — 순위 실패 판정(없으면 retrieved 로 대체) |
| `citations` | 좌표[] | 좌표 채점 시 | 답변이 근거로 든 출처 좌표. **배열** — 비교형은 여러 개. 3-4-3: 정답 위치와 대조해 "누락 / 잘못된 근거" 구분 |
| `selected_document_ids` | str[] | 선별형 | 선별형(document_set) 결과 문서 ID 배열 |
| `active_document_id` | str \| null | 대화형 | 시나리오형(2-5) 상태 |
| `abstained` | bool | ✅ **모든 응답** | 답변함=`false` / 보류·거절=`true`. **키 자체가 없으면 형식 오류** — 본문 표현으로 채점기가 추측하지 않는다(팀장 2-4) |
| `unanswerable_reason` | str \| null | 선택 | 기권 사유 자유 텍스트 |
| `route` | str \| null | 권장 | **내부 실행 경로**. ★내용 점수엔 안 쓴다. 단 **검색 평가 대상 구분**엔 쓴다 — `config.retrieval.non_search_routes`(추출표·identity 등)에 있으면 검색/citation 채점 = 해당 없음. 청크 검색 경로로 푼 문항만 검색 점수 계산(팀장). task_type≠route 감점 없음 |
| `failure` | str \| null | 오류 시 | 오류로 인한 0점(`failure` 채워짐) vs 오답 0점 구분 (4-5) |
| `latency_ms` | float \| null | 선택 | 응답 시간 (보조 지표) |
| `cost_usd` | float | 선택 | 호출 비용 (보조 지표, 기본 0.0) |

## 청크 / 좌표 객체 모양

`contexts` / `retrieved` / `reranked` 원소는 아래 둘 중 하나:

```jsonc
// (A) 박예진 청크 스키마 그대로 — grader 가 location 을 합성한다
{ "document_id": "RFP-000038", "section_path": ["Ⅰ. 사업 개요"],
  "block_type": "paragraph", "block_index": 3,          // 있으면 ref_no 에 사용
  "md_line_start": 51, "md_line_end": 51,               // 있으면 좌표 대조에 사용
  "search_text": "…", "content": "…" }

// (B) 이미 {document, section, ref_no} 로 정리한 경우
{ "document_id": "RFP-000038", "location": {"document":"RFP-000038","section":"Ⅰ. 사업 개요","ref_no":"paragraph 3"} }
```

`citations` 원소는 좌표 객체:

```jsonc
{ "document": "RFP-000038", "section": "Ⅰ. 사업 개요", "ref_no": "paragraph 3" }
// 비교형: 정답 location 이 문서×필드 배열이므로 citations 도 그만큼 낸다
```

## 예시 한 줄

```json
{"id":"PRAC-QA-002","answer":"352,000,000원(부가가치세 포함)","route":"qa",
 "contexts":[{"document_id":"RFP-000001","section_path":["2. 사업개요"],"block_type":"table","block_index":1,"md_line_start":84,"md_line_end":84,"search_text":"소요예산 352,000,000원"}],
 "citations":[{"document":"RFP-000001","section":"2. 사업개요","ref_no":"table 1"}],
 "latency_ms":1840,"cost_usd":0.0021}
```

## 채점기가 route 를 쓰지 않는다는 것의 의미

- `task_type`(평가셋) = 태스크별 점수 집계 축
- `answer_type`(평가셋) = 실제 채점기 선택 (`grade_content` 디스패치)
- `route`(모델) = 결과에 기록만. `route` 이름을 평가셋과 맞출 필요 없음 —
  같은 경로를 같은 이름으로 부르기만 하면 grader 가 경로별로 집계함
- `task_type=qa` 인데 `route=extract_table` 이어도 **내용 점수 감점 없음**
- 다만 `route` 가 `non_search_routes`(기본: extract_table / identity / table / metadata …)에 있으면
  그 문항의 **검색·출처 좌표 점수는 '해당 없음'** — 검색을 안 쓴 문항을 검색 실패로 세지 않는다
