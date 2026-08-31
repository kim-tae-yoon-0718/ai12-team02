# examples/ — 스키마 확인용 최소 샘플

여기 두 파일은 **진짜 평가셋이 아니다.** 배관(스키마·파이프라인)이 도는지 확인하는
1건짜리 샘플이다.

| 파일 | 형식 | 계약 |
| --- | --- | --- |
| `evaluation_set.jsonl` | 평가셋 문항 (스키마 v0.2) | `src/grader/models.py` → `EvaluationItem` |
| `model_responses.jsonl` | 시스템(4번) 출력 | `src/grader/models.py` → `ModelResponse` |

진짜 평가셋:
- 최종 50문항 — 임현진이 보유(누수 방지, 2-13). 이 저장소에 안 들어온다.
- practice 세트(`practice_items.jsonl`) — 임현진이 채우는 중. CI(`mode=ci`)가 이걸 쓴다.
  위치는 `$RAG_ROOT/evalset/` 아래(팀 규약). 이 저장소의 `data/` 는 `.gitignore` 로
  막혀 있다(원본 RFP 비밀유지).
