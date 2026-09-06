# tests/fixtures/ — 스키마 확인용 최소 샘플

**진짜 평가셋이 아니다.** 배관(스키마·파이프라인)이 도는지 확인하는 1건짜리 샘플.

| 파일 | 계약 |
| --- | --- |
| `evaluation_set.jsonl` | `grader.models.EvaluationItem` (스키마 v0.2) |
| `model_responses.jsonl` | `grader.models.ModelResponse` (시스템 4번 출력) |

진짜 평가셋: 최종 50문항은 임현진 보유(누수 방지 2-13, 이 저장소엔 안 옴).
practice 세트(`practice_items.jsonl`)는 `$RAG_ROOT/evalset/` 아래(`data/` 는 NDA 로 gitignore).
