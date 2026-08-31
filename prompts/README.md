# Judge prompts

이 폴더의 `judge_*.v1.md` 세 파일은 팀 확정 프롬프트 원문을 **수정 없이 그대로** 옮긴
것이다(judge_checkpoint.v1.md / judge_faithfulness.v1.md / judge_list_item.v1.md).

- `judge_faithfulness.v1.md` — 문항 전체를 한 번 채점하는 충실성 심판. "답변이 주어진
  근거만으로 뒷받침되는가"만 판정한다. 근거가 사실인지·최신인지·답이 옳은지는 판정하지
  않는다(3-3-1 D7 — 검색 재현율과 짝으로 읽을 것).
- `judge_checkpoint.v1.md` — 요약형(2-8-2) 체크포인트 하나가 답변에 담겼는지 항목 단위로
  판정하는 매처. `task_scoring.grade_summary_checkpoint`가 체크포인트마다 호출한다.
- `judge_list_item.v1.md` — 목록형(3-4-4) 정답 항목 하나가 답변에 언급됐는지 항목 단위로
  판정하는 매처. `task_scoring.grade_list`가 정답 항목마다 호출한다. 문자열 매칭으로는
  표현 변형("사업자등록증 사본" ↔ "사업자등록증")을 놓치기 때문에 LLM 판정을 쓴다.

이 저장소는 prompt 본문을 코드에 하드코딩하지 않는다. 팀 합의가 바뀌면 코드 수정 없이
`prompts/<name>.<version>.md` 파일만 버전을 올려 교체한다(【23】). 프롬프트가 바뀌면
이전 점수와의 비교가 무효가 될 수 있으므로, 버전을 올린 뒤에는 3-9 재검증을 다시 돌려야
tier=final 로 쓸 수 있다(`grader.judge.Judge.assert_ready` 가 이를 강제한다).

## 아직 연결이 안 된 것

- 사람 채점 대조 기록(3-9 verification record) — `configs/grader.yaml`의
  `judge.verification_path`가 비어 있는 동안 `tier: final` 실행은 `Judge.assert_ready()`가
  거부한다. `judge_faithfulness.v1.md`의 CHANGELOG에도 "3-9 사람 대조 미실시 →
  tier=final 사용 금지 상태"라고 명시돼 있다.
- 5개 평가 축 중 correctness/completeness(수치)/citation/abstention 은 LLM judge가 아니라
  `task_scoring.py` / `retrieval.py` / `extraction.py`의 규칙 기반 채점으로 낸다. 이 폴더의
  프롬프트가 담당하는 것은 faithfulness(전체 1회)와 completeness의 항목 단위 매칭(체크포인트/
  목록 항목)뿐이다.
