# 2026-09-08 체크리스트 — 기업 프로필 매칭 + 세션 시작 흐름

오늘 새로 만든 것들을 직접 손으로 돌려서 확인할 때 쓰는 체크리스트다. 전부
새 파일이고 공식 산출물(`extraction_table_v4`, `document_registry_v2`,
`semantic_decisions_v4.csv`)과 팀 확정 코드(`build_extraction_table.py`,
`identity_metadata.py`, `answer_pipeline.py`)는 안 건드렸다 — import만 해서
재사용.

## 0. 사전 준비

```bash
cd /home/spai1222/rfp-code
export RAG_ROOT=/srv/rfp
source /srv/rfp/venv/bin/activate   # 패키지 못 찾는다는 에러 나올 때만 필요
```

- [ ] 위 세 줄 실행에 에러 없음

---

## 1. `tools/evalset/new_document_candidates.py` — 새 문서 12필드 후보 스크리닝

새 RFP 문서(.md로 이미 변환된 상태) 1건을 넣으면, 100건 표에 쓰던 규칙엔진을
그대로 재사용해서 12필드 후보값+근거위치를 뽑아준다. 사람이 검토·확정하기
전 1차 스크리닝용 — 공식 추출표는 안 건드림.

```bash
python3 tools/evalset/new_document_candidates.py \
  --md <새문서.md 경로> \
  --document-id RFP-NEW-001 \
  --announce-date 2026-09-08 \
  --out /tmp/new_doc_check
```

- [ ] `<document_id>_candidate_decisions.csv`가 `--out` 폴더에 생김
- [ ] 화면에 12개 필드 상태(`value_present`/`field_absent`/...)가 다 나옴
- [ ] CSV 열어보면 `reviewer`/`source_review_method` 칸이 미리 채워져 있고
      `source_review_completed`/`source_context_note`는 빈칸(사람이 채울 몫)

---

## 2. `tools/company_match/company_match.py` — 기업 프로필 저장 + 기본 매칭

지역/사업분야/참가자격 3필드를 코드로(LLM 없이) 대조. 자격증은 **정확한
문구가 통째로** 들어있어야 인식한다(느슨한 대조는 3번에서).

```bash
# 저장 (같은 이름으로 다시 save하면 그 줄만 갱신됨)
python3 tools/company_match/company_match.py save \
  --company "체크기업" --region 서울 \
  --business-fields "농림수산,소프트웨어개발" \
  --certifications "소프트웨어사업자 신고필증"

# 매칭 (98건 대상)
python3 tools/company_match/company_match.py match --company "체크기업" \
  --out /tmp/check_match.csv
```

- [ ] `data/company_profiles/company_profiles.csv`에 "체크기업" 한 줄이 생김
      (회사가 늘어나도 파일 하나에 계속 누적 — 회사당 파일 따로 안 만듦)
- [ ] 매칭 결과 요약에 `적합`/`확인 필요`/`부적합`/`적합(근거 약함)` 네 종류가 나옴
- [ ] `적합(근거 약함)`은 "조건 자체가 명시 안 돼서 그냥 통과"라는 뜻이지
      "진짜 맞다고 확인됨"이 아님 — `적합`과 섞여 보이면 버그

---

## 3. `tools/company_match/loose_match_experiment.py` — 느슨한 대조 + 컨소시엄 요건

company_match.py는 안 건드리고 확장만 함. (1) 자격증을 낱말 단위로 느슨하게
대조 (2) 12필드 중 "컨소시엄 요건"을 추가로 매칭.

```bash
# 컨소시엄 필요 여부 추가(기존 프로필에 열 하나 더 채움)
python3 tools/company_match/loose_match_experiment.py set-consortium \
  --company "체크기업" --consortium-needed true

# 매칭 (느슨한 대조 + 컨소시엄 포함, 적합 우선 정렬)
python3 tools/company_match/loose_match_experiment.py match --company "체크기업"
```

- [ ] `company_profiles.csv`에 `consortium_needed` 열이 생기고 값이 채워짐
- [ ] 결과 리스트 맨 위가 `[적합]`부터 시작함(부적합이 먼저 안 나옴)
- [ ] 자격증 근거에 "느슨한 대조 — 오탐 가능성 있음" 문구가 붙어 있음(숨기지 않음)
- [ ] company_match.py 기본 매칭보다 `적합` 개수가 보통 더 많음(느슨해서 정상)

---

## 4. `tools/company_match/session_start_experiment.py` — 전체 시작 흐름

회사 조회/등록 → 추천 5 + 마감임박 5 → 검색 시작을 하나로 이어붙인 실험
진입점. `answer_pipeline.answer()`를 실제로 호출한다(select/extract는 API
키 없이 동작, QA만 키 필요).

```bash
python3 tools/company_match/session_start_experiment.py
```

**시나리오 A — 신규 등록**
- [ ] `회사 이름을 입력하세요(그만두려면 '취소'):` 프롬프트가 맨 먼저 뜸
- [ ] 없는 회사명 입력 → "등록할까요? 예/아니오/취소" 물어봄
- [ ] `아니오` → 등록 안 하고 이름 다시 물어봄(등록 안 됨)
- [ ] `예` → 지역/사업분야/자격증 하나씩 입력 → 매번 "맞나요?" 재확인
- [ ] 확인 단계에서 `아니오` 치면 그 항목만 다시 물어봄(오타 수정 가능)
- [ ] 등록 끝나면 "프로필 저장 완료" 뜨고 추천/마감임박으로 넘어감

**시나리오 B — 기존 회사 조회**
```bash
python3 tools/company_match/session_start_experiment.py --company "체크기업"
```
- [ ] 등록 없이 바로 "기존 프로필을 불러왔습니다" 뜨고 넘어감

**추천/마감임박 화면**
- [ ] "적합 상위 N건" 목록이 뜨고, 있으면 "판단 근거가 약한 공고 M건은 제외했습니다" 문구도 뜸
- [ ] "마감 임박 공고" 목록에 실제 날짜(2024년)가 찍힘 — 0건이면 이상한 것(고정
      기준일 2024-06-01 기준이라 원래 몇 건 있어야 정상)

**검색 루프**
- [ ] `질문:` 프롬프트에서 `예산 5억 이상인 공고 목록 줘` 치면 실제 목록이 나옴
- [ ] `RFP-000072의 사업분야가 뭐야` 치면 `농림수산`이 나옴
- [ ] `q` 또는 빈 입력 → "검색을 종료합니다"로 깨끗하게 끝남

**취소 기능 (아무 입력창에서나)**
- [ ] `취소`/`q`/`그만`/`종료` 입력 → 에러 트레이스백 없이 "취소했습니다.
      프로그램을 종료합니다." 뜨고 끝남

---

## 5. `src/rag/reference_clock_experiment.py` — now 토글 (미연결 부품)

지금은 세션 시작 화면에서도 안 쓴다(고정 기준일로 통일함, 4번 참고). 나중에
실사용 시점에 붙일 준비물로만 남아 있다 — 그냥 있는지만 확인.

```bash
python3 -c "
import sys; sys.path.insert(0,'src/rag')
from reference_clock_experiment import now_reference_datetime, resolve_reference_datetime
print(now_reference_datetime())
"
```

- [ ] 에러 없이 오늘 날짜·시각이 찍힘

---

## 6. 원본 무변경 확인 (커밋 전 마지막 점검)

```bash
git status --porcelain -- tools/evalset/build_extraction_table.py \
  tools/evalset/extraction_rules_v3.py
```

- [ ] 위 명령 결과가 **비어 있음**(원본 두 파일 무변경)

```bash
python -m pytest src/tests -q
```

- [ ] 전체 테스트 통과(현재 기준 1579개)

---

## 자주 헷갈리는 것 — 미리 메모

- "문서 등록부 검색 대상 98/100문서" 줄은 **시스템 전체 통계**다. 회사 매칭
  개수랑 무관 — 착각하기 쉬우니 주의.
- `적합` vs `적합(근거 약함)`: 후자는 "조건이 아예 안 적혀 있어서 못 걸렀다"는
  뜻이지 "진짜 맞다"가 아니다.
- 마감임박은 실제 현재 시각이 아니라 **고정 기준일(2024-06-01)** 기준이다
  (검색 결과와 항상 같은 기준일을 씀).
