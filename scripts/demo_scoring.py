"""실제 practice 8문항 × (정답/오답/환각) 응답 → 채점기가 제대로 가르는지 눈으로 확인."""
import json, sys
sys.path.insert(0, "/home/spai1209/work/haru/rag_grader_pipeline/src")
from checks.check_evalset import load_jsonl
from grader.models import EvaluationItem, ModelResponse, Location
from grader.task_scoring import score_item
from grader.config import load_config

import os
os.chdir("/home/spai1209/work/haru/rag_grader_pipeline")
CFG = load_config("config/grader.yaml")
cfg = {"allow_partial": CFG.grading.allow_partial, "docset_partial_credit": CFG.grading.docset_partial_credit,
       "miss_weight": CFG.grading.miss_weight, "require_table_format": CFG.grading.require_table_format,
       "grade_citations": CFG.grading.grade_citations}

items = {i["id"]: EvaluationItem.model_validate(i) for i in load_jsonl("/srv/rfp/evalset/practice_items.jsonl")}

# 문항별: (정답 응답, 오답 응답, 환각/잘못된 기권 응답)
RESP = {
 "PRAC-EXT-001": (  # 예산 248,796천원
    dict(answer="248,796천원(부가세 포함)"),
    dict(answer="500,000천원"),                         # 틀린 금액
    dict(answer="248,796천원 정도이며 유지보수 3년 포함입니다"),  # 정답+근거없는 덧붙임
 ),
 "PRAC-EXT-002": (  # 제출서류 목록 5개
    dict(answer="인감증명서, 사업자등록증 사본, 법인등기부등본, 기타 공고 서류, 제안요약서",
         structured_answer=["1. 인감증명서 1부","2. 사업자등록증 사본 1부","3. 법인등기부등본(법인) 1부",
                            "4. 기타 공고 등에서 정한 서류 각 1부","5. 제안요약서"]),
    dict(answer="인감증명서, 사업자등록증",
         structured_answer=["1. 인감증명서 1부","2. 사업자등록증 사본 1부"]),  # 3개 누락
    dict(answer="전부",
         structured_answer=["1. 인감증명서 1부","2. 사업자등록증 사본 1부","3. 법인등기부등본(법인) 1부",
                            "4. 기타 공고 등에서 정한 서류 각 1부","5. 제안요약서","6. 보안서약서(가짜)"]),  # 환각 항목 추가
 ),
 "PRAC-EXT-003": (  # 마감일 2024-06-24 (critical)
    dict(answer="2024-06-24 16:00 (입찰 참여 마감 기준)"),
    dict(answer="2024-07-15"),   # 틀린 날짜 (critical 오답!)
    dict(answer="마감일 정보를 찾을 수 없습니다", abstained=True),  # 답이 있는데 기권 (과잉거절)
 ),
 "PRAC-EXT-004": (  # "지역제한 없음" (critical)
    dict(answer="지역제한 없음"),
    dict(answer="서울특별시 소재 업체로 제한됩니다"),           # 정반대 오답 (critical!)
    dict(answer="확인 불가", abstained=True),
 ),
 "PRAC-QA-001": (  # 2024.6.1 기준 입찰 가능 (critical)
    dict(answer="입찰 참여 마감일(2024-06-24 16:00)을 넘지 않았으므로, 참여 가능합니다."),
    dict(answer="이미 마감되어 참여할 수 없습니다."),          # 틀린 판단 (critical!)
    dict(answer="참여 가능하며 낙찰 확률이 높습니다"),         # 근거 없는 추측 덧붙임
 ),
 "PRAC-QA-002": (  # 소요예산 352,000,000원
    dict(answer="352,000,000원(부가가치세 포함)"),
    dict(answer="248,796천원"),  # 다른 사업 예산 (혼동)
    dict(answer="약 3.5억원 수준으로 추정됩니다"),
 ),
 "PRAC-QA-003": (  # 비교형: 예산·사업기간 × 2문서
    dict(answer="표", structured_answer={
        "RFP-000038": {"예산": "230,000천원 이내(부가세 포함)", "사업기간": "계약체결일로부터 4개월 이내"},
        "RFP-000043": {"예산": "248,796천원(부가세 포함)", "사업기간": "계약일로부터 5개월"}}),
    dict(answer="표", structured_answer={
        "RFP-000038": {"예산": "230,000천원 이내(부가세 포함)", "사업기간": "3개월"},          # 사업기간 틀림
        "RFP-000043": {"예산": "300,000천원", "사업기간": "계약일로부터 5개월"}}),             # 예산 틀림
    dict(answer="RFP-000038이 예산·기간 모두 더 큽니다"),   # 우열판단(채점대상 아님) + 표 미제출
 ),
 "PRAC-QA-004": (  # unanswerable — 없는 사업
    dict(answer="그런 사업을 찾을 수 없습니다", abstained=True),
    dict(answer="우주정거장 데이터센터 구축사업은 예산 50억원, 기간 12개월입니다"),  # 환각!
    dict(answer="은하수정보진흥재단 사업은 확인이 어렵습니다", abstained=True),
 ),
}

LABELS = ["✅ 정답 응답", "❌ 오답 응답", "⚠️  환각/오판 응답"]
for iid, triple in RESP.items():
    it = items[iid]
    print("\n" + "═"*78)
    print(f"[{iid}]  {it.task_type}/{it.answer_type}  field_tag={it.field_tag}")
    print(f"Q: {it.question}")
    gold = it.answer_raw if isinstance(it.answer_raw, str) else json.dumps(it.answer_raw, ensure_ascii=False)
    print(f"정답: {gold[:110]}")
    print("─"*78)
    for label, rd in zip(LABELS, triple):
        rd.setdefault('abstained', bool(rd.get('abstained', False)))
        resp = ModelResponse(id=iid, **rd)
        out = score_item(it, resp, cfg)
        ts, fs, ab, final = out["task_score"], out["format_status"], out["abstention"], out["final_status"]
        sys_ans = (rd.get("answer") or "")[:70]
        det = {k: v for k, v in (ts.detail or {}).items() if k in
               ("why","reason","matched","missing","wrong_cells","covered","abstention_kind","cell_accuracy","extra")}
        fmt = "OK" if fs.passed else f"위반:{fs.violations}"
        print(f"  {label}")
        print(f"    시스템 답: {sys_ans}")
        print(f"    → task_score={ts.score:.2f}  final={final}  format={fmt}"
              + (f"  기권판정={ab.abstention_kind}" if (ab.should_abstain or ab.abstention_kind != 'ok') else ""))
        if det:
            print(f"    이유: {json.dumps(det, ensure_ascii=False)[:180]}")
