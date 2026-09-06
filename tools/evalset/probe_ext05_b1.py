"""B-1/B-3 재현 탐침 — 짧은 답 함수 + 실제 EXT-05 문항 채점."""
import json, sys; sys.path.insert(0, sys.argv[1])
from grader.models import EvaluationItem, ModelResponse
from grader.task_scoring import grade_short_answer, score_item
CFG={"residual_limit":20,"accept_natural_absence_phrasing":True}
items={json.loads(l)["id"]:json.loads(l) for l in open(sys.argv[2],encoding="utf-8") if l.strip()}
it=EvaluationItem.model_validate(items["EXT-05"])
print("EXT-05 정답:",repr(it.answer_raw),"| answer_normalized:",repr(it.answer_normalized),"| answer_source:",it.answer_source)
ACC=["6,758,571,493원","해당 사업의 예산은 6,758,571,493원입니다.","6,758,571,493원(부가가치세 등 제반 비용 포함)","6,758,571,493원($5,198,901/1$=1,300원, 2024년 기준환율)"]
REJ=["6,758,571,493달러","5,198,901원","6,758,571,494원","6,758,571,493원이며 별도 예산 3억원이 추가됩니다","확인할 수 없습니다"]
bad=0
for title,xs,want in (("정답 처리해야",ACC,1.0),("오답 처리해야",REJ,0.0)):
    print(f"--- {title} ---")
    for a in xs:
        r=ModelResponse(id="EXT-05",answer=a,abstained=False,route="추출테이블_값조회")
        s1=grade_short_answer(it,r,CFG); s2=score_item(it,r,CFG)["task_score"]
        ng=(s1.score!=want) or (s2.score!=want); bad+=ng
        print(f"  {'★NG' if ng else 'OK '} 함수={s1.score:.0f} 문항={s2.score:.0f} why={s1.detail.get('why')!s:.60}  {a}")
print(f"[{sys.argv[3]}] 문제 {bad}건")
