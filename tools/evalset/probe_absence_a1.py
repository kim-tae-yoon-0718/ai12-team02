"""A-1/A-3 재현 탐침 — 판정 함수와 실제 채점 경로 둘 다."""
import sys; sys.path.insert(0, sys.argv[1])
from grader.models import EvaluationItem, ModelResponse
from grader.normalize import classify_absence
from grader.task_scoring import score_item
ON={"residual_limit":20,"accept_natural_absence_phrasing":True}; OFF={"residual_limit":20,"accept_natural_absence_phrasing":False}
it=EvaluationItem.model_validate(dict(id="X",question="지역 제한 조건이 있어?",task_type="extraction",answer_type="value",answer_raw="없음",answer_source="table"))
A1=["없음(확인할 수 없음)","지역 제한 없음(문서를 찾지 못함)","해당 항목은 없습니다(자료 부족으로 확인할 수 없음)","지역 제한이 없지는 않습니다","지역 제한이 없는 것은 아닙니다",
    "지역 제한이 없을 수도 있습니다","지역 제한이 없다고 볼 수는 없습니다","관련 조건이 없다는 의미는 아닙니다","관련 조건이 없다고는 할 수 없습니다"]
ACC=["원문에 해당 항목이 없습니다","문서에 지역제한 항목이 기재되어 있지 않습니다","해당 필드는 미기재입니다","원문에 해당 항목 자체가 없습니다(값이 없다는 뜻으로 단정할 수 없음)"]
REJ=["없음(확인할 수 없음)","없음(문서를 찾지 못함)","없음(자료 부족)","지역 제한이 없지는 않습니다","지역 제한이 없는 것은 아닙니다","지역 제한이 없을 수도 있습니다",
     "지역 제한이 없다고 볼 수는 없습니다","지역 제한이 없다는 의미는 아닙니다","관련 조건이 없다고는 할 수 없습니다","확인할 수 없습니다","알 수 없습니다","검색 결과가 없습니다","응답이 없습니다","답변을 생성하지 못했습니다"]
def row(t):
    k=classify_absence(t); on=score_item(it,ModelResponse(id="X",answer=t,abstained=False),ON)["task_score"].score; off=score_item(it,ModelResponse(id="X",answer=t,abstained=False),OFF)["task_score"].score
    return k,on,off
bad=0
for title,xs,want in (("A-1 오류 재현(전부 0/NOT_VERIFIED 이어야)",A1,0.0),("A-3 인정해야",ACC,1.0),("A-3 인정 금지",REJ,0.0)):
    print(f"--- {title} ---")
    for t in xs:
        k,on,off=row(t); ng=(on!=want) or (want==0.0 and off!=0.0) or (want==0.0 and k=="ABSENCE_CONFIRMED") or (want==1.0 and k!="ABSENCE_CONFIRMED")
        bad+=ng; print(f"  {'★NG' if ng else 'OK '} 분류={k:<18} ON={on:.0f} OFF={off:.0f}  {t}")
print(f"[{sys.argv[2]}] 문제 {bad}건")
