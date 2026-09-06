"""§2 field_absent 채점 탐침 — 실제 후보 문항(EXT-09/10/11)으로 채점 경로 확인. 사용: probe_field_absent.py <src> <items.jsonl> <라벨>"""
import sys, json; sys.path.insert(0, sys.argv[1])
from grader.models import EvaluationItem, ModelResponse
from grader.task_scoring import score_item
ON={"residual_limit":20,"accept_natural_absence_phrasing":True}; OFF={"residual_limit":20,"accept_natural_absence_phrasing":False}
items={json.loads(l)["id"]:json.loads(l) for l in open(sys.argv[2],encoding="utf-8") if l.strip()}
GOLD={"EXT-09":"이 문서에는 지역제한 항목이 별도로 명시되어 있지 않습니다.","EXT-10":"이 문서에는 공동수급(컨소시엄) 요건 항목이 별도로 명시되어 있지 않습니다.","EXT-11":"이 문서에는 필수 제출 서류 항목이 별도로 명시되어 있지 않습니다."}
ACC={"EXT-09":["이 문서에는 지역제한 항목이 별도로 명시되어 있지 않습니다.","문서에 지역제한 항목이 명시되어 있지 않습니다","원문에 지역제한 항목이 기재되어 있지 않습니다.","원문에 해당 항목 자체가 없습니다(값이 없다는 뜻으로 단정할 수 없음)","해당 문서에는 지역제한에 관한 항목이 따로 적혀 있지 않습니다."],
     "EXT-10":["이 문서에는 공동수급(컨소시엄) 요건 항목이 별도로 명시되어 있지 않습니다.","문서에 컨소시엄 요건 항목이 명시되어 있지 않습니다","원문에 해당 항목 자체가 없습니다(값이 없다는 뜻으로 단정할 수 없음)"],
     "EXT-11":["이 문서에는 필수 제출 서류 항목이 별도로 명시되어 있지 않습니다.","문서에 필수 제출 서류 항목이 명시되어 있지 않습니다","원문에 해당 항목 자체가 없습니다(값이 없다는 뜻으로 단정할 수 없음)"]}
REJ=["없음","지역 제한 없음","공동수급 조건 없음","지역 제한이 없다고 확인했습니다","확인할 수 없습니다","자료가 부족합니다","검색 결과가 없습니다","문서를 찾지 못했습니다",
     "원문에 지역제한 항목이 없으므로 실제 지역 제한도 없습니다","문서에 공동수급 항목이 없으므로 공동수급이 허용됩니다","문서에 예산 항목이 없으므로 예산은 0원입니다","문서에 해당 항목이 미기재이므로 제한 없음",
     "없음(확인할 수 없음)","지역 제한이 없지는 않습니다","지역 제한이 없을 수도 있습니다","지역 제한이 없다는 뜻은 아닙니다","제출 서류가 존재하지 않습니다","예산은 0원입니다","지역 제한 없음(전국 입찰 가능)","해당 항목이 없어 어느 지역 업체든 참여할 수 있습니다"]
bad=0
for iid in ("EXT-09","EXT-10","EXT-11"):
    it=EvaluationItem.model_validate(items[iid]); ev=(items[iid].get("evidence") or [{}])[0]
    print(f"--- {iid} gold={it.answer_raw!r} evidence={ev.get('kind')}/{ev.get('status')} {ev.get('document')}/{ev.get('field')}")
    for t in ACC[iid]:
        on=score_item(it,ModelResponse(id=iid,answer=t,abstained=False),ON)["task_score"]; ng=on.score!=1.0; bad+=ng
        print(f"  {'★NG' if ng else 'OK '} 인정해야 ON={on.score:.0f} {t}  [{(on.detail or {}).get('why','')[:40] if isinstance(on.detail,dict) else ''}]")
    for t in REJ:
        on=score_item(it,ModelResponse(id=iid,answer=t,abstained=False),ON)["task_score"]; off=score_item(it,ModelResponse(id=iid,answer=t,abstained=False),OFF)["task_score"]; ng=(on.score!=0.0 or off.score!=0.0); bad+=ng
        print(f"  {'★NG' if ng else 'OK '} 거부해야 ON={on.score:.0f} OFF={off.score:.0f} {t}  [{(on.detail or {}).get('why','')[:40] if isinstance(on.detail,dict) else ''}]")
    for t in ("",):
        on=score_item(it,ModelResponse(id=iid,answer=t,abstained=True),ON)["task_score"]; ng=on.score!=0.0; bad+=ng; print(f"  {'★NG' if ng else 'OK '} 기권 → ON={on.score:.0f}")
print(f"[{sys.argv[3]}] 문제 {bad}건")
