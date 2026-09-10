"""§3 자연어 '없음' 채점 탐침 — 수정 전/후 같은 입력."""
import sys; sys.path.insert(0, sys.argv[1])
from grader.models import EvaluationItem, ModelResponse
from grader.task_scoring import score_item
from grader.config import load_config
cfg = load_config(sys.argv[2])
C = {"residual_limit": 20, "accept_natural_absence_phrasing": cfg.grading.accept_natural_absence_phrasing}
it = EvaluationItem.model_validate(dict(id="X", question="지역 제한 조건이 있어?", task_type="extraction",
        answer_type="value", answer_raw="없음", answer_source="table"))
ACCEPT = ["없음","해당 없음","지역 제한 없음","공동수급 조건 없음","문서에 명시되어 있지 않음",
          "관련 내용이 기재되어 있지 않음","지역 제한 조건은 없습니다.","지역 제한이 없다고 확인했습니다"]
REJECT = ["확인할 수 없음","찾지 못함","알 수 없음","자료가 부족함","현재 확인이 어려움",
          "관련 내용이 없다고 확인할 수 없음","문서를 찾지 못해 답할 수 없음",
          "관련 내용이 없는지는 알 수 없습니다","지역 제한이 없다고 확인할 수 없습니다","",
          "오류: 문서를 불러오지 못했습니다"]
print(f"설정 accept_natural_absence_phrasing={C['accept_natural_absence_phrasing']}")
ok=0; n=0
for grp, want, xs in (("정답 인정해야", 1.0, ACCEPT), ("인정하면 안 됨", 0.0, REJECT)):
    print(f"--- {grp} ---")
    for x in xs:
        s = score_item(it, ModelResponse(id="X", answer=x, abstained=False), C)["task_score"].score
        n+=1; ok += (s==want); print(f"  {'OK ' if s==want else '★NG'} score={s:.0f} {x!r}")
s = score_item(it, ModelResponse(id="X", answer="없음", abstained=True), C)["task_score"].score
n+=1; ok+=(s==0.0); print(f"  {'OK ' if s==0.0 else '★NG'} score={s:.0f} 기권(abstained=True)+'없음' → 0 이어야")
print(f"[{sys.argv[3]}] {ok}/{n}")
