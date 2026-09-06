"""합성 결함 주입 v2 — 검사기가 실제로 막는지 확인한다(§8-6).

후보 평가셋을 한 군데씩 망가뜨려 검사기가 FAIL 을 내는지 본다. 통과해 버리는
결함이 하나라도 있으면 그 검사는 있으나 마나다.
"""
from __future__ import annotations
import copy, json, subprocess, sys
from pathlib import Path

import argparse  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import add_path_args, data_root, official_paths, repo_root, require  # noqa: E402

_ap = argparse.ArgumentParser(description="평가셋 결함 주입 검사(검사기가 잡는지 확인)")
_ap.add_argument("--out", default=None, help="결과 JSON 경로")
_ap.add_argument("--items", default=None, help="후보 평가셋 items.jsonl")
_ap.add_argument("--python", default=sys.executable, help="검사기를 실행할 파이썬")
add_path_args(_ap)
_args = _ap.parse_args()
WT = repo_root(_args.repo_root)
VENV = Path(_args.python)
SRV = data_root(_args.data_root)
OFF = official_paths(SRV)
SRC = require("후보 평가셋(--items)", _args.items)
REV = SRC.parent.parent            # <산출폴더>/evalset/candidate/items.jsonl 기준


def _sel(items, n=1):
    return [i for i in items if i["task_type"] == "selection"][:n]


def _first(items, pred):
    return next(i for i in items if pred(i))


DEFECTS = {}
def defect(key, title):
    def deco(fn):
        DEFECTS[key] = (title, fn); return fn
    return deco


# ── 기존 12종 ────────────────────────────────────────────────────────
@defect("D01", "선별형 정답에 검색 제외 문서(RFP-000006) 포함")
def d01(items):
    s = _sel(items)[0]
    s["answer_raw"] = list(s["answer_raw"]) + ["RFP-000006"]; s["answer_set"] = s["answer_raw"]

@defect("D02", "선별형 정답 문서 ID 중복")
def d02(items):
    s = _sel(items)[0]
    s["answer_raw"] = list(s["answer_raw"]) + [s["answer_raw"][0]]; s["answer_set"] = s["answer_raw"]

@defect("D03", "등록부에 없는 문서 ID를 정답에 넣음")
def d03(items):
    s = _sel(items)[0]
    s["answer_raw"] = list(s["answer_raw"]) + ["RFP-000999"]; s["answer_set"] = s["answer_raw"]

@defect("D04", "선별형 reference_time 삭제")
def d04(items):
    _sel(items)[0].pop("reference_time", None)

@defect("D05", "선별형 reference_time 을 다른 날짜로 변경")
def d05(items):
    _sel(items)[0]["reference_time"] = "2025-01-01"

@defect("D06", "answer_source 라벨 삭제")
def d06(items):
    items[0].pop("answer_source", None)

@defect("D07", "근거 좌표를 존재하지 않는 ref_no 로 변경")
def d07(items):
    it = _first(items, lambda i: isinstance(i.get("location"), dict))
    it["location"]["ref_no"] = "없는섹션 · 문단 999"
    for ev in it.get("evidence") or []:
        if isinstance(ev.get("location"), dict):
            ev["location"]["ref_no"] = "없는섹션 · 문단 999"

@defect("D08", "근거 좌표 문서를 등록부에 없는 ID 로 변경")
def d08(items):
    it = _first(items, lambda i: isinstance(i.get("location"), dict))
    it["location"]["document"] = "RFP-000999"

@defect("D09", "문항 ID 중복")
def d09(items):
    items.append(copy.deepcopy(items[0]))

@defect("D10", "유형별 할당량 위반(선별형 1건 삭제)")
def d10(items):
    items.remove(_sel(items)[0])

@defect("D11", "비교형인데 비교 대상이 1건")
def d11(items):
    it = _first(items, lambda i: i.get("answer_type") == "comparison")
    keep = it["location"][0]["document"]
    it["answer_raw"] = [{k: v for k, v in row.items() if k in ("항목", keep)}
                        for row in it["answer_raw"]]
    it["location"] = [l for l in it["location"] if l.get("document") == keep]
    it["evidence"] = [e for e in (it.get("evidence") or []) if e.get("document") == keep]

@defect("D12", "선별형 정답이 문자열(리스트 아님)")
def d12(items):
    s = _sel(items)[0]
    s["answer_raw"] = s["answer_raw"][0] if s["answer_raw"] else "RFP-000002"
    s["answer_set"] = s["answer_raw"]

# ── 이번에 추가된 결함군 ─────────────────────────────────────────────
@defect("D13", "사실 문항의 근거 삭제")
def d13(items):
    it = _first(items, lambda i: i.get("evidence"))
    it.pop("evidence", None); it.pop("location", None)

@defect("D14", "유효한 문서 ID로 구성했지만 틀린 선별 정답 집합")
def d14(items):
    """★S1(범위)·S3(존재)를 전부 통과하는 '조용한' 오답 — 독립 재계산만이 잡는다.

    정답이 실제로 다른 두 문항을 골라 한쪽 정답을 다른 쪽에 덮어쓴다. 문서 ID 는
    전부 등록부에 있고 검색 대상이라 범위 검사로는 아무 문제가 없다.
    """
    sels = [i for i in items if i["task_type"] == "selection"]
    a = b = None
    for i, x in enumerate(sels):
        for y in sels[i + 1:]:
            if sorted(x["answer_raw"]) != sorted(y["answer_raw"]) and y["answer_raw"]:
                a, b = x, y
                break
        if a is not None:
            break
    if a is None:
        raise StopIteration
    a["answer_raw"] = list(b["answer_raw"]); a["answer_set"] = a["answer_raw"]
    a["evidence"] = [dict(e, document=d) for d in a["answer_raw"]
                     for e in (a.get("evidence") or [])[:1]]

@defect("D15", "정답과 무관한 청크 좌표로 교체")
def d15(items):
    """존재하는 좌표지만 다른 문서의 것 — S5(존재)는 통과, 관련성은 아니다."""
    it = _first(items, lambda i: isinstance(i.get("location"), dict) and i["id"].startswith("EXT"))
    it["location"] = {"document": "RFP-000001", "section": "2. 사업개요",
                      "ref_no": "2. 사업개요 · 문단 1-31", "line": 49}
    it["evidence"] = [{"kind": "chunk", "document": "RFP-000001",
                       "source": "chunks_v3", "location": it["location"]}]

@defect("D16", "특정 공동수급 방식 금지를 전체 금지로 분류")
def d16(items):
    """RFP-000002 는 '공동이행방식은 허용하지 않음' — 전면 금지가 아니다."""
    it = next(i for i in items if i["id"] == "SEL-024")
    it["answer_raw"] = sorted(set(it["answer_raw"]) | {"RFP-000002"})
    it["answer_set"] = it["answer_raw"]
    it["evidence"] = list(it.get("evidence") or []) + [
        {"kind": "extraction_table", "document": "RFP-000002", "field": "컨소시엄 요건",
         "status": "value_present", "source": "extraction_table_v3"}]

@defect("D17", "QA 설명형 답을 value 로 지정")
def d17(items):
    it = next(i for i in items if i["id"] == "QA-001")
    it["answer_type"] = "value"
    it["answer_raw"] = ("본 사업은 평택시 전역을 대상으로 버스정류장 안내단말기를 확충하고 "
                        "버스 운행정보를 실시간 제공하여 대중교통 서비스의 질을 높이는 것을 "
                        "목적으로 한다.")

@defect("D18", "잘못된 identity 필드 인용")
def d18(items):
    it = _first(items, lambda i: any(e.get("kind") == "identity" for e in (i.get("evidence") or [])))
    for ev in it["evidence"]:
        if ev.get("kind") == "identity":
            ev["field"] = "존재하지_않는_필드"

@defect("D19", "추출표 근거의 status 를 실제와 다르게 기록")
def d19(items):
    it = _first(items, lambda i: any(e.get("kind") == "extraction_table"
                                     and e.get("status") == "field_absent"
                                     for e in (i.get("evidence") or [])))
    for ev in it["evidence"]:
        if ev.get("status") == "field_absent":
            ev["status"] = "value_present"; break

@defect("D20", "해소 가능한 문항에 unspecified_type 을 붙임(되묻기만 해도 만점)")
def d20(items):
    it = next(i for i in items if i["id"] == "EXT-03")
    it["unspecified_type"] = "abbreviation"; it["intermediate_answer"] = "RFP-000003"

@defect("D21", "요약형 체크포인트를 원문 문장 통째로 되돌림")
def d21(items):
    it = next(i for i in items if i["id"] == "QA-003")
    it["answer_raw"] = ["기술원 환경에 최적화된 공정 및 장비 실시간 모니터링 기능 구현을 위한 "
                        "효율적이고 체계적인 설비 온라인 시스템 구축"]

@defect("D22", "검색 불가능한 청크를 유일 근거로(차단 선언 없이)")
def d22(items):
    """[2026-09-04] EXT-05 가 추출표 근거로 바뀌어 blocked_reason 제거만으로는 결함이
    생기지 않는다(주입 무효). 검색 제외 청크(RFP-000007-0006, 목차 오분류,
    retrieval_eligible=false)를 **유일한** 근거로 실제로 심는다."""
    it = next(i for i in items if i["id"] == "EXT-05")
    it.pop("blocked_reason", None)
    loc = {"document": "RFP-000007", "section": "목차", "ref_no": "목차 · 문단 19-40", "line": 103}
    it["location"] = loc
    it["evidence"] = [{"kind": "chunk", "document": "RFP-000007", "source": "chunks_v3", "location": loc}]


def run_checker(path: Path) -> tuple[int, str]:
    p = subprocess.run(
        [str(VENV), "-m", "checks.check_evalset", str(path), "--final-mode",
         "--registry", str(OFF["registry"]),
         "--chunks", str(OFF["chunks"]),
         "--identity", str(OFF["identity"]), "--table", str(_args.table or OFF["table"])],
        cwd=str(WT), env={"PYTHONPATH": str(WT / "src"), "PATH": "/usr/bin:/bin"},
        capture_output=True, text=True)
    return p.returncode, (p.stdout + p.stderr)


def main(out: Path) -> int:
    base = [json.loads(l) for l in SRC.read_text(encoding="utf-8").splitlines() if l.strip()]
    tmp = out.parent / "_injected_v2.jsonl"
    rows, blocked = [], 0
    for key in sorted(DEFECTS):
        title, fn = DEFECTS[key]
        items = copy.deepcopy(base)
        try:
            fn(items)
        except StopIteration:
            rows.append({"key": key, "defect": title, "blocked": None,
                         "first_error": "주입 대상 문항을 찾지 못함"})
            print(f"{key}  ★주입실패     {title}")
            continue
        tmp.write_text("\n".join(json.dumps(i, ensure_ascii=False) for i in items) + "\n",
                       encoding="utf-8")
        code, text = run_checker(tmp)
        caught = code != 0
        blocked += caught
        first = next((l.strip() for l in text.splitlines()
                      if l.strip().startswith("【") or l.strip().startswith("FAIL")), "")
        rows.append({"key": key, "defect": title, "blocked": caught,
                     "exit_code": code, "first_error": first[:170]})
        print(f"{key}  {'차단' if caught else '★통과(구멍)':<12} {title}\n      → {first[:150]}")
    tmp.unlink(missing_ok=True)
    out.write_text(json.dumps({"n": len(rows), "blocked": blocked, "rows": rows},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n합성 결함 {len(rows)}종 중 {blocked}종 차단")
    return 0 if blocked == len(rows) else 1


sys.exit(main(Path(_args.out) if _args.out else REV / "evalset/defect_injection_v2.json"))
