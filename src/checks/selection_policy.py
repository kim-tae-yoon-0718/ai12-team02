"""선별형 평가 문항 정책 술어 — 정답 생성과 검사기의 단일 출처.

공동수급(컨소시엄) 요건 7상태 분류 + 입찰마감 필터 + 항목 미기재 판정 + 문항 조건표.

⚠️ 모델 코드(`src/rag/*`)를 **가져다 쓰지 않는다.** 정답 생성과 모델 실행이 같은
   판정 코드를 공유하면 모델 오류가 정답에 그대로 복사되어, 채점이 모델의 실수를
   정답으로 인정하게 된다. 여기서는 공식 추출표·등록부·identity 만 읽는다.
"""
from __future__ import annotations

import re

# ────────────────────────────────────────────── 공동수급 7상태

REQUIRED_ALL = "required_all"            # 무조건 공동수급 구성 의무
REQUIRED_CONDITIONAL = "required_conditional"  # 특정 조건에서만 구성 의무
ALLOWED_EXPLICIT = "allowed_explicit"    # 공동수급 참여가 허용된다고 명시
FORBIDDEN_ALL = "forbidden_all"          # 공동수급 자체가 전면 금지
METHOD_RESTRICTED = "method_restricted"  # 특정 이행방식만 금지·제한 (전면 금지 아님)
FIELD_ABSENT = "field_absent"            # 문서에 항목 자체가 없음
UNDETERMINED = "undetermined"            # 자료만으로 판단 불가

ALL_STATES = (REQUIRED_ALL, REQUIRED_CONDITIONAL, ALLOWED_EXPLICIT,
              FORBIDDEN_ALL, METHOD_RESTRICTED, FIELD_ABSENT, UNDETERMINED)

# 【중요】 "공동수급이 금지됐다"고 말할 수 있는 상태는 FORBIDDEN_ALL 하나뿐이다.
#   METHOD_RESTRICTED 를 여기에 넣으면 "공동이행방식은 허용하지 않음" 한 줄 때문에
#   다른 방식으로는 참여할 수 있는 공고가 '공동수급 금지 공고'로 둔갑한다.
FORBIDDEN_STATES = frozenset({FORBIDDEN_ALL})

_WS = re.compile(r"\s+")

# 공동수급이라는 **참여 형태 자체**를 가리키는 말
_CONSORTIUM = r"(?:공동\s*수급(?:체)?|컨소시?시엄|컨소시엄|공동\s*계약|공동\s*도급|공동\s*참여)"
# 개별 **이행방식** 이름 — 이것 하나를 막았다고 공동수급 전체가 막힌 것은 아니다
_METHOD = r"(?:공동\s*이행|분담\s*이행|주계약자\s*관리|주계약자)"
_NEG = r"(?:불허|불가|금지|허용하지\s*않|허용되지\s*않|허용\s*안|참여를\s*제한|참가를\s*제한|배제|제외)"
_ALLOW = r"(?:허용|가능|참가할\s*수\s*있|참여할\s*수\s*있|참여가\s*가능|참가가\s*가능|참가\s*가능|참여\s*가능|구성하여\s*참[가여])"

# ① 조건부 의무 — "…하려는 경우 … 공동수급체를 구성하여 참여해야"
_COND_REQUIRED = re.compile(
    r"(?:하려는\s*경우|초과하여|초과하는\s*경우|해당하는\s*경우)[^.。\n]{0,80}?"
    r"공동\s*수급체를\s*구성하여\s*참여해야")
# ② 무조건 의무
_REQUIRED = re.compile(
    r"반드시\s*공동\s*수급"
    rf"|{_CONSORTIUM}\s*(?:구성)?\s*(?:은|는|이|가)?\s*(?:의무|필수)"
    r"|공동\s*수급체를\s*구성하여야\s*(?:만)?\s*(?:입찰|참여|참가)")

# ③ 허용·부정 동사
# ★"허용하지 않음"의 '허용'을 허용으로 읽으면 안 된다 — 부정이 뒤에 붙는다.
_ALLOW_RE = re.compile(_ALLOW + r"(?!\s*(?:하지|되지|치|하지는)?\s*(?:않|못|안\b))")
_NEG_RE = re.compile(_NEG)
_CONS_RE = re.compile(_CONSORTIUM)
_METHOD_RE = re.compile(_METHOD)

# 절 경계 — 한 문장 안에서도 대상이 바뀐다.
#   "공동수급을 허용하고 있어 하도급은 불허함" 은 앞뒤 대상이 서로 다르다.
#   절로 자르지 않고 한 정규식으로 훑으면 '공동수급 … 불허'로 잘못 이어 읽는다.
_CLAUSE_SPLIT = re.compile(
    r"[.\u3002\n;/]"
    r"|(?<=\S)\s*(?:하며|으며|되며|이며|하고\s*있어|있어|하되|단,|다만,)\s*")

_PAREN = re.compile(r"\(([^)]*)\)")


def _norm(text) -> str:
    return _WS.sub(" ", (text or "")).strip()


def _paren_kind(body: str) -> str:
    """괄호 안 성격: all_methods | one_method | apposition | other."""
    has_gongdong = re.search(r"공동\s*이행|공동(?=\s*(?:및|,|·))", body)
    has_bundam = re.search(r"분담", body)
    if has_gongdong and has_bundam:
        return "all_methods"      # 두 방식을 모두 열거 → 사실상 전면 금지
    if _CONS_RE.search(body):
        return "apposition"       # 공동수급(컨소시엄) — 같은 것을 달리 부름
    if _METHOD_RE.search(body):
        return "one_method"
    return "other"


def _negated_objects(head: str) -> dict:
    """부정어 **바로 앞** 구간에서 부정 대상 낱말을 뽑는다.

    문장 앞부분까지 끌어오면 전혀 다른 대상이 부정된 것으로 읽힌다 —
    "공동수급을 허용 … 하도급은 불허" 가 대표적이다.
    """
    tail = head[-60:]
    # ★라벨/항목 구분자 뒤부터가 진짜 부정 대상이다.
    #   "하도급/공동수급 여부 : 하도급 불허" 에서 앞의 '공동수급'은 항목 이름이지
    #   부정 대상이 아니다. 구분자를 넘어 끌어오면 허용 공고가 금지로 뒤집힌다.
    for sep in (":", "：", "여부"):
        idx = tail.rfind(sep)
        if idx != -1:
            tail = tail[idx + len(sep):]
    bare = qualified = all_methods = False
    for m in _CONS_RE.finditer(tail):
        rest = tail[m.end():].lstrip()
        pm = _PAREN.match(rest)
        kind = _paren_kind(pm.group(1)) if pm else None
        if kind == "one_method":
            qualified = True
        elif kind == "all_methods":
            all_methods = True
        else:
            bare = True           # 괄호가 없거나 동격 → 공동수급 자체가 대상
    method_only = (not (bare or qualified or all_methods)
                   and bool(_METHOD_RE.search(tail)))
    return {"bare": bare, "qualified": qualified,
            "all_methods": all_methods, "method_only": method_only}


def _allows_consortium(clause: str):
    """이 절이 공동수급 **자체**의 참여를 허용한다고 말하는가.

    ★허용 동사와 공동수급 낱말 사이에 부정어가 끼어 있으면 허용이 아니다.
      "공동수급(공동이행방식) 및 하도급은 불가함의 규정에 의한 … 과업 수행 가능자"
      에서 뒤쪽 '가능'은 과업 수행자를 가리키지 공동수급 허용이 아니다.
      이 오탐 하나가 금지 공고를 허용 공고로 뒤집는다.
    """
    for a in _ALLOW_RE.finditer(clause):
        before = clause[:a.start()]
        m = None
        for m in _CONS_RE.finditer(before[-60:]):
            pass                      # 허용 동사에 가장 가까운 공동수급 낱말
        if m is None:
            continue
        between = before[-60:][m.end():]
        if _NEG_RE.search(between):   # 사이에 부정이 끼면 허용 진술이 아니다
            continue
        return a
    return None


def classify(status: str, answer_raw, source_excerpt=None) -> tuple[str, str]:
    """(상태, 판정 근거)를 돌려준다.

    ⚠️ answer_raw 만으로 **부정의 주어**를 알 수 없을 때는 같은 추출표 행의
    source_excerpt_redacted 를 2차 근거로 한 번 더 본다. 추측이 아니라 같은 행이
    보존한 원문이다. 실제로 어떤 문서는 answer_raw 에서 주어가 잘려
    "(공동이행방식)은 허용하지 않음" 만 남아 어떤 판정기도 대상을 알 수 없다.
    """
    state, reason = classify_text(status, answer_raw)
    if state == UNDETERMINED and source_excerpt:
        alt, alt_reason = classify_text(status, source_excerpt)
        if alt != UNDETERMINED:
            return alt, f"[원문 발췌 근거] {alt_reason}"
    return state, reason


def classify_text(status: str, text) -> tuple[str, str]:
    if status == "field_absent":
        return FIELD_ABSENT, "추출표 status=field_absent (문서에 항목 없음)"
    if status != "value_present":
        return UNDETERMINED, f"추출표 status={status}"
    t = _norm(text)
    if not t:
        return UNDETERMINED, "본문 비어 있음"

    # ① 의무가 가장 강한 진술이므로 먼저 본다.
    if _COND_REQUIRED.search(t):
        return REQUIRED_CONDITIONAL, "특정 조건에서만 공동수급체 구성 의무"
    if _REQUIRED.search(t):
        return REQUIRED_ALL, "무조건 구성 의무 문구"

    # ② 절 단위로 '무엇이' 부정/허용됐는지 본다.
    allow_hit = forbid_all_hit = method_hit = None
    for clause in _CLAUSE_SPLIT.split(t):
        clause = (clause or "").strip()
        if not clause:
            continue
        if allow_hit is None and _allows_consortium(clause):
            allow_hit = clause
        for n in _NEG_RE.finditer(clause):
            obj = _negated_objects(clause[:n.start()])
            if obj["bare"] or obj["all_methods"]:
                forbid_all_hit = forbid_all_hit or clause
            elif obj["qualified"] or obj["method_only"]:
                method_hit = method_hit or clause

    # ③ 공동수급 자체를 막았으면 전면 금지 — 가장 강한 부정이다.
    if forbid_all_hit:
        return FORBIDDEN_ALL, f"공동수급 자체를 부정: …{forbid_all_hit[:90]}…"

    # ④ 참여 허용이 명시됐으면 허용 — 방식이 한정돼도 참여 자체는 가능하다.
    if allow_hit:
        note = "방식은 한정되지만 " if method_hit else ""
        return ALLOWED_EXPLICIT, f"{note}공동수급 참여 허용 명시: …{allow_hit[:90]}…"

    # ⑤ 이행방식만 부정 — 다른 방식까지 막혔다는 뜻이 아니다.
    if method_hit:
        return METHOD_RESTRICTED, (f"특정 이행방식만 제한 — 공동수급 전면 금지 아님: "
                                   f"…{method_hit[:90]}…")

    return UNDETERMINED, "허용·금지 어느 쪽도 단정할 수 없음(구성 조건만 서술 등)"


def is_forbidden_all(state: str) -> bool:
    """"공동수급이 명시적으로 금지된"에 해당하는가 — 방식 제한은 포함하지 않는다."""
    return state in FORBIDDEN_STATES


# ────────────────────────────────────────────── 입찰마감 필터

# 평가 기준 시각(2-13). 문항의 reference_time 과 같은 값이어야 한다 —
# "지금"을 실행 시각으로 잡으면 같은 평가셋이 날마다 다른 정답을 갖게 된다.
REFERENCE_TIME = "2024-06-01"

_DATE_RE = re.compile(r"(\d{4})\s*[.\-/년]\s*(\d{1,2})\s*[.\-/월]\s*(\d{1,2})")


def parse_deadline(raw) -> str | None:
    """입찰마감 문자열에서 ISO 날짜를 뽑는다. 못 뽑으면 None(=미상)."""
    if not raw:
        return None
    m = _DATE_RE.search(str(raw))
    if not m:
        return None
    y, mo, d = (int(x) for x in m.groups())
    if not (1 <= mo <= 12 and 1 <= d <= 31):
        return None
    return f"{y:04d}-{mo:02d}-{d:02d}"


def deadline_state(raw, reference_time: str = REFERENCE_TIME) -> str:
    """"open" | "expired" | "unknown".

    ★unknown 을 expired 로 몰지 않는다. 마감일을 못 읽었다는 것은 "마감이 지났다"가
      아니다. 미상을 탈락시키면 추출 실패가 그대로 정답 축소로 이어져, 추출이
      나빠질수록 정답이 작아지는(그래서 맞히기 쉬워지는) 역방향 보상이 생긴다.
    """
    iso = parse_deadline(raw)
    if iso is None:
        return "unknown"
    return "expired" if iso < reference_time else "open"


def passes_deadline(raw, reference_time: str = REFERENCE_TIME) -> bool:
    """마감 필터 통과 여부 — 미상은 통과시킨다(위 사유)."""
    return deadline_state(raw, reference_time) != "expired"


# ────────────────────────────────────────────── 항목 미기재 판정

def is_field_absent(status: str) -> bool:
    """"문서에 항목이 아예 없음"은 추출표 status 로만 판정한다.

    ★값이 '제한 없음' / '해당 없음' 인 것은 **기재된 것**이다. 문구로 미기재를
      추정하면 "지역제한 없음"이라고 적어 둔 공고가 "지역제한 항목이 없는 공고"로
      섞여 들어가, 서로 다른 두 질문의 정답이 같아진다.
    ★external_reference(다른 문서를 보라고 안내)도 미기재가 아니다 — 항목은 있고
      값의 위치만 바깥에 있다.
    """
    return status == "field_absent"


def is_value_present(status: str) -> bool:
    return status == "value_present"


def is_external_reference(status: str) -> bool:
    return status == "external_reference"


# ────────────────────────────────────────────── 공식 자료 로더

import csv as _csv          # noqa: E402  (정책 모듈 하단 유틸)
import json as _json        # noqa: E402
from pathlib import Path as _Path   # noqa: E402

SRV = _Path("/srv/rfp")
REGISTRY_PATH = SRV / "shared_data/processed/document_registry_v2/document_registry_v2.json"
IDENTITY_PATH = SRV / "shared_data/processed/document_registry_v2/document_identity_v2.csv"
TABLE_PATH = SRV / "shared_data/processed/rfp_extraction_table_v4/extraction_table_v4.json"


def load_official(registry=None, identity=None, table=None) -> dict:
    """등록부·identity·추출표를 읽어 조건 계산에 필요한 형태로 돌려준다(읽기 전용)."""
    reg = _json.loads(_Path(registry or REGISTRY_PATH).read_text(encoding="utf-8"))
    docs = reg["documents"] if isinstance(reg, dict) else reg
    eligible = {d["document_id"] for d in docs
                if d.get("retrieval_eligible") is True
                and str(d.get("active", "true")).lower() == "true"}
    with _Path(identity or IDENTITY_PATH).open(encoding="utf-8-sig") as f:
        ident = {r["document_id"]: r for r in _csv.DictReader(f)}
    tbl: dict[str, dict] = {}
    for row in _json.loads(_Path(table or TABLE_PATH).read_text(encoding="utf-8"))["rows"]:
        if str(row.get("active", "true")).lower() == "true":
            tbl.setdefault(row["document_id"], {})[row["field_name"]] = row
    return {"eligible": eligible,
            "excluded": sorted({d["document_id"] for d in docs} - eligible),
            "identity": ident, "table": tbl}


# ────────────────────────────────────────────── 선별형 문항 조건표

# 문항별 조건. 값은 (설명, 조건함수). 조건함수는 official 을 받아 문서 ID 집합을 낸다.
# ★문서 ID 를 여기 직접 적지 않는다 — 조건만 적고 공식 자료에서 매번 계산한다.
#   ID 를 박아 두면 자료가 바뀌어도 정답이 따라가지 않고, 정답이 왜 그 집합인지도
#   설명되지 않는다.

def _by_status(o, field, *states):
    return {d for d in o["eligible"]
            if (o["table"].get(d, {}).get(field) or {}).get("status") in states}


def _consortium_state(o, doc) -> str:
    row = (o["table"].get(doc, {}) or {}).get("컨소시엄 요건") or {}
    return classify(row.get("status", "missing"), row.get("answer_raw"),
                    row.get("source_excerpt_redacted"))[0]


def _by_consortium(o, *states):
    return {d for d in o["eligible"] if _consortium_state(o, d) in states}


_P = lambda o, f: _by_status(o, f, "value_present")          # noqa: E731
_A = lambda o, f: _by_status(o, f, "field_absent")           # noqa: E731
_E = lambda o, f: _by_status(o, f, "external_reference")     # noqa: E731

SELECTION_SPEC = {
  "SEL-001": ("공고일 = value_present", lambda o: _P(o, "공고일")),
  "SEL-002": ("사업기간 = value_present", lambda o: _P(o, "사업기간")),
  "SEL-003": ("사업분야 = value_present", lambda o: _P(o, "사업분야")),
  "SEL-004": ("지역제한 = value_present", lambda o: _P(o, "지역제한")),
  "SEL-005": ("필수 제출 서류 = value_present", lambda o: _P(o, "필수 제출 서류")),
  "SEL-006": ("제출 방식 = value_present", lambda o: _P(o, "제출 방식")),
  "SEL-007": ("컨소시엄 요건 = value_present", lambda o: _P(o, "컨소시엄 요건")),
  "SEL-008": ("참가 자격(면허·실적) = value_present", lambda o: _P(o, "참가 자격(면허·실적)")),
  "SEL-009": ("예산 = value_present", lambda o: _P(o, "예산")),
  "SEL-010": ("평가 배점 = value_present", lambda o: _P(o, "평가 배점")),
  "SEL-011": ("지역제한 = field_absent (항목 미기재)", lambda o: _A(o, "지역제한")),
  "SEL-012": ("필수 제출 서류 = field_absent", lambda o: _A(o, "필수 제출 서류")),
  "SEL-013": ("제출 방식 = field_absent", lambda o: _A(o, "제출 방식")),
  "SEL-014": ("컨소시엄 요건 = field_absent", lambda o: _A(o, "컨소시엄 요건")),
  "SEL-015": ("참가 자격 = field_absent", lambda o: _A(o, "참가 자격(면허·실적)")),
  "SEL-016": ("예산 = field_absent", lambda o: _A(o, "예산")),
  "SEL-017": ("평가 배점 = field_absent", lambda o: _A(o, "평가 배점")),
  "SEL-018": ("제출 방식 = field_absent AND 필수 제출 서류 = field_absent",
              lambda o: _A(o, "제출 방식") & _A(o, "필수 제출 서류")),
  "SEL-019": ("참가 자격 = value_present AND 컨소시엄 요건 = field_absent(항목 미기재)",
              lambda o: _P(o, "참가 자격(면허·실적)") & _A(o, "컨소시엄 요건")),
  "SEL-020": ("지역제한 = value_present AND 공동수급 = 참여 허용 명시(allowed_explicit)",
              lambda o: _P(o, "지역제한") & _by_consortium(o, ALLOWED_EXPLICIT)),
  "SEL-021": ("제출 방식 = external_reference AND 컨소시엄 요건 = value_present",
              lambda o: _E(o, "제출 방식") & _P(o, "컨소시엄 요건")),
  "SEL-022": ("컨소시엄 요건 = value_present AND 평가 배점 = field_absent",
              lambda o: _P(o, "컨소시엄 요건") & _A(o, "평가 배점")),
  "SEL-023": ("참가 자격 = value_present AND 공동수급 = 전면 금지(forbidden_all)",
              lambda o: _P(o, "참가 자격(면허·실적)") & _by_consortium(o, FORBIDDEN_ALL)),
  "SEL-024": ("지역제한 = value_present AND 공동수급 = 전면 금지(forbidden_all)",
              lambda o: _P(o, "지역제한") & _by_consortium(o, FORBIDDEN_ALL)),
  "SEL-025": ("지역제한 = value_present AND 평가 배점 = field_absent",
              lambda o: _P(o, "지역제한") & _A(o, "평가 배점")),
}


def compute_selection_gold(item_id: str, official: dict,
                           reference_time: str = REFERENCE_TIME) -> dict:
    """문항 하나의 정답을 공식 자료에서 독립적으로 계산한다.

    순서는 확정 정책 그대로: ①등록부 98문서 → ②문항 조건 → ③마감 필터
    (④미상은 통과) → ⑤개수 제한 없이 전체.
    """
    desc, cond = SELECTION_SPEC[item_id]
    base = cond(official)
    states = {d: deadline_state(official["identity"].get(d, {}).get("bid_deadline"),
                                reference_time) for d in base}
    expired = {d for d, v in states.items() if v == "expired"}
    unknown = {d for d, v in states.items() if v == "unknown"}
    return {
        "item_id": item_id, "condition": desc,
        "candidates": sorted(base), "candidate_count": len(base),
        "removed_by_deadline": sorted(expired),
        "passed_as_unknown_deadline": sorted(unknown),
        "gold": sorted(base - expired), "gold_count": len(base - expired),
        "reference_time": reference_time,
    }
