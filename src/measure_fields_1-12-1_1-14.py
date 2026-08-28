#!/usr/bin/env python3
# src/measure_fields_1-12-1_1-14.py   (2026-08-28 수정본 v3)
#
# 1-12-1 (6) 정의형 약칭 · 1-14 (5) 잔여물 문자 4종 · 1-14 (6) 언어 구성
# 100건 md 전수. 세기만 하고 파일은 바꾸지 않는다.
#
#   export RAG_ROOT=/srv/rfp
#   python3 src/measure_fields_1-12-1_1-14.py
#
# ── v1 대비 고친 것 (v1 결과를 보고 드러난 결함) ───────────────
# D1  약칭을 두 갈래로 쪼갠다 — 기관형(임현진 2-2-2) / 기술약어형(이태민 4-9-4)
# D2  약칭 신뢰도 mid → low. 오탐 거르개를 5종으로 늘림
# D3  특수불릿에서 가운뎃점 `·` 제외 — 한국어 정상 문장부호다(`사업명·기간·예산`).
#     v1에서 이것 때문에 5만 건이 나왔다. 별도 항목으로만 센다
# D4  영문 비율에 보정치 병기 — 이미지 마크업·URL·파일확장자를 뺀 값.
#     v1의 image(1,196)·images(597)·png·bmp·www·http는 본문 내용이 아니다
#
# ── v3에서 더한 것 ────────────────────────────────────────
# E1  약칭 자체는 다시 재지 않는다. v2가 뽑은 것을 '분류하고 표시'만 한다.
#     세 번째 정규식 시도는 하지 않는다 (1-12 13번과 같은 판단 —
#     정규식을 다듬는 비용이 얻는 것보다 크다)
# E2  꼬리 명사 추출 — `제안서에 명시된 사업관리자` → `사업관리자`.
#     원문은 그대로 두고 정리된 형태를 옆 칸에 붙인다. 지우지 않는다
# E3  충돌 목록에 등급 부여 — 확실 / 혼재 / 불명.
#     ⭐ 걸러내지 않는다. 무엇을 쓸지는 이태민 4-9-4에서 정한다.
#     여기서 거르면 남의 항목을 대신 정하는 것이다
# E4  md 출력에 충돌 목록 전체 수록 (v2는 상위 25개만 냈다)
#
# ── 안 재는 것 (일부러 뺌) ────────────────────────────────
#   글자 사이 공백(100건·1,820회) · 백슬래시 이스케이프(100건·4,181회)
#   1-12에 전수가 있다. 두 번 재면 값이 갈리고 어느 게 맞는지 다투게 된다.

import html
import json
import os
import re
import sys
import unicodedata
from collections import Counter
from datetime import datetime
from pathlib import Path

NFC = lambda s: unicodedata.normalize("NFC", str(s))

TAG = re.compile(r"<[^>]+>")
# 이미지 마크업 / URL / 파일확장자 — 언어 구성 보정용
IMG = re.compile(r'!\[[^\]]*\]\([^)]*\)')
URL = re.compile(r'(?:https?://|www\.)\S+|\b[\w.-]+@[\w.-]+\b')
EXT = re.compile(r'\.(?:png|jpe?g|bmp|gif|webp|svg|pdf|hwp|docx?|xlsx?|pptx?|zip|kr|com|net|go)\b',
                 re.I)


def strip_tags(t):
    """HTML 태그 제거 + 엔티티 복원.
    태그를 두면 table/tr/td 가 전부 영문으로 잡혀 언어 비율이 거짓이 된다."""
    return html.unescape(TAG.sub(" ", t))


def strip_markup(t):
    """이미지 마크업·URL·확장자 제거 (언어 구성 보정용)."""
    return EXT.sub(" ", URL.sub(" ", IMG.sub(" ", t)))


# ── A. 정의형 약칭 ────────────────────────────────────────
# A-1  X(이하 Y) / X(이하 "Y") / X(이하 Y라 한다)
P_IHA = re.compile(
    r'([^\s()\[\]|]{2,40})\s*[（(]\s*이하\s*["\'“”「『]?\s*'
    r'([^)）"\'“”」』]{1,25}?)\s*["\'“”」』]?\s*'
    r'(?:이?라\s*(?:고\s*)?(?:한다|칭한다|함|합니다)?)?\s*[)）]'
)
# A-2  한글기관명(ABC) — 괄호 안이 대문자 약어
#      v1은 [가-힣\s]{1,29} 라 연속 공백을 건너뛰어 앞 문장까지 붙었다
#      ("산출정보  프로젝트관리요구사항"). 최대 3어절, 공백 1칸으로 제한.
P_UPPER = re.compile(
    r'([가-힣][가-힣]{0,14}(?:\s[가-힣]{1,14}){0,2})\s*[（(]\s*'
    r'([A-Z][A-Za-z0-9\-]{1,14})\s*[)）]'
)
# A-3  X(약칭: Y) / X(약칭 Y)
P_YAK = re.compile(r'([^\s()\[\]|]{2,40})\s*[（(]\s*약칭\s*[:：]?\s*([^)）]{1,25}?)\s*[)）]')

# ── 오탐 거르개 (D2) — 지우지 않고 표시만 한다. 판정은 사람이 한다 ──
# 1. 일반명사로 끝나 기관명이 아닌 것
F_COMMON = re.compile(r'(사업|과업|용역|본건|당사|귀사|계약|제안서|입찰|시스템|제품|서비스'
                      r'|요구사항|사항|기준|방법|기간|테스트|시험|인증|점수|단말기|정보)$')
# 2. 서술어로 끝나는 것 — v1의 `한다. → 창업기업에 대한 …` 유형
F_VERB = re.compile(r'(한다|된다|같다|이다|함|됨|음)\.?$')
# 3. 조사로 시작 — v1의 `에 전원기동 → CMOS` 유형
F_PARTICLE = re.compile(r'^[은는이가을를에의로와과도만]\s')
# 4. 마스킹 문자 — `△△△` `○○○` `ㅇㅇ`
F_MASK = re.compile(r'^[△▲○●◇◆ㅇOoXx\*_\-\s]+$')
# 5. 약칭 쪽이 약칭답지 않음 — 공백 포함 6자 초과, 또는 서술어로 끝남
def bad_abbr(a):
    return (" " in a and len(a) > 6) or bool(F_VERB.search(a))


# ── 기관형 판별 (D1) ──────────────────────────────────────
ORG_TAIL = re.compile(
    r'(원|청|처|부|국|과|공사|공단|재단|센터|대학교|대학|연구소|연구원|진흥원'
    r'|협회|협의회|연합회|중앙회|진흥회|위원회|조합|공제회|학회'
    r'|본부|기술원|과학원|사업단|관리단|평가원|정보원'
    r'|박물관|병원|의료원|은행|공항|법인|단체|기관|시|도|군|구)$'
)


def classify(full, abbr):
    """기관형 / 기술약어형 / 의심 으로 가른다."""
    reasons = []
    if F_VERB.search(full):
        reasons.append("서술어로 끝남")
    if F_PARTICLE.search(full):
        reasons.append("조사로 시작")
    if F_MASK.match(full) or F_MASK.match(abbr):
        reasons.append("마스킹 문자")
    if bad_abbr(abbr):
        reasons.append("약칭이 문장에 가까움")
    if reasons:
        return "의심", reasons
    # 일반명사(요구사항·시스템·제품…)로 끝나면 기관이 아니라 기술 약어다.
    # v1에서 이것들이 대량으로 나왔고, 오탐이 아니라 진짜 기술 약어였다.
    if F_COMMON.search(full):
        return "기술약어형", []
    if ORG_TAIL.search(full) and len(full) >= 4:
        return "기관형", []
    return "기술약어형", []


# ── E2. 꼬리 명사 추출 ────────────────────────────────────
# 앞 문장이 붙어 들어온 것을 원문은 두고 옆에 정리형만 붙인다.
# `제안서에 명시된 사업관리자` → `사업관리자`
JOSA_TAIL = re.compile(
    r'^.*?(?:은|는|이|가|을|를|에|의|로|와|과|도|만|께|부터|까지|에서|에게|으로|라|고)\s+')
# 관형형 어미로 끝나는 앞 어절 — `명시된 사업관리자` → `사업관리자`
ADNOM = re.compile(r'^\S*(?:된|되는|하는|한|할|인|있는|없는|같은|따른|반드시|해당)\s+')
DET_HEAD = re.compile(r'^(?:본|해당|이|그|각|동|상기|위|아래|다음|모든|기타|일부|전체)\s+')

def tail_noun(full):
    t = full.strip()
    prev = None
    while prev != t:                      # 조사로 끝나는 앞 어절을 반복 제거
        prev = t
        t = JOSA_TAIL.sub('', t).strip()
        t = ADNOM.sub('', t).strip()
    t = DET_HEAD.sub('', t).strip()
    ws = t.split()
    if len(ws) > 3:                       # 그래도 길면 뒤 3어절만
        t = ' '.join(ws[-3:])
    return t or full


# ── E3. 충돌 등급 ────────────────────────────────────────
# 확실 : 코드가 짧고 형식이 일정해 문장이 섞일 여지가 없다
# 혼재 : 진짜 충돌 + 앞 문장 오탐이 섞여 있다. 꼬리 명사로 판단 가능
# 불명 : 앞뒤 잘림이 심해 원표기를 못 믿는다
P_CODE = re.compile(r'^[A-Z]{2,5}$')          # SFR · PMR · INR …

def grade_conflict(abbr, fulls, tails):
    if not P_CODE.match(abbr):
        return "불명", "약칭이 대문자 코드 형태가 아니다"
    n_t = len(set(tails))
    if n_t <= 1:
        return "불명", "꼬리 명사가 한 종류뿐 — 충돌이 아니라 표기 흔들림일 수 있다"
    long_ratio = sum(1 for f in fulls if len(f.split()) >= 3) / len(fulls)
    if long_ratio <= 0.2 and n_t <= 4:
        return "확실", f"원표기가 대부분 짧고 꼬리 명사 {n_t}종 — 문장 혼입 여지 없음"
    return "혼재", (f"원표기 중 {long_ratio:.0%}가 3어절 이상(앞 문장 혼입 의심). "
                    f"꼬리 명사 {n_t}종으로 판단할 것")


def find_abbr(text):
    out = []
    for pat, kind in ((P_IHA, "이하형"), (P_UPPER, "대문자약어형"), (P_YAK, "약칭형")):
        for m in pat.finditer(text):
            full = NFC(re.sub(r'\s+', ' ', m.group(1))).strip()
            abbr = NFC(re.sub(r'\s+', ' ', m.group(2))).strip()
            if not full or not abbr or full == abbr:
                continue
            bucket, reasons = classify(full, abbr)
            out.append({"pattern": kind, "bucket": bucket,
                        "full": full, "tail": tail_noun(full),
                        "abbr": abbr, "reasons": reasons})
    return out


# ── B. 잔여물 문자 4종 ────────────────────────────────────
P_CTRL = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
P_ZWSP = re.compile(r'[\u200b-\u200f\ufeff\u00ad\u2028\u2029]')      # 보이지 않는 문자
P_SPACES = re.compile(r'[ \t]{3,}')
P_BLANKS = re.compile(r'(?:\r?\n){4,}')                              # 빈 줄 3줄 이상
# D3 — 가운뎃점 `·` 제외. 한국어 정상 문장부호다(`사업명·기간·예산`).
BULLETS = "●■★▲▶○◎□▢◇◆▪▫∎❍❑❒◦▷▸►◈"
P_BULLET = re.compile('[' + re.escape(BULLETS) + ']')
P_MIDDOT = re.compile('·')                                            # 참고용으로만 센다
P_UNDER = re.compile(r'[_]{4,}|[‥…]{4,}|[─―—]{4,}')
P_DASHRUN = re.compile(r'-{4,}')
# 마크다운 표 구분선 / 수평선 — P_DASHRUN 오탐의 주범이라 제외한다.
P_TBLSEP = re.compile(r'^\s*\|?[\s|:\-]{3,}\|?\s*$')


def residue(raw):
    dash = 0
    for ln in raw.split("\n"):
        if P_TBLSEP.match(ln):          # 표 구분선은 세지 않는다
            continue
        dash += len(P_DASHRUN.findall(ln))
    bul = Counter(P_BULLET.findall(raw))
    return {
        "제어문자": len(P_CTRL.findall(raw)),
        "보이지않는문자": len(P_ZWSP.findall(raw)),
        "연속공백_3칸이상": len(P_SPACES.findall(raw)),
        "연속개행_빈줄3줄이상": len(P_BLANKS.findall(raw)),
        "특수불릿": sum(bul.values()),
        "특수불릿_종류별": dict(bul),
        "밑줄점선": len(P_UNDER.findall(raw)) + dash,
        "참고_가운뎃점": len(P_MIDDOT.findall(raw)),
    }


# ── C. 언어 구성 ──────────────────────────────────────────
P_HAN = re.compile(r'[가-힣ㄱ-ㅎㅏ-ㅣ]')
P_LAT = re.compile(r'[A-Za-z]')
P_CJK = re.compile(r'[\u4e00-\u9fff]')
P_DIG = re.compile(r'[0-9]')
P_TOKEN = re.compile(r'[A-Za-z][A-Za-z0-9\-]{1,19}')


def language(clean, clean2):
    """clean  = 태그만 제거          → 원값
       clean2 = 태그+마크업·URL 제거 → 보정값 (D4)"""
    def stat(t):
        ko, la, cj, di = (len(p.findall(t)) for p in (P_HAN, P_LAT, P_CJK, P_DIG))
        total = len(re.sub(r'\s+', '', t))
        f = lambda n: round(n / total, 4) if total else 0.0
        return {"글자수_공백제외": total, "한글": ko, "영문": la, "한자": cj, "숫자": di,
                "한글비율": f(ko), "영문비율": f(la), "한자비율": f(cj)}
    a, b = stat(clean), stat(clean2)
    return ({**a, "영문비율_보정": b["영문비율"], "한글비율_보정": b["한글비율"],
             "글자수_보정": b["글자수_공백제외"]},
            Counter(P_TOKEN.findall(clean2)))


# ── 실행 ──────────────────────────────────────────────────
def main():
    root = os.environ.get("RAG_ROOT")
    if not root:
        sys.exit("[중단] RAG_ROOT가 없습니다.  export RAG_ROOT=/srv/rfp")
    root = Path(root)
    md_dir = root / "shared_data" / "interim" / "md"
    out_dir = root / "docs"
    if not md_dir.is_dir():
        sys.exit(f"[중단] {md_dir} 없음")
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in md_dir.iterdir() if p.suffix.lower() == ".md")
    if not files:
        sys.exit(f"[중단] {md_dir} 에 md가 없습니다.")
    print(f"[대상] {len(files)}건")

    per_doc, abbr_all = [], []
    tok_all, res_docs, res_sum, bullet_docs = Counter(), Counter(), Counter(), Counter()
    lang_rows = []

    for p in files:
        raw = p.read_text(encoding="utf-8", errors="replace")
        clean = strip_tags(raw)
        clean2 = strip_markup(clean)

        for a in find_abbr(clean):
            a["file"] = p.name
            abbr_all.append(a)

        r = residue(raw)
        for k, v in r.items():
            if k == "특수불릿_종류별":
                for b in v:
                    bullet_docs[b] += 1
                continue
            res_sum[k] += v
            if v:
                res_docs[k] += 1

        lang, toks = language(clean, clean2)
        tok_all.update(toks)
        lang_rows.append((p.name, lang))

        per_doc.append({"file": p.name,
                        "약칭": Counter(a["bucket"] for a in abbr_all if a["file"] == p.name),
                        "잔여물": {k: v for k, v in r.items() if k != "특수불릿_종류별"},
                        "언어": lang})

    buckets = {b: [a for a in abbr_all if a["bucket"] == b]
               for b in ("기관형", "기술약어형", "의심")}

    def pairs_of(rows):
        c = Counter((a["full"], a["abbr"]) for a in rows)
        return [{"full": f, "abbr": a, "n": n} for (f, a), n in c.most_common()]

    # 같은 코드에 뜻이 여럿 / 같은 뜻에 코드가 여럿 — 4-9-4 재료
    # ⭐ 걸러내지 않는다. 전체를 등급 표시와 함께 넘긴다 (E3)
    by_abbr, by_abbr_tail, by_tail = {}, {}, {}
    for a in buckets["기술약어형"] + buckets["기관형"]:
        by_abbr.setdefault(a["abbr"], set()).add(a["full"])
        by_abbr_tail.setdefault(a["abbr"], set()).add(a["tail"])
        key = re.sub(r'\s+', '', a["tail"])
        by_tail.setdefault(key, {"labels": set(), "abbrs": set()})
        by_tail[key]["labels"].add(a["tail"])
        by_tail[key]["abbrs"].add(a["abbr"])

    conflict_abbr = []
    for k, v in by_abbr.items():
        tails = sorted(by_abbr_tail.get(k, set()))
        if len(v) <= 1 and len(tails) <= 1:
            continue
        g, why = grade_conflict(k, sorted(v), tails)
        conflict_abbr.append({"약칭": k, "등급": g, "등급근거": why,
                              "꼬리명사": tails, "꼬리명사_종수": len(tails),
                              "원표기_전체": sorted(v), "원표기_종수": len(v)})
    order = {"확실": 0, "혼재": 1, "불명": 2}
    conflict_abbr.sort(key=lambda d: (order[d["등급"]], -d["꼬리명사_종수"], d["약칭"]))

    conflict_tail = [{"뜻(꼬리명사)": sorted(v["labels"])[0],
                      "표기_변형": sorted(v["labels"]),
                      "약칭": sorted(v["abbrs"]), "약칭_종수": len(v["abbrs"])}
                     for v in by_tail.values() if len(v["abbrs"]) > 1]
    conflict_tail.sort(key=lambda d: (-d["약칭_종수"], d["뜻(꼬리명사)"]))

    def pct(key):
        v = sorted(l[key] for _, l in lang_rows)
        return {"min": v[0], "median": v[len(v) // 2], "max": v[-1]}

    result = {
        "item": "1-12-1 (6) · 1-14 (5)(6)",
        "version": "v2 (2026-08-28)",
        "measured_at": datetime.now().isoformat(timespec="seconds"),
        "target": str(md_dir),
        "documents": len(files),
        "정의형_약칭": {
            "총_검출": len(abbr_all),
            "검출된_문서수": len({a["file"] for a in abbr_all}),
            "갈래별": {b: len(v) for b, v in buckets.items()},
            "패턴별": dict(Counter(a["pattern"] for a in abbr_all)),
            "신뢰도": "low",
            "신뢰도_근거": "v1에서 오탐이 대량 확인됨(`한다.→창업기업에…`, `에 전원기동→CMOS`, "
                           "`△△△→수탁자`). v2에서 거르개 5종을 넣었으나 '의심' 갈래는 "
                           "육안 확인 전까지 수치로 쓸 수 없다",
            "이름_주의": "'전수'가 아니라 '정의형 약칭 전수'다. 선언 없는 약칭"
                         "(BioIN↔한국보건산업진흥원)은 원리적으로 미검출 — 임현진 2-2-2에 "
                         "이 이름 그대로 넘길 것",
            "기관형": pairs_of(buckets["기관형"]),
            "기술약어형_상위100": pairs_of(buckets["기술약어형"])[:100],
            "의심_상위100": [{"full": a["full"], "abbr": a["abbr"],
                              "reasons": a["reasons"], "file": a["file"]}
                             for a in buckets["의심"]][:100],
            "충돌_처리방침": "걸러내지 않고 전체를 등급 표시와 함께 넘긴다. "
                             "무엇을 쓸지는 이태민 4-9-4에서 정한다 — "
                             "여기서 거르면 남의 항목을 대신 정하는 것이다",
            "충돌_등급정의": {
                "확실": "원표기가 짧고 꼬리 명사가 4종 이하 — 문장 혼입 여지 없음",
                "혼재": "진짜 충돌 + 앞 문장 오탐이 섞임. 꼬리 명사로 판단할 것",
                "불명": "약칭이 코드 형태가 아니거나 꼬리 명사가 한 종류 — 참고만",
            },
            "충돌_갈래별": dict(Counter(d["등급"] for d in conflict_abbr)),
            "충돌_같은약칭_다른뜻": conflict_abbr,
            "충돌_같은뜻_다른약칭": conflict_tail,
        },
        "잔여물_문자": {
            "합계": dict(res_sum),
            "문서수": dict(res_docs),
            "특수불릿_기호별_문서수": dict(bullet_docs),
            "신뢰도": {"제어문자": "high", "보이지않는문자": "high",
                       "연속공백_3칸이상": "high", "연속개행_빈줄3줄이상": "high",
                       "특수불릿": "high", "밑줄점선": "mid"},
            "신뢰도_근거": "밑줄점선은 마크다운 표 구분선을 제외했으나 본문 구분선과 겹칠 여지가 남음",
            "주의": [
                "특수불릿은 제거 대상이 아니다 — 1-12 발견24에서 간트차트·조직도의 위치 정보로 확인(44건)",
                "가운뎃점 `·`는 불릿이 아니라 한국어 문장부호다(`사업명·기간·예산`). "
                "v1이 이걸 불릿으로 세어 5만 건이 나왔다. 참고 수치로만 둔다",
            ],
        },
        "언어_구성": {
            "한글비율": pct("한글비율"),
            "영문비율": pct("영문비율"),
            "영문비율_보정": pct("영문비율_보정"),
            "한자비율": pct("한자비율"),
            "신뢰도": "high",
            "신뢰도_근거": "HTML 태그 제거 후 계산. 보정값은 이미지 마크업·URL·파일확장자까지 제거",
            "보정_이유": "v1에서 image(1,196)·images(597)·png·bmp·www·http가 영문으로 세어졌다. "
                         "본문 내용이 아니라 변환 잔여물이다",
            "제거_금지": "B0·B-·BBB-·BB- 는 신용등급 표기다. 참가자격에 걸리는 값이라 지우면 안 된다",
            "영문토큰_고유수": len(tok_all),
            "영문토큰_상위100": [{"token": t, "n": n} for t, n in tok_all.most_common(100)],
        },
        "per_document": [{**d, "약칭": dict(d["약칭"])} for d in per_doc],
    }

    (out_dir / "measurement_1-12-1_1-14.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")

    # ── 사람이 읽는 요약 ──────────────────────────────────
    A = result["정의형_약칭"]
    L = ["# 1-12-1 (6) · 1-14 (5)(6) 측정  [v3]",
         f"- 대상 {len(files)}건 · {result['measured_at']}",
         "",
         "## 정의형 약칭 (신뢰도 low)",
         f"- 총 {A['총_검출']}건 · 갈래별 {A['갈래별']}",
         "- 주의: '전수'가 아니라 '정의형 약칭 전수'. 선언 없는 약칭은 미검출",
         "",
         f"### 기관형 {len(A['기관형'])}쌍  → 임현진 2-2-2", ""]
    if A["기관형"]:
        L += ["| 원표기 | 약칭 | 건수 |", "| --- | --- | --- |"]
        L += [f"| {d['full']} | {d['abbr']} | {d['n']} |" for d in A["기관형"][:40]]
    else:
        L += ["(0건)"]

    L += ["", f"### 기술약어형 {len(buckets['기술약어형'])}건  → 이태민 4-9-4", "",
          "| 원표기 | 약칭 | 건수 |", "| --- | --- | --- |"]
    L += [f"| {d['full']} | {d['abbr']} | {d['n']} |"
          for d in A["기술약어형_상위100"][:30]]
    L += ["", "- 원표기에 앞 문장이 붙은 것이 있다. 지우지 않고 "
          "꼬리 명사(정리형)를 충돌 표에 함께 실었다"]

    if conflict_abbr:
        L += ["", "#### 같은 약칭인데 뜻이 여럿 — 전체 "
              f"{len(conflict_abbr)}건 · 갈래별 "
              f"{dict(Counter(d['등급'] for d in conflict_abbr))}",
              "",
              "- 걸러내지 않았다. 무엇을 쓸지는 4-9-4에서 정한다",
              "- 확실 = 원표기가 짧고 꼬리 명사 4종 이하 / 혼재 = 앞 문장 오탐 섞임 / 불명 = 참고만",
              "",
              "| 등급 | 약칭 | 꼬리 명사(정리형) | 원표기 종수 | 등급 근거 |",
              "| --- | --- | --- | ---: | --- |"]
        for d in conflict_abbr:
            L.append(f"| {d['등급']} | {d['약칭']} | {' / '.join(d['꼬리명사'])} "
                     f"| {d['원표기_종수']} | {d['등급근거']} |")
    if conflict_tail:
        L += ["", f"#### 같은 뜻인데 약칭이 여럿 — 전체 {len(conflict_tail)}건", "",
              "| 뜻(꼬리 명사) | 약칭 |", "| --- | --- |"]
        L += [f"| {d['뜻(꼬리명사)']} | {' / '.join(d['약칭'])} |" for d in conflict_tail]

    L += ["", f"### 의심 {len(buckets['의심'])}건 — 육안 확인 필요", "",
          "| 원표기 | 약칭 | 이유 |", "| --- | --- | --- |"]
    L += [f"| {a['full']} | {a['abbr']} | {', '.join(a['reasons'])} |"
          for a in buckets["의심"][:25]]

    L += ["", "## 잔여물 문자 4종", "",
          "| 항목 | 총 출현 | 문서수 |", "| --- | ---: | ---: |"]
    L += [f"| {k} | {v:,} | {res_docs.get(k, 0)} |" for k, v in res_sum.items()]
    L += ["", f"- 특수불릿 기호별 문서수: {dict(bullet_docs)}",
          "- 주의: 특수불릿은 제거 대상이 아니다 (1-12 발견24 — 간트차트·조직도 44건)",
          "- 가운뎃점 `·`는 불릿이 아니라 문장부호라 별도 집계(참고_가운뎃점)"]

    L += ["", "## 언어 구성 (신뢰도 high)", "",
          "| | min | median | max |", "| --- | ---: | ---: | ---: |"]
    for k in ("한글비율", "영문비율", "영문비율_보정", "한자비율"):
        d = result["언어_구성"][k]
        L.append(f"| {k} | {d['min']} | {d['median']} | {d['max']} |")
    L += ["", "- 보정 = 이미지 마크업·URL·파일확장자 제거 후",
          "- 제거 금지: B0 · B- · BBB- · BB- 는 신용등급 표기(참가자격에 걸림)",
          "", f"- 영문 토큰 고유 {len(tok_all)}종", "",
          "| 토큰 | 건수 |", "| --- | ---: |"]
    L += [f"| {t} | {n} |" for t, n in tok_all.most_common(40)]

    (out_dir / "measurement_1-12-1_1-14.md").write_text("\n".join(L), encoding="utf-8")

    print("\n".join(L[:70]))
    print(f"\n[완료] {out_dir}/measurement_1-12-1_1-14.json")
    print(f"[완료] {out_dir}/measurement_1-12-1_1-14.md")


if __name__ == "__main__":
    main()
