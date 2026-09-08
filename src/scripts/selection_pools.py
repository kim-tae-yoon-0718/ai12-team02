import json
import re
import pandas as pd

"""
선별형 문항 배분을 위한, 데이터 밀집도 확인
"""
RAG_ROOT = "/srv/rfp"
EXTRACTION_CSV = f"{RAG_ROOT}/shared_data/processed/rfp_extraction_table_v5/extraction_table_v5.csv"
OUT_PATH = f"{RAG_ROOT}/evalset/v1/_work/selection_pools.json"

# 수정: 중복만 제외. practice 문서도 제외하지 않음
EXCLUDE_IDS = ["RFP-000006", "RFP-000017"]

df = pd.read_csv(EXTRACTION_CSV)
df = df[~df["document_id"].isin(EXCLUDE_IDS)]
wide = df.pivot(index="document_id", columns="field_name", values="status")

def find_field(key):
    key_norm = key.replace(" ", "")
    matches = [c for c in wide.columns if key_norm in c.replace(" ", "")]
    assert len(matches) == 1, f"'{key}' 매칭 실패: {matches}"
    return matches[0]

# 수정: "컨소시엄요건 = value_present"는 "조항이 있다"는 뜻일 뿐, "컨소시엄을 허용/요구한다"는
# 뜻이 아님(불허 조항도 value_present로 잡힘). 조항 본문에서 불허 표현을 찾아
# 실제로 공동수급을 허용/요구하는 문서만 별도로 구분한다.
consortium_col = find_field("컨소시엄요건")
consortium_raw = df[df["field_name"] == consortium_col].set_index("document_id")["answer_raw"]
DISALLOW_PAT = re.compile(r"불허|불가|금지|허용하지\s*않|허용되지\s*않|제외한\s*단독|단독입찰로\s*진행")

def consortium_required(doc_id):
    if wide.loc[doc_id, consortium_col] != "value_present":
        return False
    text = consortium_raw.get(doc_id, "") or ""
    return not DISALLOW_PAT.search(text)

wide["__컨소시엄요구__"] = [consortium_required(d) for d in wide.index]

pools = {}

for f in wide.columns:
    if f == "__컨소시엄요구__":
        continue
    pools[f"single_present__{f}"] = sorted(wide.index[wide[f] == "value_present"].tolist())
    pools[f"negative_absent__{f}"] = sorted(wide.index[wide[f] == "field_absent"].tolist())

# "명시"(조항 존재)와 "요구"(실제 허용/요구)를 구분해서 둘 다 별도 풀로 남긴다.
pools["consortium_required__컨소시엄요건"] = sorted(wide.index[wide["__컨소시엄요구__"]].tolist())
pools["consortium_mentioned_but_not_required__컨소시엄요건"] = sorted(
    wide.index[(wide[consortium_col] == "value_present") & (~wide["__컨소시엄요구__"])].tolist()
)

combos = {
    "compound_1__제출방식absent_필수제출서류absent": {"제출방식": "field_absent", "필수제출서류": "field_absent"},
    "compound_2__참가자격present_컨소시엄absent":   {"참가자격": "value_present", "컨소시엄요건": "field_absent"},
    "compound_3__지역제한present_컨소시엄요구":      {"지역제한": "value_present", "__컨소시엄요구__": True},
    "compound_4__제출방식extref_컨소시엄요구":       {"제출방식": "external_reference", "__컨소시엄요구__": True},
    "compound_5__컨소시엄present_평가배점absent":    {"컨소시엄요건": "value_present", "평가배점": "field_absent"},
    "zero_1__지역제한present_컨소시엄absent":        {"지역제한": "value_present", "컨소시엄요건": "field_absent"},
}
for name, cond in combos.items():
    mask = pd.Series(True, index=wide.index)
    for key, status in cond.items():
        col = key if key == "__컨소시엄요구__" else find_field(key)
        mask &= (wide[col] == status)
    pools[name] = sorted(wide.index[mask].tolist())

for k, v in pools.items():
    print(k, len(v))

with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(pools, f, ensure_ascii=False, indent=2)
