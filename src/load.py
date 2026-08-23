import json
from pathlib import Path

DATA_DIR  = "/srv/rfp/shared_data/raw/files/"          # 원본 RFP 폴더
META_PATH = "/srv/rfp/shared_data/raw/data_list.csv"   # 메타데이터
OUT_DIR   = "/srv/rfp/shared_data/interim/"            # 중간 산출물 저장위치

DOCS = Path()