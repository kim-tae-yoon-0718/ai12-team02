"""
오픈AI 임베딩 클라이언트.
2026-08-31 트랙 전환: embedding_provider=openai, 모델은 text-embedding-3-small
로 저비용 시작(config.embedding_model). e5 계열과 달리 쿼리/문서 접두어
규약이 없다.

⚠️ 실제 실행 전 확인 필요: base.yaml의 data_egress_confirmed가 true인지
   반드시 확인할 것 — RFP 문서를 외부 API로 보내는 데 대한 팀 확인이 아직
   안 끝났으면 이 클라이언트를 실제로 호출하면 안 된다.
"""
from __future__ import annotations
import os
import time
from typing import Any

from pricing import Usage

try:
    from openai import OpenAI
except ImportError:  # 아직 설치 안 됐을 수 있음
    OpenAI = None  # type: ignore


class EmbeddingClient:
    def __init__(self, cfg: dict[str, Any]):
        if cfg.get("embedding_provider") != "openai":
            raise NotImplementedError(
                "이 클라이언트는 오픈AI 트랙 전용입니다. "
                "로컬 트랙은 김태윤님 코드를 참고하세요."
            )
        if not cfg.get("data_egress_confirmed", False):
            raise RuntimeError(
                "data_egress_confirmed=false — RFP 문서를 외부 API로 보내는 것에 "
                "대한 확인이 아직 안 끝났습니다. base.yaml에서 확인 후 true로 "
                "바꾸기 전까지는 실행하지 않습니다."
            )
        if OpenAI is None:
            raise ImportError(
                "openai 패키지가 없습니다. "
                "pip install openai --break-system-packages"
            )
        if not os.environ.get("OPENAI_API_KEY"):
            # 값을 읽어 보관하지 않는다 — 존재 여부만 확인하고 SDK가 직접 읽게 둔다
            raise RuntimeError("OPENAI_API_KEY 환경변수가 없습니다.")
        self.client = OpenAI()
        model = cfg.get("embedding_model")
        if not model:
            raise RuntimeError(
                "embedding_model이 비어 있습니다. 코드가 조용히 기본값"
                "(text-embedding-3-small 등)을 선택하지 않습니다 — base.yaml에 "
                "사람이 직접 채워야 합니다(message.txt 4번 확정)."
            )
        self.model = model
        self.max_length = cfg.get("embedding_max_length")
        # 문항 단위 사용량 — run_eval이 문항 시작 때마다 reset_usage()로 초기화한다.
        # ⚠️ 2026-09-02 버그 수정: 예전엔 last_query_usage가 클라이언트에 계속
        # 남아 있어서, API를 부르지 않은 다음 문항까지 같은 비용이 복사됐다.
        self.usage = Usage()

    def reset_usage(self) -> None:
        """문항 시작 시 호출 — 이전 문항 사용량이 다음 문항으로 새지 않게."""
        self.usage = Usage()

    def embed_batch(
        self, texts: list[str], batch_size: int = 100, max_retries: int = 3
    ) -> list[list[float]]:
        """텍스트 목록을 배치로 임베딩. 실패 시 지수 백오프로 재시도.

        ⚠️ 2026-09-02 버그 수정: 예전엔 **마지막 배치**의 usage만 기록해서
        인덱싱처럼 배치가 여러 번인 경우 임베딩 토큰·요청 수가 크게 과소
        집계됐다(14,828청크 = 149요청인데 1요청분만 기록). 이제 응답마다
        누적한다 — usage.embedding_requests는 실제 API 요청 수와 같다."""
        vectors: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            for attempt in range(max_retries):
                try:
                    resp = self.client.embeddings.create(
                        model=self.model, input=batch
                    )
                    vectors.extend([d.embedding for d in resp.data])
                    if resp.usage:
                        self.usage.add_embedding(resp.usage.total_tokens)
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    wait = 2**attempt
                    print(f"⚠️  임베딩 실패({e}), {wait}초 후 재시도...")
                    time.sleep(wait)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        """단일 쿼리 임베딩 (검색 시점 사용)."""
        return self.embed_batch([text])[0]
