"""
Embedding-based Document Retriever

OpenAI text-embedding-3-small로 문서 벡터화 → 코사인 유사도 검색.
BM25의 키워드 한계를 보완하여 의미 기반 매칭 제공.
"""

import json
import os
from pathlib import Path

import numpy as np

from src.document_layer.retriever import DocBM25Retriever, GDC_DESCRIPTIONS

# 캐시 경로
CACHE_DIR = Path("data")
EMBED_PATH = CACHE_DIR / "doc_embeddings.npy"
DOC_IDS_PATH = CACHE_DIR / "doc_ids.json"

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536


class DocEmbeddingRetriever:
    """OpenAI embedding 기반 문서 검색."""

    def __init__(self, docs: dict, graph=None):
        self.docs = docs
        self._graph = graph
        self._doc_ids: list[str] = []
        self._embeddings: np.ndarray | None = None
        self._client = None
        self._ready = False

        self._load_or_build()

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
                from dotenv import load_dotenv
                load_dotenv("/home/bbo/RAG_Baseline/.env")
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    return None
                self._client = OpenAI(api_key=api_key)
            except Exception:
                return None
        return self._client

    def _load_or_build(self):
        """캐시에서 로드하거나 없으면 빌드."""
        if EMBED_PATH.exists() and DOC_IDS_PATH.exists():
            try:
                self._embeddings = np.load(EMBED_PATH)
                with open(DOC_IDS_PATH, 'r') as f:
                    self._doc_ids = json.load(f)
                if len(self._doc_ids) == self._embeddings.shape[0]:
                    self._ready = True
                    return
            except Exception:
                pass

        # 캐시 없음 → 빌드 시도
        self._build_embeddings()

    def _build_embeddings(self):
        """모든 문서의 enriched text를 임베딩하여 캐시에 저장."""
        client = self._get_client()
        if client is None:
            return

        doc_ids = []
        texts = []
        for doc_id, meta in self.docs.items():
            text = self._build_doc_text(meta)
            doc_ids.append(doc_id)
            texts.append(text)

        if not texts:
            return

        try:
            # 배치 임베딩 (OpenAI는 한 번에 최대 2048개 지원)
            response = client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=texts,
            )
            embeddings = np.array(
                [item.embedding for item in response.data],
                dtype=np.float32,
            )

            # 캐시 저장
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            np.save(EMBED_PATH, embeddings)
            with open(DOC_IDS_PATH, 'w') as f:
                json.dump(doc_ids, f)

            self._doc_ids = doc_ids
            self._embeddings = embeddings
            self._ready = True
            print(f"[embedding] Built and cached {len(doc_ids)} doc embeddings")

        except Exception as e:
            print(f"[embedding] Build failed: {e}")

    def _build_doc_text(self, meta) -> str:
        """BM25와 동일한 enriched text 생성."""
        text = f"{meta.title} {meta.display_name} {meta.doc_type} {meta.section_number}"

        chapter = meta.section_number.split(".")[0] if meta.section_number else ""
        chapter_topic = DocBM25Retriever.CHAPTER_TOPICS.get(chapter, "")
        if chapter_topic:
            text += f" {chapter_topic}"

        for gdc in meta.gdc_refs:
            desc = GDC_DESCRIPTIONS.get(gdc, "")
            text += f" GDC {gdc} general design criterion {gdc} {desc}"

        for ref in meta.rg_refs:
            text += f" regulatory guide {ref}"
        for ref in meta.srp_refs:
            text += f" SRP {ref}"

        return text

    def search(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        """쿼리 임베딩 → 코사인 유사도 → top-k 문서."""
        if not self._ready:
            return []

        query_vec = self._embed_query(query)
        if query_vec is None:
            return []

        # 코사인 유사도 (OpenAI 임베딩은 이미 정규화됨)
        scores = self._embeddings @ query_vec

        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score > 0:
                results.append((self._doc_ids[idx], score))

        return results

    def _embed_query(self, query: str) -> np.ndarray | None:
        """단일 쿼리 임베딩."""
        client = self._get_client()
        if client is None:
            return None

        try:
            response = client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=[query],
            )
            return np.array(response.data[0].embedding, dtype=np.float32)
        except Exception:
            return None
