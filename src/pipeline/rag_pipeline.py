"""
RAG Pipeline: 2-Layer Search + LLM Answer Generation

Query → Layer 1 (Document Layer KG 라우팅)
     → Layer 2 (Attribute Layer BM25 검색)
     → Context + Query → LLM → Answer

Usage:
    from src.pipeline.rag_pipeline import RAGPipeline

    rag = RAGPipeline()
    result = rag.ask("What are the NRC acceptance criteria for containment heat removal?")
    print(result.answer)
"""

import os
from dataclasses import dataclass, field

from openai import OpenAI
from dotenv import load_dotenv

from src.pipeline.two_layer_search import TwoLayerSearchPipeline, SearchResult, build_pipeline

load_dotenv("/home/bbo/RAG_Baseline/.env")


SYSTEM_PROMPT = """You are an expert on U.S. Nuclear Regulatory Commission (NRC) regulations and review standards.

Answer the user's question using ONLY the retrieved context provided below.
- If the context contains the answer, provide a clear, specific response with regulatory references.
- If the context is insufficient, say so honestly and explain what information is missing.
- Be precise and cite specific sections, criteria, or regulatory guides when possible.
- Answer in English."""

USER_PROMPT = """Retrieved Context:
{context}

Question: {query}

Answer based on the context above:"""


@dataclass
class RAGResult:
    """RAG 파이프라인 최종 결과"""
    query: str
    answer: str = ""
    # 검색 결과
    search: SearchResult = None
    # LLM 토큰 사용량
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # 디버그
    trace: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "answer": self.answer,
            "n_docs": len(self.search.selected_docs) if self.search else 0,
            "n_sections": len(self.search.retrieved_sections) if self.search else 0,
            "context_len": len(self.search.context) if self.search else 0,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "trace": self.trace,
        }


class RAGPipeline:
    """
    2-Layer Search + LLM 답변 생성 RAG 파이프라인.

    Layer 1: Document Layer KG → 관련 문서 선택
    Layer 2: Attribute Layer PageIndex → 섹션 검색
    LLM:     검색 컨텍스트 기반 답변 생성
    """

    def __init__(
        self,
        model: str = "gpt-4.1",
        search_pipeline: TwoLayerSearchPipeline = None,
        system_prompt: str = SYSTEM_PROMPT,
        max_tokens: int = 1000,
        temperature: float = 0.2,
    ):
        self.model = model
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.search_pipeline = search_pipeline or build_pipeline()
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.temperature = temperature

    def ask(self, query: str) -> RAGResult:
        """
        질문 → 검색 → 답변 생성.

        Args:
            query: 사용자 질문

        Returns:
            RAGResult with answer, search results, and trace
        """
        result = RAGResult(query=query)

        # ── 검색 ──────────────────────────────────────
        search_result = self.search_pipeline.search(query)
        result.search = search_result
        result.trace.extend(search_result.trace)

        if not search_result.context.strip():
            result.answer = "검색 결과가 없습니다. 다른 질문을 시도해주세요."
            result.trace.append("[LLM] Skipped — no context.")
            return result

        # ── LLM 답변 생성 ─────────────────────────────
        result.trace.append(f"[LLM] Generating answer with {self.model}...")

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": USER_PROMPT.format(
                        context=search_result.context, query=query,
                    )},
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            result.answer = response.choices[0].message.content.strip()
            result.prompt_tokens = response.usage.prompt_tokens
            result.completion_tokens = response.usage.completion_tokens
            result.trace.append(
                f"[LLM] Done. {result.prompt_tokens}+{result.completion_tokens} tokens."
            )
        except Exception as e:
            result.answer = f"[LLM ERROR] {str(e)}"
            result.trace.append(f"[LLM] Error: {str(e)}")

        return result
