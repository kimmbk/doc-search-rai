"""
Two-Layer Search Pipeline

Query → Document Layer KG (문서 선택) → Attribute Layer (섹션 탐색) → 컨텍스트 수집

Layer 1: Document Layer KG에서 쿼리와 관련된 문서 N개 선택
Layer 2: 선택된 문서들의 PageIndex 트리에서 BM25 검색 + 구조 탐색
"""

import json
import os
from dataclasses import dataclass, field
from typing import Optional

from src.document_layer.kg_builder import DocumentLayerKG
from src.attribute_layer.environment import AttributeLayerEnv

PAGEINDEX_DIR = "/home/bbo/Document_pageindex/pageindex"


@dataclass
class SearchResult:
    """검색 결과"""
    query: str
    selected_docs: list[dict] = field(default_factory=list)
    retrieved_sections: list[dict] = field(default_factory=list)
    context: str = ""
    trace: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "selected_docs": self.selected_docs,
            "retrieved_sections": [
                {k: v for k, v in s.items() if k != "content"}
                for s in self.retrieved_sections
            ],
            "context_length": len(self.context),
            "trace": self.trace,
        }


class TwoLayerSearchPipeline:
    """
    2-Layer 검색 파이프라인.

    1. Document Layer: KG 기반 쿼리 라우팅 → 관련 문서 선택
    2. Attribute Layer: 선택된 문서 내 BM25 + 구조 탐색 → 컨텍스트 수집
    """

    def __init__(
        self,
        kg: DocumentLayerKG,
        trees_dir: str,
        max_docs: int = 10,
        max_sections: int = 5,
        max_context_chars: int = 8000,
    ):
        self.kg = kg
        self.trees_dir = trees_dir
        self.env = AttributeLayerEnv(trees_dir)
        self.max_docs = max_docs
        self.max_sections = max_sections
        self.max_context_chars = max_context_chars

    def search(self, query: str) -> SearchResult:
        """
        2-Layer 검색을 실행한다.

        Args:
            query: 사용자 질의

        Returns:
            SearchResult with context and trace
        """
        result = SearchResult(query=query)

        # ── Layer 1: Document Layer KG 라우팅 ────────
        result.trace.append("[Layer 1] Document Layer KG routing...")

        doc_ids = self.kg.route_query(query, max_docs=self.max_docs)

        if not doc_ids:
            result.trace.append("[Layer 1] No documents matched.")
            return result

        for doc_id in doc_ids:
            doc_meta = self.kg.docs.get(doc_id)
            if doc_meta:
                result.selected_docs.append({
                    "doc_id": doc_id,
                    "display_name": doc_meta.display_name,
                    "title": doc_meta.title,
                    "doc_type": doc_meta.doc_type,
                })

        result.trace.append(
            f"[Layer 1] Selected {len(doc_ids)} docs: "
            + ", ".join(d["display_name"] for d in result.selected_docs[:5])
            + ("..." if len(result.selected_docs) > 5 else "")
        )

        # ── 트리 존재 여부 확인 ─────────────────────
        available_ids = []
        for doc_id in doc_ids:
            tree_path = os.path.join(self.trees_dir, f"{doc_id}_structure.json")
            if os.path.exists(tree_path):
                available_ids.append(doc_id)

        if not available_ids:
            result.trace.append("[Layer 2] No trees available for selected docs.")
            # 트리 없이도 Document Layer 정보로 컨텍스트 구성
            result.context = self._build_doc_only_context(doc_ids)
            return result

        # ── Layer 2: Attribute Layer 탐색 ────────────
        result.trace.append(f"[Layer 2] Loading {len(available_ids)} document trees...")

        self.env.load_documents(available_ids)
        result.trace.append(f"[Layer 2] {self.env.node_count} nodes indexed.")

        # BM25 검색
        search_results = self.env.search(query, max_results=self.max_sections)

        if search_results:
            result.trace.append(
                f"[Layer 2] Search found {len(search_results)} sections."
            )
        else:
            result.trace.append("[Layer 2] No search results, browsing top-level.")

        # 검색 결과 읽기
        seen = set()
        for sr in search_results:
            key = f"{sr['doc_id']}::{sr['node_id']}"
            if key in seen:
                continue
            seen.add(key)

            node_data = self.env.read(sr["doc_id"], sr["node_id"])
            if node_data and node_data.get("content"):
                result.retrieved_sections.append(node_data)

        # 검색 결과 부족 시 browse로 보완
        if len(result.retrieved_sections) < 2:
            for doc_id in available_ids[:3]:
                items = self.env.browse(doc_id, None, depth=1)
                for item in items[:2]:
                    key = f"{item['doc_id']}::{item['node_id']}"
                    if key in seen:
                        continue
                    seen.add(key)
                    node_data = self.env.read(item["doc_id"], item["node_id"])
                    if node_data and node_data.get("content"):
                        result.retrieved_sections.append(node_data)

        result.trace.append(
            f"[Layer 2] Retrieved {len(result.retrieved_sections)} sections total."
        )

        # ── 컨텍스트 조합 ────────────────────────────
        result.context = self._build_context(query, result)
        result.trace.append(
            f"[Done] Context: {len(result.context)} chars from "
            f"{len(result.selected_docs)} docs, "
            f"{len(result.retrieved_sections)} sections."
        )

        return result

    def _build_context(self, query: str, result: SearchResult) -> str:
        """검색 결과를 LLM 프롬프트용 컨텍스트로 조합한다."""
        parts = []
        parts.append(f"=== Retrieved Context for: {query} ===\n")

        # Document Layer 정보
        parts.append("--- Selected Documents ---")
        for d in result.selected_docs:
            parts.append(f"  [{d['doc_type']}] {d['display_name']}: {d['title']}")

        parts.append("\n--- Retrieved Sections ---")
        total_chars = 0
        for section in result.retrieved_sections:
            if total_chars >= self.max_context_chars:
                parts.append("\n(context limit reached)")
                break

            content = section.get("content", "")
            remaining = self.max_context_chars - total_chars
            if len(content) > remaining:
                content = content[:remaining] + "..."

            doc_meta = self.kg.docs.get(section["doc_id"])
            doc_name = doc_meta.display_name if doc_meta else section["doc_id"]

            page_range = section.get("page_range", "?")
            parts.append(
                f"\n[{doc_name} > {section['title'][:60]}] "
                f"(p.{page_range})\n"
                f"{content}"
            )
            total_chars += len(content)

        return "\n".join(parts)

    def _build_doc_only_context(self, doc_ids: list[str]) -> str:
        """트리가 없을 때 Document Layer 정보만으로 컨텍스트를 구성한다."""
        parts = ["=== Document Layer Context (no trees available) ===\n"]
        for doc_id in doc_ids:
            meta = self.kg.docs.get(doc_id)
            if meta:
                parts.append(
                    f"[{meta.doc_type}] {meta.display_name}: {meta.title}\n"
                    f"  File: {meta.filename} ({meta.page_count}p)\n"
                    f"  Cross-refs: RG={meta.rg_refs[:3]}, "
                    f"GDC={meta.gdc_refs[:3]}, "
                    f"SRP={meta.srp_refs[:3]}"
                )
        return "\n".join(parts)


def build_pipeline(
    meta_path: str = "data/doc_metadata.json",
    kg_path: str = "data/document_layer_kg.json",
    trees_dir: str = PAGEINDEX_DIR,
    **kwargs,
) -> TwoLayerSearchPipeline:
    """사전 구축된 데이터로 파이프라인을 초기화한다."""
    kg = DocumentLayerKG.load(kg_path, meta_path)
    return TwoLayerSearchPipeline(kg=kg, trees_dir=trees_dir, **kwargs)
