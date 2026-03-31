"""
Document Layer KG Builder

추출된 메타데이터와 cross-reference로 NetworkX 기반 Knowledge Graph를 구축한다.

노드: 문서 (REG, RG, SRP, DSRS) + GDC 엔티티
엣지: 문서 간 관계 (스키마 제약)
"""

import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import networkx as nx

from src.document_layer.classifier import DocumentMeta


# ─── 엣지 관계 스키마 ──────────────────────────────────
# (source_type, target_type) → 허용 관계 목록
RELATION_SCHEMA = {
    # REG → others
    ("REG", "RG"):   "ESTABLISHES_REQUIREMENT",
    ("REG", "SRP"):  "ESTABLISHES_CRITERIA",
    ("REG", "DSRS"): "ESTABLISHES_CRITERIA",

    # GDC → others (GDC는 REG의 하위 엔티티)
    ("GDC", "RG"):   "PROVIDES_GUIDANCE",
    ("GDC", "SRP"):  "IMPOSES_REQUIREMENT",
    ("GDC", "DSRS"): "IMPOSES_REQUIREMENT",

    # RG → SRP/DSRS
    ("SRP", "RG"):   "ACCEPTED_METHOD",
    ("DSRS", "RG"):  "ACCEPTED_METHOD",

    # SRP ↔ DSRS
    ("SRP", "DSRS"): "ADAPTED_FOR_NUSCALE",
    ("DSRS", "SRP"): "BASED_ON",

    # 같은 유형 간
    ("SRP", "SRP"):   "CROSS_REFERENCES",
    ("DSRS", "DSRS"): "CROSS_REFERENCES",
    ("RG", "RG"):     "CROSS_REFERENCES",
}


class DocumentLayerKG:
    """
    Document Layer Knowledge Graph.

    노드:
      - doc:<doc_id>  (REG, RG, SRP, DSRS 문서)
      - gdc:<number>  (GDC 엔티티)

    엣지:
      - 스키마 제약에 따른 관계
    """

    def __init__(self):
        self.graph = nx.DiGraph()
        self.docs: dict[str, DocumentMeta] = {}      # doc_id → DocumentMeta
        self.section_index: dict[str, list[str]] = defaultdict(list)  # (type, section) → [doc_id]

    # ─── 구축 ─────────────────────────────────────────
    def build(self, docs: list[DocumentMeta]) -> None:
        """메타데이터 리스트로부터 KG를 구축한다."""
        self._add_document_nodes(docs)
        self._add_gdc_nodes(docs)
        self._add_edges_from_crossrefs(docs)
        self._add_srp_dsrs_section_edges()

        print(f"\n[DocumentLayerKG] Built: {self.graph.number_of_nodes()} nodes, "
              f"{self.graph.number_of_edges()} edges")

    def _add_document_nodes(self, docs: list[DocumentMeta]) -> None:
        """문서 노드를 추가한다."""
        for doc in docs:
            node_id = f"doc:{doc.doc_id}"
            self.docs[doc.doc_id] = doc
            self.graph.add_node(
                node_id,
                doc_type=doc.doc_type,
                section_number=doc.section_number,
                title=doc.title,
                display_name=doc.display_name,
                filename=doc.filename,
                pdf_path=doc.pdf_path,
                page_count=doc.page_count,
                node_kind="document",
            )
            # 섹션 인덱스 구축 (같은 섹션 번호의 SRP↔DSRS 매칭용)
            key = f"{doc.doc_type}:{doc.section_number}"
            self.section_index[key].append(doc.doc_id)

    def _add_gdc_nodes(self, docs: list[DocumentMeta]) -> None:
        """모든 문서에서 참조된 GDC를 엔티티 노드로 추가한다."""
        all_gdcs = set()
        for doc in docs:
            all_gdcs.update(doc.gdc_refs)

        for gdc_num in sorted(all_gdcs, key=int):
            node_id = f"gdc:{gdc_num}"
            self.graph.add_node(
                node_id,
                doc_type="GDC",
                section_number=gdc_num,
                title=f"General Design Criterion {gdc_num}",
                display_name=f"GDC {gdc_num}",
                node_kind="entity",
            )
            # GDC → REG 연결 (GDC는 10 CFR 50 Appendix A의 일부)
            reg_nodes = [n for n, d in self.graph.nodes(data=True)
                         if d.get("doc_type") == "REG"]
            for reg_node in reg_nodes:
                self.graph.add_edge(
                    reg_node, node_id,
                    relation="CONTAINS_GDC",
                )

    def _add_edges_from_crossrefs(self, docs: list[DocumentMeta]) -> None:
        """cross-reference 기반으로 엣지를 추가한다."""
        for doc in docs:
            src_node = f"doc:{doc.doc_id}"
            src_type = doc.doc_type

            # GDC 참조 → GDC 엔티티 연결
            for gdc_num in doc.gdc_refs:
                gdc_node = f"gdc:{gdc_num}"
                if self.graph.has_node(gdc_node):
                    # 문서 → GDC: CITES_GDC
                    self.graph.add_edge(src_node, gdc_node, relation="CITES_GDC")
                    # GDC → 문서: 스키마에 따른 관계
                    rel = RELATION_SCHEMA.get(("GDC", src_type))
                    if rel:
                        self.graph.add_edge(gdc_node, src_node, relation=rel)

            # RG 참조 → RG 문서 연결
            for rg_num in doc.rg_refs:
                target_ids = self._find_docs_by_section("RG", rg_num)
                for tid in target_ids:
                    tgt_node = f"doc:{tid}"
                    rel = RELATION_SCHEMA.get((src_type, "RG"), "REFERENCES")
                    self.graph.add_edge(src_node, tgt_node, relation=rel)

            # SRP 참조 → SRP 문서 연결
            for srp_num in doc.srp_refs:
                target_ids = self._find_docs_by_section("SRP", srp_num)
                for tid in target_ids:
                    tgt_node = f"doc:{tid}"
                    rel = RELATION_SCHEMA.get((src_type, "SRP"), "REFERENCES")
                    self.graph.add_edge(src_node, tgt_node, relation=rel)

            # DSRS 참조 → DSRS 문서 연결
            for dsrs_num in doc.dsrs_refs:
                target_ids = self._find_docs_by_section("DSRS", dsrs_num)
                for tid in target_ids:
                    tgt_node = f"doc:{tid}"
                    rel = RELATION_SCHEMA.get((src_type, "DSRS"), "REFERENCES")
                    self.graph.add_edge(src_node, tgt_node, relation=rel)

            # 10 CFR 참조 → REG 연결
            if doc.cfr_refs and doc.doc_type != "REG":
                reg_nodes = [n for n, d in self.graph.nodes(data=True)
                             if d.get("doc_type") == "REG"]
                for reg_node in reg_nodes:
                    rel = RELATION_SCHEMA.get(("REG", src_type))
                    if rel:
                        self.graph.add_edge(reg_node, src_node, relation=rel)
                        # 역방향
                        self.graph.add_edge(
                            src_node, reg_node, relation="CITES_REGULATION",
                        )

    def _add_srp_dsrs_section_edges(self) -> None:
        """같은 섹션번호의 SRP↔DSRS를 ADAPTED_FOR_NUSCALE로 연결한다."""
        dsrs_sections = {
            doc.section_number: doc.doc_id
            for doc in self.docs.values()
            if doc.doc_type == "DSRS"
        }
        srp_sections = {
            doc.section_number: doc.doc_id
            for doc in self.docs.values()
            if doc.doc_type == "SRP"
        }

        matched = 0
        for section, dsrs_id in dsrs_sections.items():
            if section in srp_sections:
                srp_id = srp_sections[section]
                src = f"doc:{srp_id}"
                tgt = f"doc:{dsrs_id}"
                if not self.graph.has_edge(src, tgt):
                    self.graph.add_edge(src, tgt, relation="ADAPTED_FOR_NUSCALE")
                if not self.graph.has_edge(tgt, src):
                    self.graph.add_edge(tgt, src, relation="BASED_ON")
                matched += 1

        print(f"[KG] SRP↔DSRS section match: {matched} pairs")

    def _find_docs_by_section(self, doc_type: str, section: str) -> list[str]:
        """섹션번호로 문서를 찾는다."""
        key = f"{doc_type}:{section}"
        return self.section_index.get(key, [])

    # ─── 쿼리 ─────────────────────────────────────────
    def get_related_docs(self, doc_id: str, max_hops: int = 2) -> list[dict]:
        """특정 문서에서 max_hops 내의 관련 문서를 반환한다."""
        start = f"doc:{doc_id}"
        if not self.graph.has_node(start):
            return []

        visited = set()
        results = []
        queue = [(start, 0, "")]

        while queue:
            node, hop, path = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)

            if hop > 0 and self.graph.nodes[node].get("node_kind") == "document":
                results.append({
                    "doc_id": node.replace("doc:", ""),
                    "display_name": self.graph.nodes[node].get("display_name", ""),
                    "hop": hop,
                    "path": path,
                })

            if hop < max_hops:
                for neighbor in self.graph.neighbors(node):
                    edge_data = self.graph.edges[node, neighbor]
                    rel = edge_data.get("relation", "")
                    new_path = f"{path} → {rel} → {self.graph.nodes[neighbor].get('display_name', neighbor)}"
                    queue.append((neighbor, hop + 1, new_path.strip(" → ")))

        return results

    def find_docs_by_keyword(self, keyword: str) -> list[str]:
        """키워드로 문서 노드를 검색한다 (제목, 섹션번호 매칭)."""
        keyword_lower = keyword.lower()
        results = []
        for node, data in self.graph.nodes(data=True):
            if data.get("node_kind") != "document":
                continue
            title = data.get("title", "").lower()
            section = data.get("section_number", "").lower()
            display = data.get("display_name", "").lower()
            if (keyword_lower in title or keyword_lower in section
                    or keyword_lower in display):
                results.append(node.replace("doc:", ""))
        return results

    def route_query(self, query: str, max_docs: int = 10,
                    use_decomposition: bool = True) -> list[str]:
        """
        KAG-style 4-retriever + LLM decomposition + merger 기반 문서 라우팅.

        1. [optional] LLM query decomposition → sub-queries + GDC/chapter hints
        2. kg_cs: 명시적 참조 매칭 → KG 1-2홉
        3. kg_fr: 쿼리 키워드 fuzzy 매칭 → PPR 전파
        4. rc:    BM25 문서 제목 검색
        5. rc_embed: 임베딩 코사인 유사도 검색
        6. merger: min-max 정규화 → 가중 합산 → top-k
        """
        if not hasattr(self, '_retrievers_initialized'):
            self._init_retrievers()

        pool_size = max(max_docs * 5, 50)

        # ── Query Decomposition ───────────────────────
        queries_to_run = [query]
        synthetic_cs = []

        if use_decomposition and self._decomposer is not None:
            decomposed = self._decomposer.decompose(query)
            if decomposed.sub_queries:
                queries_to_run = [query] + decomposed.sub_queries
            # likely_gdcs → synthetic kg_cs results
            for gdc_num in decomposed.likely_gdcs:
                for doc_id, meta in self.docs.items():
                    if gdc_num in meta.gdc_refs:
                        synthetic_cs.append((doc_id, 0.8))

        # ── Run 4 retrievers on all queries ───────────
        from collections import defaultdict
        all_cs = list(synthetic_cs)
        all_fr = []
        all_rc = []
        all_embed = []

        for q in queries_to_run:
            all_cs.extend(self._kg_cs.search(q, top_k=pool_size))
            all_fr.extend(self._kg_fr.search(q, top_k=pool_size))
            all_rc.extend(self._rc.search(q, top_k=pool_size))
            all_embed.extend(self._rc_embed.search(q, top_k=pool_size))

        # 중복 doc_id는 최대 점수 유지
        def dedup(results):
            best = {}
            for doc_id, score in results:
                if doc_id not in best or score > best[doc_id]:
                    best[doc_id] = score
            return list(best.items())

        merged = self._merger.merge(
            dedup(all_cs), dedup(all_fr), dedup(all_rc),
            rc_embed_results=dedup(all_embed),
            query=query, top_k=max_docs,
        )

        return [doc_id for doc_id, _ in merged]

    def _init_retrievers(self):
        """KAG retrievers를 lazy 초기화한다."""
        from src.document_layer.retriever import (
            KGConstraintRetriever, KGFuzzyPPRRetriever,
            DocBM25Retriever, KAGMerger,
        )
        from src.document_layer.embedding_retriever import DocEmbeddingRetriever

        self._kg_cs = KGConstraintRetriever(self.docs, self.graph, self.section_index)
        self._kg_fr = KGFuzzyPPRRetriever(self.docs, self.graph)
        self._rc = DocBM25Retriever(self.docs, self.graph)
        self._rc_embed = DocEmbeddingRetriever(self.docs, self.graph)
        self._merger = KAGMerger(docs=self.docs)

        try:
            from src.document_layer.query_decomposer import QueryDecomposer
            self._decomposer = QueryDecomposer()
        except Exception:
            self._decomposer = None

        self._retrievers_initialized = True

    # ─── 저장/로드 ────────────────────────────────────
    def save(self, path: str) -> None:
        """KG를 JSON으로 저장한다."""
        data = {
            "nodes": [],
            "edges": [],
        }
        for node, attrs in self.graph.nodes(data=True):
            data["nodes"].append({"id": node, **attrs})
        for src, tgt, attrs in self.graph.edges(data=True):
            data["edges"].append({"source": src, "target": tgt, **attrs})

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[OK] Saved KG to {path}")

    @classmethod
    def load(cls, kg_path: str, meta_path: str) -> "DocumentLayerKG":
        """저장된 KG와 메타데이터를 로드한다."""
        from src.document_layer.classifier import load_metadata

        kg = cls()
        # 메타데이터 로드
        docs = load_metadata(meta_path)
        for doc in docs:
            kg.docs[doc.doc_id] = doc
            key = f"{doc.doc_type}:{doc.section_number}"
            kg.section_index[key].append(doc.doc_id)

        # 그래프 로드
        with open(kg_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for node_data in data["nodes"]:
            nid = node_data.pop("id")
            kg.graph.add_node(nid, **node_data)
        for edge_data in data["edges"]:
            src = edge_data.pop("source")
            tgt = edge_data.pop("target")
            kg.graph.add_edge(src, tgt, **edge_data)

        return kg

    # ─── 통계 ─────────────────────────────────────────
    def stats(self) -> dict:
        """KG 통계를 반환한다."""
        node_types = defaultdict(int)
        for _, data in self.graph.nodes(data=True):
            node_types[data.get("doc_type", "unknown")] += 1

        edge_types = defaultdict(int)
        for _, _, data in self.graph.edges(data=True):
            edge_types[data.get("relation", "unknown")] += 1

        return {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "node_types": dict(node_types),
            "edge_types": dict(edge_types),
        }
