"""
쿼리 경로 시각화: 쿼리가 어떻게 문서에 도달했는지 추적

Usage:
    cd /home/bbo/doc_search
    /home/bbo/RAG_Baseline/.venv/bin/python scripts/visualize_query_path.py "containment heat removal"
    /home/bbo/RAG_Baseline/.venv/bin/python scripts/visualize_query_path.py "natural phenomena geographic scope"
"""

import json
import sys
import os

sys.path.insert(0, ".")

from pyvis.network import Network

from src.document_layer.kg_builder import DocumentLayerKG
from src.document_layer.retriever import (
    KGConstraintRetriever, KGFuzzyPPRRetriever,
    DocBM25Retriever, KAGMerger, GDC_DESCRIPTIONS,
)
from src.document_layer.embedding_retriever import DocEmbeddingRetriever
from src.document_layer.query_decomposer import QueryDecomposer


# 색상 설정
COLORS = {
    "query": "#FF6B6B",
    "sub_query": "#FFA07A",
    "decomposer": "#FFD93D",
    "kg_cs": "#4ECDC4",
    "kg_fr": "#45B7D1",
    "rc": "#96CEB4",
    "rc_embed": "#DDA0DD",
    "merger": "#F7DC6F",
    "GDC": "#FF8C00",
    "REG": "#8B0000",
    "SRP": "#2E86C1",
    "DSRS": "#27AE60",
    "RG": "#8E44AD",
    "selected": "#FFD700",
    "evidence": "#FF1744",
}


def build_visualization(query: str, evidence_docs: list[str] = None):
    kg = DocumentLayerKG.load("data/document_layer_kg.json", "data/doc_metadata.json")

    # ── Initialize retrievers ────────────────────
    cs = KGConstraintRetriever(kg.docs, kg.graph, kg.section_index)
    fr = KGFuzzyPPRRetriever(kg.docs, kg.graph)
    rc = DocBM25Retriever(kg.docs, kg.graph)
    embed = DocEmbeddingRetriever(kg.docs, kg.graph)
    decomposer = QueryDecomposer()
    merger = KAGMerger(docs=kg.docs)

    # ── Decompose ────────────────────────────────
    decomposed = decomposer.decompose(query)
    queries_to_run = [query] + (decomposed.sub_queries or [])

    # ── Run each retriever ───────────────────────
    # Track which retriever found which doc
    retriever_hits = {
        "kg_cs": {}, "kg_fr": {}, "rc": {}, "rc_embed": {},
    }

    for q in queries_to_run:
        for name, retriever in [("kg_cs", cs), ("kg_fr", fr), ("rc", rc), ("rc_embed", embed)]:
            results = retriever.search(q, top_k=15)
            for doc_id, score in results:
                if doc_id not in retriever_hits[name] or score > retriever_hits[name][doc_id]:
                    retriever_hits[name][doc_id] = score

    # ── Get final merged result ──────────────────
    final_results = kg.route_query(query, max_docs=10)

    # ── Build pyvis network ──────────────────────
    net = Network(
        height="900px", width="100%",
        bgcolor="#1a1a2e", font_color="white",
        directed=True,
    )
    net.barnes_hut(gravity=-3000, central_gravity=0.3, spring_length=200)

    # -- Query node --
    net.add_node(
        "query", label=query[:60] + "...",
        color=COLORS["query"], size=35, shape="box",
        title=f"Original Query:\n{query}",
        font={"size": 14, "color": "white"},
    )

    # -- Decomposer node --
    if decomposed.sub_queries:
        net.add_node(
            "decomposer", label="LLM Decomposer\n(GPT-4.1)",
            color=COLORS["decomposer"], size=25, shape="diamond",
            title=f"GDCs: {decomposed.likely_gdcs}\nChapters: {decomposed.likely_chapters}\nConcepts: {decomposed.key_concepts}",
            font={"size": 12, "color": "#333"},
        )
        net.add_edge("query", "decomposer", color="#FFD93D", width=2)

        # Sub-query nodes
        for i, sq in enumerate(decomposed.sub_queries):
            sq_id = f"sq_{i}"
            net.add_node(
                sq_id, label=sq[:50],
                color=COLORS["sub_query"], size=18, shape="box",
                title=f"Sub-query {i+1}:\n{sq}",
                font={"size": 10, "color": "white"},
            )
            net.add_edge("decomposer", sq_id, color="#FFA07A", width=1.5)

    # -- Retriever nodes --
    retriever_labels = {
        "kg_cs": "kg_cs\n(Regex→KG)",
        "kg_fr": "kg_fr\n(PPR)",
        "rc": "rc\n(BM25)",
        "rc_embed": "rc_embed\n(Embedding)",
    }
    for name, label in retriever_labels.items():
        net.add_node(
            name, label=label,
            color=COLORS[name], size=22, shape="ellipse",
            title=f"{name}: {len(retriever_hits[name])} docs found",
            font={"size": 11, "color": "white"},
        )
        net.add_edge("query", name, color=COLORS[name], width=1.5, dashes=True)

    # -- Merger node --
    net.add_node(
        "merger", label="KAG Merger\n(min-max + weighted)",
        color=COLORS["merger"], size=25, shape="diamond",
        title="Merges 4 retriever results\n+ concept intersection bonus",
        font={"size": 12, "color": "#333"},
    )
    for name in retriever_labels:
        net.add_edge(name, "merger", color=COLORS[name], width=1.5)

    # -- GDC entity nodes (seeded by kg_fr or decomposer) --
    gdc_nodes_added = set()
    for gdc_num in decomposed.likely_gdcs:
        gdc_node = f"gdc:{gdc_num}"
        if kg.graph.has_node(gdc_node):
            desc = GDC_DESCRIPTIONS.get(gdc_num, "")
            net.add_node(
                gdc_node, label=f"GDC {gdc_num}",
                color=COLORS["GDC"], size=18, shape="hexagon",
                title=f"General Design Criterion {gdc_num}\n{desc}",
                font={"size": 11, "color": "white"},
            )
            gdc_nodes_added.add(gdc_node)
            # Connect decomposer → GDC
            if decomposed.sub_queries:
                net.add_edge("decomposer", gdc_node, color="#FF8C00", width=1.5, label="likely_gdc")

    # -- Document nodes (final top-10) --
    evidence_set = set(evidence_docs or [])
    for rank, doc_id in enumerate(final_results, 1):
        meta = kg.docs.get(doc_id)
        if not meta:
            continue

        is_evidence = doc_id in evidence_set
        doc_type = meta.doc_type

        # Determine which retrievers found this doc
        found_by = []
        for name in ["kg_cs", "kg_fr", "rc", "rc_embed"]:
            if doc_id in retriever_hits[name]:
                found_by.append(f"{name}: {retriever_hits[name][doc_id]:.3f}")

        border_color = COLORS["evidence"] if is_evidence else COLORS.get(doc_type, "#888")
        border_width = 4 if is_evidence else 2

        label = f"#{rank} {meta.display_name}"
        if is_evidence:
            label += "\n★ EVIDENCE"

        net.add_node(
            f"doc:{doc_id}",
            label=label,
            color={
                "background": COLORS.get(doc_type, "#888"),
                "border": border_color,
            },
            size=20 + (10 - rank) * 2,
            shape="box",
            borderWidth=border_width,
            title=f"{meta.display_name}: {meta.title}\n\nFound by:\n" + "\n".join(found_by),
            font={"size": 11, "color": "white"},
        )

        # Connect merger → doc
        net.add_edge("merger", f"doc:{doc_id}", color="#F7DC6F", width=2, label=f"#{rank}")

        # Connect GDC → doc if GDC cited
        for gdc_node in gdc_nodes_added:
            gdc_num = gdc_node.replace("gdc:", "")
            if gdc_num in meta.gdc_refs:
                net.add_edge(
                    gdc_node, f"doc:{doc_id}",
                    color="#FF8C00", width=1, dashes=True,
                    title=f"CITES GDC {gdc_num}",
                )

    # -- Legend --
    legend_html = """
    <div style="position:fixed;top:10px;right:10px;background:rgba(0,0,0,0.8);padding:15px;border-radius:8px;color:white;font-size:12px;z-index:1000;">
        <b>Query Path Visualization</b><br><br>
        <span style="color:#FF6B6B">■</span> Query<br>
        <span style="color:#FFD93D">◆</span> LLM Decomposer<br>
        <span style="color:#FFA07A">■</span> Sub-queries<br>
        <span style="color:#4ECDC4">●</span> kg_cs (Regex→KG)<br>
        <span style="color:#45B7D1">●</span> kg_fr (PPR)<br>
        <span style="color:#96CEB4">●</span> rc (BM25)<br>
        <span style="color:#DDA0DD">●</span> rc_embed (Embedding)<br>
        <span style="color:#FF8C00">⬡</span> GDC Entity<br>
        <span style="color:#F7DC6F">◆</span> KAG Merger<br>
        <span style="color:#2E86C1">■</span> SRP Doc<br>
        <span style="color:#27AE60">■</span> DSRS Doc<br>
        <span style="color:#8E44AD">■</span> RG Doc<br>
        <span style="color:#FF1744">━</span> Evidence Doc (target)<br>
    </div>
    """

    # 파일명: 쿼리 첫 3단어
    import re as _re
    slug = "_".join(_re.sub(r"[^a-z0-9 ]", "", query.lower()).split()[:4])
    output_path = f"data/query_path_{slug}.html"
    net.save_graph(output_path)

    # Inject legend
    with open(output_path, "r") as f:
        html = f.read()
    html = html.replace("</body>", legend_html + "</body>")
    with open(output_path, "w") as f:
        f.write(html)

    print(f"Saved to {output_path}")
    return output_path


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/visualize_query_path.py <query> [evidence_doc_id ...]")
        print("Example: python scripts/visualize_query_path.py \"containment heat removal\" ML15356A267")
        sys.exit(1)

    query = sys.argv[1]
    evidence = sys.argv[2:] if len(sys.argv) > 2 else []
    build_visualization(query, evidence)


if __name__ == "__main__":
    main()
