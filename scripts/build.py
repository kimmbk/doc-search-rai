"""
전체 빌드 스크립트: Document Layer KG 구축

Attribute Layer 트리는 /home/bbo/Document_pageindex/pageindex/ 에 사전 구축됨.

Usage:
    cd /home/bbo/doc_search
    /home/bbo/RAG_Baseline/.venv/bin/python scripts/build.py
"""

import sys
sys.path.insert(0, ".")

from src.document_layer.classifier import DocumentClassifier, save_metadata
from src.document_layer.cross_ref import extract_all_cross_refs
from src.document_layer.kg_builder import DocumentLayerKG


DOCS_DIR = "/home/bbo/RAG_Baseline/documents"
META_PATH = "data/doc_metadata.json"
KG_PATH = "data/document_layer_kg.json"


def main():
    print("=" * 60)
    print("Step 1: 문서 분류 + 메타데이터 추출")
    print("=" * 60)
    classifier = DocumentClassifier(DOCS_DIR)
    docs = classifier.classify_all()
    save_metadata(docs, META_PATH)

    print("\n" + "=" * 60)
    print("Step 2: Cross-reference 추출")
    print("=" * 60)
    docs = extract_all_cross_refs(docs)
    save_metadata(docs, META_PATH)

    print("\n" + "=" * 60)
    print("Step 3: Document Layer KG 구축")
    print("=" * 60)
    kg = DocumentLayerKG()
    kg.build(docs)
    kg.save(KG_PATH)
    stats = kg.stats()
    print(f"  Nodes: {stats['total_nodes']}, Edges: {stats['total_edges']}")

    print("\n" + "=" * 60)
    print("Build complete!")
    print(f"  Metadata: {META_PATH}")
    print(f"  KG:       {KG_PATH}")
    print(f"  Trees:    /home/bbo/Document_pageindex/pageindex/ (pre-built)")
    print("=" * 60)


if __name__ == "__main__":
    main()
