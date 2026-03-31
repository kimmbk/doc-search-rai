"""
임베딩 캐시 빌드 스크립트

Usage:
    cd /home/bbo/doc_search
    /home/bbo/RAG_Baseline/.venv/bin/python scripts/build_embeddings.py
"""

import sys
sys.path.insert(0, ".")

from src.document_layer.kg_builder import DocumentLayerKG
from src.document_layer.embedding_retriever import DocEmbeddingRetriever, EMBED_PATH


def main():
    print("=" * 60)
    print("Building document embedding cache...")
    print("=" * 60)

    kg = DocumentLayerKG.load("data/document_layer_kg.json", "data/doc_metadata.json")
    print(f"Loaded {len(kg.docs)} docs")

    if EMBED_PATH.exists():
        print(f"Cache already exists at {EMBED_PATH}")
        print("Delete it first to rebuild: rm data/doc_embeddings.npy data/doc_ids.json")
        return

    retriever = DocEmbeddingRetriever(kg.docs, kg.graph)
    if retriever._ready:
        print(f"Done! {EMBED_PATH} ({EMBED_PATH.stat().st_size / 1024:.0f} KB)")
    else:
        print("Failed to build embeddings. Check OpenAI API key.")


if __name__ == "__main__":
    main()
