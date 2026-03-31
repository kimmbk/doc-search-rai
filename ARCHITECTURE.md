# doc_search RAG Architecture

## Overview

NRC(미국 원자력규제위원회) 규제문서 151개를 대상으로 한 2-Layer RAG 시스템.
KAG(Knowledge Augmented Generation) 논문 기반 Document Layer 라우팅 + PageIndex 기반 Attribute Layer 탐색.

## Pipeline Flow

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────┐
│         LLM Query Decomposition (GPT-4.1)           │
│                                                     │
│  원본 쿼리 → sub-queries 2~4개                       │
│            + likely_gdcs + likely_chapters           │
│  ※ API 실패 시 원본 쿼리 그대로 fallback              │
└──────────────────────┬──────────────────────────────┘
                       │
═══════════════════════════════════════════════════════
              LAYER 1: Document Layer
             (KAG-style 문서 선택)
═══════════════════════════════════════════════════════
                       │
    ┌──────────┬───────┼───────┬──────────┐
    ▼          ▼       ▼       ▼          │
┌────────┐┌────────┐┌──────┐┌─────────┐  │
│ kg_cs  ││ kg_fr  ││  rc  ││rc_embed │  │
│(regex) ││(PPR)   ││(BM25)││(vector) │  │
│        ││        ││      ││         │  │
│GDC/RG/ ││fuzzy   ││title+││OpenAI   │  │
│SRP/DSRS││keyword ││chap- ││text-emb-│  │
│패턴매칭 ││→KG노드 ││ter   ││3-small  │  │
│→KG홉   ││→PPR   ││topic ││cosine   │  │
│→직접매칭││→entity ││+GDC  ││similar- │  │
│        ││travers.││desc  ││ity      │  │
├────────┤├────────┤├──────┤├─────────┤  │
│ w=0.35 ││ w=0.30 ││w=0.15││ w=0.20  │  │
└───┬────┘└───┬────┘└──┬───┘└────┬────┘  │
    └─────────┴────┬───┴─────────┘       │
                   ▼                      │
           ┌──────────────┐               │
           │  KAG Merger  │               │
           │              │               │
           │ 1. min-max   │               │
           │    정규화     │               │
           │ 2. 가중 합산  │               │
           │ 3. multi-src │               │
           │    보너스     │               │
           │ 4. concept   │               │
           │  intersection│               │
           │ (GDC×Chapter)│               │
           └──────┬───────┘               │
                  ▼
           Top-10 Documents
                  │
═══════════════════════════════════════════════════════
              LAYER 2: Attribute Layer
        (PageIndex 트리 기반 섹션 탐색)
═══════════════════════════════════════════════════════
                  │
                  ▼
          문서 트리 로드 (선택된 10개)
          /home/bbo/Document_pageindex/
          pageindex/{doc_id}_structure.json
                  │
        ┌─────────┴─────────┐
        ▼                   ▼
  ┌──────────┐      ┌───────────┐
  │ BM25 검색 │      │ 구조 탐색   │
  │(summary+ │      │(browse/   │
  │ title)   │      │ read)     │
  │ → top-5  │      │ fallback  │
  └─────┬────┘      └─────┬────┘
        └────────┬─────────┘
                 ▼
        Retrieved Sections (5개)
        title, summary, page_range
                 │
         Context 조합 (max 8000 chars)
                 │
═══════════════════════════════════════════════════════
               LLM 답변 생성
═══════════════════════════════════════════════════════
                 │
                 ▼
          ┌──────────────┐
          │   GPT-4.1    │
          │              │
          │ System: NRC  │
          │  regulatory  │
          │  expert      │
          │              │
          │ Context +    │
          │ Question →   │
          │ Answer       │
          └──────┬───────┘
                 │
                 ▼
          ┌──────────────┐
          │  RAG Result  │
          │ - answer     │
          │ - trace      │
          │ - sources    │
          └──────────────┘
```

## File Structure

```
doc_search/
├── src/
│   ├── document_layer/              # LAYER 1
│   │   ├── classifier.py                문서 메타데이터 (DocumentMeta)
│   │   ├── cross_ref.py                 교차참조 추출
│   │   ├── kg_builder.py                KG 구축 + route_query() 오케스트레이션
│   │   ├── retriever.py                 kg_cs + kg_fr + rc(BM25) + KAGMerger
│   │   ├── embedding_retriever.py       rc_embed (벡터 검색)
│   │   └── query_decomposer.py          LLM 쿼리 분해
│   ├── attribute_layer/             # LAYER 2
│   │   └── environment.py               PageIndex 트리 로드 + BM25 섹션 검색
│   └── pipeline/                    # 통합
│       ├── two_layer_search.py          Layer1 + Layer2 연결
│       └── rag_pipeline.py              검색 + LLM 답변 생성
├── data/
│   ├── doc_metadata.json                151개 문서 메타
│   ├── document_layer_kg.json           KG (203 nodes, 1906 edges)
│   ├── doc_embeddings.npy               임베딩 캐시
│   └── doc_ids.json                     임베딩 ID 매핑
└── scripts/
    ├── build.py                         KG 빌드
    ├── build_embeddings.py              임베딩 캐시 빌드
    ├── query.py                         CLI 쿼리
    └── run_eval.py                      베이스라인 호환 평가
```

## Data Flow Summary

| Stage | Input | Output | Role |
|-------|-------|--------|------|
| Query Decomposition | 자연어 쿼리 | sub-queries + GDC/chapter hints | 암묵적 참조 명시화 |
| Layer 1 (4 retrievers) | sub-queries | doc_id × score 리스트 | 관련 문서 선택 |
| KAG Merger | 4개 retriever 결과 | top-10 doc_ids | 점수 합산 + 교차 보너스 |
| Layer 2 (PageIndex) | top-10 doc_ids + query | 5개 섹션 context | 섹션 수준 탐색 |
| LLM Generation | context + query | answer | 최종 답변 |

## Key Design Decisions

1. **KAG-style 4-retriever fusion**: regex(kg_cs) + PPR(kg_fr) + BM25(rc) + embedding(rc_embed) 병렬 실행 후 점수 합산
2. **Concept Intersection Bonus**: 쿼리에서 매칭된 GDC와 챕터의 교집합 문서에 대폭 보너스
3. **LLM Query Decomposition**: 자연어 쿼리를 구조화된 sub-query로 분해하여 검색 정확도 향상
4. **Graceful Fallback**: OpenAI API 실패 시 BM25 + PPR + regex만으로 동작
5. **Attribute Layer PageIndex**: 사전 구축된 페이지인덱스 트리(/home/bbo/Document_pageindex/)로 섹션 탐색

## External Dependencies

- **Documents**: `/home/bbo/RAG_Baseline/documents/` (151 NRC PDFs)
- **PageIndex**: `/home/bbo/Document_pageindex/pageindex/` (152 structure JSONs)
- **QA Dataset**: `/home/bbo/RAG_Baseline/qadataset/rag_qa_dataset_final.json` (571 queries)
- **Evaluation**: `/home/bbo/RAG_Baseline/04_evaluate.py` (9 metrics)
- **Python venv**: `/home/bbo/RAG_Baseline/.venv/`
- **LLM**: OpenAI GPT-4.1 (generation + decomposition)
- **Embedding**: OpenAI text-embedding-3-small (1536 dims)
