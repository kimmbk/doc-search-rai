# doc_search RAG Architecture (v2)

## Overview

NRC(미국 원자력규제위원회) 규제문서 151개를 대상으로 한 2-Layer RAG 시스템.
**Layer 1**은 OpenSPG/KAG 프레임워크 기반 Document Layer, **Layer 2**는 PageIndex 기반 Attribute Layer.

v1(자체 구현)에서 v2(KAG 프레임워크 도입)로 전환하는 설계.

### v1 → v2 변경 요약

| 구분 | v1 (현재) | v2 (목표) |
|------|-----------|-----------|
| Layer 1 엔진 | NetworkX + 자체 retriever | **OpenSPG/KAG** (Docker) |
| KG 저장소 | JSON 파일 (메모리 로드) | **OpenSPG Graph DB** |
| KG 구축 | 자체 regex + cross_ref.py | **KAG kg-builder** (LLM 기반 추출) |
| Retrieval | 자체 kg_cs/kg_fr/rc/rc_embed | **KAG kg-solver** (KG_CS + KG_FR + RC) |
| Query 분해 | 자체 QueryDecomposer | **KAG LF Planner** |
| Reasoning | 없음 | **KAG Deduce Executor** (multi-hop) |
| Layer 2 | PageIndex BM25 | PageIndex BM25 (유지) |
| 배포 | 단일 머신 | **KAG 서버: 별도 머신** / 클라이언트: 로컬 |

---

## 시스템 구성

```
┌─────────────────────────────┐     ┌──────────────────────────────────┐
│  Client (로컬 or 평가 서버)   │     │  KAG Server (별도 머신, Docker)    │
│                             │     │                                  │
│  ┌───────────────────────┐  │     │  ┌────────────────────────────┐  │
│  │ rag_pipeline.py       │  │     │  │ OpenSPG Engine             │  │
│  │                       │  │     │  │  - Graph DB (NRC 스키마)    │  │
│  │ 1. KAG API 호출 ──────┼──┼─────┼──│  - Chunk Index (벡터)      │  │
│  │    (Layer 1)          │  │     │  │  - Entity Index            │  │
│  │                       │  │     │  └────────────────────────────┘  │
│  │ 2. PageIndex 검색     │  │     │  ┌────────────────────────────┐  │
│  │    (Layer 2, 로컬)    │  │     │  │ KAG Services               │  │
│  │                       │  │     │  │  - kg-builder API           │  │
│  │ 3. LLM 답변 생성      │  │     │  │  - kg-solver API (:8887)   │  │
│  │    (OpenAI API)       │  │     │  │  - LLM (Qwen/GPT)         │  │
│  └───────────────────────┘  │     │  │  - Embedding (bge-m3)     │  │
│                             │     │  └────────────────────────────┘  │
│  ┌───────────────────────┐  │     └──────────────────────────────────┘
│  │ PageIndex Data        │  │
│  │ (structure.json × 152)│  │
│  └───────────────────────┘  │
└─────────────────────────────┘
```

---

## Pipeline Flow

```
User Query
    │
════════════════════════════════════════════════════════════════
          LAYER 1: Document Layer (KAG Server)
════════════════════════════════════════════════════════════════
    │
    ▼
┌──────────────────────────────────────────────────────┐
│              KAG Solver Pipeline                      │
│                                                      │
│  ┌──────────────────────────────────────────┐        │
│  │ LF Planner (쿼리 분해 + 실행 계획)         │        │
│  │                                          │        │
│  │ "containment heat removal criteria"      │        │
│  │  → Plan:                                 │        │
│  │    1. retrieval(GDC 38, containment)     │        │
│  │    2. retrieval(SRP 6.2.2, heat removal) │        │
│  │    3. deduce(교차 규제 요건 추론)           │        │
│  └──────────────┬───────────────────────────┘        │
│                 │                                     │
│    ┌────────────┼────────────┐                        │
│    ▼            ▼            ▼                        │
│ ┌────────┐ ┌────────┐ ┌──────────┐                   │
│ │ KG_CS  │ │ KG_FR  │ │    RC    │                   │
│ │        │ │        │ │          │                    │
│ │ exact  │ │ fuzzy  │ │ vector   │                    │
│ │ 1-hop  │ │ 1-hop  │ │ chunk    │                    │
│ │ entity │ │ + PPR  │ │ retrieval│                    │
│ │ linking│ │ chunk  │ │ (bge-m3) │                    │
│ │        │ │retriev.│ │          │                    │
│ ├────────┤ ├────────┤ ├──────────┤                   │
│ │스키마   │ │인식     │ │유사도     │                   │
│ │기반    │ │임계값   │ │임계값     │                   │
│ │정확매칭 │ │ 0.8    │ │ 0.65    │                    │
│ └───┬────┘ └───┬────┘ └────┬─────┘                   │
│     └──────────┼───────────┘                          │
│                ▼                                      │
│     ┌─────────────────┐                               │
│     │ Hybrid Merger   │                               │
│     │ + Deduce        │                               │
│     │   Executor      │                               │
│     │ (multi-hop      │                               │
│     │  reasoning)     │                               │
│     └────────┬────────┘                               │
│              ▼                                        │
│     Top-N Documents + 근거 경로(trace)                 │
└──────────────┬───────────────────────────────────────┘
               │
════════════════════════════════════════════════════════════════
          LAYER 2: Attribute Layer (로컬 PageIndex)
════════════════════════════════════════════════════════════════
               │
               ▼
       문서 트리 로드 (선택된 N개)
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
════════════════════════════════════════════════════════════════
             LLM 답변 생성 (GPT-4.1)
════════════════════════════════════════════════════════════════
               │
               ▼
        ┌──────────────┐
        │  RAG Result  │
        │ - answer     │
        │ - trace      │
        │ - sources    │
        │ - reasoning  │
        └──────────────┘
```

---

## Phase 1: KAG 서버 환경 구축 (별도 머신)

### 1.1 Docker 기동

```bash
# KAG 프로젝트 클론
git clone https://github.com/OpenSPG/KAG.git
cd KAG

# Docker Compose로 OpenSPG + KAG 서비스 기동
docker-compose up -d
# → OpenSPG Engine + KAG API 서버 (:8887)
```

### 1.2 NRC 도메인 스키마 정의

OpenSPG 스키마로 NRC 규제문서 도메인을 정의한다.

```
# schema/nrc_schema.spg (또는 kag_config.yaml 내 정의)

# ── 엔티티 타입 ──────────────────────────
Entity/NRCDocument:
  properties:
    doc_id:          Text        # ML번호 또는 파일명 기반
    doc_type:        Enum(REG, RG, SRP, DSRS)
    section_number:  Text        # e.g., "50.46", "1.82", "6.2.2"
    title:           Text
    revision:        Text
    page_count:      Integer
    filename:        Text
    chapter:         Text        # section_number 첫 자리

Entity/GDC:
  properties:
    number:          Text        # "1" ~ "64"
    title:           Text        # "General Design Criterion N"
    description:     Text        # 주제 설명

Entity/Chapter:
  properties:
    number:          Text        # "2" ~ "19"
    title:           Text        # 챕터 주제

# ── 관계 타입 ──────────────────────────
Relation/CITES_GDC:             NRCDocument → GDC
Relation/ESTABLISHES_REQUIREMENT: NRCDocument(REG) → NRCDocument(RG)
Relation/ESTABLISHES_CRITERIA:  NRCDocument(REG) → NRCDocument(SRP|DSRS)
Relation/PROVIDES_GUIDANCE:     GDC → NRCDocument(RG|SRP|DSRS)
Relation/ACCEPTED_METHOD:       NRCDocument(SRP|DSRS) → NRCDocument(RG)
Relation/ADAPTED_FOR_NUSCALE:   NRCDocument(SRP) → NRCDocument(DSRS)
Relation/BASED_ON:              NRCDocument(DSRS) → NRCDocument(SRP)
Relation/CROSS_REFERENCES:      NRCDocument → NRCDocument (같은 유형)
Relation/BELONGS_TO_CHAPTER:    NRCDocument → Chapter
```

### 1.3 kag_config.yaml

```yaml
project:
  namespace: NRC_RegDoc
  host_addr: http://0.0.0.0:8887
  language: en

# ── kg-builder ─────────────────────────
kag_builder_pipeline:
  # 비정형 문서 (151 NRC PDFs)
  unstructured_builder_chain:
    reader:
      type: pdf                       # PyMuPDF 기반 PDF 리더
    splitter:
      type: length
      split_length: 50000             # 규제문서는 대형 → 큰 청크
      window: 200
    extractor:
      type: schema_free               # NER + Triple 추출
      llm:
        type: openai                   # 또는 로컬 Qwen
        model: gpt-4.1-mini           # 비용 절감용 mini
        api_key: ${OPENAI_API_KEY}
      ner_prompt: prompt/nrc_ner.py
      triple_prompt: prompt/nrc_triple.py
    vectorizer:
      type: openai                     # 또는 bge-m3
      model: text-embedding-3-small
      dimensions: 1536
    writer:
      type: kg
      parallel_num: 4

  # 구조화 데이터 (기존 doc_metadata.json → 관계 주입)
  domain_knowledge_inject:
    external_graph_loader:
      type: json
      source: data/doc_metadata.json   # 기존 cross-ref 데이터 활용
    vectorizer:
      type: openai
      model: text-embedding-3-small

# ── kg-solver ──────────────────────────
kag_solver_pipeline:
  type: kag_static_pipeline

  planner:
    type: lf_planner
    llm:
      type: openai
      model: gpt-4.1

  executors:
    retriever:
      type: hybrid
      kg_cs:                           # exact 1-hop entity linking
        enabled: true
      kg_fr:                           # fuzzy 1-hop + PPR
        enabled: true
        recognition_threshold: 0.8
      rc:                              # vector chunk retrieval
        enabled: true
        similarity_threshold: 0.65
    deduce:
      type: llm_based
      llm:
        type: openai
        model: gpt-4.1
    math:
      enabled: false                   # NRC 도메인에서 불필요

  generator:
    type: llm
    llm:
      type: openai
      model: gpt-4.1
```

---

## Phase 2: kg-builder로 KG 구축

### 2.1 PDF 문서 인제스트

```
151 NRC PDFs
    │
    ▼
┌─────────┐   ┌──────────┐   ┌────────────────┐   ┌───────────┐   ┌────────┐
│  Reader  │ → │ Splitter │ → │   Extractor    │ → │Vectorizer │ → │ Writer │
│  (PDF)   │   │ (length) │   │ (schema_free)  │   │(embedding)│   │  (KG)  │
│          │   │          │   │                │   │           │   │        │
│ PyMuPDF  │   │ 50K char │   │ NER: 문서유형,  │   │text-emb-  │   │OpenSPG │
│ 텍스트   │   │ chunks   │   │  GDC, 규제참조  │   │3-small    │   │Graph   │
│ 추출     │   │          │   │ Triple: 관계    │   │1536 dims  │   │DB 저장 │
└─────────┘   └──────────┘   │ 추출           │   └───────────┘   └────────┘
                              └────────────────┘
```

### 2.2 기존 메타데이터 주입 (DomainKnowledgeInject)

현재 `doc_metadata.json`에 이미 추출된 cross-reference 정보를 활용한다.
LLM 추출 결과와 기존 regex 추출 결과를 병합하여 KG 품질을 높인다.

```python
# scripts/inject_metadata.py (작성 예정)
#
# doc_metadata.json의 각 문서에 대해:
# 1. NRCDocument 노드 생성/업데이트
# 2. GDC 엔티티 노드 생성
# 3. gdc_refs → CITES_GDC 관계 생성
# 4. rg_refs → ACCEPTED_METHOD 관계 생성
# 5. srp_refs/dsrs_refs → CROSS_REFERENCES 관계 생성
# 6. 같은 section_number의 SRP↔DSRS → ADAPTED_FOR_NUSCALE 관계
```

### 2.3 구축 결과 기대치

| 항목 | v1 (현재) | v2 (KAG) |
|------|-----------|----------|
| 노드 수 | 203 (151 문서 + 52 GDC) | 203 + **LLM 추출 엔티티** (규제 개념, 시스템 등) |
| 엣지 수 | 1,906 | 1,906 + **LLM 추출 관계** |
| 청크 인덱스 | 없음 (문서 단위만) | **PDF 청크 벡터 인덱스** (chunk 단위 검색 가능) |
| 스키마 | 암묵적 (코드 내 하드코딩) | **명시적 SPG 스키마** |

---

## Phase 3: kg-solver로 Document Layer 교체

### 3.1 KAG Solver API 호출

클라이언트에서 KAG 서버의 solver API를 호출하여 Layer 1 결과를 얻는다.

```python
# src/document_layer/kag_client.py (신규)

import requests
from dataclasses import dataclass, field

KAG_SERVER = "http://<kag-server-ip>:8887"

@dataclass
class KAGResult:
    """KAG Solver 응답"""
    doc_ids: list[str] = field(default_factory=list)
    chunks: list[dict] = field(default_factory=list)   # KAG RC 결과
    reasoning: str = ""                                 # Deduce 결과
    trace: list[str] = field(default_factory=list)

class KAGClient:
    """KAG Solver API 클라이언트."""

    def __init__(self, server_url: str = KAG_SERVER):
        self.server_url = server_url

    def query(self, question: str, top_k: int = 10) -> KAGResult:
        """
        KAG Solver에 쿼리를 보내고 관련 문서를 받는다.

        KAG 내부 흐름:
        1. LF Planner → 쿼리 분해 + 실행 계획
        2. KG_CS (exact) + KG_FR (fuzzy+PPR) + RC (vector) 병렬 검색
        3. Deduce Executor → multi-hop 추론
        4. 결과 병합 → top-k 문서 + 근거 경로
        """
        resp = requests.post(
            f"{self.server_url}/api/v1/solver/qa",
            json={"query": question, "top_k": top_k},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        return KAGResult(
            doc_ids=data.get("doc_ids", []),
            chunks=data.get("chunks", []),
            reasoning=data.get("reasoning", ""),
            trace=data.get("trace", []),
        )
```

### 3.2 KAG의 3-Retriever 비교 (v1 자체구현 vs v2 KAG 내장)

| Retriever | v1 자체 구현 | v2 KAG 내장 | 차이 |
|-----------|-------------|-------------|------|
| **KG_CS** | regex 패턴매칭 → NetworkX 1-2홉 | **스키마 기반 entity linking → OpenSPG 1-hop** | KAG는 LLM entity linking 사용, 더 정확 |
| **KG_FR** | 토큰 매칭 → PPR (전체 그래프) | **LLM fuzzy 인식 → PPR + chunk retrieval** | KAG는 인식 임계값(0.8) 기반, chunk 단위 |
| **RC** | BM25 (문서 title+GDC desc) | **벡터 chunk retrieval (similarity ≥ 0.65)** | KAG는 chunk 단위 semantic 검색 |
| **rc_embed** | OpenAI embedding (문서 단위) | RC에 통합 (chunk 단위) | KAG RC가 더 세밀 |
| **Merger** | 자체 min-max + 가중합산 + concept bonus | **KAG hybrid merger** | KAG 자체 병합 로직 |
| **Query 분해** | 자체 QueryDecomposer (GPT-4.1) | **LF Planner** (KAG 내장) | 동일 역할, KAG가 실행계획까지 생성 |
| **Reasoning** | 없음 | **Deduce Executor** (multi-hop) | v2에서 추가 |

---

## Phase 4: Attribute Layer 연결 + 통합 파이프라인

### 4.1 통합 파이프라인

```python
# src/pipeline/two_layer_search.py (수정)

class TwoLayerSearchPipeline:
    def __init__(self, kag_client, trees_dir, ...):
        self.kag = kag_client           # KAG Server 클라이언트
        self.env = AttributeLayerEnv(trees_dir)  # 기존 유지

    def search(self, query: str) -> SearchResult:
        result = SearchResult(query=query)

        # ── Layer 1: KAG Solver ──────────────
        kag_result = self.kag.query(query, top_k=10)
        result.selected_docs = kag_result.doc_ids
        result.trace.extend(kag_result.trace)

        # KAG chunks를 보조 context로 보존 (Layer 2와 병합)
        kag_chunks = kag_result.chunks

        # ── Layer 2: PageIndex (기존 로직) ────
        self.env.load_documents(kag_result.doc_ids)
        sections = self.env.search(query, max_results=5)
        # ... (기존 browse fallback 등 유지)

        # ── Context 조합 ─────────────────────
        # PageIndex sections + KAG chunks 병합
        result.context = self._build_context(
            query, sections, kag_chunks, kag_result.reasoning
        )
        return result
```

### 4.2 KAG chunks와 PageIndex의 역할 분리

```
KAG RC chunks (Layer 1)          PageIndex sections (Layer 2)
───────────────────────          ──────────────────────────────
- PDF 원문 chunk (벡터 검색)      - 사전 구축된 구조 트리
- 문서 선택의 근거                 - 섹션 단위 summary
- semantic 매칭에 강함             - 목차 구조 탐색에 강함
- 세부 내용 포함                   - 페이지 범위 정보 포함

         ↓ 병합 전략 ↓
─────────────────────────────────────────
1. PageIndex sections를 우선 사용 (구조화됨)
2. KAG chunks로 보완 (PageIndex에 없는 세부 내용)
3. 중복 제거 (같은 페이지 범위 → PageIndex 우선)
4. max_context_chars 내에서 조합
```

---

## 파일 구조 (v2)

```
doc_search/
├── src/
│   ├── document_layer/              # LAYER 1 (KAG 클라이언트)
│   │   ├── kag_client.py               KAG Solver API 클라이언트 (신규)
│   │   ├── classifier.py               문서 메타데이터 (유지, 주입용)
│   │   └── cross_ref.py                교차참조 추출 (유지, 주입용)
│   ├── attribute_layer/             # LAYER 2 (유지)
│   │   └── environment.py              PageIndex 트리 + BM25 검색
│   └── pipeline/                    # 통합
│       ├── two_layer_search.py         KAG(L1) + PageIndex(L2) 연결
│       └── rag_pipeline.py             검색 + LLM 답변 생성
├── kag_server/                      # KAG 서버 설정 (별도 머신에 배포)
│   ├── kag_config.yaml                 KAG 프로젝트 설정
│   ├── schema/
│   │   └── nrc_schema.spg              NRC 도메인 스키마
│   ├── prompt/
│   │   ├── nrc_ner.py                  NRC 도메인 NER 프롬프트
│   │   └── nrc_triple.py              NRC 도메인 Triple 추출 프롬프트
│   ├── builder/
│   │   └── inject_metadata.py          기존 메타데이터 → KAG KG 주입
│   └── docker-compose.override.yml     서버 환경 커스텀
├── data/
│   ├── doc_metadata.json               151개 문서 메타 (주입 소스)
│   └── (v1 파일들은 deprecated)
└── scripts/
    ├── query.py                        CLI 쿼리 (KAG 클라이언트 사용)
    └── run_eval.py                     평가 (출력 포맷 유지)
```

---

## 구현 순서

### Step 1: KAG 서버 기동 (별도 머신)

```
1-1. Docker/Docker Compose 설치
1-2. OpenSPG/KAG 클론 + docker-compose up
1-3. 브라우저에서 http://<server>:8887 접속 확인
1-4. NRC 프로젝트 생성 + 스키마 등록
```

### Step 2: KG 구축

```
2-1. 기존 doc_metadata.json → DomainKnowledgeInject로 관계 그래프 주입
     (이미 추출된 GDC/RG/SRP cross-ref 활용 → 빠르게 기본 KG 완성)
2-2. 151개 NRC PDF → kg-builder 비정형 파이프라인 실행
     (LLM 기반 NER + Triple 추출 → 기본 KG에 엔티티/관계 보강)
2-3. 벡터 인덱스 자동 생성 (chunk 단위 embedding)
2-4. KAG 웹 UI에서 그래프 시각화 + 품질 점검
```

### Step 3: Solver 검증

```
3-1. KAG 웹 UI에서 샘플 쿼리 테스트
     - "What are the NRC acceptance criteria for containment heat removal?"
     - "How do regulatory requirements for natural phenomena define site parameters?"
3-2. Solver API 응답 포맷 확인 → KAGClient 맞춤 구현
3-3. Layer 1 단독 Recall@10 측정 (v1 대비 비교)
```

### Step 4: 클라이언트 통합 (로컬)

```
4-1. kag_client.py 구현 (API 호출 + 응답 파싱)
4-2. two_layer_search.py 수정 (KAGClient → PageIndex 연결)
4-3. rag_pipeline.py 수정 (context 병합 로직)
4-4. run_eval.py로 전체 파이프라인 평가
```

### Step 5: v1 코드 정리

```
5-1. 자체 retriever 코드 제거 (retriever.py, embedding_retriever.py, query_decomposer.py)
5-2. kg_builder.py 제거 (KAG가 대체)
5-3. v1 데이터 파일 정리 (document_layer_kg.json, doc_embeddings.npy 등)
```

---

## Data Flow Summary

| Stage | 위치 | Input | Output | Role |
|-------|------|-------|--------|------|
| LF Planner | KAG Server | 자연어 쿼리 | 실행 계획 + sub-tasks | 쿼리 분해 + 검색 전략 |
| KG_CS | KAG Server | entity mention | exact 1-hop 문서 | 명시적 참조 매칭 |
| KG_FR | KAG Server | fuzzy keyword | PPR 전파 + chunks | 암묵적 관계 탐색 |
| RC | KAG Server | query embedding | similar chunks | 의미 기반 검색 |
| Deduce | KAG Server | 검색 결과 | multi-hop 추론 | 교차 규제 요건 추론 |
| Hybrid Merge | KAG Server | 3 retriever 결과 | top-N doc_ids + chunks | Layer 1 최종 결과 |
| PageIndex | 로컬 | top-N doc_ids + query | 5개 섹션 context | 구조 기반 섹션 탐색 |
| LLM Gen | 로컬 (OpenAI) | context + query | answer | 최종 답변 |

---

## Fallback 전략

| 상황 | Fallback |
|------|----------|
| KAG 서버 접속 불가 | v1 자체 retriever로 fallback (코드 보존 가능) |
| KAG solver 타임아웃 (>30s) | RC (벡터 검색) 결과만 사용 |
| PageIndex 트리 없음 | KAG chunks로 context 구성 |
| LLM API 실패 | context + trace만 반환 (답변 없이) |

---

## 기대 효과

| 지표 | v1 (현재) | v2 (KAG) 기대치 |
|------|-----------|-----------------|
| Layer 1 Recall@10 | ~20% | **60%+** (chunk 벡터 + LLM entity linking) |
| Multi-hop 질의 | 지원 안함 | **Deduce Executor로 지원** |
| KG 품질 | regex 기반 (누락 多) | **LLM 추출 + regex 병합** (높은 커버리지) |
| 검색 단위 | 문서 단위 | **chunk 단위** (더 세밀) |
| 유지보수 | 자체 코드 관리 필요 | **KAG 프레임워크 업데이트** 활용 |
| 배포 | 단일 프로세스 | **서버 분리** (확장 가능) |

---

## External Dependencies

- **KAG Server**: Docker 기반, 별도 머신 (OpenSPG + KAG 서비스)
- **Documents**: 151 NRC PDFs (KAG 서버에 인제스트)
- **PageIndex**: `/home/bbo/Document_pageindex/pageindex/` (로컬, 152 structure JSONs)
- **QA Dataset**: `/home/bbo/RAG_Baseline/qadataset/rag_qa_dataset_final.json` (571 queries)
- **Evaluation**: `/home/bbo/RAG_Baseline/04_evaluate.py` (9 metrics)
- **LLM**: OpenAI GPT-4.1 (KAG solver + 답변 생성)
- **Embedding**: OpenAI text-embedding-3-small 또는 bge-m3
