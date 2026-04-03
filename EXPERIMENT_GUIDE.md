# KAG 2-Layer RAG 실험 가이드라인

## 실험 목적

OpenSPG/KAG 프레임워크를 Document Layer로 도입했을 때,
자체 구현(v1) 대비 검색 정확도와 답변 품질이 어느 정도 개선되는지 정량적으로 검증한다.

---

## 실험 환경

### 머신 구성

| 역할 | 스펙 요건 | 설치 항목 |
|------|-----------|-----------|
| **KAG 서버** (별도 머신) | RAM 16GB+, 디스크 50GB+, Docker/Docker Compose | OpenSPG, KAG, LLM API 키 |
| **클라이언트** (현재 머신) | 기존 doc_search 환경 | kag_client.py, PageIndex 데이터 |

### 데이터셋

- **문서**: NRC 규제문서 151개 PDF (`/home/bbo/RAG_Baseline/documents/`)
- **QA**: 571개 질의 (easy 228 / medium 266 / hard 77), 전부 answerable
  - 경로: `/home/bbo/RAG_Baseline/qadataset/rag_qa_dataset_final.json`
- **PageIndex**: 152개 structure.json (`/home/bbo/Document_pageindex/pageindex/`)
- **기존 메타데이터**: `doc_search/data/doc_metadata.json` (151개 문서 cross-ref)

### 평가 도구

- `RAG_Baseline/04_evaluate.py` — 9개 지표 자동 평가
- Judge LLM: vLLM 서버 (Qwen3 등)
- Embedding: OpenAI text-embedding-3-small

---

## 실험 설계

### 비교 대상 (4개 조건)

| ID | 조건 | Document Layer | Attribute Layer | 설명 |
|----|------|----------------|-----------------|------|
| **A** | Baseline (RAG_Baseline) | 없음 (chunk 직접 검색) | 없음 | 기존 벡터 RAG |
| **B** | v1 자체구현 | NetworkX KG + 4-retriever | PageIndex BM25 | 현재 doc_search |
| **C** | KAG only | KAG solver (KG_CS+KG_FR+RC) | 없음 (KAG chunks만) | KAG 단독 성능 |
| **D** | KAG + PageIndex | KAG solver | PageIndex BM25 | 최종 목표 시스템 |

### 평가 지표 (9개)

```
Step A — Retrieval (검색 품질)
  1. Recall@2     : top-2 검색 결과에 정답 문서 포함 여부
  2. Recall@5     : top-5 검색 결과에 정답 문서 포함 여부
  3. MRR          : 정답 문서의 역순위 평균

Step B — Generation (답변 품질)
  4. Exact Match  : 정규화 후 정확 일치
  5. Accuracy     : LLM-Judge 기반 정확도 (0~10점)
  6. ROUGE-L      : GT 답변과의 ROUGE-L F1

Step C — Faithfulness (충실도)
  7. Faithfulness      : 답변의 각 claim이 context에서 지지되는 비율
  8. Answer Relevance  : 답변→역생성 질문↔원본 질문 임베딩 유사도
  9. Context Precision  : 검색된 chunk들의 rank-weighted 관련성
```

### 추가 측정 지표

| 지표 | 측정 방법 | 목적 |
|------|-----------|------|
| **L1 Recall@10** | Layer 1 선택 문서 10개 중 정답 문서 포함 여부 | Document Layer 단독 성능 |
| **Latency (p50/p95)** | 쿼리별 검색+생성 시간 | 실용성 검증 |
| **KG 커버리지** | 추출된 엔티티/관계 수 vs v1 | KG 품질 비교 |
| **난이도별 분석** | easy/medium/hard 각각의 Recall, Accuracy | 어려운 쿼리에서의 개선 확인 |

---

## 실험 절차

### Phase 0: 베이스라인 확보

```
목표: 조건 A(Baseline)와 조건 B(v1)의 결과를 먼저 확보한다.
위치: 클라이언트 머신 (현재 환경)
```

**Step 0-1. Baseline (A) 결과 확인**

```bash
# 기존 RAG_Baseline 결과가 있는지 확인
ls /home/bbo/RAG_Baseline/results/

# 없으면 RAG_Baseline 파이프라인으로 전체 평가 실행
cd /home/bbo/RAG_Baseline
.venv/bin/python 03_query.py --all
.venv/bin/python 04_evaluate.py
```

**Step 0-2. v1 (B) 전체 평가 실행**

```bash
cd /home/bbo/doc_search
/home/bbo/RAG_Baseline/.venv/bin/python scripts/run_eval.py --all

# 평가
cd /home/bbo/RAG_Baseline
python 04_evaluate.py --input doc_search_2layer/query_results_full.json
```

**Step 0-3. 결과 기록**

결과를 `results/baseline_comparison.json`에 기록:

```json
{
  "condition_A_baseline": {
    "recall@2": 0.XX, "recall@5": 0.XX, "mrr": 0.XX,
    "accuracy": 0.XX, "rouge_l": 0.XX,
    "faithfulness": 0.XX, "context_precision": 0.XX
  },
  "condition_B_v1": {
    "recall@2": 0.XX, "recall@5": 0.XX, "mrr": 0.XX,
    "accuracy": 0.XX, "rouge_l": 0.XX,
    "faithfulness": 0.XX, "context_precision": 0.XX
  }
}
```

---

### Phase 1: KAG 서버 구축 (별도 머신)

```
목표: KAG 서버를 기동하고 NRC 도메인 KG를 구축한다.
위치: KAG 서버 머신
소요 시간 예상: 1~2일
```

**Step 1-1. Docker 환경 준비**

```bash
# Docker + Docker Compose 설치 확인
docker --version        # 20.10+ 필요
docker-compose --version  # 2.0+ 필요

# KAG 클론
git clone https://github.com/OpenSPG/KAG.git
cd KAG
```

**Step 1-2. KAG 서비스 기동**

```bash
# docker-compose로 전체 서비스 기동
docker-compose up -d

# 서비스 확인
curl http://localhost:8887/health    # OpenSPG + KAG API
```

**Step 1-3. NRC 프로젝트 생성**

```bash
# 브라우저에서 http://<server>:8887 접속
# 또는 API로 프로젝트 생성

# 1. 프로젝트 생성 (namespace: NRC_RegDoc)
# 2. 스키마 등록 (ARCHITECTURE.md의 nrc_schema 참고)
# 3. LLM/Embedding 설정 (kag_config.yaml)
```

**Step 1-4. 기존 메타데이터 주입 (DomainKnowledgeInject)**

먼저 확실한 데이터(기존 regex 추출 cross-ref)로 기본 KG를 구성한다.

```bash
# 클라이언트에서 서버로 데이터 전송
scp /home/bbo/doc_search/data/doc_metadata.json <server>:~/KAG/nrc_project/data/

# 서버에서 주입 스크립트 실행
# → 151 NRCDocument 노드 + 52 GDC 노드 + 1,906 관계
python inject_metadata.py
```

검증:
```bash
# KAG 웹 UI에서 그래프 노드/엣지 수 확인
# 기대: 203 노드, 1,906+ 엣지 (v1과 동일 수준)
```

**Step 1-5. PDF 문서 인제스트 (kg-builder)**

```bash
# 클라이언트에서 서버로 PDF 전송
scp /home/bbo/RAG_Baseline/documents/*.pdf <server>:~/KAG/nrc_project/data/pdfs/

# kg-builder 비정형 파이프라인 실행
# → PDF 청크 분할 → LLM NER/Triple 추출 → 벡터 인덱스 생성
python -m kag.builder.main_builder --config kag_config.yaml
```

주의사항:
- 151개 PDF × LLM 추출 = 상당한 API 비용/시간 발생
- `gpt-4.1-mini` 사용 시 비용 절감 가능
- 로컬 Qwen 사용 시 GPU 필요 (VRAM 16GB+)

검증:
```bash
# KG 통계 확인
# - 노드 수: 203 + α (LLM 추출 엔티티)
# - 엣지 수: 1,906 + α (LLM 추출 관계)
# - 청크 인덱스: 벡터 수 확인 (수천 개 예상)
```

**Step 1-6. Solver 기본 동작 검증**

KAG 웹 UI 또는 API에서 샘플 쿼리 3개를 테스트한다.

```
테스트 쿼리 1 (명시적 참조):
  "What are the NRC acceptance criteria for containment heat removal?"
  → 기대: GDC 38/39/40 관련 문서, SRP 6.2.2

테스트 쿼리 2 (암묵적 참조):
  "How do regulatory requirements define the geographic scope for site parameters?"
  → 기대: GDC 2, SRP 2.3.x 관련 문서

테스트 쿼리 3 (multi-hop):
  "What seismic design parameters must be established for NuScale structures?"
  → 기대: GDC 2 → SRP 3.7.1 → DSRS 3.7.1 경로 추론
```

각 쿼리에 대해 기록:
- 반환된 doc_ids (순서대로)
- 검색 소요 시간
- KAG trace/reasoning 출력

---

### Phase 2: 클라이언트 통합 + 조건 C 평가

```
목표: KAG solver 단독 성능(조건 C)을 측정한다.
위치: 클라이언트 머신
```

**Step 2-1. kag_client.py 구현**

```python
# src/document_layer/kag_client.py
# ARCHITECTURE.md의 KAGClient 클래스 참고
# KAG 서버 API 엔드포인트에 맞춰 구현

KAG_SERVER = "http://<server-ip>:8887"
```

**Step 2-2. KAG-only 평가 스크립트 작성**

```bash
# scripts/run_eval_kag.py (신규)
# - KAG solver에서 doc_ids + chunks 반환
# - chunks를 직접 context로 사용 (PageIndex 없이)
# - LLM 답변 생성
# - 출력 포맷: run_eval.py와 동일 (04_evaluate.py 호환)
```

**Step 2-3. 조건 C 실행 + 평가**

```bash
cd /home/bbo/doc_search

# 전체 571개 쿼리 실행 (KAG only)
/home/bbo/RAG_Baseline/.venv/bin/python scripts/run_eval_kag.py --all \
  --experiment-id kag_only

# 평가
cd /home/bbo/RAG_Baseline
python 04_evaluate.py --input kag_only/query_results_full.json
```

**Step 2-4. L1 Recall@10 별도 측정**

```bash
# scripts/eval_l1_recall.py (신규)
# KAG solver가 반환한 top-10 doc_ids에 GT evidence_docs가 포함되는지 측정
# 난이도별 (easy/medium/hard) 분리 분석

/home/bbo/RAG_Baseline/.venv/bin/python scripts/eval_l1_recall.py --all
```

---

### Phase 3: 조건 D 평가 (KAG + PageIndex)

```
목표: 최종 시스템(KAG + PageIndex)의 성능을 측정한다.
위치: 클라이언트 머신
```

**Step 3-1. two_layer_search.py 수정**

```python
# KAG solver → PageIndex 연결
# ARCHITECTURE.md의 통합 파이프라인 코드 참고
```

**Step 3-2. 조건 D 실행 + 평가**

```bash
cd /home/bbo/doc_search

# 전체 571개 쿼리 실행 (KAG + PageIndex)
/home/bbo/RAG_Baseline/.venv/bin/python scripts/run_eval.py --all \
  --experiment-id kag_pageindex

# 평가
cd /home/bbo/RAG_Baseline
python 04_evaluate.py --input kag_pageindex/query_results_full.json
```

---

### Phase 4: 결과 분석 + 소거 실험

```
목표: 각 컴포넌트의 기여도를 분석하고 최적 구성을 찾는다.
위치: 클라이언트 머신
```

**Step 4-1. 4개 조건 비교표 작성**

```
┌──────────────────┬─────────┬─────────┬─────────┬──────────────┐
│ 지표              │ A(Base) │ B(v1)   │ C(KAG)  │ D(KAG+PI)    │
├──────────────────┼─────────┼─────────┼─────────┼──────────────┤
│ Recall@2         │         │         │         │              │
│ Recall@5         │         │         │         │              │
│ MRR              │         │         │         │              │
│ Accuracy         │         │         │         │              │
│ ROUGE-L          │         │         │         │              │
│ Faithfulness     │         │         │         │              │
│ Answer Relevance │         │         │         │              │
│ Context Precision│         │         │         │              │
│ L1 Recall@10     │   N/A   │         │         │              │
│ Latency (p50)    │         │         │         │              │
│ Latency (p95)    │         │         │         │              │
└──────────────────┴─────────┴─────────┴─────────┴──────────────┘
```

**Step 4-2. 난이도별 분석**

```
┌──────────────────┬──────────────────┬──────────────────┬──────────────────┐
│                  │ Easy (228)       │ Medium (266)     │ Hard (77)        │
│ 지표              │ B→D 변화         │ B→D 변화         │ B→D 변화          │
├──────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ Recall@5         │                  │                  │                  │
│ Accuracy         │                  │                  │                  │
│ Faithfulness     │                  │                  │                  │
└──────────────────┴──────────────────┴──────────────────┴──────────────────┘
```

**Step 4-3. 소거 실험 (Ablation)**

KAG solver의 각 retriever 기여도를 확인한다.

| 소거 조건 | KG_CS | KG_FR | RC | PageIndex | 측정 지표 |
|-----------|-------|-------|-----|-----------|-----------|
| Full (D) | O | O | O | O | (기준) |
| -KG_CS | X | O | O | O | KG_CS 기여도 |
| -KG_FR | O | X | O | O | KG_FR 기여도 |
| -RC | O | O | X | O | RC 기여도 |
| -PageIndex (C) | O | O | O | X | PageIndex 기여도 |
| RC only | X | X | O | O | 벡터 검색 단독 |

각 소거 조건은 KAG config를 수정하거나 클라이언트에서 해당 결과를 제외하여 실행한다.

```yaml
# kag_config.yaml에서 retriever 개별 비활성화 예시
executors:
  retriever:
    type: hybrid
    kg_cs:
      enabled: false    # ← 이 retriever 비활성화
    kg_fr:
      enabled: true
    rc:
      enabled: true
```

**Step 4-4. 오류 분석 (Error Analysis)**

조건 D에서도 실패한 쿼리를 분석한다.

```
분류 기준:
1. L1 실패 (문서 선택 실패) → KG 구조/스키마 문제
2. L2 실패 (섹션 검색 실패) → PageIndex 커버리지 문제
3. LLM 실패 (context 있으나 답변 부정확) → 프롬프트/모델 문제

각 카테고리별:
- 실패 쿼리 수 / 비율
- 대표 실패 사례 3~5개 기록
- 원인 분석 + 개선 방향
```

---

## 실험 체크리스트

### Phase 0: 베이스라인 확보
- [ ] 조건 A (Baseline) 9개 지표 결과 확보
- [ ] 조건 B (v1) 전체 571개 쿼리 평가 완료
- [ ] `results/baseline_comparison.json` 작성

### Phase 1: KAG 서버 구축
- [ ] Docker 환경 준비 완료
- [ ] KAG 서비스 기동 + 웹 UI 접속 확인
- [ ] NRC 스키마 등록
- [ ] `doc_metadata.json` → DomainKnowledgeInject 완료 (203 노드, 1906 엣지)
- [ ] 151개 PDF → kg-builder 인제스트 완료
- [ ] KG 통계 기록 (노드 수, 엣지 수, 청크 수)
- [ ] 샘플 쿼리 3개 정상 동작 확인

### Phase 2: 조건 C (KAG only)
- [ ] `kag_client.py` 구현 + API 연결 확인
- [ ] `run_eval_kag.py` 작성
- [ ] 전체 571개 쿼리 실행 완료
- [ ] 9개 지표 평가 완료
- [ ] L1 Recall@10 측정 완료

### Phase 3: 조건 D (KAG + PageIndex)
- [ ] `two_layer_search.py` KAG 통합 수정
- [ ] 전체 571개 쿼리 실행 완료
- [ ] 9개 지표 평가 완료

### Phase 4: 분석
- [ ] 4개 조건 비교표 완성
- [ ] 난이도별 분석 완료
- [ ] 소거 실험 최소 3개 조건 완료
- [ ] 오류 분석 (실패 쿼리 분류 + 대표 사례)
- [ ] 최종 보고서 작성

---

## 성공/실패 판단 기준

### 최소 성공 기준 (Must)

| 지표 | 조건 B (v1) 대비 | 절대 목표 |
|------|-----------------|-----------|
| **Recall@5** | +20%p 이상 개선 | ≥ 0.50 |
| **L1 Recall@10** | +30%p 이상 개선 | ≥ 0.60 |
| **Accuracy** | 하락 없음 | ≥ 0.50 |
| **Latency (p50)** | — | ≤ 15초 |

### 목표 기준 (Want)

| 지표 | 절대 목표 |
|------|-----------|
| Recall@5 | ≥ 0.70 |
| L1 Recall@10 | ≥ 0.80 |
| Accuracy | ≥ 0.65 |
| Faithfulness | ≥ 0.70 |
| Hard 쿼리 Recall@5 | ≥ 0.40 |

### 실패 판단

아래 중 하나라도 해당되면 KAG 도입 재검토:
- Recall@5가 v1 대비 개선 없음 (±5%p 이내)
- Latency p95 > 30초 (실용성 부족)
- KG 구축 비용이 정당화되지 않는 수준의 미미한 개선

---

## 결과 저장 구조

```
/home/bbo/RAG_Baseline/results/
├── baseline/                        # 조건 A
│   ├── query_results_full.json
│   └── evaluation_results.json
├── doc_search_2layer/               # 조건 B (v1)
│   ├── query_results_full.json
│   └── evaluation_results.json
├── kag_only/                        # 조건 C
│   ├── query_results_full.json
│   ├── evaluation_results.json
│   └── l1_recall.json
├── kag_pageindex/                   # 조건 D
│   ├── query_results_full.json
│   ├── evaluation_results.json
│   └── l1_recall.json
├── ablation/                        # 소거 실험
│   ├── no_kg_cs/
│   ├── no_kg_fr/
│   ├── no_rc/
│   └── rc_only/
└── final_comparison.json            # 전체 비교 요약
```

---

## 실험 일정 (예상)

| Phase | 작업 | 소요 시간 |
|-------|------|-----------|
| Phase 0 | 베이스라인 확보 | 반나절 (이미 일부 완료) |
| Phase 1 | KAG 서버 + KG 구축 | 1~2일 |
| Phase 2 | 클라이언트 통합 + 조건 C 평가 | 반나절 |
| Phase 3 | 조건 D 평가 | 반나절 |
| Phase 4 | 분석 + 소거 실험 + 보고서 | 1일 |
| **합계** | | **3~4일** |

---

## 주의사항

1. **API 비용**: KAG builder의 LLM 추출은 151개 PDF 전부 처리 시 상당한 비용 발생.
   `gpt-4.1-mini`로 먼저 소규모(10개 PDF) 테스트 후 전체 확장 권장.

2. **재현성**: 모든 LLM 호출은 `temperature=0.0`으로 고정.
   KAG solver도 동일 설정 필요.

3. **공정 비교**: 조건 B와 D는 같은 LLM(GPT-4.1)으로 답변 생성.
   Judge LLM도 동일 모델/동일 vLLM 서버 사용.

4. **네트워크 지연**: KAG 서버가 별도 머신이므로 API 호출 지연 발생.
   Latency 측정 시 네트워크 오버헤드를 별도 기록.

5. **KG 구축 순서**: DomainKnowledgeInject(Step 1-4)를 **먼저** 하고
   PDF 인제스트(Step 1-5)를 **나중에** 한다.
   기존 cross-ref 기반 관계가 있어야 LLM 추출 결과와 비교/보완 가능.
