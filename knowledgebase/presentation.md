# mtRAG: Multi-Turn RAG Benchmark

---

## Slide 1: Title

### mtRAG: A Multi-Turn Conversational Benchmark for Evaluating RAG Systems

**IBM Research, 2025**

*Understanding where RAG systems break down in real conversations*

---

## Slide 2: The Challenge

### The Research Problem

**Current RAG Benchmarks:**
- Focus on single-turn question answering
- Don't reflect real conversation properties
- Limited evaluation of multi-turn dependencies

**What's Missing:**
- Questions that reference conversation history
- Changing relevant passages across turns
- Unanswerable questions (hallucination risk)
- Multi-domain evaluation

**The Challenge:** Build a benchmark that tests RAG systems on realistic multi-turn conversations

---

## Slide 3: Timeline & Publication

### Project Information

| Item | Details |
|------|---------|
| **Published** | January 2025 |
| **Venue** | arXiv preprint |
| **Paper ID** | arXiv:2501.03468 |
| **Institution** | IBM Research |
| **GitHub** | github.com/ibm/mt-rag-benchmark |

**Benchmark Availability:**
- ✅ Conversations (110 human-generated)
- ✅ Corpora (4 domains, 366K passages)  
- ✅ Evaluation tasks (842 retrieval + generation)
- ✅ Baseline results (9 LLMs tested)
- ✅ Evaluation scripts (BEIR-compatible)

**Status:** Publicly available, ready for research use

---

## Slide 4: The Benchmark at a Glance

| Component | Count |
|-----------|-------|
| Conversations | 110 human-generated |
| Avg turns/conversation | 7.7 |
| Evaluation tasks | 842 |
| Domains | 4 diverse corpora |
| Total passages | 366,479 |
| Unique passages/conv | 16.9 avg |

**What makes it unique:**
- First end-to-end human-generated multi-turn RAG benchmark
- Active retrieval (passages change per turn)
- Real-world conversation properties

---

## Slide 5: Four Diverse Corpora

| Domain | Source | Documents | Passages | Style |
|--------|--------|-----------|----------|-------|
| **ClapNQ** | Wikipedia | 4,293 | 183,408 | Encyclopedic |
| **FiQA** | StackExchange | 7,661 | 49,607 | Conversational Q&A |
| **Cloud** | IBM Docs (NEW) | 57,638 | 61,022 | Technical |
| **Govt** | .gov/.mil (NEW) | 8,578 | 72,422 | Official/Legal |

**Design:**
- 512-token passages, 100-token overlap
- Inter-connected pages for natural flow
- Tests generalization across domains

---

## Slide 6: Question Enrichments

### Three Classification Dimensions

**1. Answerability (4 types)**
- Answerable, Unanswerable, Partial, Conversational

**2. Question Type (10 types)**
- Factoid, Explanation, Composite, Comparative, How-To, Keyword, Opinion, Summarization, Troubleshooting, Non-Question

**3. Multi-Turn (3 types)**
- Follow-up, Clarification, N/A (first turn)

**Purpose:** Fine-grained evaluation of model capabilities

---

## Slide 7: Human Data Creation

**Process:**
1. Annotators interact with live RAG system
2. Check retrieved passages → modify for relevance & diversity
3. Review & repair generated responses
4. Add enrichments (answerability, question type, multi-turn)

**Quality Control:**
- All conversations reviewed
- Average 7.3 edited responses per conversation
- Ensures natural conversation flow

**Distribution:**
- ClapNQ: 29 conv | FiQA: 27 conv
- Cloud: 26 conv | Govt: 28 conv

---

## Slide 8: Retrieval Evaluation

### Three Methods Tested

| Method | Type | Example |
|--------|------|---------|
| BM25 | Lexical | Keyword matching |
| SPLADE | Sparse | Learned sparse vectors |
| BGE/Elser | Dense | Neural embeddings |

### Two Settings

| Setting | Description | Challenge |
|---------|-------------|-----------|
| **Last Turn** | Use current question only | Loses context |
| **Query Rewrite** | Rewrite with full context | Requires understanding |

**Format:** BEIR (standardized IR evaluation)

---

## Slide 9: Generation Evaluation

### Three Retrieval Settings

| Setting | Passages Provided | Purpose | Tasks |
|---------|------------------|---------|-------|
| **Reference** | Gold standard (human-curated) | Test pure generation | 842 |
| **Reference + RAG** | Gold + auto-retrieved | Test if RAG helps/hurts | 436 |
| **Full RAG** | Auto-retrieved only | Real-world scenario | 842 |

**Models Tested:** 9 LLMs evaluated

**Key Questions:**
- Where do systems break? Retrieval or generation?
- Do perfect passages solve the problem?
- How do errors propagate?

---

## Slide 10: Key Insights

### Where RAG Systems Fail

**Retrieval Bottleneck:**
- If retrieval fails → LLM can't answer

**Generation Bottleneck:**
- Even with perfect passages → LLMs struggle with:
  - Later turns
  - Unanswerable questions
  - Questions referencing conversation history

**Error Propagation:**
- Retrieval errors compound generation problems

---

## Slide 11: Automation Exploration

### Automatic Evaluation

**Reference-Based Metrics:**
- RB_llm (LLM-as-Judge)
- RB_alg (Algorithmic)

**Reference-Less Metrics:**
- Faithfulness (grounds in passages?)
- Answer Relevance
- Multi-Turn Bias

**Finding:** Some metrics correlate with human scores, many don't

### Synthetic Data (mtRAG-S)

| Metric | Human | Synthetic |
|--------|-------|-----------|
| Avg turns | 7.7 | 5.9 |
| Unique passages | 16.9 | 4.6 |
| Human edits | 7.3 | 0 |

**Challenge:** Hard to generate unanswerable questions

---

## Slide 12: Benchmark Structure

### Available Data

**Conversations:**
- `conversations.json` - 110 full conversations with metadata

**Generation Tasks:**
- `reference.jsonl` - 842 tasks with gold passages
- `reference+RAG.jsonl` - 436 tasks (≤2 contexts)
- `RAG.jsonl` - 842 tasks with auto-retrieval

**Retrieval Tasks (BEIR format):**
- Queries: lastturn, rewrite, questions
- Qrels: 2,132 relevance judgments
- Per domain organization

**Evaluations:**
- Pre-computed results from 9 LLMs
- Human evaluation subset included

---

## Slide 13: Research Contributions

### What mtRAG Enables

**1. Comprehensive Evaluation**
- Full RAG pipeline (retrieval + generation)
- Multi-domain generalization
- Real conversation properties

**2. Comparative Analysis**
- Retrieval methods (lexical, sparse, dense)
- LLM capabilities across settings
- Human vs synthetic data

**3. Fine-Grained Analysis**
- Performance by question type
- Answerability handling
- Multi-turn dependencies

**4. Open Benchmark**
- Available at: `github.com/ibm/mt-rag-benchmark`
- Reproducible evaluation
- Community resource

---

## Slide 14: Key Takeaways

### Summary

✅ **First end-to-end human multi-turn RAG benchmark**

✅ **110 conversations → 842 tasks across 4 domains**

✅ **Reveals both retrieval and generation bottlenecks**

✅ **Shows current models struggle with:**
- Unanswerable questions
- Later conversation turns
- Multi-turn dependencies

✅ **Provides path for future improvements**

**Paper:** arxiv.org/abs/2501.03468

