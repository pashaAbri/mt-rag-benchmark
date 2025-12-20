# Context Summarization for Multi-Turn RAG

## Overview

This experiment implements the **conversation summarization technique** from the SELF-multi-RAG paper, adapted for the MT-RAG benchmark using Claude Sonnet 4.5.

### Key Insight from the Paper

Traditional query rewriting methods (like T5QR or human rewrites) compress the current question into a standalone query but **lose important context**. The SELF-multi-RAG paper proposes instead:

> "Summarize the conversation history in 40-50 words and ask a question so that the summary and the question can be used without the conversation history to generate a meaningful response."

### Expected Benefits

According to the paper:
- **+13.5% retrieval effectiveness** over query rewriting (R@5)
- **+13% response quality improvement** over single-turn baselines
- Works well for both **sparse (BM25) and dense (BGE, ELSER)** retrieval

## Method

### Input Format (Multi-turn Conversation)

```
User: Tell me about the Arizona Cardinals.
Agent: The Arizona Cardinals are an NFL team based in Glendale, Arizona...
User: Do they play outside the US?
Agent: Yes, they played in London in 2017 against the Los Angeles Rams...
User: Are the Arizona Cardinals and Chicago Cardinals the same team?
```

### Output Format (Summary + Question)

```
Summary: The Arizona Cardinals are an NFL team that has played games 
internationally, including a 2017 game in London against the Los Angeles Rams. 
The conversation has covered their schedule and game locations.

Question: Are the Arizona Cardinals and the Chicago Cardinals the same franchise?
```

### Why This Works Better

| Approach | Example | Issue |
|----------|---------|-------|
| **Last Turn Only** | "Are they the same team?" | ❌ Missing context about *which* teams |
| **Query Rewrite** | "Are the Arizona and Chicago Cardinals the same?" | ❌ Loses international games context |
| **Full Conversation** | [All 6 turns verbatim] | ❌ Too verbose, includes noise |
| **Context Summary** | [40-50 word summary + question] | ✅ Optimal balance of context and focus |

## Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│  Multi-Turn Conversation                                     │
│  (User/Agent alternating turns)                             │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 1: Extract Conversation History                        │
│  • All turns before current question                        │
│  • Includes both user questions and agent responses         │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 2: LLM Summarization (Claude Sonnet 4.5)              │
│  • System prompt with 40-50 word constraint                 │
│  • Few-shot examples from multiple domains                  │
│  • Output: "Summary: ... Question: ..."                     │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 3: Retrieval Query                                     │
│  "Summary: [context] Question: [reformulated question]"     │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 4: Retrieval & Evaluation                              │
│  • BM25 (sparse), BGE (dense), ELSER (learned sparse)       │
│  • Compare R@K, nDCG@K against baselines                    │
└─────────────────────────────────────────────────────────────┘
```

## Usage

### Step 1: Generate Summaries

```bash
# Test on first 10 conversations
python run_context_summarization.py --max_conversations 10

# Run on specific domain
python run_context_summarization.py --domains clapnq

# Run on all conversations
python run_context_summarization.py
```

### Step 2: Run Retrieval

```bash
# Run with BM25 and BGE
python run_retrieval.py --domains clapnq --retrievers bm25 bge

# Run all retrievers
python run_retrieval.py --domains clapnq cloud fiqa govt
```

### Step 3: Evaluate and Compare

```bash
# Evaluate and compare with baselines
python evaluate_and_compare.py --domains clapnq --compare_baselines
```

## Files

| File | Description |
|------|-------------|
| `prompt_template.py` | System prompt, few-shot examples, response parsing |
| `run_context_summarization.py` | Main summarization script using Claude Sonnet 4.5 |
| `run_retrieval.py` | Run BM25/BGE/ELSER retrieval with summaries |
| `evaluate_and_compare.py` | Evaluate results and compare with baselines |
| `intermediate/` | Generated summaries in BEIR format |
| `retrieval_results/` | Retrieval results for evaluation |

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--llm_model` | `claude-sonnet-4-5-20250929` | LLM for summarization |
| `--workers` | 5 | Concurrent API calls |
| `--max_conversations` | None | Limit for testing |
| `--domains` | all | Domains to process |

## Expected Results

Based on the SELF-multi-RAG paper (Table 5), we expect:

| Method | Contriever R@5 | Contriever R@10 | BM25 R@5 | BM25 R@10 |
|--------|----------------|-----------------|----------|-----------|
| Full Conversation | 0.53 | 0.61 | 0.50 | 0.58 |
| T5QR Rewrite | 0.53 | 0.64 | 0.45 | 0.55 |
| **Context Summary** | **0.61** | **0.71** | **0.56** | **0.66** |
| GPT-4 Summary | 0.62 | 0.72 | 0.60 | 0.70 |

Key finding: The learned summarization nearly matches GPT-4 quality (0.61 vs 0.62 R@5).

## Differences from Targeted Rewrite

| Aspect | Targeted Rewrite | Context Summary |
|--------|------------------|-----------------|
| **Turn Selection** | Similarity-based filtering | Include all history |
| **Output Format** | Standalone question | Summary + Question |
| **Context Handling** | Filter irrelevant turns | Summarize all context |
| **Word Budget** | No constraint | 40-50 words |
| **Based On** | Semantic similarity heuristic | SELF-multi-RAG paper |

## References

- [SELF-multi-RAG Paper](https://arxiv.org/abs/YOUR_PAPER_LINK) - Original summarization technique
- [MT-RAG Benchmark](../../../README.md) - Benchmark dataset
- [Targeted Rewrite](../targeted_rewrite/) - Alternative approach using similarity filtering

