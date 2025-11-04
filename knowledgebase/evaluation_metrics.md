# MT-RAG Evaluation Metrics Reference

This document provides a comprehensive reference for all evaluation metrics in the MT-RAG benchmark, including which evaluation step adds each metric, what it measures, and how to interpret it.

---

## Table of Contents
1. [Evaluation Pipeline Overview](#evaluation-pipeline-overview)
2. [Step 1: Algorithmic Metrics](#step-1-algorithmic-metrics)
3. [Step 2: IDK Judge](#step-2-idk-judge)
4. [Step 3: RAGAS Faithfulness](#step-3-ragas-faithfulness)
5. [Step 4: RADBench LLM Judge](#step-4-radbench-llm-judge)
6. [Step 5: IDK Conditioning](#step-5-idk-conditioning)
7. [Quick Reference Table](#quick-reference-table)

---

## Evaluation Pipeline Overview

The MT-RAG evaluation consists of 5 sequential steps:

```
Step 1: Algorithmic Judges (ROUGE, BERTScore, etc.)
   ↓
Step 2: IDK Judge (I Don't Know detection)
   ↓
Step 3: RAGAS Faithfulness (LLM-based faithfulness)
   ↓
Step 4: RADBench Judge (Comprehensive LLM evaluation)
   ↓
Step 5: IDK Conditioning (Adjust metrics for answerability)
```

**Total Metrics**: 15 fields in the `metrics` object of each evaluated record.

---

## Step 1: Algorithmic Metrics

**Script**: `scripts/evaluation/run_algorithmic.py`  
**Purpose**: Compute algorithmic (rule-based) evaluation metrics  
**Metrics Added**: 9

### `Recall`
- **What it measures**: Recall/overlap between model response and reference answer
- **Range**: 0.0 to 1.0 (higher is better)
- **Type**: Algorithmic
- **Use case**: Measures how much of the reference information appears in the response

### `RougeL_stemFalse`
- **What it measures**: ROUGE-L (Longest Common Subsequence) without stemming
- **Range**: 0.0 to 1.0 (higher is better)
- **Type**: Algorithmic
- **Use case**: Measures phrase-level overlap with reference, approximates appropriateness

### `BertscoreP`
- **What it measures**: BERTScore Precision - semantic similarity between response and reference
- **Range**: -1.0 to 1.0 (higher is better, typically 0.0 to 1.0)
- **Type**: Algorithmic (BERT embeddings)
- **Use case**: Semantic precision - how much of the response is relevant to the reference

### `BertscoreR`
- **What it measures**: BERTScore Recall - semantic similarity between reference and response
- **Range**: -1.0 to 1.0 (higher is better, typically 0.0 to 1.0)
- **Type**: Algorithmic (BERT embeddings)
- **Use case**: Semantic recall - how much of the reference is covered by the response

### `BertKPrec`
- **What it measures**: BERT-based keyword precision comparing response to **passages**
- **Range**: -1.0 to 1.0 (higher is better, typically 0.0 to 1.0)
- **Type**: Algorithmic (BERT embeddings)
- **Use case**: Approximates faithfulness and completeness relative to source passages
- **Note**: Will be `null` when `contexts` array is empty (no passages available)

### `Extractiveness_RougeL`
- **What it measures**: ROUGE-L based extractiveness from **passages**
- **Range**: 0.0 to 1.0 (higher = more extractive)
- **Type**: Algorithmic
- **Use case**: Measures how much the response copies directly from passages
- **Note**: Will be `null` when `contexts` array is empty (no passages available)

### `Length`
- **What it measures**: Number of words in the model's response
- **Range**: Integer ≥ 0
- **Type**: Count
- **Use case**: Response verbosity analysis

### `RB_agg`
- **What it measures**: Reference-Based Aggregate score
- **Formula**: Harmonic mean of `BertscoreR`, `RougeL_stemFalse`, and extractiveness
- **Range**: 0.0 to 1.0 (higher is better)
- **Type**: Composite algorithmic metric
- **Use case**: Overall algorithmic quality score combining completeness, appropriateness, and faithfulness
- **Details**: 
  ```python
  recall = (BertscoreR + 1) / 2
  rouge = RougeL_stemFalse
  extractiveness = (max(BertKPrec) + 1) / 2  # or 0 if BertKPrec missing
  
  denominator = (recall × rouge) + (recall × extractiveness) + (rouge × extractiveness)
  RB_agg = 3 × recall × rouge × extractiveness / denominator
  ```

### `RB_agg_zero_denominator`
- **What it measures**: Flag indicating if RB_agg calculation had zero denominator
- **Values**: `true` or `false`
- **Type**: Boolean flag
- **Use case**: Debugging/quality control for RB_agg calculation
- **Note**: When `true`, RB_agg is set to 0

---

## Step 2: IDK Judge

**Script**: `scripts/evaluation/judge_wrapper.py` → `run_idk_judge()`  
**Purpose**: Detect "I Don't Know" responses using LLM  
**Metrics Added**: 1

### `idk_eval`
- **What it measures**: Whether the model response indicates it doesn't know the answer
- **Values**: 
  - `0.0` = Response provides a full/partial answer
  - `0.5` = Response provides a partial answer with some IDK indication
  - `1.0` = Response fully indicates "I don't know"
- **Type**: LLM-based classification
- **Model**: GPT-4o-mini (default)
- **Use case**: Identify when models appropriately decline to answer unanswerable questions
- **Accuracy**: >97% on human evaluation

---

## Step 3: RAGAS Faithfulness

**Script**: `scripts/evaluation/judge_wrapper.py` → `run_ragas_judges_openai()` or `run_ragas_judges_local()`  
**Purpose**: Evaluate faithfulness using RAGAS framework  
**Metrics Added**: 1

### `RL_F`
- **What it measures**: Reference-Less Faithfulness score
- **Range**: 0.0 to 1.0 (higher = more faithful)
- **Type**: LLM-based (reference-less)
- **Model**: OpenAI API or local LLM
- **Use case**: Evaluates if the response is faithful to the provided documents/context
- **Method**: LLM analyzes response against passages to detect hallucinations
- **Note**: "Reference-less" means it doesn't require the reference answer, only passages

---

## Step 4: RADBench LLM Judge

**Script**: `scripts/evaluation/judge_wrapper.py` → `run_radbench_judge()`  
**Purpose**: Comprehensive LLM-based evaluation  
**Metrics Added**: 1

### `RB_llm`
- **What it measures**: Reference-Based LLM judge score evaluating overall quality
- **Range**: 0.0 to 1.0 (normalized from 1-10 scale)
- **Type**: LLM-based (reference-based)
- **Model**: GPT-4o-mini (default)
- **Evaluates**:
  - **Faithfulness**: Is the answer faithful to documents and conversation?
  - **Appropriateness**: Is the answer relevant to the question?
  - **Completeness**: Does the answer include all important information?
- **Method**: LLM compares model response to reference answer and passages
- **Inputs**: Previous conversation, current question, reference answer, model response, passages
- **Correlation**: Correlates well with human judgments (see MT-RAG paper)

---

## Step 5: IDK Conditioning

**Script**: `scripts/evaluation/judge_wrapper.py` → `get_idk_conditioned_metrics()`  
**Purpose**: Adjust metrics based on question answerability  
**Metrics Added**: 3

### `RB_agg_idk`
- **What it measures**: IDK-conditioned version of `RB_agg`
- **Range**: 0.0 to 1.0 (higher is better)
- **Type**: Conditioned algorithmic metric
- **Logic**:
  - If question is unanswerable AND model says IDK → `1.0` (correct behavior)
  - If question is unanswerable AND model answers → `0.0` (hallucination)
  - If question is answerable → use original `RB_agg` score

### `RB_llm_idk`
- **What it measures**: IDK-conditioned version of `RB_llm`
- **Range**: 0.0 to 1.0 (higher is better)
- **Type**: Conditioned LLM metric
- **Logic**: Same as `RB_agg_idk` but applied to `RB_llm`

### `RL_F_idk`
- **What it measures**: IDK-conditioned version of `RL_F`
- **Range**: 0.0 to 1.0 (higher is better)
- **Type**: Conditioned LLM metric
- **Logic**: Same as `RB_agg_idk` but applied to `RL_F`

**Importance**: IDK-conditioned metrics properly reward models that correctly decline to answer unanswerable questions, which is critical for trustworthy RAG systems.

---

## Quick Reference Table

| Metric | Step | Type | Range | Requires Reference | Requires Passages | Description |
|--------|------|------|-------|-------------------|-------------------|-------------|
| `Recall` | 1 | Algo | 0-1 | ✅ | ❌ | Overlap with reference |
| `RougeL_stemFalse` | 1 | Algo | 0-1 | ✅ | ❌ | ROUGE-L score |
| `BertscoreP` | 1 | Algo | -1 to 1 | ✅ | ❌ | BERT semantic precision |
| `BertscoreR` | 1 | Algo | -1 to 1 | ✅ | ❌ | BERT semantic recall |
| `BertKPrec` | 1 | Algo | -1 to 1 | ❌ | ✅ | BERT keyword precision vs passages |
| `Extractiveness_RougeL` | 1 | Algo | 0-1 | ❌ | ✅ | Extractiveness from passages |
| `Length` | 1 | Count | ≥0 | ❌ | ❌ | Response word count |
| `RB_agg` | 1 | Composite | 0-1 | ✅ | ⚠️ | Harmonic mean (recall, rouge, extract) |
| `RB_agg_zero_denominator` | 1 | Flag | bool | ✅ | ❌ | Zero denominator flag |
| `idk_eval` | 2 | LLM | 0, 0.5, 1 | ❌ | ❌ | IDK detection |
| `RL_F` | 3 | LLM | 0-1 | ❌ | ✅ | RAGAS faithfulness |
| `RB_llm` | 4 | LLM | 0-1 | ✅ | ✅ | RADBench comprehensive score |
| `RB_agg_idk` | 5 | Conditioned | 0-1 | ✅ | ⚠️ | IDK-conditioned RB_agg |
| `RB_llm_idk` | 5 | Conditioned | 0-1 | ✅ | ✅ | IDK-conditioned RB_llm |
| `RL_F_idk` | 5 | Conditioned | 0-1 | ❌ | ✅ | IDK-conditioned RL_F |

**Legend**:
- ✅ = Required
- ❌ = Not required
- ⚠️ = Uses if available, but can compute without
- Algo = Algorithmic
- LLM = LLM-based evaluation

---

## Notes on Missing Values

### When `BertKPrec` and `Extractiveness_RougeL` are `null`:
These metrics require passages (from the `contexts` field) to compute. When `contexts = []` (empty), these metrics cannot be calculated and will be `null`.

**Common scenarios**:
- Unanswerable questions in the reference scenario (no relevant passages exist)
- Conversational questions where no retrieval was performed
- Approximately 65/842 records per model in the reference scenario have empty contexts

**Impact**: `RB_agg` can still be computed but uses `extractiveness = 0` when `BertKPrec` is missing.

---

## Recommended Metrics for Analysis

### For Overall Quality:
- **`RB_llm_idk`** - Best overall metric (LLM-based, handles answerability)
- **`RB_agg_idk`** - Best algorithmic metric (fast, handles answerability)

### For Specific Properties:
- **Faithfulness**: `RL_F_idk`, `BertKPrec`
- **Completeness**: `BertscoreR`, `Recall`
- **Appropriateness**: `RougeL_stemFalse`, `RB_llm`
- **Extractiveness**: `Extractiveness_RougeL`
- **Hallucination Detection**: `idk_eval`, `RL_F`

### For Comparison with MT-RAG Paper:
The paper primarily uses:
- `RBalg` (our `RB_agg_idk`)
- `RBllm` (our `RB_llm_idk`)
- `RLF` (our `RL_F_idk`)
- Answerability accuracy (based on `idk_eval`)

---

## Running the Evaluation Pipeline

### Full Pipeline (All Steps):
```bash
python scripts/evaluation/run_step_by_step.py \
  --step all \
  -i <input>.jsonl \
  -o <output>.jsonl \
  --provider openai
```

### Individual Steps:
```bash
# Step 1: Algorithmic
python scripts/evaluation/run_step_by_step.py --step algorithmic -i <file>.jsonl -o <file>.jsonl -e scripts/evaluation/config.yaml

# Step 2: IDK Judge
python scripts/evaluation/run_step_by_step.py --step idk -i <file>.jsonl -o <file>.jsonl --provider openai

# Step 3: RAGAS
python scripts/evaluation/run_step_by_step.py --step ragas -i <file>.jsonl -o <file>.jsonl --provider openai

# Step 4: RADBench
python scripts/evaluation/run_step_by_step.py --step radbench -i <file>.jsonl -o <file>.jsonl --provider openai

# Step 5: IDK Conditioning
python scripts/evaluation/run_step_by_step.py --step idk_condition -i <file>.jsonl -o <file>.jsonl
```

---

## References

- **MT-RAG Paper**: "MT RAG: A Multi-Turn Conversational Benchmark for Evaluating Retrieval-Augmented Generation Systems"
- **RAGAS**: Es et al., 2024 - "RAGAs: Automated evaluation of retrieval augmented generation"
- **RADBench**: Kuo et al., 2024 - "RAD-Bench: Evaluating large language models capabilities in retrieval augmented dialogues"
- **BERTScore**: Zhang et al., 2020 - "BERTScore: Evaluating Text Generation with BERT"

---

**Last Updated**: November 4, 2024  
**Version**: 1.0

