# QPP-GenRE Experiments to Improve Retrieval Results

## Executive Summary

Based on analysis of current retrieval results, we've identified several key struggles:
1. **BM25 underperformance** (R@10: 0.27-0.35, nDCG@10: 0.21-0.26)
2. **BGE-base 1.5 room for improvement** (R@10: 0.38-0.50, nDCG@10: 0.30-0.40)
3. **Oracle routing potential** (+13-29% improvement if we could predict best strategy per query)
4. **MMR Cluster strategies** consistently underperform Query Rewrite

**QPP-GenRE** offers a solution: predict retrieval quality using LLM-generated relevance judgments, enabling:
- Post-retrieval quality validation
- Per-query strategy selection
- Interpretable error diagnosis

---

## Pre-trained Fine-Tuned Checkpoints

We will use **pre-trained models fine-tuned on MS MARCO V1** passage ranking dataset, which are ready to use immediately:

- **LLaMA-7B** checkpoint: [Download](https://drive.google.com/file/d/1UCoHRhm4n3bYl07eVvLRHmxABuEQknXU/view?usp=share_link)
- **Llama-3-8B** checkpoint: [Download](https://drive.google.com/file/d/1AOw73Um8p-d_MOn9CxNZSe_ufvYXRlgs/view?usp=share_link)
- **Llama-3-8B-Instruct** checkpoint: [Download](https://drive.google.com/file/d/15VvHS9jV-1J8RwfGJCmaJTNEgaP-5_qC/view?usp=share_link)

**Why Pre-trained Checkpoints:**
- **Best performance**: State-of-the-art results on TREC-DL 19-22 datasets
- **Proven approach**: Already validated on IR tasks
- **Domain transfer**: MS MARCO is general enough that it should transfer reasonably well to conversational search
- **Time savings**: Validate the approach immediately, then fine-tune if needed

**Recommended Model:** Llama-3-8B-Instruct (best balance of performance and ease of use)

---

## Experiment 1: QPP-Based Strategy Selection

### Problem
Oracle routing shows +13-29% improvement potential, but we lack a reliable way to predict which strategy (Last Turn, Rewrite, Questions) will perform best for each query.

### Hypothesis
QPP-GenRE can predict nDCG@10 for each strategy's retrieval results, enabling us to select the best-performing strategy per query.

### Experimental Design

**Setup:**
1. For each query, run all three strategies (Last Turn, Rewrite, Questions) through first-stage retrieval (ELSER)
2. Use QPP-GenRE (pre-trained checkpoint) to predict nDCG@10 for each strategy's top-10 results
3. Select the strategy with highest predicted nDCG@10
4. Compare against:
   - Best single strategy baseline (Rewrite)
   - Oracle routing (ground truth best strategy)

**Metrics:**
- R@1, R@3, R@5, R@10
- nDCG@1, nDCG@3, nDCG@5, nDCG@10
- Strategy selection accuracy (% of queries where QPP-GenRE selects the oracle strategy)
- Correlation between predicted nDCG@10 and actual nDCG@10

**Expected Outcome:**
- Achieve 50-70% of oracle routing improvement (target: +6-15% over Rewrite baseline)
- Better than current fusion approaches (+0.4-3%)

**Implementation Steps:**
1. Download pre-trained Llama-3-8B-Instruct checkpoint
2. Run inference for all three strategies on test set using `judge_relevance.py`
3. Compute predicted nDCG@10 using `predict_measures.py`
4. Implement selection logic: `best_strategy = argmax(predicted_ndcg@10)`
5. Evaluate on held-out test set
6. **If needed**: Fine-tune on domain data (Experiment 2) for better performance

---

## Experiment 2: Domain-Specific Fine-Tuning

### Problem
Different domains show different strategy preferences and confidence patterns:
- **FIQA**: Rewrite-heavy (financial context benefits from LLM rewriting)
- **CLOUD**: Questions-preferring (technical docs align with explicit formats)
- **GOVT**: Low confidence overall (15-20 range, potential domain mismatch)

### Hypothesis
Fine-tuning QPP-GenRE on domain-specific data will improve:
1. Prediction accuracy for that domain
2. Calibration (predicted vs actual nDCG alignment)
3. Strategy selection accuracy

### Experimental Design

**Setup:**
1. Create training data from each domain:
   - Use existing qrels (relevance judgments)
   - Generate query-document pairs from retrieval runs
   - Format as QPP-GenRE training examples

2. Fine-tune QPP-GenRE separately for each domain:
   - FIQA: Financial Q&A pairs
   - GOVT: Policy documents
   - CLOUD: Technical documentation
   - CLAPNQ: General conversational search

3. Evaluate:
   - In-domain: Test on same domain
   - Cross-domain: Test generalization
   - Compare against pre-trained checkpoint baseline (MS MARCO fine-tuned)

**Metrics:**
- Domain-specific nDCG@10 prediction accuracy
- Strategy selection accuracy per domain
- Cross-domain generalization (fine-tuned on Domain A, test on Domain B)
- Improvement over pre-trained checkpoint baseline

**Expected Outcome:**
- Domain-specific models improve prediction accuracy by 10-20% over pre-trained checkpoint
- Better calibration for domain-specific confidence patterns
- GOVT domain shows largest improvement (addressing low confidence issue)

**Implementation Steps:**
1. Extract query-document pairs with relevance labels from each domain
2. Format as QPP-GenRE training data
3. Fine-tune using QLoRA (4-bit quantization) for efficiency
4. Evaluate on held-out test sets per domain
5. Compare against pre-trained checkpoint baseline

---

## Resources Needed

### For Experiment 1 (Pre-trained Checkpoints)

1. **Compute**: 
   - GPU for inference (A100 or similar, can use smaller GPUs)
   - ~8GB GPU memory for Llama-3-8B-Instruct (4-bit quantized)

2. **Models**:
   - Download pre-trained checkpoint: [Llama-3-8B-Instruct](https://drive.google.com/file/d/15VvHS9jV-1J8RwfGJCmaJTNEgaP-5_qC/view?usp=share_link) (~7-8GB)

3. **Data**:
   - Retrieval run files (already have)
   - Query-document pairs (already have)
   - Qrels for evaluation (already have)

4. **Infrastructure**:
   - Clone QPP-GenRE repository
   - Install dependencies (`pip install -r requirements.txt`)
   - Run `judge_relevance.py` and `predict_measures.py`

### For Experiment 2 (Domain Fine-Tuning)

1. **Compute**: 
   - GPU for fine-tuning (A100 40GB recommended)
   - ~1.5 hours per domain on A100

2. **Data**:
   - Existing qrels (relevance judgments)
   - Query-document pairs formatted for training

3. **Models**:
   - Base model from Hugging Face (Llama-3-8B or Llama-3-8B-Instruct)
   - Fine-tuned checkpoints per domain (created during training)

---

## Related Work

- QPP-GenRE paper: [Query Performance Prediction using Relevance Judgments Generated by Large Language Models](https://dl.acm.org/doi/10.1145/3736402)
- QPP-GenRE repository: [https://github.com/ChuanMeng/QPP-GenRE](https://github.com/ChuanMeng/QPP-GenRE)
- Current analysis: `knowledgebase/retrieval/relevance_scores_analysis.md`
- Oracle routing analysis: `scripts/ideas/retrieval_tasks/oracle/ORACLE_ROUTING_ANALYSIS.md`
- QPP-GenRE integration plan: `knowledgebase/retrieval/QPP_GenRE_plan.md`
