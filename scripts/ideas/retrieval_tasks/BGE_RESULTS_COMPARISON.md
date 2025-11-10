# BGE Dense Retrieval Results: Extractive Query Rewriting

**Date:** November 10, 2024  
**Retrieval Method:** BGE-base-en-v1.5 (Dense/Semantic Retrieval)  
**Evaluation Metrics:** NDCG@k and Recall@k (k=1,3,5,10)

## Executive Summary

This document evaluates extractive query rewriting with **semantic/dense retrieval (BGE)** instead of lexical retrieval (BM25). Key question: **Do templates and well-formed queries help semantic retrieval more than lexical retrieval?**

---

## Overall Performance Summary (NDCG@10 / Recall@10)

| Domain | Lastturn | Human Rewrite | Pure Extractive | Hybrid Extractive | Best Method |
|--------|----------|---------------|-----------------|-------------------|-------------|
| **clapnq** | 0.424 / 0.522 | **0.498 / 0.606** | 0.406 / 0.511 | 0.399 / 0.496 | Human ✅ |
| **cloud** | 0.307 / 0.381 | **0.342 / 0.423** | 0.285 / 0.362 | 0.290 / 0.365 | Human ✅ |
| **fiqa** | 0.291 / 0.367 | **0.341 / 0.418** | 0.234 / 0.294 | 0.236 / 0.309 | Human ✅ |
| **govt** | 0.344 / 0.427 | **0.420 / 0.528** | 0.306 / 0.414 | 0.303 / 0.404 | Human ✅ |
| **Average** | 0.342 / 0.424 | **0.400 / 0.494** | 0.308 / 0.395 | 0.307 / 0.393 | Human ✅ |

**Key Finding:** BGE performance is **significantly higher** than BM25 across all methods, but extractive methods still lag behind baselines.

---

## Detailed Results by Domain

### ClapNQ (Wikipedia Q&A) - 208 queries

| Method | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 | Recall@1 | Recall@3 | Recall@5 | Recall@10 |
|--------|--------|--------|--------|---------|----------|----------|----------|-----------|
| Lastturn | 0.394 | 0.341 | 0.367 | 0.424 | 0.155 | 0.304 | 0.388 | 0.522 |
| Human Rewrite | **0.471** | **0.412** | **0.438** | **0.498** | **0.174** | **0.375** | **0.462** | **0.606** |
| Pure Extractive | 0.332 | 0.321 | 0.353 | 0.406 | 0.127 | 0.296 | 0.390 | 0.511 |
| Hybrid Extractive | 0.351 | 0.322 | 0.344 | 0.399 | 0.134 | 0.296 | 0.367 | 0.496 |

**Ranking:** Human (0.498) > Lastturn (0.424) > Pure (0.406) > Hybrid (0.399)

**Analysis:**
- ❌ Both extractive methods **underperform lastturn** with BGE
- 🔴 **Pure performs better than Hybrid** (opposite of expected!)
- 📉 18.5% behind human rewrites (vs 7% with BM25)
- 💡 Keyword spam (Pure) works better than templates (Hybrid) even for semantic retrieval

---

### Cloud (Technical Documentation) - 188 queries

| Method | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 | Recall@1 | Recall@3 | Recall@5 | Recall@10 |
|--------|--------|--------|--------|---------|----------|----------|----------|-----------|
| Lastturn | 0.250 | 0.253 | 0.276 | 0.307 | 0.122 | 0.254 | 0.312 | 0.381 |
| Human Rewrite | **0.293** | **0.276** | **0.303** | **0.342** | **0.148** | **0.271** | **0.338** | **0.423** |
| Pure Extractive | 0.229 | 0.222 | 0.252 | 0.285 | 0.106 | 0.222 | 0.285 | 0.362 |
| Hybrid Extractive | 0.234 | 0.231 | 0.253 | 0.290 | 0.109 | 0.226 | 0.277 | 0.365 |

**Ranking:** Human (0.342) > Lastturn (0.307) > Hybrid (0.290) > Pure (0.285)

**Analysis:**
- ❌ Both extractive methods underperform both baselines
- 🤔 Hybrid slightly better than Pure (+1.8%)
- 📊 Still worse than lastturn
- 💡 Technical queries still don't benefit from extractive rewriting

---

### FiQA (Financial Q&A) - 180 queries

| Method | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 | Recall@1 | Recall@3 | Recall@5 | Recall@10 |
|--------|--------|--------|--------|---------|----------|----------|----------|-----------|
| Lastturn | 0.239 | 0.227 | 0.248 | 0.291 | 0.103 | 0.207 | 0.263 | 0.367 |
| Human Rewrite | **0.311** | **0.274** | **0.294** | **0.341** | **0.130** | **0.249** | **0.308** | **0.418** |
| Pure Extractive | 0.206 | 0.187 | 0.202 | 0.234 | 0.079 | 0.176 | 0.216 | 0.294 |
| Hybrid Extractive | 0.183 | 0.178 | 0.197 | 0.236 | 0.079 | 0.173 | 0.218 | 0.309 |

**Ranking:** Human (0.341) > Lastturn (0.291) > Hybrid (0.236) > Pure (0.234)

**Analysis:**
- ❌ Both extractive methods significantly underperform baselines
- 🔴 **Worst domain** for extractive approaches
- 📉 19-31% behind baselines
- 💭 Financial semantic queries don't work well with keyword extraction

---

### Govt (Government/Policy) - 201 queries

| Method | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 | Recall@1 | Recall@3 | Recall@5 | Recall@10 |
|--------|--------|--------|--------|---------|----------|----------|----------|-----------|
| Lastturn | 0.299 | 0.277 | 0.305 | 0.344 | 0.133 | 0.264 | 0.334 | 0.427 |
| Human Rewrite | **0.343** | **0.329** | **0.368** | **0.420** | **0.157** | **0.313** | **0.404** | **0.528** |
| Pure Extractive | 0.219 | 0.222 | 0.260 | 0.306 | 0.096 | 0.218 | 0.307 | 0.414 |
| Hybrid Extractive | 0.229 | 0.228 | 0.256 | 0.303 | 0.103 | 0.219 | 0.289 | 0.404 |

**Ranking:** Human (0.420) > Lastturn (0.344) > Pure (0.306) > Hybrid (0.303)

**Analysis:**
- ❌ Both extractive methods underperform baselines
- 🔴 Even govt domain (best for BM25) struggles with BGE
- 📉 11-27% behind baselines
- 💡 Semantic retrieval needs better query formulation

---

## Critical Finding: BGE vs BM25 Comparison

### Pure Extractive Performance

| Domain | BM25 NDCG@10 | BGE NDCG@10 | BGE Improvement | BGE Better? |
|--------|--------------|-------------|-----------------|-------------|
| clapnq | 0.290 | 0.406 | **+40%** | ✅ YES |
| cloud | 0.239 | 0.285 | **+19%** | ✅ YES |
| fiqa | 0.152 | 0.234 | **+54%** | ✅ YES |
| govt | 0.339 | 0.306 | **-10%** | ❌ NO |

**Insight:** BGE dramatically improves performance for Pure Extractive in 3/4 domains!

### Hybrid Extractive Performance

| Domain | BM25 NDCG@10 | BGE NDCG@10 | BGE Improvement | BGE Better? |
|--------|--------------|-------------|-----------------|-------------|
| clapnq | 0.284 | 0.399 | **+41%** | ✅ YES |
| cloud | 0.241 | 0.290 | **+20%** | ✅ YES |
| fiqa | 0.155 | 0.236 | **+52%** | ✅ YES |
| govt | 0.336 | 0.303 | **-10%** | ❌ NO |

**Insight:** BGE also dramatically improves Hybrid performance (except govt domain).

---

## Pure vs Hybrid with BGE

| Domain | Pure BGE | Hybrid BGE | Difference | Winner |
|--------|----------|------------|------------|---------|
| clapnq | **0.406** | 0.399 | -1.7% | Pure ✅ |
| cloud | 0.285 | **0.290** | +1.8% | Hybrid ✅ |
| fiqa | 0.234 | **0.236** | +0.9% | Hybrid ✅ |
| govt | **0.306** | 0.303 | -1.0% | Pure ✅ |
| **Average** | **0.308** | 0.307 | **-0.3%** | **Tie** |

**SURPRISING RESULT:** Pure and Hybrid still perform **nearly identically** with BGE! 

**This contradicts the hypothesis** that well-formed queries would help semantic retrieval.

---

## Key Insights

### 1. BGE Dramatically Improves Performance

**Average NDCG@10 gains from BM25 → BGE:**
- Pure Extractive: +26% average (huge!)
- Hybrid Extractive: +26% average (huge!)
- But both still lag behind baselines

### 2. Pure ≈ Hybrid (Even with BGE!)

Despite expectations that:
- ✅ Templates would help semantic understanding
- ✅ Well-formed questions would encode better
- ✅ Named entities would be preserved

**Reality:** Performance is nearly identical (0.3% difference)

**Why?**
- BGE embeddings capture semantic meaning regardless of syntax
- Keyword repetition doesn't hurt (embeddings are averaged)
- Templates add structure that BGE doesn't need

### 3. Both Methods Still Underperform Baselines

**Gap from Human Rewrite:**
- Pure: -23% (BGE) vs -7.5% (BM25)
- Hybrid: -23% (BGE) vs -7.6% (BM25)

**Gap widened with BGE!** This suggests:
- Semantic retrieval is more sensitive to query quality
- Keyword extraction isn't enough for dense retrieval
- Human rewrites add semantic meaning that extractive methods miss

### 4. Government Domain Exception

**Only domain where BM25 > BGE for extractive methods:**
- Pure: BM25 (0.339) > BGE (0.306)
- Hybrid: BM25 (0.336) > BGE (0.303)

**Possible reasons:**
- Government queries benefit from keyword density
- Policy terminology needs exact matches
- Semantic similarity may introduce false positives

---

## Comparison: BM25 vs BGE Overall

### Which Retrieval Method is Better?

| Scenario | BM25 | BGE | Winner |
|----------|------|-----|---------|
| **Human Rewrite** | 0.272 | **0.400** | BGE (+47%) |
| **Lastturn** | 0.244 | **0.342** | BGE (+40%) |
| **Pure Extractive** | 0.255 | **0.308** | BGE (+21%) |
| **Hybrid Extractive** | 0.254 | **0.307** | BGE (+21%) |

**Clear winner:** BGE outperforms BM25 for **all** query rewriting approaches.

**BUT:** The **gap from human performance grows** with BGE:
- BM25: Extractive is 93% of human performance
- BGE: Extractive is only 77% of human performance

---

## Recommendations

### For Extractive Query Rewriting

1. **❌ Don't use with BGE currently** - Large performance gap from baselines
   - Pure: -23% vs human
   - Hybrid: -23% vs human
   - Better to use lastturn or human rewrites

2. **✅ Stick with BM25** - Extractive methods are more competitive
   - Pure: -7.5% vs human (acceptable)
   - Hybrid: -7.6% vs human (acceptable)

3. **🔧 Major improvements needed** for BGE compatibility:
   - Current keyword extraction doesn't capture semantic meaning
   - Need better term selection that considers semantic coherence
   - May need completely different approach for dense retrieval

### Pure vs Hybrid (Again!)

**Even with semantic retrieval:**
- Pure ≈ Hybrid (0.3% difference)
- Templates don't help
- **Conclusion:** Pure is still the winner (simpler, equivalent performance)

### Domain-Specific Insights

**ClapNQ (Wikipedia):**
- Best for extractive methods
- But still 15-20% behind baselines with BGE

**Cloud (Technical):**
- Consistent underperformance
- BGE helps (+20%) but still not enough
- Avoid extractive for technical domains

**FiQA (Financial):**
- Worst domain for extractive methods
- 19-31% behind baselines
- Semantic understanding crucial

**Govt (Policy):**
- Paradox: Better with BM25 than BGE for extractive
- Suggests keyword-based matching is better for policy queries
- BGE introduces too many false positives

---

## Combined BM25 + BGE Summary

### Overall Rankings (NDCG@10 Average)

**With BM25:**
1. Human: 0.272
2. Pure: 0.255 (-6.3%)
3. Hybrid: 0.254 (-6.6%)
4. Lastturn: 0.244

**With BGE:**
1. Human: 0.400
2. Lastturn: 0.342 (-14.5%)
3. Pure: 0.308 (-23.0%)
4. Hybrid: 0.307 (-23.3%)

**Key Observation:** Ranking changes! Lastturn beats extractive methods with BGE.

### Extractive Methods: BM25 vs BGE

| Method | Best With | Reason |
|--------|-----------|--------|
| Pure Extractive | **BGE** | +21% average improvement |
| Hybrid Extractive | **BGE** | +21% average improvement |
| **BUT** | BM25 is more **competitive** | Smaller gap from baselines |

---

## The Big Picture

### What We Learned

1. **Templates don't matter** - Pure ≈ Hybrid for both BM25 and BGE
2. **BGE is powerful** - Dramatically improves all methods
3. **But extractive falls behind** - Gap from baselines grows with BGE
4. **Domain matters** - Technical/financial struggle most

### Why Extractive Methods Struggle with BGE

**Hypothesis:** Extractive methods produce **keyword-rich but semantically incoherent** queries:

```
Original: "Do the Arizona Cardinals play outside the US?"
Human: "Where do the Arizona Cardinals play, regardless of location, this week?"
Pure: "where arizona cardinals play outside us cardinals play outside..."
```

**With BM25:** Keywords match, TF-IDF weights them  
**With BGE:** Semantic embedding is confused by repetition and lack of coherence

The model tries to create a semantic representation of:
- "cardinals play outside arizona cardinals play"

vs

- "Where do the Arizona Cardinals play, regardless of location?"

The second has **semantic coherence** that BGE can encode properly.

---

## Conclusion & Future Work

### Current Status

✅ **Successfully tested** both extractive methods with BM25 and BGE  
⚠️ **Mixed results:**
- BM25: Competitive (-7% from human)
- BGE: Not competitive (-23% from human)

### Recommended Path Forward

**Short term (Use now):**
- Use **Pure Extractive + BM25** for non-technical domains
- Skip extractive methods for Cloud/FiQA
- 93% of human performance at zero annotation cost

**Long term (Research):**
1. **Improve semantic coherence:**
   - Better n-gram selection (prefer longer, semantically complete phrases)
   - Sentence-level context preservation
   - Avoid keyword repetition

2. **Test deduplication:**
   - Remove overlapping n-grams
   - May significantly improve BGE performance

3. **Hybrid approaches:**
   - Use extractive for BM25
   - Use human/LLM rewrites for BGE
   - Ensemble both retrievers

4. **Investigate govt domain:**
   - Why does BM25 > BGE for extractive methods?
   - May inform better keyword selection strategies

### Final Recommendation

For the MT-RAG benchmark:
- **Production use:** Human rewrites with BGE (best overall)
- **Low-resource alternative:** Pure Extractive with BM25 (good enough)
- **Avoid:** Extractive methods with BGE (current implementation)
- **Skip Hybrid:** No advantage over Pure in either retrieval method

