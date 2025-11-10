# ELSER Retrieval Results: Extractive Query Rewriting

**Date:** November 10, 2024  
**Retrieval Method:** ELSER v2 (Elastic Learned Sparse Encoder)  
**Evaluation Metrics:** NDCG@k and Recall@k (k=1,3,5,10)

## Executive Summary

ELSER is a learned sparse retrieval method that combines benefits of lexical (BM25) and semantic (BGE) retrieval. This evaluation tests how extractive query rewriting performs with ELSER.

---

## Overall Performance Summary (NDCG@10 / Recall@10)

| Domain | Lastturn | Human Rewrite | Pure Extractive | Hybrid Extractive | Best Method |
|--------|----------|---------------|-----------------|-------------------|-------------|
| **clapnq** | 0.527 / 0.630 | **0.578 / 0.701** | 0.460 / 0.579 | 0.458 / 0.566 | Human ✅ |
| **cloud** | 0.427 / 0.504 | **0.438 / 0.528** | 0.328 / 0.409 | 0.308 / 0.383 | Human ✅ |
| **fiqa** | 0.391 / 0.472 | **0.436 / 0.536** | 0.333 / 0.407 | 0.302 / 0.368 | Human ✅ |
| **govt** | 0.449 / 0.559 | **0.517 / 0.651** | 0.446 / 0.562 | 0.428 / 0.548 | Human ✅ |
| **Average** | 0.449 / 0.541 | **0.492 / 0.604** | 0.392 / 0.489 | 0.374 / 0.466 | Human ✅ |

**Key Finding:** ELSER shows **highest absolute performance** of all retrieval methods, but extractive methods still lag behind baselines significantly.

---

## Detailed Results by Domain

### ClapNQ (Wikipedia Q&A) - 208 queries

| Method | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 | Recall@1 | Recall@3 | Recall@5 | Recall@10 |
|--------|--------|--------|--------|---------|----------|----------|----------|-----------|
| Lastturn | 0.476 | 0.437 | 0.475 | 0.527 | 0.198 | 0.397 | 0.511 | 0.630 |
| Human Rewrite | **0.524** | **0.470** | **0.513** | **0.578** | **0.209** | **0.424** | **0.552** | **0.701** |
| Pure Extractive | 0.389 | 0.363 | 0.400 | 0.460 | 0.149 | 0.333 | 0.440 | 0.579 |
| Hybrid Extractive | 0.394 | 0.368 | 0.396 | 0.458 | 0.169 | 0.332 | 0.418 | 0.566 |

**Ranking:** Human (0.578) > Lastturn (0.527) > Pure (0.460) > Hybrid (0.458)

**Analysis:**
- ❌ Both extractive methods underperform both baselines
- 🔴 12.7% and 13.4% behind lastturn
- 📉 20.4% and 20.8% behind human
- 💡 Even with ELSER, extractive methods struggle

---

### Cloud (Technical Documentation) - 188 queries

| Method | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 | Recall@1 | Recall@3 | Recall@5 | Recall@10 |
|--------|--------|--------|--------|---------|----------|----------|----------|-----------|
| Lastturn | 0.372 | 0.367 | 0.389 | 0.427 | 0.179 | 0.353 | 0.420 | 0.504 |
| Human Rewrite | **0.378** | **0.365** | **0.394** | **0.438** | **0.179** | **0.353** | **0.430** | **0.528** |
| Pure Extractive | 0.271 | 0.255 | 0.286 | 0.328 | 0.125 | 0.245 | 0.317 | 0.409 |
| Hybrid Extractive | 0.261 | 0.241 | 0.276 | 0.308 | 0.114 | 0.235 | 0.314 | 0.383 |

**Ranking:** Human (0.438) > Lastturn (0.427) > Pure (0.328) > Hybrid (0.308)

**Analysis:**
- ❌ Extractive methods perform **very poorly** on technical domain
- 🔴 23.2% (pure) and 27.9% (hybrid) behind lastturn
- 💡 Technical queries don't benefit from keyword extraction even with ELSER

---

### FiQA (Financial Q&A) - 180 queries

| Method | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 | Recall@1 | Recall@3 | Recall@5 | Recall@10 |
|--------|--------|--------|--------|---------|----------|----------|----------|-----------|
| Lastturn | 0.344 | 0.320 | 0.348 | 0.391 | 0.146 | 0.293 | 0.370 | 0.472 |
| Human Rewrite | **0.389** | **0.344** | **0.378** | **0.436** | **0.163** | **0.310** | **0.402** | **0.536** |
| Pure Extractive | 0.317 | 0.268 | 0.292 | 0.333 | 0.140 | 0.244 | 0.311 | 0.407 |
| Hybrid Extractive | 0.289 | 0.244 | 0.262 | 0.302 | 0.129 | 0.228 | 0.276 | 0.368 |

**Ranking:** Human (0.436) > Lastturn (0.391) > Pure (0.333) > Hybrid (0.302)

**Analysis:**
- ❌ Extractive significantly behind both baselines
- 🔴 14.8% (pure) and 22.8% (hybrid) behind lastturn
- 💡 Financial domain continues to be challenging

---

### Govt (Government/Policy) - 201 queries

| Method | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 | Recall@1 | Recall@3 | Recall@5 | Recall@10 |
|--------|--------|--------|--------|---------|----------|----------|----------|-----------|
| Lastturn | 0.353 | 0.361 | 0.400 | 0.449 | 0.164 | 0.345 | 0.445 | 0.559 |
| Human Rewrite | **0.413** | **0.407** | **0.454** | **0.517** | **0.194** | **0.392** | **0.508** | **0.651** |
| Pure Extractive | 0.348 | 0.359 | 0.396 | 0.446 | 0.166 | 0.357 | 0.444 | 0.562 |
| Hybrid Extractive | 0.333 | 0.347 | 0.382 | 0.428 | 0.155 | 0.350 | 0.441 | 0.548 |

**Ranking:** Human (0.517) > Lastturn (0.449) > Pure (0.446) > Hybrid (0.428)

**Analysis:**
- ✅ **Pure Extractive nearly ties lastturn!** (-0.7%)
- 🎯 Best domain for extractive approaches with ELSER
- 📊 Still 13.7% behind human rewrites
- 💡 Government queries benefit from comprehensive keyword coverage

---

## Critical Findings

### 1. Pure vs Hybrid with ELSER

| Domain | Pure NDCG@10 | Hybrid NDCG@10 | Difference | Winner |
|--------|--------------|----------------|------------|---------|
| clapnq | **0.460** | 0.458 | -0.4% | Pure ✅ |
| cloud | **0.328** | 0.308 | -6.1% | Pure ✅ |
| fiqa | **0.333** | 0.302 | -9.3% | Pure ✅ |
| govt | **0.446** | 0.428 | -4.0% | Pure ✅ |
| **Average** | **0.392** | 0.374 | **-4.6%** | **Pure ✅** |

**INTERESTING:** With ELSER, Pure clearly outperforms Hybrid (4.6% advantage)!

This is different from BM25 and BGE where they were tied. Templates/structure actually **hurts** ELSER performance.

### 2. Extractive Methods: Best Retrieval System

**For Pure Extractive:**

| Retriever | NDCG@10 Avg | vs Human | Competitive? |
|-----------|-------------|----------|--------------|
| BM25 | 0.255 | -6.3% | ✅ Yes |
| BGE | 0.308 | -23.0% | ❌ No |
| **ELSER** | **0.392** | **-20.3%** | ⚠️ Borderline |

**For Hybrid Extractive:**

| Retriever | NDCG@10 Avg | vs Human | Competitive? |
|-----------|-------------|----------|--------------|
| BM25 | 0.254 | -6.6% | ✅ Yes |
| BGE | 0.307 | -23.3% | ❌ No |
| ELSER | 0.374 | -24.0% | ❌ No |

**Key Insight:** ELSER has highest absolute scores but extractive methods still lag significantly behind baselines.

### 3. ELSER Performance Characteristics

**ELSER vs BM25:**
- Pure: +53% improvement (0.255 → 0.392)
- Hybrid: +47% improvement (0.254 → 0.374)
- **Massive gains** from learned sparse encoding

**ELSER vs BGE:**
- Pure: +27% improvement (0.308 → 0.392)
- Hybrid: +22% improvement (0.307 → 0.374)
- ELSER outperforms dense retrieval for extractive queries

**But:**
- Gap from human widens (BM25: -7%, ELSER: -20-24%)
- Gap from lastturn widens (BM25: +4%, ELSER: -13%)

---

## Retrieval System Rankings

### For Extractive Methods

**Best to Worst (by gap from baseline):**

1. **BM25** - Smallest gap (-6-7% from human) ✅
2. **BGE** - Medium gap (-23% from human) ❌
3. **ELSER** - Medium gap (-20-24% from human) ❌

**But for absolute performance:**

1. **ELSER** - Highest scores (0.374-0.392) ✅
2. **BGE** - Medium scores (0.307-0.308)
3. **BM25** - Lowest scores (0.254-0.255)

### Interpretation:

**Extractive methods:**
- Work well with **simple lexical matching** (BM25)
- Struggle with **learned/semantic matching** (BGE, ELSER)
- Don't capture semantic relationships that these methods exploit

---

## Key Observations

### 1. Government Domain Success

**Pure Extractive with ELSER on Govt:**
- NDCG@10: 0.446 vs Lastturn 0.449 (-0.7%)
- **Nearly ties the baseline!**
- Only domain where extractive is competitive with ELSER

**Why govt domain works:**
- Policy queries benefit from comprehensive keyword coverage
- ELSER's learned sparse terms align well with policy terminology
- Less reliance on semantic coherence

### 2. Cloud Domain Failure

**All extractive methods fail on Cloud:**
- BM25: -5% vs lastturn
- BGE: -7% vs lastturn  
- ELSER: -23% vs lastturn

**Consistent pattern:** Technical documentation needs precise queries, not keyword extraction.

### 3. The Template Penalty

**Hybrid underperforms Pure across all ELSER tests:**
- Average: -4.6% below Pure
- Worst gap: -9.3% (fiqa)
- Best gap: -0.4% (clapnq)

**Why templates hurt with ELSER:**
- ELSER learns importance weighting from text
- Templates add structural words ("what is", "do", etc.) that dilute importance
- Keyword-rich queries let ELSER focus on content terms

---

## Complete Comparison: All Three Systems

### Pure Extractive Performance

| Domain | BM25 | BGE | ELSER | Best |
|--------|------|-----|-------|------|
| clapnq | 0.290 | 0.406 | **0.460** | ELSER |
| cloud | 0.239 | 0.285 | **0.328** | ELSER |
| fiqa | 0.152 | 0.234 | **0.333** | ELSER |
| govt | **0.339** | 0.306 | 0.446 | ELSER |
| **Avg** | 0.255 | 0.308 | **0.392** | ELSER |

**ELSER wins for Pure Extractive in all domains!**

### Hybrid Extractive Performance

| Domain | BM25 | BGE | ELSER | Best |
|--------|------|-----|-------|------|
| clapnq | 0.284 | 0.399 | **0.458** | ELSER |
| cloud | 0.241 | 0.290 | **0.308** | ELSER |
| fiqa | 0.155 | 0.236 | **0.302** | ELSER |
| govt | **0.336** | 0.303 | 0.428 | ELSER |
| **Avg** | 0.254 | 0.307 | **0.374** | ELSER |

**ELSER wins for Hybrid Extractive in all domains except govt!**

---

## Gap Analysis: Extractive vs Baselines

### Pure Extractive Gap from Human

| Domain | BM25 Gap | BGE Gap | ELSER Gap | Worst System |
|--------|----------|---------|-----------|--------------|
| clapnq | -3.7% | -18.5% | **-20.4%** | ELSER |
| cloud | -3.6% | -16.7% | **-25.1%** | ELSER |
| fiqa | -18.3% | -31.4% | **-23.6%** | BGE |
| govt | -4.2% | -27.1% | **-13.7%** | BGE |
| **Avg** | **-7.5%** ✅ | -23.4% | -20.7% | BGE |

**Insight:** BM25 has smallest gap, making extractive most competitive for lexical retrieval.

### Hybrid Extractive Gap from Human

| Domain | BM25 Gap | BGE Gap | ELSER Gap | Worst System |
|--------|----------|---------|-----------|--------------|
| clapnq | -5.6% | -19.9% | **-20.8%** | ELSER |
| cloud | -2.8% | -15.2% | **-29.7%** | ELSER |
| fiqa | -16.7% | -30.8% | **-30.7%** | ELSER/BGE |
| govt | -5.1% | -27.9% | **-17.2%** | BGE |
| **Avg** | **-7.6%** ✅ | -23.5% | -24.6% | ELSER |

**Insight:** Hybrid has larger gaps with ELSER than Pure, suggesting templates are counterproductive.

---

## Surprising Results

### 1. ELSER ≠ Best for Extractive

Despite ELSER being the best retrieval system overall:
- Pure Extractive: BM25 most competitive (-7.5% gap)
- Hybrid Extractive: BM25 most competitive (-7.6% gap)

**Why?** ELSER expects semantically coherent queries, but extractive methods produce keyword lists.

### 2. Templates Hurt More with ELSER

| Retriever | Pure vs Hybrid Gap |
|-----------|-------------------|
| BM25 | -0.4% (tie) |
| BGE | -0.3% (tie) |
| **ELSER** | **-4.6%** (pure wins) |

Templates are **most harmful** with ELSER, suggesting learned sparse encoding prefers raw keywords over structured questions.

### 3. Government Domain Resilience

**Govt is the best domain for extractive with all systems:**
- BM25: -4.2% gap (excellent)
- ELSER: -13.7% gap (acceptable)
- BGE: -27.1% gap (poor)

Policy/government queries are most amenable to extractive keyword-based rewriting.

---

## Recommendations

### Which Retrieval System for Extractive Methods?

**For Production:**

1. **BM25 + Pure Extractive** ← **Best choice**
   - Smallest gap from human (-6.3%)
   - Simple, fast, no infrastructure needed
   - Competitive performance

2. **ELSER + Pure Extractive** ← Consider if using ELSER already
   - Highest absolute scores
   - But 20% behind human
   - Only use for govt domain (-13.7% gap acceptable)

3. **BGE + Pure Extractive** ← Avoid
   - 23% behind human
   - Not worth the complexity

**Never use:**
- ❌ Any Hybrid approach (consistently worse than Pure)

### Domain-Specific Recommendations

| Domain | Use Extractive? | Best Combo | Gap |
|--------|-----------------|------------|-----|
| **govt** | ✅ Yes | ELSER + Pure | -13.7% |
| **clapnq** | ⚠️ Maybe | BM25 + Pure | -3.7% |
| **fiqa** | ❌ Avoid | BM25 + Pure | -18.3% |
| **cloud** | ❌ Avoid | Use lastturn | N/A |

---

## Final Verdict

### Extractive Query Rewriting Assessment

**Works well with:**
- ✅ BM25 (lexical retrieval)
- ⚠️ ELSER for govt domain only
- ❌ Not recommended for BGE

**Best configuration:**
- **Pure Extractive + BM25** (simple, competitive)
- Average: 93% of human performance
- Govt domain: 96% of human performance

**Avoid:**
- ❌ Hybrid approaches (no benefit, more complexity)
- ❌ BGE/ELSER with extractive (large performance gaps)
- ❌ Cloud/technical domains (use lastturn instead)

### Overall Recommendation

**For the MT-RAG benchmark:**
- Use **human rewrites + ELSER** for maximum performance (0.492 avg)
- Use **Pure Extractive + BM25** for low-resource scenarios (0.255 avg, 93% of human)
- **Don't use Hybrid** - no scenario where it wins

**Research contribution:**
- Demonstrated that simple extractive methods can achieve 93% of human performance with BM25
- Showed that templates/NER don't improve keyword-based retrieval
- Identified domain-specific patterns (govt benefits, cloud suffers)

