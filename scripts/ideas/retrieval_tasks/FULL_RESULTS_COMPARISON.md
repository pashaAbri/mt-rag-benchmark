# Complete Results Comparison: Extractive Query Rewriting

**Date:** November 9, 2024  
**Retrieval Method:** BM25  
**Evaluation Metrics:** NDCG@k and Recall@k (k=1,3,5,10)

## Executive Summary

This document compares four query rewriting approaches:
1. **Lastturn** - Baseline: Current query only (no context)
2. **Human Rewrite** - Gold standard: Human-annotated rewrites
3. **Pure Extractive** - MMR-based keyword selection (ours)
4. **Hybrid Extractive** - MMR + NER + Templates (ours)

---

## Overall Performance Summary (NDCG@10 / Recall@10)

| Domain | Lastturn | Human Rewrite | Pure Extractive | Hybrid Extractive | Best Method |
|--------|----------|---------------|-----------------|-------------------|-------------|
| **clapnq** | 0.269 / 0.361 | **0.301 / 0.399** | 0.290 / 0.389 | 0.284 / 0.385 | Human ✅ |
| **cloud** | **0.252 / 0.320** | 0.248 / 0.327 | 0.239 / 0.309 | 0.241 / 0.319 | Lastturn ✅ |
| **fiqa** | 0.136 / 0.194 | **0.186 / 0.255** | 0.152 / 0.207 | 0.155 / 0.215 | Human ✅ |
| **govt** | 0.319 / 0.402 | **0.354 / 0.452** | 0.339 / 0.418 | 0.336 / 0.419 | Human ✅ |
| **Average** | 0.244 / 0.319 | **0.272 / 0.358** | 0.255 / 0.331 | 0.254 / 0.335 | Human ✅ |

---

## Detailed Results by Domain

### ClapNQ (Wikipedia Q&A) - 208 queries

| Method | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 | Recall@1 | Recall@3 | Recall@5 | Recall@10 |
|--------|--------|--------|--------|---------|----------|----------|----------|-----------|
| Lastturn | 0.222 | 0.197 | 0.216 | 0.269 | 0.083 | 0.175 | 0.231 | 0.361 |
| Human Rewrite | **0.250** | **0.224** | **0.253** | **0.301** | **0.092** | **0.199** | **0.280** | **0.399** |
| Pure Extractive | 0.226 | 0.213 | 0.242 | 0.290 | 0.088 | 0.193 | 0.272 | 0.389 |
| Hybrid Extractive | 0.221 | 0.202 | 0.232 | 0.284 | 0.082 | 0.177 | 0.255 | 0.385 |

**Ranking:** Human (0.301) > Pure (0.290) > Hybrid (0.284) > Lastturn (0.269)

**Analysis:**
- ✅ Both extractive methods beat lastturn
- ✅ Pure Extractive performs better than Hybrid
- 🤔 Hybrid's templates may over-complicate Wikipedia queries
- 📊 Pure Extractive is only 3.7% behind human rewrites

---

### Cloud (Technical Documentation) - 188 queries

| Method | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 | Recall@1 | Recall@3 | Recall@5 | Recall@10 |
|--------|--------|--------|--------|---------|----------|----------|----------|-----------|
| Lastturn | **0.216** | **0.201** | **0.220** | **0.252** | **0.112** | **0.191** | **0.239** | **0.320** |
| Human Rewrite | 0.202 | 0.195 | 0.211 | 0.248 | 0.103 | 0.188 | 0.234 | 0.327 |
| Pure Extractive | 0.176 | 0.190 | 0.203 | 0.239 | 0.088 | 0.190 | 0.222 | 0.309 |
| Hybrid Extractive | 0.176 | 0.185 | 0.205 | 0.241 | 0.083 | 0.185 | 0.230 | 0.319 |

**Ranking:** Lastturn (0.252) > Human (0.248) > Hybrid (0.241) > Pure (0.239)

**Analysis:**
- ❌ ALL rewriting methods underperform lastturn
- 🤔 Technical queries are already precise; rewriting hurts
- 📉 Pure and Hybrid perform nearly identically (both weak)
- 💡 Suggests technical domain needs different approach

---

### FiQA (Financial Q&A) - 180 queries

| Method | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 | Recall@1 | Recall@3 | Recall@5 | Recall@10 |
|--------|--------|--------|--------|---------|----------|----------|----------|-----------|
| Lastturn | 0.083 | 0.094 | 0.111 | 0.136 | 0.032 | 0.095 | 0.133 | 0.194 |
| Human Rewrite | **0.139** | **0.131** | **0.156** | **0.186** | **0.057** | **0.125** | **0.183** | **0.255** |
| Pure Extractive | 0.089 | 0.113 | 0.127 | 0.152 | 0.033 | 0.118 | 0.151 | 0.207 |
| Hybrid Extractive | 0.089 | 0.117 | 0.128 | 0.155 | 0.031 | 0.125 | 0.152 | 0.215 |

**Ranking:** Human (0.186) > Hybrid (0.155) > Pure (0.152) > Lastturn (0.136)

**Analysis:**
- ✅ Both extractive methods beat lastturn
- ✅ Hybrid slightly outperforms Pure (+2.0%)
- ❌ Still 16.7% behind human rewrites
- 💡 Financial domain shows largest improvement from rewriting

---

### Govt (Government/Policy) - 201 queries

| Method | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 | Recall@1 | Recall@3 | Recall@5 | Recall@10 |
|--------|--------|--------|--------|---------|----------|----------|----------|-----------|
| Lastturn | 0.268 | 0.253 | 0.282 | 0.319 | 0.117 | 0.238 | 0.312 | 0.402 |
| Human Rewrite | **0.289** | **0.280** | **0.305** | **0.354** | **0.125** | **0.271** | **0.338** | **0.452** |
| Pure Extractive | 0.269 | 0.283 | 0.304 | 0.339 | 0.120 | 0.275 | 0.335 | 0.418 |
| Hybrid Extractive | 0.264 | 0.274 | 0.300 | 0.336 | 0.112 | 0.266 | 0.335 | 0.419 |

**Ranking:** Human (0.354) > Pure (0.339) > Hybrid (0.336) > Lastturn (0.319)

**Analysis:**
- ✅ Both extractive methods beat lastturn
- ✅ Pure and Hybrid perform nearly identically
- 🎯 Very competitive with human performance (-4-5%)
- 🏆 Best domain for extractive approaches

---

## Key Findings

### 1. Pure vs Hybrid Extractive Comparison

| Domain | Pure NDCG@10 | Hybrid NDCG@10 | Difference | Winner |
|--------|--------------|----------------|------------|---------|
| clapnq | 0.290 | 0.284 | -2.1% | Pure ✅ |
| cloud | 0.239 | 0.241 | +0.8% | Hybrid ✅ |
| fiqa | 0.152 | 0.155 | +2.0% | Hybrid ✅ |
| govt | 0.339 | 0.336 | -0.9% | Pure ✅ |
| **Average** | **0.255** | **0.254** | **-0.4%** | **Tie** |

**Insight:** Pure and Hybrid perform **nearly identically** across all domains. The added complexity of templates and NER doesn't provide significant benefit.

### 2. When Extractive Methods Win

**✅ Extractive > Lastturn (3/4 domains):**
- ClapNQ: +7-8%
- FiQA: +11-14%
- Govt: +5-6%

**❌ Extractive < Lastturn (1/4 domains):**
- Cloud: -5% (technical queries don't benefit from rewriting)

### 3. Gap from Human Performance

| Domain | Pure Gap | Hybrid Gap | Average Gap |
|--------|----------|------------|-------------|
| clapnq | -3.7% | -5.6% | -4.7% |
| cloud | -3.6% | -2.8% | -3.2% |
| fiqa | -18.3% | -16.7% | -17.5% |
| govt | -4.2% | -5.1% | -4.7% |
| **Average** | **-7.5%** | **-7.6%** | **-7.5%** |

**Insight:** Extractive methods are consistently ~7.5% behind human rewrites. The gap is largest in FiQA (financial domain).

### 4. Repetition Analysis

**Pure Extractive Output (repetitive):**
```
"where arizona cardinals play outside us cardinals play outside arizona cardinals play..."
```

**Hybrid Extractive Output (cleaner):**
```
"Do Arizona Cardinals play outside US?"
```

**Performance Impact:** Despite cleaner output, Hybrid doesn't outperform Pure, suggesting:
- BM25 handles repetitive keywords reasonably well (term saturation)
- Well-formed questions aren't necessarily better for lexical retrieval
- Templates may remove useful keyword variants

---

## Domain-Specific Insights

### ClapNQ (Wikipedia)
- **Best approach:** Human rewrite
- **Extractive performance:** Good (only 4-6% behind human)
- **Observation:** General knowledge queries benefit from context but extractive is competitive

### Cloud (Technical)
- **Best approach:** Lastturn (no rewriting!)
- **Extractive performance:** Poor (underperforms baseline)
- **Observation:** Technical queries are self-contained; rewriting adds noise

### FiQA (Financial)
- **Best approach:** Human rewrite (by far)
- **Extractive performance:** Moderate (+11-14% vs lastturn, but -17% vs human)
- **Observation:** Financial concepts need semantic understanding that extractive methods lack

### Govt (Policy)
- **Best approach:** Human rewrite
- **Extractive performance:** Excellent (only 4-5% behind human)
- **Observation:** Policy queries benefit from comprehensive keyword coverage

---

## Recommendations

### When to Use Each Method

**Use Lastturn when:**
- ✅ Technical/documentation queries
- ✅ Self-contained questions
- ✅ Domain expertise encoded in query wording

**Use Pure Extractive when:**
- ✅ Government/policy domains
- ✅ Need better-than-baseline without human annotation
- ✅ Recall is more important than precision
- ✅ Low-resource scenarios

**Use Hybrid Extractive when:**
- 🤷 Similar performance to Pure
- 🤷 Want cleaner, more readable output
- ❌ **Not recommended** - added complexity doesn't improve performance

**Use Human Rewrite when:**
- ✅ Maximum accuracy needed
- ✅ Financial or semantic-heavy domains
- ✅ Resources available for annotation

### Recommended Improvements

**For Pure Extractive:**
1. ✅ **Deduplicate overlapping n-grams** - Main issue identified
2. ✅ **Reduce max_terms** from 10 → 5-7 for technical domains
3. ✅ **Domain-specific tuning** - Different λ for different domains
4. ⚠️ **Skip Cloud domain** - Use lastturn instead

**For Hybrid Extractive:**
1. ❌ **May not be worth the complexity** - Performs similarly to Pure
2. 🤔 **Revisit templates** - May be over-structuring queries
3. 🤔 **Test without post-processing** - Capitalization/question marks may hurt BM25

---

## Statistical Summary

### Average Performance Across All Domains

| Metric | Lastturn | Human | Pure | Hybrid | Best |
|--------|----------|-------|------|--------|------|
| **NDCG@1** | 0.197 | **0.220** | 0.190 | 0.189 | Human |
| **NDCG@3** | 0.186 | **0.207** | 0.200 | 0.195 | Human |
| **NDCG@5** | 0.207 | **0.231** | 0.219 | 0.217 | Human |
| **NDCG@10** | 0.244 | **0.272** | 0.255 | 0.254 | Human |
| **Recall@1** | 0.086 | **0.095** | 0.080 | 0.076 | Human |
| **Recall@3** | 0.173 | **0.196** | 0.194 | 0.189 | Human |
| **Recall@5** | 0.229 | **0.265** | 0.245 | 0.243 | Human |
| **Recall@10** | 0.319 | **0.358** | 0.331 | 0.335 | Human |

**Key Takeaway:** Human rewrites consistently outperform all methods, but extractive methods provide 93% of human performance on average.

---

## Wins/Losses Matrix

|  | vs Lastturn | vs Human | vs Pure |
|---|-------------|----------|---------|
| **Pure Extractive** | 3 wins, 1 loss | 0 wins, 4 losses | baseline |
| **Hybrid Extractive** | 3 wins, 1 loss | 0 wins, 4 losses | 2 wins, 2 losses |

**Observation:** Pure and Hybrid are nearly equivalent in performance, with neither showing clear superiority.

---

## Conclusion

### Main Findings

1. **Extractive methods work** - Beat lastturn in 75% of domains
2. **Pure ≈ Hybrid** - Added complexity of templates/NER doesn't help BM25 retrieval
3. **Domain matters** - Technical queries don't benefit from rewriting
4. **Human advantage persists** - But only by ~7.5% on average

### Best Practices

**For Research/Production:**
- Use **Pure Extractive** - simpler, equivalent performance to Hybrid
- Skip rewriting for **technical/documentation domains**
- Focus improvement efforts on **deduplication**, not templates

**For Future Work:**
- Test with **semantic retrieval** (BGE, dense encoders) - templates may help there
- Implement **n-gram deduplication** - likely to improve Pure Extractive
- Explore **domain-specific models** - Technical vs conversational

### Final Verdict

**Pure Extractive is the winner** among our methods:
- ✅ Same performance as Hybrid
- ✅ Much simpler implementation  
- ✅ No dependency on spaCy NER
- ✅ Faster execution
- ✅ More interpretable

**Recommended approach:** Pure Extractive with n-gram deduplication for non-technical domains.

