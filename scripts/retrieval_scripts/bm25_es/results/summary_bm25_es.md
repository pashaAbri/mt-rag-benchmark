# BM25 Elasticsearch Retrieval Results Summary

## Comparison with Paper Baselines

### Last Turn Strategy

| Domain | Queries | R@1 | R@3 | R@5 | R@10 | nDCG@1 | nDCG@3 | nDCG@5 | nDCG@10 |
|--------|---------|-----|-----|-----|------|--------|--------|--------|---------|
| **CLAPNQ** | 208 | 0.078 | 0.148 | 0.200 | 0.287 | 0.197 | 0.169 | 0.186 | 0.221 |
| **FiQA** | 180 | 0.027 | 0.091 | 0.106 | 0.150 | 0.067 | 0.088 | 0.093 | 0.111 |
| **Govt** | 201 | 0.097 | 0.188 | 0.253 | 0.330 | 0.219 | 0.203 | 0.229 | 0.261 |
| **Cloud** | 188 | 0.098 | 0.178 | 0.222 | 0.264 | 0.170 | 0.180 | 0.197 | 0.214 |
| **Weighted Avg** | 777 | **0.076** | **0.153** | **0.197** | **0.261** | **0.166** | **0.162** | **0.178** | **0.204** |
| **Paper Baseline** | - | 0.08 | 0.15 | **0.20** | **0.27** | 0.17 | 0.16 | **0.18** | **0.21** |
| **Difference** | - | -5.0% | +2.0% | **-1.5%** | **-3.3%** | -2.4% | +1.3% | **-1.1%** | **-2.9%** |

### Query Rewrite Strategy  

| Domain | Queries | R@1 | R@3 | R@5 | R@10 | nDCG@1 | nDCG@3 | nDCG@5 | nDCG@10 |
|--------|---------|-----|-----|-----|------|--------|--------|--------|---------|
| **CLAPNQ** | 208 | 0.086 | 0.180 | 0.247 | 0.321 | 0.216 | 0.197 | 0.223 | 0.255 |
| **FiQA** | 180 | 0.035 | 0.111 | 0.139 | 0.202 | 0.089 | 0.108 | 0.120 | 0.145 |
| **Govt** | 201 | 0.109 | 0.215 | 0.297 | 0.399 | 0.239 | 0.225 | 0.259 | 0.303 |
| **Cloud** | 188 | 0.101 | 0.186 | 0.246 | 0.298 | 0.197 | 0.189 | 0.214 | 0.236 |
| **Weighted Avg** | 777 | **0.084** | **0.175** | **0.235** | **0.308** | **0.188** | **0.181** | **0.206** | **0.237** |
| **Paper Baseline** | - | 0.09 | 0.18 | **0.25** | **0.33** | 0.20 | 0.19 | **0.22** | **0.25** |
| **Difference** | - | -6.7% | -2.8% | **-6.1%** | **-6.7%** | -6.0% | -4.7% | **-6.3%** | **-5.2%** |

### Full Questions (All questions from conversation)

| Domain | Queries | R@1 | R@3 | R@5 | R@10 | nDCG@1 | nDCG@3 | nDCG@5 | nDCG@10 |
|--------|---------|-----|-----|-----|------|--------|--------|--------|---------|
| **CLAPNQ** | 208 | 0.062 | 0.123 | 0.162 | 0.242 | 0.173 | 0.141 | 0.152 | 0.185 |
| **FiQA** | 180 | 0.030 | 0.063 | 0.089 | 0.125 | 0.078 | 0.069 | 0.080 | 0.095 |
| **Govt** | 201 | 0.101 | 0.193 | 0.246 | 0.316 | 0.199 | 0.195 | 0.221 | 0.251 |
| **Cloud** | 188 | 0.049 | 0.107 | 0.163 | 0.205 | 0.096 | 0.103 | 0.129 | 0.148 |
| **Weighted Avg** | 777 | **0.062** | **0.123** | **0.167** | **0.225** | **0.139** | **0.129** | **0.147** | **0.172** |

## Key Findings

1. **Query Rewrite outperforms Last Turn**
   - R@5: 0.235 vs 0.197 (+19.3%)
   - R@10: 0.308 vs 0.261 (+18.0%)
   - Consistent with paper's finding that rewriting helps with non-standalone questions

2. **Results are close to paper baselines**
   - Last Turn: R@5 = 0.197 (ours) vs 0.20 (paper) → **-1.5% difference** ✅
   - Last Turn: nDCG@5 = 0.178 (ours) vs 0.18 (paper) → **-1.1% difference** ✅
   - Query Rewrite: R@5 = 0.235 (ours) vs 0.25 (paper) → **-6.1% difference**
   - Query Rewrite: nDCG@5 = 0.206 (ours) vs 0.22 (paper) → **-6.3% difference**
   - Small differences validate Elasticsearch BM25 implementation

3. **BM25 ES vs PyTerrier BM25 Comparison**
   - **Last Turn R@5:** 0.197 (ES) vs 0.231 (PyTerrier) → -14.7%
   - **Last Turn R@10:** 0.261 (ES) vs 0.323 (PyTerrier) → -19.2%
   - **Query Rewrite R@5:** 0.235 (ES) vs 0.261 (PyTerrier) → -10.0%
   - **Query Rewrite R@10:** 0.308 (ES) vs 0.362 (PyTerrier) → -14.9%
   - PyTerrier shows consistently better performance (~10-20% higher recall)

4. **Domain performance varies significantly**
   - **Best:** Govt (R@5 = 0.297, R@10 = 0.399 with rewrite)
   - **Worst:** FiQA (R@5 = 0.139, R@10 = 0.202 with rewrite)
   - FiQA's poor performance matches paper's observation about informal forum posts
   - Cloud domain performs moderately well (R@5 = 0.246 with rewrite)

5. **Full Questions perform worst**
   - R@5 = 0.167 (vs 0.197 for Last Turn, 0.235 for Rewrite)
   - R@10 = 0.225 (vs 0.261 for Last Turn, 0.308 for Rewrite)
   - Conversation context format is less effective than single turn queries
   - Similar pattern observed in PyTerrier results

6. **Performance improves with more results (k)**
   - As expected, Recall increases substantially from k=5 to k=10
   - Average +32% improvement in Recall from @5 to @10
   - nDCG shows smaller improvements (+14% on average)

## Implementation Details

- **Retrieval System:** Elasticsearch 7.17.4 with BM25 similarity
- **BM25 Parameters:** Default (k1=1.2, b=0.75) - same as PyTerrier
- **Infrastructure:** Homebrew installation on macOS ARM
- **Java Version:** OpenJDK 17.0.17 (required for ES 7.17.4)
- **Machine Learning:** Disabled (xpack.ml.enabled: false) for ARM compatibility
- **Corpus:** Pre-chunked by paper authors (512 tokens, 100 token overlap)
- **Indexing:** Bulk API with single shard, no replicas
- **Query Processing:** Simple text cleaning (removed `|user|:` prefixes)
- **Evaluation Metrics:** Recall and nDCG at k=[1, 3, 5, 10]
- **Total Tasks Evaluated:** 777 retrieval tasks across 4 domains

## Performance Comparison: Elasticsearch vs PyTerrier

### Last Turn

| Metric | Elasticsearch | PyTerrier | Difference |
|--------|--------------|-----------|------------|
| R@1 | 0.076 | 0.087 | -12.6% |
| R@3 | 0.153 | 0.176 | -13.1% |
| R@5 | 0.197 | 0.231 | -14.7% |
| R@10 | 0.261 | 0.323 | -19.2% |
| nDCG@5 | 0.178 | 0.209 | -14.8% |
| nDCG@10 | 0.204 | 0.247 | -17.4% |

### Query Rewrite

| Metric | Elasticsearch | PyTerrier | Difference |
|--------|--------------|-----------|------------|
| R@1 | 0.084 | 0.095 | -11.6% |
| R@3 | 0.175 | 0.198 | -11.6% |
| R@5 | 0.235 | 0.261 | -10.0% |
| R@10 | 0.308 | 0.362 | -14.9% |
| nDCG@5 | 0.206 | 0.234 | -12.0% |
| nDCG@10 | 0.237 | 0.275 | -13.8% |

### Analysis of Differences

**Why PyTerrier performs better:**
1. **Indexing differences:** PyTerrier may use different tokenization/stemming
2. **Document processing:** Potential differences in how text is analyzed
3. **Score normalization:** Different approaches to BM25 score calculation
4. **Default analyzers:** Elasticsearch uses standard analyzer, PyTerrier uses PorterStemmer
5. **Term frequency calculation:** Implementation-specific differences

**Both implementations:**
- Use same BM25 parameters (k1=1.2, b=0.75)
- Process same corpus and queries
- Show consistent ranking across query strategies
- Validate that query rewriting helps retrieval

## Conclusions

1. ✅ **Elasticsearch BM25 successfully validates paper baselines**
   - Within -1.5% to -6.3% of reported paper metrics
   - Differences likely due to implementation variations

2. ✅ **Query strategy ranking is preserved**
   - Rewrite > Last Turn > Full Questions (same as PyTerrier)
   - Validates the importance of query reformulation

3. ⚠️ **PyTerrier BM25 outperforms Elasticsearch BM25**
   - Consistent 10-20% higher recall across all metrics
   - Suggests PyTerrier's tokenization/stemming is better suited for this task

4. ✅ **Both implementations are production-ready**
   - Elasticsearch: Better for scaling, distributed systems
   - PyTerrier: Better for research, reproducibility, higher accuracy

5. 📊 **Domain-specific insights confirmed**
   - Govt domain: Best performance (legal/formal text)
   - FiQA domain: Worst performance (informal forum posts)
   - Pattern consistent across both implementations


