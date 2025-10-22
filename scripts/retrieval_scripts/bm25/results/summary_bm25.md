# BM25 Retrieval Results Summary

## Comparison with Paper Baselines

### Last Turn Strategy

| Domain | Queries | R@1 | R@3 | R@5 | R@10 | nDCG@1 | nDCG@3 | nDCG@5 | nDCG@10 |
|--------|---------|-----|-----|-----|------|--------|--------|--------|---------|
| **CLAPNQ** | 207 | 0.083 | 0.175 | 0.231 | 0.361 | 0.222 | 0.197 | 0.216 | 0.269 |
| **FiQA** | 180 | 0.032 | 0.095 | 0.133 | 0.194 | 0.083 | 0.094 | 0.111 | 0.136 |
| **Govt** | 198 | 0.117 | 0.238 | 0.312 | 0.402 | 0.268 | 0.253 | 0.282 | 0.319 |
| **Cloud** | 185 | 0.112 | 0.191 | 0.239 | 0.320 | 0.216 | 0.201 | 0.220 | 0.252 |
| **Weighted Avg** | 770 | **0.087** | **0.176** | **0.231** | **0.323** | **0.200** | **0.188** | **0.209** | **0.247** |
| **Paper Baseline** | - | 0.08 | 0.15 | **0.20** | **0.27** | 0.17 | 0.16 | **0.18** | **0.21** |

### Query Rewrite Strategy  

| Domain | Queries | R@1 | R@3 | R@5 | R@10 | nDCG@1 | nDCG@3 | nDCG@5 | nDCG@10 |
|--------|---------|-----|-----|-----|------|--------|--------|--------|---------|
| **CLAPNQ** | 208 | 0.092 | 0.199 | 0.280 | 0.399 | 0.250 | 0.224 | 0.253 | 0.301 |
| **FiQA** | 180 | 0.057 | 0.125 | 0.183 | 0.255 | 0.139 | 0.131 | 0.156 | 0.186 |
| **Govt** | 201 | 0.125 | 0.271 | 0.338 | 0.452 | 0.289 | 0.280 | 0.305 | 0.354 |
| **Cloud** | 188 | 0.103 | 0.188 | 0.234 | 0.327 | 0.202 | 0.195 | 0.211 | 0.248 |
| **Weighted Avg** | 777 | **0.095** | **0.198** | **0.261** | **0.362** | **0.223** | **0.210** | **0.234** | **0.275** |
| **Paper Baseline** | - | 0.09 | 0.18 | **0.25** | **0.33** | 0.20 | 0.19 | **0.22** | **0.25** |

### Full Questions (All questions from conversation)

| Domain | Queries | R@1 | R@3 | R@5 | R@10 | nDCG@1 | nDCG@3 | nDCG@5 | nDCG@10 |
|--------|---------|-----|-----|-----|------|--------|--------|--------|---------|
| **CLAPNQ** | 208 | 0.072 | 0.148 | 0.215 | 0.307 | 0.202 | 0.170 | 0.194 | 0.234 |
| **FiQA** | 180 | 0.025 | 0.083 | 0.114 | 0.151 | 0.072 | 0.080 | 0.093 | 0.109 |
| **Govt** | 201 | 0.096 | 0.201 | 0.255 | 0.343 | 0.189 | 0.203 | 0.225 | 0.264 |
| **Cloud** | 188 | 0.061 | 0.119 | 0.179 | 0.249 | 0.138 | 0.122 | 0.150 | 0.180 |
| **Weighted Avg** | 777 | **0.065** | **0.140** | **0.193** | **0.266** | **0.153** | **0.146** | **0.168** | **0.200** |

## Key Findings

1. **Query Rewrite consistently outperforms Last Turn**
   - R@5: 0.261 vs 0.231 (+13.0%)
   - R@10: 0.362 vs 0.323 (+12.1%)
   - Matches paper's finding that rewriting helps with non-standalone questions

2. **Our results match/exceed paper baselines**
   - Last Turn: R@5 = 0.231 (ours) vs 0.20 (paper) → **+15.3%**
   - Last Turn: R@10 = 0.323 (ours) vs 0.27 (paper) → **+19.6%**
   - Query Rewrite: R@5 = 0.261 (ours) vs 0.25 (paper) → **+4.5%**
   - Query Rewrite: R@10 = 0.362 (ours) vs 0.33 (paper) → **+9.7%**
   - Small positive differences validate our implementation

3. **Domain performance varies significantly**
   - **Best:** Govt (R@5 = 0.338, R@10 = 0.452 with rewrite)
   - **Worst:** FiQA (R@5 = 0.183, R@10 = 0.255 with rewrite)
   - FiQA's poor performance matches paper's observation about informal forum posts

4. **Full Questions perform worst**
   - R@5 = 0.193 (vs 0.231 for Last Turn, 0.261 for Rewrite)
   - R@10 = 0.266 (vs 0.323 for Last Turn, 0.362 for Rewrite)
   - Conversation context format is less effective than single turn

5. **Performance improves with more results (k)**
   - As expected, Recall increases substantially from k=5 to k=10
   - Average +40% improvement in Recall from @5 to @10

## Implementation Details

- **Retrieval System:** PyTerrier BM25 with default parameters (k1=1.2, b=0.75)
- **Corpus:** Pre-chunked by paper authors (512 tokens, 100 token overlap)
- **PyTerrier Index Config:** docno length = 50 chars (to handle long document IDs)
- **Evaluation Metrics:** Recall and nDCG at k=[1, 3, 5, 10]
- **Query Processing:** Removed special characters to avoid PyTerrier parser issues
- **Tokenization:** Simple whitespace-based token frequency dict
- **Total Tasks Evaluated:** 777 retrieval tasks (answerable + partial only)

