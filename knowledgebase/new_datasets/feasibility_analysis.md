## Feasibility Analysis: Using QReCC & TopiOCQA for MT-RAG Experiments

### What You Currently Have (MT-RAG)

| Component   | Format                                | Example                                |
| ----------- | ------------------------------------- | -------------------------------------- | ---- | ------------- |
| **Queries** | JSONL with `_id`, `text`              | `{"\_id": "conv_id<::>turn", "text": " | user | : question"}` |
| **Qrels**   | TSV: `query-id`, `corpus-id`, `score` | Maps queries to relevant passages      |
| **Corpus**  | JSONL per domain                      | Passages with `_id`, `text`, `title`   |
| **Domains** | 4                                     | clapnq, cloud, fiqa, govt              |

---

## QReCC: HIGH Compatibility

| Aspect                       | QReCC                    | MT-RAG Compatible?            |
| ---------------------------- | ------------------------ | ----------------------------- |
| **Multi-turn conversations** | 5-6 turns avg            | Similar to MT-RAG             |
| **Question rewrites**        | Built-in                 | Directly usable               |
| **Reference answers**        | 91% coverage             | Can evaluate generation       |
| **Corpus**                   | 54M passages (10M pages) | Very large — needs subsetting |
| **Gold passages**            | Answer_URL available     | Can derive qrels              |
| **Train/Test split**         | 63K train / 16K test     |                               |

### What's Needed for QReCC:

1. **Convert data format**:

   ```
   QReCC → MT-RAG format
   - Conversation_no + Turn_no → task_id (e.g., "1<::>3")
   - Question → lastturn query
   - Rewrite → rewrite query
   - Context → can generate "questions" query
   ```

2. **Build/subset corpus**:

   - Full corpus is **54M passages** (very large)
   - Option A: Use only passages linked to Answer_URLs (~91% of questions)
   - Option B: Download and index a subset

3. **Generate qrels**:
   - Use Answer_URL to find gold passages
   - Map to corpus passage IDs

### Estimated Effort: **Medium** (2-3 days)

---

## TopiOCQA: HIGH Compatibility

| Aspect                       | TopiOCQA                                 | MT-RAG Compatible?                   |
| ---------------------------- | ---------------------------------------- | ------------------------------------ |
| **Multi-turn conversations** | 13 turns avg                             | Longer than MT-RAG                   |
| **Question rewrites**        | T5-QReCC rewrites available              | Directly usable                      |
| **Reference answers**        | Answer + Rationale                       | Can evaluate generation              |
| **Corpus**                   | 5.8M Wikipedia articles → 25.7M passages | Large but manageable                 |
| **Gold passages**            | Provided explicitly                      | Direct qrels available               |
| **Topic switching**          | 97% have switches                        | Novel challenge for MT-RAG           |
| **Unanswerable**             | 8% marked                                | Matches MT-RAG's answerability focus |

### What's Needed for TopiOCQA:

1. **Convert data format**:

   ```
   TopiOCQA → MT-RAG format
   - Conversation_no + Turn_no → task_id
   - Question → lastturn query
   - rewrites_t5_qrecc → rewrite query
   - all_history → questions query (optional)
   - Answer → reference answer
   - Rationale → gold passage
   ```

2. **Use provided corpus**:

   - Wikipedia segments (~25.7M passages) already chunked
   - TSV format: `id`, `text`, `title`

3. **Use provided gold passages**:
   - `gold_passages_info` files directly map to qrels

### Estimated Effort: **Low-Medium** (1-2 days)

---

## Comparison Summary

| Feature                  | MT-RAG (Current) | QReCC              | TopiOCQA              |
| ------------------------ | ---------------- | ------------------ | --------------------- |
| **Size (test)**          | 842 turns        | 16K turns          | 2.5K turns            |
| **Avg turns/conv**       | 3-4              | 5-6                | 13                    |
| **Domains**              | 4 specialized    | Open (web)         | Open (Wikipedia)      |
| **Corpus size**          | ~100K passages   | 54M passages       | 25.7M passages        |
| **Rewrites provided**    | Yes              | Yes                | Yes                   |
| **Gold passages**        | Yes              | Partial (from URL) | Yes                   |
| **Answerability labels** | Yes              | No                 | Yes (8% unanswerable) |
| **Topic switching**      | No               | No                 | Yes (unique feature)  |

---

## Recommendation

### Start with **TopiOCQA**

**Why:**

1. **Gold passages provided** — no need to derive qrels
2. **Corpus already segmented** — 200-word chunks ready to use
3. **Rewrites available** — three query variants ready
4. **Unanswerable questions** — matches your IDK evaluation
5. **Topic switching** — novel challenge that tests retrieval robustness
6. **Wikipedia-based** — familiar domain, easier to debug

### Then Add **QReCC**

**Why later:**

1. Large corpus (54M) requires infrastructure
2. Need to derive gold passages from URLs
3. But: 16K test samples gives statistical power
4. Web-based domain adds diversity

---

## Implementation Plan

### Phase 1: TopiOCQA Integration (1-2 days)

```bash
# Structure to create
human/retrieval_tasks/topiocqa/
├── topiocqa_lastturn.jsonl      # Original questions
├── topiocqa_rewrite.jsonl       # T5-QReCC rewrites
├── topiocqa_allhistory.jsonl    # Full context (optional)
├── qrels/
│   └── dev.tsv                  # From gold_passages_info

corpora/
├── topiocqa.jsonl               # From Wikipedia segments
```

**Scripts needed:**

1. `convert_topiocqa.py` — Convert dataset to MT-RAG format
2. `build_topiocqa_corpus.py` — Convert Wikipedia TSV to JSONL
3. `build_topiocqa_qrels.py` — Convert gold_passages to qrels

### Phase 2: Run Experiments

Once converted, your existing pipelines work:

```bash
# Retrieval (existing scripts work)
python scripts/baselines/retrieval_scripts/bge/bge_retrieval.py --domain topiocqa ...

# Reranking (Mono-T5)
python scripts/ideas/retrieval_tasks/mono-t5-as-reranker/...

# Generation (existing scripts work)
python scripts/baselines/generation_scripts/full_rag/...

# Evaluation (existing scripts work)
python scripts/evaluation/run_step_by_step.py --step all ...
```

---

## Key Considerations

1. **Corpus size**: TopiOCQA's 25.7M passages is ~250x larger than MT-RAG's domains. Indexing and retrieval will be slower.

2. **Topic switching**: This is a unique challenge — your Mono-T5 reranker may struggle when the topic changes mid-conversation.

3. **Answer style**: TopiOCQA has abstractive answers (not extractive spans), which may affect RB_agg scores differently.

4. **Evaluation subset**: Consider starting with a subset (e.g., 500-1000 turns) for faster iteration.

---

**Bottom line**: Yes, both datasets are highly compatible with your pipeline. **TopiOCQA is the easier starting point** due to its ready-made gold passages and corpus. Would you like me to create the conversion scripts?
