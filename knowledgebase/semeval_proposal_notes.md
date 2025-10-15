# SemEval Proposal Notes - MTRAGEval

## Section-by-Section Summary

### 1. Overview
- **Goal:** Evaluate Multi-Turn RAG Conversations addressing answerability and later turns
- **Based on:** MTRAG benchmark - first to use active retrieval, long answers, unanswerables, multiple domains
- **Key challenges:** Answerability (knowing when not to answer) and later turns (non-standalone questions)
- **Three subtasks:** A) Retrieval, B) Generation with Reference Passages, C) Full RAG
- **Expected participation:** 24-45 submissions (based on prior TREC RAG and iKAT tasks)
- **Impact:** Will provide additional multi-turn conversations as public resource

### 2. Data and Resources
- **Trial data:** MTRAG benchmark with 110 conversations across 4 domains (ClapNQ, FiQA, Cloud, Govt)
- **Test data:** ~200 new tasks from unseen conversations targeting challenging areas + hidden challenges
- **Task definition:** Conversation turn containing all previous turns + last user question
- **Quality:** All conversations manually created and reviewed by human annotators
- **Restrictions:** Cannot use Mixtral 8x7B (used during dataset creation)
- **Allowed:** New trained models, prompt engineering, query rewriting, agentic RAG
- **Release plan:** Full conversations released after evaluation period (not during, to avoid answer leakage)

### 3. Subtasks
**Subtask A - Retrieval:**
- Given: Task (answerable with relevant passages)
- Evaluate: Whether retrieval system retrieves relevant passages
- Focus: Retrieval performance in multi-turn context

**Subtask B - Generation with Reference Passages:**
- Given: Task + reference passages (gold) + target answer
- Evaluate: Generated answer vs reference answer
- Focus: Pure generation capability with perfect passages

**Subtask C - Full RAG:**
- Given: Task + target answer
- Process: First retrieve 5 passages, then generate answer
- Evaluate: Generated answer vs reference answer
- Focus: End-to-end RAG performance (real-world scenario)

**Flexibility:** Participants can join one or multiple subtasks; creative submissions encouraged

### 4. Evaluation
**Process:**
- **Two phases:** 1) Retrieval phase (Subtasks A, C), 2) Generation phase (Subtask B)
- **Submission:** Only one per subtask evaluated (final submission); can resubmit until deadline
- **Transparency:** Evaluation script provided for dev set; test leaderboard visible only after competition
- **Script:** Same evaluation script used by organizers and participants

**Retrieval Metrics (Subtask A):**
- nDCG (Normalized Discounted Cumulative Gain)
- Recall
- Both measured @5 and @10

**Generation Metrics (Subtasks B & C):**
1. **RB_alg:** Harmonic mean of BERT-Recall, Rouge-L, BERT-K-Precision
2. **RB_llm:** Reference-Based LLM judge (adapted from RAD-Bench)
3. **RL_f:** RAGAS Faithfulness LLM judge (reference-less)
- **IDK conditioning:** All metrics conditioned on "I Don't Know" LLM judge (checks if response contains answer)

**Human Evaluation:**
- ~20 tasks on all participating models
- Applied to full RAG task (Subtask C) due to reference-less nature
- Provides qualitative assessment

**Full details:** Available in MTRAG benchmark paper

### 5. Trial Data and Baselines
**Available resources:**
- MTRAG benchmark at https://github.com/IBM/mt-rag-benchmark
- 110 conversations = 842 tasks
- Evaluation scripts included

**Retrieval baselines (Table 1):**
- BM25: Recall@5=0.20, nDCG@5=0.18 (lexical)
- BGE-base1.5: Recall@5=0.30, nDCG@5=0.27 (dense)
- Elser: Recall@5=0.49, nDCG@5=0.45 (best, but still low - room for improvement)

**Generation baselines (Table 2):**
- Reference setting (•): GPT-4o, Llama3.1 405B/8B, Qwen2.5 7B
- RAG setting (◦): All models show performance drop vs reference
- Key finding: Room for improvement even with reference answers

**Leaderboard plans:** Different baselines by model size (small/large)

### 6. Task Organizers and Roles
**Sara Rosenthal (Lead Organizer):**
- Staff Research Scientist, IBM Research New York
- 8 prior SemEval tasks (Sentiment Analysis, OffensEval, Table Fact Verification)
- Current SemEval Workshop 2024-2025 organizer
- Experience: Benchmarks, human annotation

**Yannis Katsis (Co-Organizer):**
- Senior Research Scientist, IBM Research Almaden
- Creator of MTRAG benchmark
- Focus: Dataset selection and quality
- Experience: RAG systems, knowledge extraction

**Vraj Shah (Co-Organizer):**
- Staff Research Scientist, IBM Almaden
- Focus: Evaluation metrics and running evaluation
- Experience: RAG systems, LLM-based evaluation, data management

**Marina Danilevsky (Advisory Organizer):**
- Senior Research Scientist, IBM Almaden (Manager, Core Language Technologies)
- Prior SemEval task co-organizer (Table Fact Verification)
- Focus: Evaluation, model explainability, human-in-the-loop
- Multiple tutorials and online courses

### 7. Ethical Considerations

**7.1 Impact:**
- **Problem addressed:** Hallucination in LLMs
- **Solution:** RAG framework for faithful, grounded responses
- **Benefit:** Prevent misinformation spread; significant impact for LLM development

**7.2 Data and Annotators:**
- **Annotators:** Highly skilled professionals paid well above minimum wage
- **Privacy:** All annotator information anonymized (no PII)
- **Content:** Questions are general, not individual-specific
- **Fictitious data:** Any personal-looking information is fabricated (e.g., "How can I avoid bankruptcy?")
- **License:** Apache 2.0 for all released data
- **Funding:** All costs (annotation, evaluation resources) covered by IBM ongoing work processes

---

## Quick Reference Tables

### Baseline Performance Summary

**Retrieval (only 777 answerable/partial tasks):**
| Model | Recall@5 | Recall@10 | nDCG@5 | nDCG@10 |
|-------|----------|-----------|--------|---------|
| BM25 | 0.20 | 0.27 | 0.18 | 0.21 |
| BGE-base1.5 | 0.30 | 0.38 | 0.27 | 0.30 |
| Elser | 0.49 | 0.58 | 0.45 | 0.49 |

**Generation (all 842 tasks, with IDK conditioning):**
| Model | RL_f (•/◦) | RB_llm (•/◦) | RB_alg (•/◦) |
|-------|-----------|-------------|-------------|
| Reference | 0.87/0.65 | 0.95/0.95 | 0.88/0.85 |
| GPT-4o | 0.76/0.71 | 0.76/0.70 | 0.45/0.40 |
| Llama3.1 405B | 0.75/0.72 | 0.74/0.68 | 0.48/0.42 |
| Llama3.1 8B | 0.55/0.56 | 0.59/0.59 | 0.37/0.35 |
| Qwen2.5 7B | 0.68/0.67 | 0.66/0.68 | 0.44/0.39 |

*• = Reference setting (gold passages), ◦ = RAG setting (retrieved passages)*

---

## Key Takeaways

1. **Novel benchmark:** First multi-turn RAG evaluation with active retrieval, unanswerables, and multiple domains
2. **Low baseline performance:** Even best models show significant room for improvement
3. **Dual evaluation:** Both retrieval and generation bottlenecks identified
4. **Hidden challenges:** Test set includes undisclosed difficulties beyond answerability/later turns
5. **Two-phase evaluation:** Sequential retrieval → generation evaluation process
6. **Practical approach:** Evaluation scripts provided; only final submission counts
7. **Resource contribution:** New conversations will be released publicly post-competition
8. **Ethical foundation:** Fair annotator pay, privacy protection, misinformation prevention focus

