# MTRAGEval Competition Plan

## Overview

This document outlines a 4-week plan for each of the three subtasks in the MTRAGEval SemEval task.

**Competition Structure:**
- **Subtask A:** Retrieval
- **Subtask B:** Generation with Reference Passages
- **Subtask C:** Full RAG (End-to-End)

**Strategy:** Focus on one subtask at a time, or run parallel efforts if resources allow.

---

## Plan Summary

| Week | Subtask A: Retrieval | Subtask B: Generation | Subtask C: Full RAG |
|------|---------------------|----------------------|---------------------|
| **Week 1** | Setup & Baseline<br>• Environment setup<br>• Data analysis<br>• BM25 baseline<br>• Failure analysis | Setup & Baseline<br>• Environment setup<br>• Reference answer analysis<br>• Simple prompts<br>• IDK conditioning | Setup & Baseline<br>• End-to-end pipeline<br>• Pipeline design<br>• Baseline implementation<br>• Error propagation analysis |
| **Week 2** | Query Enhancement<br>• Query rewriting<br>• Conversation history<br>• Context strategies<br>• Evaluation | Prompt Engineering<br>• Multi-turn prompts<br>• Answerability handling<br>• Faithfulness prompts<br>• Answer quality | Retrieval Optimization<br>• Query rewriting<br>• Dense retrieval<br>• Hybrid approaches<br>• Passage selection |
| **Week 3** | Advanced Retrieval<br>• Dense retrieval (BGE)<br>• Hybrid systems<br>• Re-ranking<br>• Multi-turn features | Model Selection<br>• Model comparison<br>• Optional fine-tuning<br>• Post-processing<br>• Ensemble methods | Generation Optimization<br>• Adapt prompts<br>• Agentic RAG<br>• Error recovery<br>• Hallucination prevention |
| **Week 4** | Optimization & Submit<br>• Hyperparameter tuning<br>• Error analysis<br>• Format output<br>• **Submit** | Optimization & Submit<br>• Metric optimization<br>• Error analysis<br>• System description<br>• **Submit** | Final Testing & Submit<br>• Full system testing<br>• Human eval prep<br>• System description<br>• **Submit** |

**Target Performance by Subtask:**
- **Subtask A:** Recall@5 > 0.49 (beat Elser baseline)
- **Subtask B:** RB_alg > 0.48 (beat Llama 405B reference baseline)
- **Subtask C:** RB_alg > 0.40, RL_f > 0.70 (beat GPT-4o RAG baseline)

---

## Subtask A: Retrieval - 4 Week Plan

**Goal:** Given a multi-turn conversation task, retrieve the 5 most relevant passages for the last turn.

**Metrics:** Recall@5, Recall@10, nDCG@5, nDCG@10

### Week 1: Setup & Baseline

**Tasks:**
1. [ ] Clone mt-rag-benchmark repository
2. [ ] Install dependencies and evaluation scripts
3. [ ] Load and explore the 4 corpora (ClapNQ, FiQA, Cloud, Govt)
4. [ ] Understand data format: 842 tasks from 110 conversations
5. [ ] Review 777 answerable/partial tasks used for retrieval evaluation
6. [ ] Analyze conversation structure and turn patterns
7. [ ] Identify "later turns" and their characteristics
8. [ ] Study reference passages for answerable questions
9. [ ] Look for patterns in multi-turn dependencies
10. [ ] Implement BM25 baseline (target: Recall@5 = 0.20)
11. [ ] Run evaluation script on dev set
12. [ ] Analyze failure cases
13. [ ] Document baseline performance per domain

**Deliverable:** Working baseline + analysis report

### Week 2: Query Enhancement

**Tasks:**
1. [ ] Implement conversation history concatenation (simple approach)
2. [ ] Design prompts for LLM-based query rewriting
3. [ ] Test: "Rewrite this question to be standalone given conversation history"
4. [ ] Compare performance: last-turn-only vs rewritten queries
5. [ ] Experiment with different history lengths (1 turn, 3 turns, full history)
6. [ ] Weight recent turns higher than older turns
7. [ ] Test hybrid: question + last answer only
8. [ ] Run full dev set evaluation
9. [ ] Analyze per-domain performance
10. [ ] Identify best query formulation strategy
11. [ ] Document improvements over baseline

**Deliverable:** Query rewriting system + evaluation results

### Week 3: Advanced Retrieval

**Tasks:**
1. [ ] Implement BGE-base or similar dense retriever (target: Recall@5 = 0.30)
2. [ ] Compare with BM25 on multi-turn scenarios
3. [ ] Test query rewriting with dense retrieval
4. [ ] Analyze: Does dense help with later turns?
5. [ ] Implement BM25 + Dense hybrid (combine scores)
6. [ ] Add cross-encoder re-ranker for top-20 results
7. [ ] Test different fusion strategies (weighted sum, RRF)
8. [ ] Boost passages that appeared in previous turns
9. [ ] Penalize duplicate passages already shown
10. [ ] Test conversation-aware scoring

**Deliverable:** Best retrieval system + ablation study

### Week 4: Optimization & Submission

**Tasks:**
1. [ ] Hyperparameter optimization (weights, thresholds)
2. [ ] Test on full dev set (all domains)
3. [ ] Ensure @5 retrieval (required for Subtask C)
4. [ ] Deep dive into failures: answerability, later turns
5. [ ] Check performance by question type
6. [ ] Document limitations and edge cases
7. [ ] Format output according to submission requirements
8. [ ] Test submission format with provided scripts
9. [ ] Write system description (methods, experiments, results)
10. [ ] Submit to evaluation platform
11. [ ] Archive code and configuration
12. [ ] Prepare for potential revisions

**Target Performance:** Recall@5 > 0.49 (beat Elser baseline)

---

## Subtask B: Generation with Reference Passages - 4 Week Plan

**Goal:** Given a task + reference passages (gold), generate a high-quality answer.

**Metrics:** RB_alg, RB_llm, RL_f (all with IDK conditioning)

### Week 1: Setup & Baseline

**Tasks:**
1. [ ] Setup generation environment (OpenAI API / local LLM)
2. [ ] Load generation tasks (842 tasks with reference passages)
3. [ ] Understand input format: conversation history + question + passages
4. [ ] Review evaluation metrics and IDK conditioning
5. [ ] Analyze reference answers (length, style, structure)
6. [ ] Study how answers use passages (direct quotes vs paraphrasing)
7. [ ] Identify unanswerable questions and how they're handled
8. [ ] Check patterns in later turns vs first turns
9. [ ] Implement simple prompt: "Answer based on passages"
10. [ ] Test with Llama 3.1 8B or similar (target: RB_alg ~ 0.37)
11. [ ] Run evaluation script on dev set
12. [ ] Analyze IDK conditioning impact

**Deliverable:** Working baseline generator + metrics

### Week 2: Prompt Engineering

**Tasks:**
1. [ ] Design prompts that emphasize conversation context
2. [ ] Include examples of good multi-turn answers
3. [ ] Test: "Reference previous turns when relevant"
4. [ ] Compare few-shot vs zero-shot performance
5. [ ] Design prompt for identifying unanswerable questions
6. [ ] Test explicit: "Say 'I don't know' if passages don't contain answer"
7. [ ] Add calibration for partial answers
8. [ ] Measure impact on IDK-conditioned metrics
9. [ ] Prompt for faithfulness: "Only use information from passages"
10. [ ] Test different answer lengths/styles
11. [ ] Add instruction: "Be conversational, not robotic"
12. [ ] Evaluate RL_f (faithfulness) specifically

**Deliverable:** Optimized prompts + performance analysis

### Week 3: Model Selection & Fine-Tuning

**Tasks:**
1. [ ] Test multiple models: Llama 3.1 70B/405B, GPT-4, Qwen
2. [ ] Compare performance on later turns specifically
3. [ ] Analyze cost/performance tradeoffs
4. [ ] Select best model for final system
5. [ ] Prepare training data from MTRAG (use some dev as train)
6. [ ] Fine-tune smaller model on multi-turn QA task
7. [ ] Test if fine-tuning helps with later turns/answerability
8. [ ] Compare with strong prompted baseline
9. [ ] Add answer formatting/cleaning
10. [ ] Test response length optimization
11. [ ] Ensure factual grounding (citation checking)
12. [ ] Test ensemble: multiple models → consensus

**Deliverable:** Best generation system + model comparison

### Week 4: Optimization & Submission

**Tasks:**
1. [ ] Optimize prompts for each metric (balance RB_alg, RB_llm, RL_f)
2. [ ] Test different temperature/sampling parameters
3. [ ] Validate IDK threshold calibration
4. [ ] Analyze failures by question type
5. [ ] Check later turn performance specifically
6. [ ] Test on unanswerable questions
7. [ ] Document patterns in low-scoring answers
8. [ ] Generate answers for full dev set
9. [ ] Verify output format
10. [ ] Write system description
11. [ ] Test submission pipeline
12. [ ] Submit to evaluation platform
13. [ ] Archive model/prompts/config
14. [ ] Prepare for test phase

**Target Performance:** RB_alg > 0.48 (beat Llama 405B reference baseline)

---

## Subtask C: Full RAG (End-to-End) - 4 Week Plan

**Goal:** Given only a task, retrieve 5 passages AND generate answer (combines A + B).

**Metrics:** RB_alg, RB_llm, RL_f (with IDK conditioning) + Human Eval

### Week 1: Setup & Baseline

**Tasks:**
1. [ ] Setup end-to-end RAG pipeline
2. [ ] Integrate retrieval + generation components
3. [ ] Load full task data (842 tasks)
4. [ ] Review evaluation: automatic metrics + human eval (~20 tasks)
5. [ ] Design data flow: Task → Query → Retrieval → Generation → Answer
6. [ ] Implement basic pipeline: BM25 + Simple prompt
7. [ ] Test on sample conversations
8. [ ] Measure end-to-end latency
9. [ ] Run full baseline: BM25 retrieval + Llama 8B generation
10. [ ] Evaluate on dev set
11. [ ] Separate retrieval errors from generation errors
12. [ ] Measure error propagation (bad retrieval → bad generation)

**Deliverable:** End-to-end baseline + error analysis

### Week 2: Retrieval Optimization

**Tasks:**
1. [ ] Implement query rewriting for retrieval
2. [ ] Test: conversation-aware query formulation
3. [ ] Optimize for 5-passage retrieval (quality over quantity)
4. [ ] Measure impact on downstream generation
5. [ ] Upgrade to dense retriever (BGE or better)
6. [ ] Implement hybrid BM25 + Dense
7. [ ] Focus on precision@5 (all passages should be high quality)
8. [ ] Test: Does better retrieval improve generation metrics?
9. [ ] Experiment with diversity in top-5 passages
10. [ ] Test conversation-aware deduplication
11. [ ] Optimize passage ranking for generation usefulness
12. [ ] Validate retrieval component is production-ready

**Deliverable:** Optimized retrieval for generation

### Week 3: Generation Optimization

**Tasks:**
1. [ ] Adapt prompts from Subtask B to work with retrieved passages
2. [ ] Handle cases where retrieved passages are irrelevant
3. [ ] Test answerability detection with imperfect passages
4. [ ] Optimize faithfulness (only use retrieved passages)
5. [ ] Test agentic RAG: iterative query refinement
6. [ ] Implement: If answer uncertain → refine query → retrieve again
7. [ ] Balance retrieval quality vs generation quality
8. [ ] Optimize for all three metrics simultaneously
9. [ ] Add fallback strategies for failed retrieval
10. [ ] Improve IDK detection (better to say "I don't know" than hallucinate)
11. [ ] Test graceful degradation with poor passages
12. [ ] Ensure no hallucinations in final answers

**Deliverable:** Optimized end-to-end RAG system

### Week 4: Final Testing & Submission

**Tasks:**
1. [ ] Run complete pipeline on full dev set
2. [ ] Test all domains (ClapNQ, FiQA, Cloud, Govt)
3. [ ] Verify performance meets targets
4. [ ] Measure latency and resource usage
5. [ ] Review answers that will go to human eval
6. [ ] Ensure conversational quality
7. [ ] Check naturalness and appropriateness
8. [ ] Test with human reviewers if possible
9. [ ] Format outputs correctly
10. [ ] Test submission pipeline end-to-end
11. [ ] Write comprehensive system description
12. [ ] Document architecture and design decisions
13. [ ] Submit final system
14. [ ] Archive complete pipeline
15. [ ] Prepare for potential human evaluation results
16. [ ] Document lessons learned

**Target Performance:** 
- RB_alg > 0.40 in RAG setting (beat GPT-4o baseline)
- RL_f > 0.70 (strong faithfulness)
- Strong human evaluation scores

