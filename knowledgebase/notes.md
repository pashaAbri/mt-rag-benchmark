# Notes from the Paper
* goals: handle later turns, unanswerable questions, non-standalone questions, and multiple domains
* multi-turn conversation benchmark:
    * Retreival: the relevant pasages should change during the conversation causing repeated retrieval
    * Generation: The generation should struggle to answer many of the questions that refer to and rely on previous turns
* MTRAG: 
    * human generated conversations
    * across 4 diff domains
    * vary in style topic and source
    * our conversations comprise turns that vary along the dimentions of question type, multi-turn, and answerability
* annotators:
    * at every turn, after issuing their quesitons, annotators checked the passages retrived by the RAG system and modified the pasage set to imrprove relevance and diversity
    * next, they reviewed and repaired the generated response to improve its quality
    * the resulting conversations average 7-8 turns in length and 16.9 unique relevant passages per conversation
* evaluate our benchmark on Retrieval and generation components of RAG systems
    * Retrieval performance of lexical, sparse, and dense retrieval under two settings (last turn and query rewrite)
        * **Lexical Retrieval** (e.g., BM25): Traditional keyword/term matching, like Ctrl+F search - looks for exact word matches. Fast but can miss semantic meaning.
        * **Sparse Retrieval** (e.g., SPLADE): Uses learned representations but still sparse vectors. Better at understanding synonyms and related terms. Middle ground between lexical and dense.
        * **Dense Retrieval** (e.g., embedding models): Uses neural networks to create dense vector embeddings. Captures semantic meaning (e.g., "car" and "automobile" are similar). Better at conceptual matching.
        * **Two Settings/Scenarios:**
            * **Last Turn**: Use only the most recent question as the search query. Example: If turn 5 asks "What about their pricing?", search using just that phrase. Problem: Loses context from earlier turns.
            * **Query Rewrite**: Reformulate the question to include necessary context from conversation history. Example: Rewrite "What about their pricing?" → "What is the pricing for AWS Lambda functions?". Attempts to make each turn standalone.
    * analyze generative performance of 9 LLMs under three retrieval settings (reference, reference+RAG, and full RAG)
        * **Reference**: LLM gets the gold standard (perfect) passages curated by human annotators. Best case scenario - tests pure generation ability. Shows maximum possible performance.
        * **Reference + RAG**: LLM gets the reference passages PLUS automatically retrieved passages. Tests if adding RAG results helps or hurts when you already have good passages. Can reveal if extra (potentially noisy) passages confuse the model.
        * **Full RAG**: LLM gets ONLY automatically retrieved passages (no human-curated ones). Real-world scenario - what actual RAG systems do. Most challenging - tests the complete pipeline.
    * **Why This Matters:**
        * This dual evaluation reveals:
            * **Retrieval bottleneck**: If retrieval fails, even the best LLM can't answer correctly
            * **Generation bottleneck**: Even with perfect passages (reference), LLMs struggle with multi-turn questions
            * **Error propagation**: How retrieval errors impact generation quality
            * **Comparative analysis**: Which component needs more improvement
        * The goal is to understand where RAG systems break down in multi-turn conversations and guide future improvements.

* **Automation Paths**: Human data generation and evaluation doesn't scale well - they explore automation via:
    * **Automatic Evaluation (LLM-as-a-Judge + Algorithmic Metrics)**:
        * Reference-Based Metrics:
            * RB_llm: Adapted RAD-Bench using LLMs to compare responses against gold standard answers
            * RB_alg: Algorithmic/statistical comparison to reference answers
        * Reference-Less Metrics (no gold answer needed):
            * Faithfulness (RL_F): Checks if response is grounded in retrieved passages (avoids hallucinations)
            * Answer Relevance (RL_R): LLM guesses what question generated the response, compares to actual question
            * Multi-Turn Bias (RL_MTB): Detects if model ignores conversation history
        * **Finding**: Some automated metrics correlate with human scores, many do not - more work needed in automatic evaluation
    * **Synthetic Data Generation (mtRAG-S)**:
        * Companion benchmark with synthetically generated conversations using LLMs
        * Extended Lee et al. 2024 framework for multi-turn generation
        * Key differences from human data:
            * Shorter conversations (5.9 vs 7.7 avg turns)
            * Lower passage diversity (4.6 vs 16.9 unique passages per conv)
            * Longer questions but shorter responses
            * No human-edited responses
        * Challenges:
            * Hard to generate truly unanswerable questions (LLM often creates partial answers)
            * Longer turns lead to repetitive questions and hallucinated responses
            * Lower passage diversity
        * **Purpose**: By providing both human and synthetic benchmarks over same corpora, enables comparison and understanding of tradeoffs

* **Corpora - Four Diverse Domains**:
    * **Total**: 366,479 passages across 4 domains varying in style, topic, and source
    * **ClapNQ (Wikipedia)**: 
        * Existing corpus from QA/IR dataset
        * 4,293 documents → 183,408 passages (largest corpus)
        * General knowledge, encyclopedic content
        * Writing style: Formal, encyclopedic, well-structured
    * **FiQA (Finance)**:
        * Existing corpus - StackExchange posts discussing financial advice
        * 7,661 documents → 49,607 passages
        * Personal finance, investing, stock market advice
        * Writing style: Conversational, Q&A format, community-driven
    * **Cloud (Technical Documentation)**:
        * NEW corpus - crawled from major cloud provider (IBM Cloud)
        * 57,638 documents → 61,022 passages
        * Cloud computing, technical guides, video tutorials
        * Writing style: Technical, instructional, product documentation
    * **Govt (Government)**:
        * NEW corpus - crawled from .gov and .mil domains
        * 8,578 documents → 72,422 passages
        * Government services, policies, municipal information
        * Writing style: Official, bureaucratic, public service information
    * **Key Design Features**:
        * Two existing corpora (ClapNQ, FiQA) + two new custom corpora (Cloud, Govt)
        * Documents split into 512-token passages with 100-token overlap
        * Passage-level corpus recommended for experiments

* **Human Data** (human/ directory):
    * **Conversations** (human/conversations/):
        * 110 human-generated conversations in single JSON file (conversations.json)
        * Distribution across domains:
            * ClapNQ: 29 conversations
            * Cloud: 26 conversations
            * FiQA: 27 conversations
            * Govt: 28 conversations
        * Each conversation contains:
            * Author, domain, generator, retriever metadata
            * Messages array with user/agent exchanges
            * Each message has: speaker, text, timestamp, enrichments (answerability, question type, multi-turn type)
            * Review and status information
    * **Generation Tasks** (human/generation_tasks/):
        * 842 evaluation tasks (one per conversation turn)
        * Three settings:
            * reference.jsonl (842 tasks): Gold standard human-curated passages
            * reference+RAG.jsonl (436 tasks): Reference + auto-retrieved passages (restricted to ≤2 contexts)
            * RAG.jsonl (842 tasks): Only auto-retrieved passages (real-world scenario)
        * Each task includes:
            * task_id, conversation_id, turn number
            * Collection (domain name)
            * contexts: retrieved passages with relevance feedback
            * input: conversation history + current question
            * targets: reference response
            * Enrichments: answerability, question type, multi-turn type
    * **Retrieval Tasks** (human/retrieval_tasks/):
        * BEIR format for standard IR evaluation
        * Organized by domain (clapnq/, cloud/, fiqa/, govt/)
        * Each domain has ~208 queries in 3 formats:
            * {domain}_lastturn.jsonl: Current question only (no context)
            * {domain}_rewrite.jsonl: Question rewritten to be standalone
            * {domain}_questions.jsonl: All questions in original form
        * Relevance judgments (qrels/dev.tsv):
            * Total: 2,132 relevance judgments across all domains
            * ClapNQ: 579, Cloud: 495, FiQA: 536, Govt: 522
            * Format: query-id, corpus-id, relevance score (TSV)
        * Only includes answerable and partial questions (unanswerable excluded)
    * **Evaluations** (human/evaluations/):
        * Pre-computed experimental results from paper (large JSON files)
        * reference.json (14 MB): Results with gold standard passages
        * reference+RAG.json (8.5 MB): Results with reference + RAG passages
        * RAG.json (16 MB): Results with only RAG-retrieved passages
        * reference_subset_with_human_evaluations.json (2.1 MB): Subset with human quality judgments
        * Contains model responses from 9 LLMs tested in paper with automatic and human evaluation scores

* **Evaluation Metrics** (human/evaluations/):
    * **All Files** (reference.json, reference+RAG.json, RAG.json):
        * **Algorithmic Metrics Only:**
            * **Rouge-L**: Word overlap score measuring longest common subsequence between generated and reference response (range: 0-1)
            * **Bert-Rec (BERT Recall)**: BERT-based recall score measuring semantic overlap with reference (range: -1 to 1)
            * **Bert-KPrec (BERT K-Precision)**: BERT-based knowledge precision score (range: -1 to 1)
            * **Conditional IDK**: Measures if model appropriately says "I don't know" for unanswerable questions
            * **RB-ALG (Reference-Based Algorithmic)**: Harmonic mean of Bert-Recall, Bert-K-Precision, and Rouge-L (from Adlakha et al., 2024)
            * **RB-LLM (Reference-Based LLM Judge)**: LLM-as-judge comparing response to gold standard (inspired by RADBench - Kuo et al., 2024)
            * **RL-F (Reference-Less Faithfulness)**: LLM judge checking if response is grounded in retrieved passages (from RAGAS - Es et al., 2024)
    * **reference_subset_with_human_evaluations.json ONLY:**
        * **All algorithmic metrics above PLUS:**
        * **Human Evaluation Metrics** (4-point Likert scale: No=1, Mostly No=2, Mostly Yes=3, Yes=4):
            * **Naturalness**: Is the response coherent, natural, and not dismissive?
            * **Appropriateness**: Does the response provide an appropriate amount of useful information?
            * **Completeness**: Does the response include all information relevant to the question found in the context?
            * **Faithfulness**: Is the response faithful and grounded in the context (not hallucinated)?
            * **Win Rate**: Head-to-head comparison - percentage of times this model's response was preferred over other models' responses for the same task (range: 0-100)
    * **Key Differences:**
        * **reference.json, reference+RAG.json, RAG.json**: Only automated metrics (no human judgments)
        * **reference_subset_with_human_evaluations.json**: Both automated metrics AND human quality judgments on a subset of data