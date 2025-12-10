#!/usr/bin/env python3
"""
Analyze mismatches between mono-T5 and oracle choices.
Shows detailed mono-T5 scores for top documents to understand why wrong choices were made.
"""
import json
from pathlib import Path
import sys

# Use absolute paths
script_dir = Path("/Users/pastil/Dev/Github/mt-rag-benchmark/scripts/ideas/retrieval_tasks/mono_t5_oracle_selection")
project_root = Path("/Users/pastil/Dev/Github/mt-rag-benchmark")

# Import utils and mono-T5 scorer
sys.path.insert(0, str(script_dir))
from utils import (
    load_retrieval_results_with_texts,
    load_queries,
    STRATEGY,
    DOMAINS
)
from mono_t5_oracle_selection import MonoT5Scorer, CACHE_DIR

# Load mismatches
results_dir = script_dir / "results"
with open(results_dir / "mismatches.json", 'r') as f:
    mismatches = json.load(f)

# Load retrieval results
bm25_dir = project_root / "scripts" / "baselines" / "retrieval_scripts" / "bm25" / "results"
bge_dir = project_root / "scripts" / "baselines" / "retrieval_scripts" / "bge" / "results"
elser_dir = project_root / "scripts" / "baselines" / "retrieval_scripts" / "elser" / "results"

print("Loading retrieval results...")
bm25_results = load_retrieval_results_with_texts(bm25_dir, 'bm25', STRATEGY, domains=DOMAINS)
bge_results = load_retrieval_results_with_texts(bge_dir, 'bge', STRATEGY, domains=DOMAINS)
elser_results = load_retrieval_results_with_texts(elser_dir, 'elser', STRATEGY, domains=DOMAINS)

# Load queries
queries = load_queries(STRATEGY, project_root, domains=DOMAINS)

# Initialize mono-T5 scorer
print("\nLoading mono-T5 model...")
scorer = MonoT5Scorer(cache_dir=CACHE_DIR)

# Analyze top 5 worst mismatches
print("\n" + "=" * 100)
print("DETAILED ANALYSIS OF TOP 5 WORST MISMATCHES")
print("=" * 100)

for i, m in enumerate(mismatches[:5], 1):
    task_id = m['task_id']
    query = queries.get(task_id, 'N/A')
    
    print(f"\n{'='*100}")
    print(f"Example {i}: Task ID: {task_id}")
    print(f"Query: {query}")
    print(f"\nMono-T5 chose: {m['monot5_choice'].upper()} (R@10 = {m['monot5_score']:.4f})")
    print(f"Oracle chose:   {m['oracle_choice'].upper()} (R@10 = {m['oracle_score']:.4f})")
    print(f"Performance gap: {m['gap']:.4f}")
    print(f"\nAll retriever scores - BM25: {m['bm25_score']:.4f}, BGE: {m['bge_score']:.4f}, ELSER: {m['elser_score']:.4f}")
    
    # Score top 5 documents from each retriever with mono-T5
    print(f"\n{'─'*100}")
    print("MONO-T5 SCORES FOR TOP 5 DOCUMENTS FROM EACH RETRIEVER:")
    print(f"{'─'*100}")
    
    for retriever_name, results in [('bm25', bm25_results), ('bge', bge_results), ('elser', elser_results)]:
        if task_id in results:
            contexts = results[task_id].get('contexts', [])[:5]
            print(f"\n{retriever_name.upper()} (Oracle R@10 = {m[f'{retriever_name}_score']:.4f}):")
            
            doc_texts = [ctx.get('text', '') for ctx in contexts]
            if doc_texts:
                # Score with mono-T5
                mono_scores = scorer.score_batch(query, doc_texts)
                
                for j, (ctx, mono_score) in enumerate(zip(contexts, mono_scores), 1):
                    retrieval_score = ctx.get('score', 0)
                    doc_text = ctx.get('text', '')[:150]
                    print(f"  {j}. Retrieval Score: {retrieval_score:.4f} | Mono-T5 Score: {mono_score:.4f}")
                    print(f"     Text: {doc_text}...")
                
                # Calculate predicted recall@10 for this retriever
                from utils import calculate_predicted_recall_at_k
                predicted_recall = calculate_predicted_recall_at_k(
                    contexts, mono_scores, k=10, threshold=0.9
                )
                avg_mono_score = sum(mono_scores) / len(mono_scores) if mono_scores else 0
                print(f"  → Predicted R@10 (threshold=0.9): {predicted_recall:.4f}")
                print(f"  → Average Mono-T5 Score: {avg_mono_score:.4f}")

print(f"\n{'='*100}")
print(f"Analysis complete! Analyzed {len(mismatches[:5])} worst mismatches.")
print(f"{'='*100}")
