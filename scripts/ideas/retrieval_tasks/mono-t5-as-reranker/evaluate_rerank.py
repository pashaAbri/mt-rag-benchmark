#!/usr/bin/env python3
"""
Evaluate combined retrieval + MonoT5 reranking approach.

This script:
1. Combines retrieval results from all three strategies (lastturn, rewrite, questions)
2. Deduplicates documents by document_id
3. Uses MonoT5 to rerank the combined results
4. Saves results in standard format and runs standard evaluation script
"""

import sys
import json
import torch
import subprocess
from pathlib import Path
from tqdm import tqdm

from utils import (
    MonoT5Scorer,
    load_queries,
    load_retrieval_results,
    combine_and_deduplicate,
    rerank_with_monot5,
    save_results_for_evaluation,
    DOMAINS,
    STRATEGIES,
    EVAL_SCRIPT_PATH
)

# Add project root to path (needed if importing other project modules, though utils handles most)
script_dir = Path(__file__).parent
project_root = script_dir.parents[3]
sys.path.insert(0, str(project_root))

# Configuration
MODEL_NAME = "castorini/monot5-base-msmarco"
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
INTERMEDIATE_DIR = script_dir / "intermediate"


def main():
    print("="*80)
    print("Combined Retrieval + MonoT5 Reranking Evaluation")
    print("="*80)
    
    # Ensure intermediate directory exists
    INTERMEDIATE_DIR.mkdir(exist_ok=True)
    
    # Initialize MonoT5 scorer
    try:
        scorer = MonoT5Scorer(MODEL_NAME, DEVICE)
    except NameError:
        print("Error: MonoT5Scorer not found. Check if torch/transformers are installed.")
        return
    
    # Process each domain
    for domain in DOMAINS:
        print(f"\n{'='*60}")
        print(f"Processing domain: {domain.upper()}")
        print(f"{'='*60}")
        
        # Load queries (use rewrite as the query for reranking)
        queries_rewrite = load_queries(domain, 'rewrite')
        print(f"Loaded {len(queries_rewrite)} rewrite queries")
        
        # Load retrieval results for all strategies
        results_by_strategy = {}
        for strategy in STRATEGIES:
            results = load_retrieval_results(domain, strategy)
            results_by_strategy[strategy] = results
            print(f"  {strategy}: {len(results)} tasks")
        
        # Combine and deduplicate
        print(f"\nCombining and deduplicating results...")
        combined_docs = combine_and_deduplicate(results_by_strategy)
        
        # Filter to tasks that are in queries_rewrite
        tasks_to_process = [tid for tid in combined_docs.keys() if tid in queries_rewrite]
        print(f"Combined documents for {len(tasks_to_process)} tasks")
        
        # Rerank with MonoT5
        print(f"\nReranking with MonoT5...")
        reranked_results = {}
        
        for task_id in tqdm(tasks_to_process, desc="Reranking"):
            documents = combined_docs[task_id]
            query = queries_rewrite[task_id]
            
            # Rerank
            reranked = rerank_with_monot5(scorer, query, documents, top_k=100)
            
            # Convert to results format
            doc_scores = {doc['document_id']: score for doc, score in reranked}
            if doc_scores:
                reranked_results[task_id] = doc_scores
        
        # Save results for standard evaluation in intermediate directory
        output_file = INTERMEDIATE_DIR / f"reranked_{domain}.jsonl"
        evaluated_file = INTERMEDIATE_DIR / f"reranked_{domain}_evaluated.jsonl"
        
        save_results_for_evaluation(reranked_results, domain, output_file)
        
        # Run standard evaluation script
        print(f"\nRunning standard evaluation for {domain}...")
        cmd = [
            "python", str(EVAL_SCRIPT_PATH),
            "--input_file", str(output_file),
            "--output_file", str(evaluated_file)
        ]
        
        try:
            subprocess.run(cmd, check=True)
            print(f"Evaluation complete. Results saved to {evaluated_file}")
            
            # Print aggregate results if available
            agg_file = evaluated_file.with_suffix('') 
            agg_file = Path(f"{agg_file}_aggregate.csv")
            
            if agg_file.exists():
                print("\nAggregate Results:")
                with open(agg_file, 'r') as f:
                    print(f.read())
            else:
                print(f"Aggregate file not found at expected path: {agg_file}")
                    
        except subprocess.CalledProcessError as e:
            print(f"Error running evaluation: {e}")


if __name__ == '__main__':
    main()
