#!/usr/bin/env python3
"""
Unified script to evaluate all retrieval strategy combinations with MonoT5 reranking.

This script:
1. Supports all strategy combinations (single, pairs, or all three)
2. Combines retrieval results from specified strategies
3. Deduplicates documents by document_id
4. Uses MonoT5 to rerank the combined results
5. Saves results in standard format and runs standard evaluation script

Usage:
    python evaluate_all_combinations.py                    # Run all combinations
    python evaluate_all_combinations.py --combinations lastturn+rewrite  # Run specific combination
"""

import sys
import json
import torch
import subprocess
import argparse
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
    EVAL_SCRIPT_PATH
)

# Add project root to path
script_dir = Path(__file__).parent
project_root = script_dir.parents[3]
sys.path.insert(0, str(project_root))

# Configuration
MODEL_NAME = "castorini/monot5-base-msmarco"
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
INTERMEDIATE_DIR = script_dir / "intermediate"

# All possible combinations
ALL_COMBINATIONS = [
    ['lastturn', 'rewrite'],
    ['lastturn', 'questions'],
    ['rewrite', 'questions'],
    ['lastturn', 'rewrite', 'questions'],
]

def get_combination_name(strategies):
    """Generate a filename-safe name for the combination."""
    return "_".join(sorted(strategies))

def parse_combination(combo_str):
    """Parse combination string like 'lastturn+rewrite' into list."""
    return sorted(combo_str.split('+'))

def main():
    parser = argparse.ArgumentParser(description='Evaluate retrieval strategy combinations with MonoT5')
    parser.add_argument(
        '--combinations',
        nargs='+',
        help='Specific combinations to run (e.g., "lastturn+rewrite" "rewrite+questions"). If not specified, runs all.',
        default=None
    )
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='Skip reranking if output file already exists'
    )
    args = parser.parse_args()
    
    # Determine which combinations to run
    if args.combinations:
        combinations = [parse_combination(c) for c in args.combinations]
    else:
        combinations = ALL_COMBINATIONS
    
    print("="*80)
    print("Unified Retrieval Strategy Combination Evaluation")
    print("="*80)
    print(f"Combinations to evaluate: {[get_combination_name(c) for c in combinations]}")
    print(f"Skip existing files: {args.skip_existing}")
    print("="*80)
    
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
        
        # Load all retrieval results upfront
        all_strategies = set()
        for combo in combinations:
            all_strategies.update(combo)
        
        loaded_results = {}
        for strategy in all_strategies:
            results = load_retrieval_results(domain, strategy)
            loaded_results[strategy] = results
            print(f"  {strategy}: {len(results)} tasks")
        
        # Process each combination
        for combo in combinations:
            combo_name = get_combination_name(combo)
            print(f"\n--- Combination: {combo} ({combo_name}) ---")
            
            output_file = INTERMEDIATE_DIR / f"reranked_{combo_name}_{domain}.jsonl"
            evaluated_file = INTERMEDIATE_DIR / f"reranked_{combo_name}_{domain}_evaluated.jsonl"
            
            # Check if exists to skip reranking
            if args.skip_existing and output_file.exists():
                print(f"Output file {output_file.name} already exists. Skipping reranking step.")
            else:
                # Prepare results dict for this combination
                results_by_strategy = {s: loaded_results[s] for s in combo}
                
                # Combine and deduplicate
                combined_docs = combine_and_deduplicate(results_by_strategy)
                
                # Filter to tasks that are in queries_rewrite
                tasks_to_process = [tid for tid in combined_docs.keys() if tid in queries_rewrite]
                print(f"Combined documents for {len(tasks_to_process)} tasks")
                
                # Rerank with MonoT5
                reranked_results = {}
                for task_id in tqdm(tasks_to_process, desc=f"Reranking {combo_name}"):
                    documents = combined_docs[task_id]
                    query = queries_rewrite[task_id]
                    
                    reranked = rerank_with_monot5(scorer, query, documents, top_k=100)
                    
                    doc_scores = {doc['document_id']: score for doc, score in reranked}
                    if doc_scores:
                        reranked_results[task_id] = doc_scores
                
                # Save results
                save_results_for_evaluation(reranked_results, domain, output_file)
            
            # Evaluate
            print(f"Running evaluation for {combo_name} on {domain}...")
            cmd = [
                sys.executable, str(EVAL_SCRIPT_PATH),
                "--input_file", str(output_file),
                "--output_file", str(evaluated_file)
            ]
            
            try:
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                
                # Print aggregate results if available
                agg_file = evaluated_file.with_suffix('') 
                agg_file = Path(f"{agg_file}_aggregate.csv")
                
                if agg_file.exists():
                    import pandas as pd
                    import ast
                    df = pd.read_csv(agg_file)
                    all_rows = df[df['collection'] == 'all']
                    if len(all_rows) > 0:
                        all_row = all_rows.iloc[0]
                        ndcg = ast.literal_eval(all_row['nDCG'])
                        recall = ast.literal_eval(all_row['Recall'])
                        print(f"  nDCG@10: {ndcg[3]:.4f}, Recall@10: {recall[3]:.4f}")
                
            except subprocess.CalledProcessError as e:
                print(f"Error running evaluation: {e}")
                if e.stderr:
                    print(f"Stderr: {e.stderr}")
    
    print("\n" + "="*80)
    print("All combinations evaluated!")
    print("="*80)

if __name__ == '__main__':
    main()
