#!/usr/bin/env python3
"""
Unified script to run MonoT5 reranking on various retrieval strategy combinations (Targeted Rewrite Edition).

This script:
1. Supports all strategy combinations (single, pairs, or all three)
2. Combines retrieval results from specified strategies
3. Deduplicates documents by document_id
4. Uses MonoT5 to rerank the combined results
5. Saves results in standard format and runs standard evaluation script

Usage:
    python run_rerank.py                    # Run all combinations
    python run_rerank.py --combinations lastturn+targeted_rewrite  # Run specific combination
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
MODEL_NAME = str(project_root / "scripts/ideas/retrieval_tasks/mono_t5_oracle_selection/.cache")
if not Path(MODEL_NAME).exists():
    print(f"Error: Local model path not found: {MODEL_NAME}")
    print("Please run scripts/ideas/retrieval_tasks/mono_t5_oracle_selection/download_model.py first.")
    sys.exit(1)

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
# INTERMEDIATE_DIR determined in main() based on args

# All possible combinations
ALL_COMBINATIONS = [
    ['lastturn', 'targeted_rewrite'],
    ['lastturn', 'questions'],
    ['targeted_rewrite', 'questions'],
    ['lastturn', 'targeted_rewrite', 'questions'],
]

def get_combination_name(strategies):
    """Generate a filename-safe name for the combination."""
    return "_".join(sorted(strategies))

def parse_combination(combo_str):
    """Parse combination string like 'lastturn+targeted_rewrite' into list."""
    return sorted(combo_str.split('+'))

def main():
    parser = argparse.ArgumentParser(description='Evaluate retrieval strategy combinations with MonoT5')
    parser.add_argument(
        '--combinations',
        nargs='+',
        help='Specific combinations to run (e.g., "lastturn+targeted_rewrite"). If not specified, runs all.',
        default=None
    )
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='Skip reranking if output file already exists'
    )
    parser.add_argument(
        '--rerank-query',
        type=str,
        default='targeted_rewrite',
        choices=['targeted_rewrite', 'lastturn', 'questions'],
        help='Query type to use for reranking (default: targeted_rewrite)'
    )
    parser.add_argument(
        '--domains',
        nargs='+',
        choices=DOMAINS,
        help='Specific domains to process. If not specified, runs all.',
        default=None
    )
    args = parser.parse_args()
    
    # Determine output directory
    intermediate_dir = script_dir / "intermediate" / f"using_{args.rerank_query}_query"
    intermediate_dir.mkdir(exist_ok=True, parents=True)
    
    # Determine which combinations to run
    if args.combinations:
        combinations = [parse_combination(c) for c in args.combinations]
    else:
        combinations = ALL_COMBINATIONS
    
    print("="*80)
    print("Unified Retrieval Strategy Combination Evaluation (Targeted Rewrite)")
    print("="*80)
    print(f"Combinations to evaluate: {[get_combination_name(c) for c in combinations]}")
    print(f"Reranking query type: {args.rerank_query}")
    print(f"Output directory: {intermediate_dir}")
    print(f"Skip existing files: {args.skip_existing}")
    print("="*80)
    
    # Initialize MonoT5 scorer
    try:
        scorer = MonoT5Scorer(MODEL_NAME, DEVICE)
    except NameError:
        print("Error: MonoT5Scorer not found. Check if torch/transformers are installed.")
        return
    
    # Filter domains if specified
    domains_to_process = args.domains if args.domains else DOMAINS
    
    # Process each domain
    for domain in domains_to_process:
        print(f"\n{'='*60}")
        print(f"Processing domain: {domain.upper()}")
        print(f"{'='*60}")
        
        # Load queries for reranking
        queries_rerank = load_queries(domain, args.rerank_query)
        print(f"Loaded {len(queries_rerank)} {args.rerank_query} queries for reranking")
        
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
            
            output_file = intermediate_dir / f"reranked_{combo_name}_{domain}.jsonl"
            evaluated_file = intermediate_dir / f"reranked_{combo_name}_{domain}_evaluated.jsonl"
            
            # Check if exists to skip reranking
            if args.skip_existing and output_file.exists():
                print(f"Output file {output_file.name} already exists. Skipping reranking step.")
            else:
                # Prepare results dict for this combination
                results_by_strategy = {s: loaded_results[s] for s in combo}
                
                # Combine and deduplicate
                combined_docs = combine_and_deduplicate(results_by_strategy)
                
                # Filter to tasks that are in queries_rerank
                tasks_to_process = [tid for tid in combined_docs.keys() if tid in queries_rerank]
                print(f"Combined documents for {len(tasks_to_process)} tasks")
                
                # Rerank with MonoT5
                reranked_results = {}
                for task_id in tqdm(tasks_to_process, desc=f"Reranking {combo_name}"):
                    documents = combined_docs[task_id]
                    query = queries_rerank[task_id]
                    
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

