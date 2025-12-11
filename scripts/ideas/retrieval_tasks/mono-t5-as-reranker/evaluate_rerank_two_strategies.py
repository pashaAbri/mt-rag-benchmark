#!/usr/bin/env python3
"""
Evaluate combined retrieval + MonoT5 reranking with only lastturn and rewrite strategies.

This script:
1. Combines retrieval results from two strategies (lastturn, rewrite)
2. Deduplicates documents by document_id
3. Uses MonoT5 to rerank the combined results
4. Compares results with original 3-strategy version and individual strategies
"""

import sys
import json
import torch
import subprocess
import pandas as pd
import ast
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm

from utils import (
    MonoT5Scorer,
    load_queries,
    load_retrieval_results,
    rerank_with_monot5,
    save_results_for_evaluation,
    DOMAINS,
    EVAL_SCRIPT_PATH,
    COLLECTION_NAMES
)

# Add project root to path
script_dir = Path(__file__).parent
project_root = script_dir.parents[3]
sys.path.insert(0, str(project_root))

# Configuration
MODEL_NAME = "castorini/monot5-base-msmarco"
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
INTERMEDIATE_DIR = script_dir / "intermediate"
COMPARISON_DIR = script_dir / "comparison_results"

# Only use two strategies
STRATEGIES = ['lastturn', 'rewrite']


def combine_and_deduplicate_two_strategies(
    results_by_strategy: Dict[str, Dict[str, List[Dict]]]
) -> Dict[str, List[Dict]]:
    """
    Combine retrieval results from two strategies (lastturn, rewrite) and deduplicate by document_id.
    
    Returns:
        Dictionary mapping task_id to list of unique documents
    """
    combined = {}
    
    # Find common task_ids
    common_task_ids = None
    for strategy in STRATEGIES:
        if strategy in results_by_strategy:
            task_ids = set(results_by_strategy[strategy].keys())
            if common_task_ids is None:
                common_task_ids = task_ids
            else:
                common_task_ids &= task_ids
    
    if common_task_ids is None:
        return {}
    
    for task_id in common_task_ids:
        # Collect all documents from both strategies
        doc_dict = {}  # document_id -> document info
        
        for strategy in STRATEGIES:
            if strategy in results_by_strategy and task_id in results_by_strategy[strategy]:
                contexts = results_by_strategy[strategy][task_id]
                for ctx in contexts:
                    doc_id = ctx.get('document_id')
                    if doc_id:
                        # Keep the first occurrence (or could keep highest score)
                        if doc_id not in doc_dict:
                            doc_dict[doc_id] = {
                                'document_id': doc_id,
                                'text': ctx.get('text', ''),
                                'title': ctx.get('title', ''),
                                'source': ctx.get('source', ''),
                                'original_score': ctx.get('score', 0.0),
                                'strategies': [strategy]
                            }
                        else:
                            # Track which strategies retrieved this doc
                            if strategy not in doc_dict[doc_id]['strategies']:
                                doc_dict[doc_id]['strategies'].append(strategy)
        
        combined[task_id] = list(doc_dict.values())
    
    return combined


def load_evaluation_results(file_path: Path) -> Dict[str, float]:
    """Load evaluation results from aggregate CSV file."""
    if not file_path.exists():
        return {}
    
    try:
        df = pd.read_csv(file_path)
        # Get the 'all' row which has weighted averages
        all_rows = df[df['collection'] == 'all']
        if len(all_rows) == 0:
            # If no 'all' row, use the first row
            all_row = df.iloc[0]
        else:
            all_row = all_rows.iloc[0]
        
        # Parse the list strings
        recall = ast.literal_eval(all_row['Recall'])
        ndcg = ast.literal_eval(all_row['nDCG'])
        
        # Convert to dict with metric names matching pytrec_eval format
        results = {
            'recall_1': recall[0],
            'recall_3': recall[1],
            'recall_5': recall[2],
            'recall_10': recall[3] if len(recall) > 3 else 0.0,
            'ndcg_cut_1': ndcg[0],
            'ndcg_cut_3': ndcg[1],
            'ndcg_cut_5': ndcg[2],
            'ndcg_cut_10': ndcg[3] if len(ndcg) > 3 else 0.0,
        }
        
        return results
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return {}


def compare_results():
    """Compare results between 2-strategy and 3-strategy versions."""
    print("\n" + "="*80)
    print("COMPARISON: 2-Strategy vs 3-Strategy vs Individual Strategies")
    print("="*80)
    
    comparison_data = []
    
    for domain in DOMAINS:
        # Load 2-strategy results
        two_strategy_file = INTERMEDIATE_DIR / f"reranked_two_strategies_{domain}_evaluated_aggregate.csv"
        two_strategy_results = load_evaluation_results(two_strategy_file)
        
        if not two_strategy_results:
            print(f"Warning: No 2-strategy results found for {domain}, skipping comparison")
            continue
        
        # Load 3-strategy results (original) - optional
        three_strategy_file = INTERMEDIATE_DIR / f"reranked_{domain}_evaluated_aggregate.csv"
        three_strategy_results = load_evaluation_results(three_strategy_file)
        
        # Load individual strategy results - optional
        lastturn_file = INTERMEDIATE_DIR / f"baseline_lastturn_{domain}_evaluated_aggregate.csv"
        rewrite_file = INTERMEDIATE_DIR / f"baseline_rewrite_{domain}_evaluated_aggregate.csv"
        lastturn_results = load_evaluation_results(lastturn_file)
        rewrite_results = load_evaluation_results(rewrite_file)
        
        # Key metrics to compare
        metrics = ['ndcg_cut_10', 'ndcg_cut_5', 'ndcg_cut_3', 'ndcg_cut_1', 
                   'recall_10', 'recall_5', 'recall_3', 'recall_1']
        
        for metric in metrics:
            comparison_data.append({
                'domain': domain,
                'metric': metric,
                '2_strategy': two_strategy_results.get(metric, 0.0),
                '3_strategy': three_strategy_results.get(metric, None) if three_strategy_results else None,
                'lastturn': lastturn_results.get(metric, None) if lastturn_results else None,
                'rewrite': rewrite_results.get(metric, None) if rewrite_results else None,
            })
    
    if not comparison_data:
        print("No comparison data available. Run evaluations first.")
        return
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(comparison_data)
    
    # Save comparison
    COMPARISON_DIR.mkdir(exist_ok=True)
    comparison_file = COMPARISON_DIR / "comparison_2_vs_3_strategies.csv"
    comparison_df.to_csv(comparison_file, index=False)
    print(f"\nComparison saved to {comparison_file}")
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY: nDCG@10 Comparison")
    print("="*80)
    
    summary_data = []
    for domain in DOMAINS:
        two_strategy_file = INTERMEDIATE_DIR / f"reranked_two_strategies_{domain}_evaluated_aggregate.csv"
        three_strategy_file = INTERMEDIATE_DIR / f"reranked_{domain}_evaluated_aggregate.csv"
        lastturn_file = INTERMEDIATE_DIR / f"baseline_lastturn_{domain}_evaluated_aggregate.csv"
        rewrite_file = INTERMEDIATE_DIR / f"baseline_rewrite_{domain}_evaluated_aggregate.csv"
        
        two_results = load_evaluation_results(two_strategy_file)
        if not two_results:
            continue
            
        two_ndcg10 = two_results.get('ndcg_cut_10', 0.0)
        three_results = load_evaluation_results(three_strategy_file)
        three_ndcg10 = three_results.get('ndcg_cut_10', None) if three_results else None
        lastturn_results = load_evaluation_results(lastturn_file)
        lastturn_ndcg10 = lastturn_results.get('ndcg_cut_10', None) if lastturn_results else None
        rewrite_results = load_evaluation_results(rewrite_file)
        rewrite_ndcg10 = rewrite_results.get('ndcg_cut_10', None) if rewrite_results else None
        
        row = {
            'domain': domain,
            '2_strategy': f"{two_ndcg10:.4f}",
        }
        
        if three_ndcg10 is not None:
            row['3_strategy'] = f"{three_ndcg10:.4f}"
            row['diff_2_vs_3'] = f"{two_ndcg10 - three_ndcg10:+.4f}"
        else:
            row['3_strategy'] = "N/A"
            row['diff_2_vs_3'] = "N/A"
        
        if lastturn_ndcg10 is not None:
            row['lastturn'] = f"{lastturn_ndcg10:.4f}"
        else:
            row['lastturn'] = "N/A"
            
        if rewrite_ndcg10 is not None:
            row['rewrite'] = f"{rewrite_ndcg10:.4f}"
        else:
            row['rewrite'] = "N/A"
        
        if lastturn_ndcg10 is not None and rewrite_ndcg10 is not None:
            row['diff_2_vs_best'] = f"{two_ndcg10 - max(lastturn_ndcg10, rewrite_ndcg10):+.4f}"
        else:
            row['diff_2_vs_best'] = "N/A"
        
        summary_data.append(row)
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))
        
        summary_file = COMPARISON_DIR / "summary_ndcg10_comparison.csv"
        summary_df.to_csv(summary_file, index=False)
        print(f"\nSummary saved to {summary_file}")
    else:
        print("No summary data available.")


def evaluate_baseline_strategy(domain: str, strategy: str, scorer: MonoT5Scorer):
    """Evaluate a single strategy baseline with MonoT5 reranking."""
    print(f"\nEvaluating baseline: {strategy}")
    
    queries_rewrite = load_queries(domain, 'rewrite')
    results = load_retrieval_results(domain, strategy)
    
    reranked_results = {}
    for task_id in tqdm(queries_rewrite.keys(), desc=f"Reranking {strategy}"):
        if task_id not in results:
            continue
        
        documents = results[task_id]
        query = queries_rewrite[task_id]
        
        # Rerank
        reranked = rerank_with_monot5(scorer, query, documents, top_k=100)
        
        # Convert to results format
        doc_scores = {doc['document_id']: score for doc, score in reranked}
        if doc_scores:
            reranked_results[task_id] = doc_scores
    
    # Save results
    output_file = INTERMEDIATE_DIR / f"baseline_{strategy}_{domain}.jsonl"
    evaluated_file = INTERMEDIATE_DIR / f"baseline_{strategy}_{domain}_evaluated.jsonl"
    
    save_results_for_evaluation(reranked_results, domain, output_file)
    
    # Run evaluation
    cmd = [
        "python", str(EVAL_SCRIPT_PATH),
        "--input_file", str(output_file),
        "--output_file", str(evaluated_file)
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print(f"Error evaluating {strategy}: {e}")


def main():
    print("="*80)
    print("Combined Retrieval (2 Strategies: lastturn + rewrite) + MonoT5 Reranking")
    print("="*80)
    
    # Ensure directories exist
    INTERMEDIATE_DIR.mkdir(exist_ok=True)
    COMPARISON_DIR.mkdir(exist_ok=True)
    
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
        
        # Load retrieval results for two strategies
        results_by_strategy = {}
        for strategy in STRATEGIES:
            results = load_retrieval_results(domain, strategy)
            results_by_strategy[strategy] = results
            print(f"  {strategy}: {len(results)} tasks")
        
        # Combine and deduplicate
        print(f"\nCombining and deduplicating results from {len(STRATEGIES)} strategies...")
        combined_docs = combine_and_deduplicate_two_strategies(results_by_strategy)
        
        # Filter to tasks that are in queries_rewrite
        tasks_to_process = [tid for tid in combined_docs.keys() if tid in queries_rewrite]
        print(f"Combined documents for {len(tasks_to_process)} tasks")
        
        # Rerank with MonoT5
        print("\nReranking with MonoT5...")
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
        
        # Save results for standard evaluation
        output_file = INTERMEDIATE_DIR / f"reranked_two_strategies_{domain}.jsonl"
        evaluated_file = INTERMEDIATE_DIR / f"reranked_two_strategies_{domain}_evaluated.jsonl"
        
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
        except subprocess.CalledProcessError as e:
            print(f"Error running evaluation: {e}")
        
        # Also evaluate individual baselines for comparison
        print("\nEvaluating individual baselines for comparison...")
        for strategy in STRATEGIES:
            evaluate_baseline_strategy(domain, strategy, scorer)
    
    # Compare results
    compare_results()


if __name__ == '__main__':
    main()
