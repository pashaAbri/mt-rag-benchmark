#!/usr/bin/env python3
"""
Extract per-task oracle selections from cross-retriever analysis.

This script calculates which retriever+strategy combination is optimal for each task
and saves the results for feature analysis.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Tuple, Optional
from collections import defaultdict


RETRIEVERS = ['bm25', 'elser', 'bge']
STRATEGIES = ['lastturn', 'rewrite', 'questions']
PRIMARY_METRIC = 'ndcg_cut_5'


def load_evaluated_results(filepath: Path) -> Dict[str, Dict[str, float]]:
    """Load evaluated results from a JSONL file."""
    results = {}
    
    if not filepath.exists():
        return results
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                task_id = data.get('task_id')
                retriever_scores = data.get('retriever_scores', {})
                
                if task_id and retriever_scores:
                    results[task_id] = retriever_scores
            except (json.JSONDecodeError, KeyError):
                continue
    
    return results


def find_best_combination(
    task_id: str,
    all_results: Dict[str, Dict[str, Dict[str, float]]],
    metric: str
) -> Tuple[Optional[str], Optional[float]]:
    """Find the best retriever+strategy combination for a given task."""
    scores = {}
    
    for retriever in RETRIEVERS:
        for strategy in STRATEGIES:
            if retriever in all_results and strategy in all_results[retriever]:
                task_scores = all_results[retriever][strategy].get(task_id, {})
                if metric in task_scores:
                    combination = f"{retriever}_{strategy}"
                    scores[combination] = task_scores[metric]
    
    if not scores:
        return None, None
    
    best_combination = max(scores.items(), key=lambda x: x[1])
    return best_combination[0], best_combination[1]


def main():
    parser = argparse.ArgumentParser(
        description="Extract per-task oracle selections"
    )
    parser.add_argument(
        "--results-base-dir",
        type=str,
        default="scripts/baselines/retrieval_scripts",
        help="Base directory containing retriever results"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="oracle_selections.json",
        help="Output JSON file"
    )
    parser.add_argument(
        "--domain",
        type=str,
        choices=['all', 'clapnq', 'cloud', 'fiqa', 'govt'],
        default='all',
        help="Domain to analyze"
    )
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent
    project_root = script_dir.parents[4]
    results_base_dir = (project_root / args.results_base_dir).resolve()
    output_file = script_dir / args.output
    
    print("=" * 80)
    print("EXTRACTING ORACLE SELECTIONS")
    print("=" * 80)
    print(f"Domain: {args.domain}")
    print(f"Output: {output_file}")
    
    # Determine file pattern
    if args.domain == 'all':
        file_pattern = '{retriever}_all_{strategy}_evaluated.jsonl'
    else:
        file_pattern = '{retriever}_{domain}_{strategy}_evaluated.jsonl'
    
    # Load all results
    print("\nLoading results...")
    all_results = {}
    
    for retriever in RETRIEVERS:
        all_results[retriever] = {}
        retriever_dir = results_base_dir / retriever / "results"
        
        for strategy in STRATEGIES:
            filename = file_pattern.format(retriever=retriever, strategy=strategy, domain=args.domain)
            filepath = retriever_dir / filename
            
            results = load_evaluated_results(filepath)
            all_results[retriever][strategy] = results
            print(f"  {retriever.upper():<8} {strategy.capitalize():<12} {len(results):>4} tasks")
    
    # Get all task_ids
    all_task_ids = set()
    for retriever in RETRIEVERS:
        for strategy in STRATEGIES:
            if retriever in all_results and strategy in all_results[retriever]:
                all_task_ids.update(all_results[retriever][strategy].keys())
    
    print(f"\nTotal unique tasks: {len(all_task_ids)}")
    
    # Calculate oracle selections
    print("Calculating oracle selections...")
    oracle_selections = {}
    combination_counts = defaultdict(int)
    
    for task_id in all_task_ids:
        best_combination, best_score = find_best_combination(
            task_id, all_results, PRIMARY_METRIC
        )
        
        if best_combination:
            oracle_selections[task_id] = {
                'combination': best_combination,
                'score': best_score,
                'retriever': best_combination.split('_')[0],
                'strategy': best_combination.split('_', 1)[1]
            }
            combination_counts[best_combination] += 1
    
    print(f"\nOracle selections calculated: {len(oracle_selections)} tasks")
    print("\nCombination distribution:")
    total = sum(combination_counts.values())
    for combo, count in sorted(combination_counts.items(), key=lambda x: x[1], reverse=True):
        pct = (count / total * 100) if total > 0 else 0.0
        retriever, strategy = combo.split('_', 1)
        print(f"  {retriever.upper():<8} {strategy.capitalize():<12} {count:>4} ({pct:>5.1f}%)")
    
    # Save results
    output_data = {
        'domain': args.domain,
        'total_tasks': len(oracle_selections),
        'oracle_selections': oracle_selections,
        'combination_distribution': dict(combination_counts)
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()

