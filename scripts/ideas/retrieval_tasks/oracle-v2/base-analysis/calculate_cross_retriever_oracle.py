#!/usr/bin/env python3
"""
Calculate Cross-Retriever Oracle Performance

This script calculates the oracle performance if we could perfectly select
the best retriever+strategy combination (9 total: 3 retrievers × 3 strategies)
for each turn.

The oracle picks the combination that maximizes a primary metric (nDCG@10) for each task,
then reports the aggregate performance across all metrics.

Usage:
    python calculate_cross_retriever_oracle.py
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import statistics


# Retrievers and strategies
RETRIEVERS = ['bm25', 'elser', 'bge']
STRATEGIES = ['lastturn', 'rewrite', 'questions']

# Metrics to track
METRICS = {
    'recall_1': 'R@1',
    'recall_3': 'R@3',
    'recall_5': 'R@5',
    'recall_10': 'R@10',
    'ndcg_cut_1': 'nDCG@1',
    'ndcg_cut_3': 'nDCG@3',
    'ndcg_cut_5': 'nDCG@5',
    'ndcg_cut_10': 'nDCG@10',
}

# Primary metric for oracle selection
PRIMARY_METRIC = 'ndcg_cut_10'


def load_evaluated_results(filepath: Path) -> Dict[str, Dict[str, float]]:
    """
    Load evaluated results from a JSONL file.
    
    Args:
        filepath: Path to the evaluated JSONL file
        
    Returns:
        Dictionary mapping task_id to metrics dictionary
    """
    results = {}
    
    if not filepath.exists():
        print(f"  Warning: File not found: {filepath}")
        return results
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                task_id = data.get('task_id')
                
                if not task_id:
                    continue
                
                # Extract retriever scores
                retriever_scores = data.get('retriever_scores', {})
                
                if retriever_scores:
                    results[task_id] = retriever_scores
                    
            except json.JSONDecodeError as e:
                print(f"  Warning: Failed to parse line {line_num} in {filepath.name}: {e}")
                continue
            except Exception as e:
                print(f"  Warning: Error processing line {line_num} in {filepath.name}: {e}")
                continue
    
    return results


def find_best_combination(
    task_id: str,
    all_results: Dict[str, Dict[str, Dict[str, float]]],
    metric: str
) -> Tuple[Optional[str], Optional[float]]:
    """
    Find the best retriever+strategy combination for a given task.
    
    Args:
        task_id: Task identifier
        all_results: Dict mapping retriever to strategy to task_id to scores
        metric: Metric name to optimize
        
    Returns:
        Tuple of (best_combination, best_score) where combination is "retriever_strategy"
    """
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


def get_chosen_scores(
    combination: str,
    all_results: Dict[str, Dict[str, Dict[str, float]]],
    task_id: str
) -> Optional[Dict[str, float]]:
    """Get scores for the chosen retriever+strategy combination."""
    parts = combination.split('_', 1)
    if len(parts) != 2:
        return None
    
    retriever, strategy = parts
    if retriever in all_results and strategy in all_results[retriever]:
        return all_results[retriever][strategy].get(task_id, {})
    return None


def process_task_for_oracle(
    task_id: str,
    all_results: Dict[str, Dict[str, Dict[str, float]]],
    metric: str,
    oracle_choices: defaultdict,
    oracle_scores_by_metric: defaultdict,
    per_combination_scores: defaultdict
):
    """Process a single task for oracle calculation."""
    best_combination, _ = find_best_combination(task_id, all_results, metric)
    
    if best_combination is None:
        return
    
    oracle_choices[best_combination] += 1
    
    chosen_scores = get_chosen_scores(best_combination, all_results, task_id)
    
    if chosen_scores:
        for m in METRICS.keys():
            if m in chosen_scores:
                oracle_scores_by_metric[m].append(chosen_scores[m])
                per_combination_scores[best_combination][m].append(chosen_scores[m])


def calculate_cross_retriever_oracle(
    all_results: Dict[str, Dict[str, Dict[str, float]]],
    metric: str
) -> Tuple[Dict[str, float], Dict[str, int], Dict[str, List[float]]]:
    """
    Calculate oracle performance by always picking the best retriever+strategy combination.
    
    Args:
        all_results: Dict mapping retriever to strategy to task_id to scores
        metric: Primary metric to optimize
        
    Returns:
        Tuple of:
        - oracle_scores: Dictionary mapping metric to oracle score
        - oracle_choices: Dictionary mapping combination to count of times chosen
        - per_combination_scores: Dictionary mapping combination to list of scores
    """
    # Get all task_ids that appear in any combination
    all_task_ids = set()
    for retriever in RETRIEVERS:
        for strategy in STRATEGIES:
            if retriever in all_results and strategy in all_results[retriever]:
                all_task_ids.update(all_results[retriever][strategy].keys())
    
    oracle_choices = defaultdict(int)
    oracle_scores_by_metric = defaultdict(list)
    per_combination_scores = defaultdict(lambda: defaultdict(list))
    
    for task_id in all_task_ids:
        process_task_for_oracle(
            task_id, all_results, metric,
            oracle_choices, oracle_scores_by_metric, per_combination_scores
        )
    
    # Calculate averages
    oracle_avg_scores = {}
    for m, scores in oracle_scores_by_metric.items():
        if scores:
            oracle_avg_scores[m] = statistics.mean(scores)
        else:
            oracle_avg_scores[m] = 0.0
    
    return oracle_avg_scores, dict(oracle_choices), dict(per_combination_scores)


def calculate_individual_performance(results: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """
    Calculate average performance for a single retriever+strategy combination.
    
    Args:
        results: Dictionary mapping task_id to metrics
        
    Returns:
        Dictionary mapping metric to average score
    """
    metric_scores = defaultdict(list)
    
    for task_id, scores in results.items():
        for metric in METRICS.keys():
            if metric in scores:
                metric_scores[metric].append(scores[metric])
    
    avg_scores = {}
    for metric, scores in metric_scores.items():
        if scores:
            avg_scores[metric] = statistics.mean(scores)
        else:
            avg_scores[metric] = 0.0
    
    return avg_scores


def find_best_static_combination(individual_perfs: Dict[str, Dict[str, float]]) -> Optional[str]:
    """Find the best static combination based on nDCG@10."""
    static_scores = {}
    for combination, perf in individual_perfs.items():
        static_scores[combination] = perf.get('ndcg_cut_10', 0.0)
    
    if not static_scores:
        return None
    return max(static_scores.items(), key=lambda x: x[1])[0]


def print_improvements_table(
    best_static_perf: Dict[str, float],
    oracle_perf: Dict[str, float]
):
    """Print improvements table."""
    print("\n" + "-" * 120)
    print("ORACLE IMPROVEMENTS OVER BEST STATIC COMBINATION")
    print("-" * 120)
    
    print(f"{'Metric':<12} {'Best Static':<12} {'Oracle':<12} {'Absolute':<12} {'Relative %':<12}")
    print("-" * 60)
    
    for metric_key, metric_name in METRICS.items():
        static_val = best_static_perf.get(metric_key, 0.0)
        oracle_val = oracle_perf.get(metric_key, 0.0)
        absolute_improvement = oracle_val - static_val
        relative_improvement = (absolute_improvement / static_val * 100) if static_val > 0 else 0.0
        
        print(f"{metric_name:<12} {static_val:<12.4f} {oracle_val:<12.4f} "
              f"{absolute_improvement:<+12.4f} {relative_improvement:<+12.2f}%")


def print_performance_table(
    individual_perfs: Dict[str, Dict[str, float]],
    oracle_perf: Dict[str, float],
    oracle_choices: Dict[str, int]
):
    """Print a formatted performance comparison table."""
    print("\n" + "=" * 120)
    print("CROSS-RETRIEVER ORACLE PERFORMANCE COMPARISON")
    print("=" * 120)
    
    # Find best static combination
    best_static = find_best_static_combination(individual_perfs)
    
    # Header
    header = f"{'Metric':<12} {'Oracle':<12} {'Best Static':<15}"
    if best_static:
        header += f" {'Best Static Val':<15}"
    print(header)
    print("-" * len(header))
    
    # Print each metric
    for metric_key, metric_name in METRICS.items():
        oracle_val = oracle_perf.get(metric_key, 0.0)
        best_static_val = individual_perfs[best_static].get(metric_key, 0.0) if best_static else 0.0
        
        print(f"{metric_name:<12} {oracle_val:<12.4f} {best_static:<15} {best_static_val:<15.4f}")
    
    # Summary
    print("\n" + "-" * 120)
    if best_static:
        print(f"Best Static Combination: {best_static}")
    
    print("\nOracle Combination Distribution:")
    total_choices = sum(oracle_choices.values())
    for combination, count in sorted(oracle_choices.items(), key=lambda x: x[1], reverse=True):
        pct = (count / total_choices * 100) if total_choices > 0 else 0.0
        retriever, strategy = combination.split('_', 1)
        print(f"  {retriever.upper():<8} {strategy.capitalize():<12} {count:>4} ({pct:>5.1f}%)")
    
    # Calculate improvements
    if best_static:
        best_static_perf = individual_perfs[best_static]
        print_improvements_table(best_static_perf, oracle_perf)


def save_results(
    individual_perfs: Dict[str, Dict[str, float]],
    oracle_perf: Dict[str, float],
    oracle_choices: Dict[str, int],
    output_file: Path
):
    """Save results to JSON file."""
    results = {
        'individual_combinations': individual_perfs,
        'oracle': {
            'performance': oracle_perf,
            'combination_distribution': oracle_choices,
            'total_tasks': sum(oracle_choices.values()),
        }
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Calculate cross-retriever oracle performance"
    )
    parser.add_argument(
        "--results-base-dir",
        type=str,
        default="scripts/baselines/retrieval_scripts",
        help="Base directory containing retriever results (relative to project root)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="cross_retriever_oracle_results.json",
        help="Output JSON file for results"
    )
    parser.add_argument(
        "--domain",
        type=str,
        choices=['all', 'clapnq', 'cloud', 'fiqa', 'govt'],
        default='all',
        help="Domain to analyze (default: all)"
    )
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent
    # Go up from analysis/oracle-v2/retrieval_tasks/ideas/scripts to project root (5 levels)
    project_root = script_dir.parents[4]
    
    # Resolve paths
    results_base_dir = (project_root / args.results_base_dir).resolve()
    output_file = script_dir / args.output
    
    print("=" * 120)
    print("CROSS-RETRIEVER ORACLE PERFORMANCE CALCULATION")
    print("=" * 120)
    print(f"Results base directory: {results_base_dir}")
    print(f"Output file: {output_file}")
    print(f"Domain: {args.domain}")
    
    # Determine file patterns
    if args.domain == 'all':
        file_pattern = '{retriever}_all_{strategy}_evaluated.jsonl'
    else:
        file_pattern = '{retriever}_{domain}_{strategy}_evaluated.jsonl'
    
    # Load all results
    print("\nLoading evaluated results...")
    all_results = {}
    individual_perfs = {}
    
    for retriever in RETRIEVERS:
        all_results[retriever] = {}
        retriever_dir = results_base_dir / retriever / "results"
        
        for strategy in STRATEGIES:
            filename = file_pattern.format(retriever=retriever, strategy=strategy, domain=args.domain)
            filepath = retriever_dir / filename
            
            results = load_evaluated_results(filepath)
            all_results[retriever][strategy] = results
            
            combination = f"{retriever}_{strategy}"
            if results:
                individual_perfs[combination] = calculate_individual_performance(results)
                print(f"  {retriever.upper():<8} {strategy.capitalize():<12} {len(results):>4} tasks")
            else:
                print(f"  {retriever.upper():<8} {strategy.capitalize():<12} {len(results):>4} tasks (no data)")
    
    if not any(all_results[r][s] for r in RETRIEVERS for s in STRATEGIES):
        print("\nError: No results found. Please check the results directory path.")
        return
    
    # Calculate oracle performance
    print("\nCalculating cross-retriever oracle performance...")
    oracle_perf, oracle_choices, _ = calculate_cross_retriever_oracle(
        all_results, PRIMARY_METRIC
    )
    
    print(f"  Oracle selected {len(oracle_choices)} different combinations")
    print(f"  Total tasks: {sum(oracle_choices.values())}")
    
    # Print performance table
    print_performance_table(individual_perfs, oracle_perf, oracle_choices)
    
    # Save results
    save_results(individual_perfs, oracle_perf, oracle_choices, output_file)


if __name__ == "__main__":
    main()

