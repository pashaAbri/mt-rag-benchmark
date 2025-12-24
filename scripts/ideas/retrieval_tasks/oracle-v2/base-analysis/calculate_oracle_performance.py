#!/usr/bin/env python3
"""
Calculate Oracle Performance for Baseline Retrieval Strategies

This script calculates the oracle performance if we could perfectly select
the best retrieval strategy (lastturn, rewrite, or questions) for each turn.

The oracle picks the strategy that maximizes a primary metric (nDCG@10) for each task,
then reports the aggregate performance across all metrics.

Usage:
    python calculate_oracle_performance.py
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import statistics


# Strategy names
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


def find_oracle_choice(
    lastturn_scores: Dict[str, float],
    rewrite_scores: Dict[str, float],
    questions_scores: Dict[str, float],
    metric: str
) -> Tuple[Optional[str], Optional[float]]:
    """
    Find the best strategy for a given task based on a metric.
    
    Args:
        task_id: Task identifier
        lastturn_scores: Metrics for lastturn strategy
        rewrite_scores: Metrics for rewrite strategy
        questions_scores: Metrics for questions strategy
        metric: Metric name to optimize
        
    Returns:
        Tuple of (best_strategy, best_score) or (None, None) if no data
    """
    scores = {}
    
    if lastturn_scores and metric in lastturn_scores:
        scores['lastturn'] = lastturn_scores[metric]
    if rewrite_scores and metric in rewrite_scores:
        scores['rewrite'] = rewrite_scores[metric]
    if questions_scores and metric in questions_scores:
        scores['questions'] = questions_scores[metric]
    
    if not scores:
        return None, None
    
    best_strategy = max(scores.items(), key=lambda x: x[1])
    return best_strategy[0], best_strategy[1]


def get_chosen_scores(
    best_strategy: str,
    lastturn_scores: Dict[str, float],
    rewrite_scores: Dict[str, float],
    questions_scores: Dict[str, float]
) -> Optional[Dict[str, float]]:
    """Get scores for the chosen strategy."""
    if best_strategy == 'lastturn':
        return lastturn_scores
    elif best_strategy == 'rewrite':
        return rewrite_scores
    elif best_strategy == 'questions':
        return questions_scores
    return None


def process_task_for_oracle(
    lastturn_scores: Dict[str, float],
    rewrite_scores: Dict[str, float],
    questions_scores: Dict[str, float],
    metric: str,
    oracle_choices: defaultdict,
    oracle_scores_by_metric: defaultdict,
    per_task_scores: defaultdict
):
    """Process a single task for oracle calculation."""
    best_strategy, _ = find_oracle_choice(
        lastturn_scores, rewrite_scores, questions_scores, metric
    )
    
    if best_strategy is None:
        return
    
    oracle_choices[best_strategy] += 1
    
    chosen_scores = get_chosen_scores(
        best_strategy, lastturn_scores, rewrite_scores, questions_scores
    )
    
    if chosen_scores:
        for m in METRICS.keys():
            if m in chosen_scores:
                oracle_scores_by_metric[m].append(chosen_scores[m])
                per_task_scores[best_strategy][m].append(chosen_scores[m])


def calculate_oracle_performance(
    lastturn_results: Dict[str, Dict[str, float]],
    rewrite_results: Dict[str, Dict[str, float]],
    questions_results: Dict[str, Dict[str, float]],
    metric: str
) -> Tuple[Dict[str, float], Dict[str, int], Dict[str, List[float]]]:
    """
    Calculate oracle performance by always picking the best strategy for each task.
    
    Args:
        lastturn_results: Results for lastturn strategy
        rewrite_results: Results for rewrite strategy
        questions_results: Results for questions strategy
        metric: Primary metric to optimize
        
    Returns:
        Tuple of:
        - oracle_scores: Dictionary mapping metric to oracle score
        - oracle_choices: Dictionary mapping strategy to count of times chosen
        - per_task_scores: Dictionary mapping strategy to list of scores for tasks where it was chosen
    """
    # Get all task_ids that appear in at least one strategy
    all_task_ids = set()
    all_task_ids.update(lastturn_results.keys())
    all_task_ids.update(rewrite_results.keys())
    all_task_ids.update(questions_results.keys())
    
    oracle_choices = defaultdict(int)
    oracle_scores_by_metric = defaultdict(list)
    per_task_scores = defaultdict(lambda: defaultdict(list))
    
    for task_id in all_task_ids:
        lastturn_scores = lastturn_results.get(task_id, {})
        rewrite_scores = rewrite_results.get(task_id, {})
        questions_scores = questions_results.get(task_id, {})
        
        process_task_for_oracle(
            lastturn_scores, rewrite_scores, questions_scores,
            metric, oracle_choices, oracle_scores_by_metric, per_task_scores
        )
    
    # Calculate averages
    oracle_avg_scores = {}
    for m, scores in oracle_scores_by_metric.items():
        if scores:
            oracle_avg_scores[m] = statistics.mean(scores)
        else:
            oracle_avg_scores[m] = 0.0
    
    return oracle_avg_scores, dict(oracle_choices), dict(per_task_scores)


def calculate_individual_performance(results: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """
    Calculate average performance for a single strategy.
    
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


def print_performance_table(
    lastturn_perf: Dict[str, float],
    rewrite_perf: Dict[str, float],
    questions_perf: Dict[str, float],
    oracle_perf: Dict[str, float],
    oracle_choices: Dict[str, int]
):
    """Print a formatted performance comparison table."""
    print("\n" + "=" * 100)
    print("PERFORMANCE COMPARISON")
    print("=" * 100)
    
    # Header
    header = f"{'Metric':<12} {'Last Turn':<12} {'Rewrite':<12} {'Questions':<12} {'Oracle':<12} {'Best Static':<12}"
    print(header)
    print("-" * len(header))
    
    # Find best static strategy (based on nDCG@10)
    static_scores = {
        'lastturn': lastturn_perf.get('ndcg_cut_10', 0.0),
        'rewrite': rewrite_perf.get('ndcg_cut_10', 0.0),
        'questions': questions_perf.get('ndcg_cut_10', 0.0),
    }
    best_static = max(static_scores.items(), key=lambda x: x[1])[0]
    
    # Print each metric
    for metric_key, metric_name in METRICS.items():
        lastturn_val = lastturn_perf.get(metric_key, 0.0)
        rewrite_val = rewrite_perf.get(metric_key, 0.0)
        questions_val = questions_perf.get(metric_key, 0.0)
        oracle_val = oracle_perf.get(metric_key, 0.0)
        
        # Get best static value
        if best_static == 'lastturn':
            best_static_val = lastturn_val
        elif best_static == 'rewrite':
            best_static_val = rewrite_val
        else:
            best_static_val = questions_val
        
        print(f"{metric_name:<12} {lastturn_val:<12.4f} {rewrite_val:<12.4f} {questions_val:<12.4f} "
              f"{oracle_val:<12.4f} {best_static_val:<12.4f}")
    
    # Summary
    print("\n" + "-" * 100)
    print(f"Best Static Strategy: {best_static.capitalize()}")
    print("Oracle Strategy Distribution:")
    total_choices = sum(oracle_choices.values())
    for strategy, count in sorted(oracle_choices.items()):
        pct = (count / total_choices * 100) if total_choices > 0 else 0.0
        print(f"  {strategy.capitalize()}: {count} ({pct:.1f}%)")
    
    # Calculate improvements
    print("\n" + "-" * 100)
    print("ORACLE IMPROVEMENTS OVER BEST STATIC STRATEGY")
    print("-" * 100)
    
    best_static_perf = {
        'lastturn': lastturn_perf,
        'rewrite': rewrite_perf,
        'questions': questions_perf,
    }[best_static]
    
    print(f"{'Metric':<12} {'Best Static':<12} {'Oracle':<12} {'Absolute':<12} {'Relative %':<12}")
    print("-" * 60)
    
    for metric_key, metric_name in METRICS.items():
        static_val = best_static_perf.get(metric_key, 0.0)
        oracle_val = oracle_perf.get(metric_key, 0.0)
        absolute_improvement = oracle_val - static_val
        relative_improvement = (absolute_improvement / static_val * 100) if static_val > 0 else 0.0
        
        print(f"{metric_name:<12} {static_val:<12.4f} {oracle_val:<12.4f} "
              f"{absolute_improvement:<+12.4f} {relative_improvement:<+12.2f}%")


def save_results(
    lastturn_perf: Dict[str, float],
    rewrite_perf: Dict[str, float],
    questions_perf: Dict[str, float],
    oracle_perf: Dict[str, float],
    oracle_choices: Dict[str, int],
    output_file: Path
):
    """Save results to JSON file."""
    results = {
        'individual_strategies': {
            'lastturn': lastturn_perf,
            'rewrite': rewrite_perf,
            'questions': questions_perf,
        },
        'oracle': {
            'performance': oracle_perf,
            'strategy_distribution': oracle_choices,
            'total_tasks': sum(oracle_choices.values()),
        }
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Calculate oracle performance for baseline retrieval strategies"
    )
    parser.add_argument(
        "--retriever",
        type=str,
        choices=['elser', 'bm25', 'bge'],
        default='elser',
        help="Retriever to analyze (default: elser)"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Directory containing evaluated results (relative to project root). If not specified, uses default based on retriever."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="oracle_performance_results.json",
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
    if args.results_dir:
        if Path(args.results_dir).is_absolute():
            results_dir = Path(args.results_dir)
        else:
            # Resolve relative paths from project root
            results_dir = (project_root / args.results_dir).resolve()
    else:
        # Use default based on retriever
        results_dir = (project_root / f"scripts/baselines/retrieval_scripts/{args.retriever}/results").resolve()
    
    output_file = script_dir / args.output
    
    print("=" * 100)
    print("ORACLE PERFORMANCE CALCULATION")
    print("=" * 100)
    print(f"Retriever: {args.retriever.upper()}")
    print(f"Results directory: {results_dir}")
    print(f"Output file: {output_file}")
    print(f"Domain: {args.domain}")
    
    # Determine which files to load
    retriever_prefix = args.retriever
    if args.domain == 'all':
        file_patterns = {
            'lastturn': f'{retriever_prefix}_all_lastturn_evaluated.jsonl',
            'rewrite': f'{retriever_prefix}_all_rewrite_evaluated.jsonl',
            'questions': f'{retriever_prefix}_all_questions_evaluated.jsonl',
        }
    else:
        file_patterns = {
            'lastturn': f'{retriever_prefix}_{args.domain}_lastturn_evaluated.jsonl',
            'rewrite': f'{retriever_prefix}_{args.domain}_rewrite_evaluated.jsonl',
            'questions': f'{retriever_prefix}_{args.domain}_questions_evaluated.jsonl',
        }
    
    # Load results
    print("\nLoading evaluated results...")
    lastturn_results = load_evaluated_results(results_dir / file_patterns['lastturn'])
    print(f"  Last Turn: {len(lastturn_results)} tasks")
    
    rewrite_results = load_evaluated_results(results_dir / file_patterns['rewrite'])
    print(f"  Rewrite: {len(rewrite_results)} tasks")
    
    questions_results = load_evaluated_results(results_dir / file_patterns['questions'])
    print(f"  Questions: {len(questions_results)} tasks")
    
    if not (lastturn_results or rewrite_results or questions_results):
        print("\nError: No results found. Please check the results directory path.")
        return
    
    # Calculate individual performances
    print("\nCalculating individual strategy performances...")
    lastturn_perf = calculate_individual_performance(lastturn_results)
    rewrite_perf = calculate_individual_performance(rewrite_results)
    questions_perf = calculate_individual_performance(questions_results)
    
    # Calculate oracle performance
    print("Calculating oracle performance...")
    oracle_perf, oracle_choices, _ = calculate_oracle_performance(
        lastturn_results, rewrite_results, questions_results, PRIMARY_METRIC
    )
    
    print("  Oracle selected strategies:")
    total = sum(oracle_choices.values())
    for strategy, count in sorted(oracle_choices.items()):
        pct = (count / total * 100) if total > 0 else 0.0
        print(f"    {strategy.capitalize()}: {count} ({pct:.1f}%)")
    
    # Print performance table
    print_performance_table(
        lastturn_perf, rewrite_perf, questions_perf, oracle_perf, oracle_choices
    )
    
    # Save results
    save_results(
        lastturn_perf, rewrite_perf, questions_perf, oracle_perf, oracle_choices, output_file
    )


if __name__ == "__main__":
    main()

