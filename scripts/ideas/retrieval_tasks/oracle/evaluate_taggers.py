#!/usr/bin/env python3
"""
Evaluate Context Dependency Taggers

This script evaluates the rule-based and LLM-based taggers against the oracle
data (which strategy actually performs best for each query).

Metrics:
- Routing accuracy: Does the predicted strategy match the oracle best?
- Expected R@5: What R@5 would we achieve with this routing?
- Oracle gap captured: What % of the improvement from baseline to oracle?

Usage:
    python evaluate_taggers.py
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np


def load_oracle_data(oracle_file: Path) -> pd.DataFrame:
    """Load the oracle best strategy data."""
    df = pd.read_csv(oracle_file)
    return df


def load_tagger_results_from_files(tagged_queries_dir: Path, tag_key: str) -> Dict[str, dict]:
    """
    Load tagger results from individual JSON files.
    
    Args:
        tagged_queries_dir: Path to tagged_queries directory
        tag_key: Key in oracle_metadata (e.g., 'rule_based_tags' or 'llm_based_tags')
        
    Returns:
        Dict mapping task_id to tagger output
    """
    results = {}
    domains = ['clapnq', 'cloud', 'fiqa', 'govt']
    
    for domain in domains:
        domain_dir = tagged_queries_dir / domain
        if not domain_dir.exists():
            continue
        
        for json_file in domain_dir.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                task_id = data.get('task_id')
                if not task_id:
                    continue
                
                # Get tags from oracle_metadata
                oracle_meta = data.get('oracle_metadata', {})
                tags = oracle_meta.get(tag_key, {})
                
                if tags:
                    results[task_id] = tags
                    
            except Exception as e:
                print(f"Error loading {json_file.name}: {e}")
    
    return results


def evaluate_tagger(
    tagger_results: Dict[str, dict],
    oracle_df: pd.DataFrame,
    tagger_name: str,
) -> dict:
    """
    Evaluate a tagger against oracle data.
    
    Args:
        tagger_results: Dict mapping task_id to tagger output
        oracle_df: DataFrame with oracle best strategy and R@5 scores
        tagger_name: Name of the tagger for reporting
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Match tagger predictions to oracle data
    matched_results = []
    missing = 0
    
    for _, row in oracle_df.iterrows():
        task_id = row['task_id']
        
        if task_id not in tagger_results:
            missing += 1
            continue
        
        tagger_pred = tagger_results[task_id]
        predicted_strategy = tagger_pred.get('recommended_strategy', 'rewrite')
        # Normalize strategy names
        if predicted_strategy in ['standalone', 'self-contained']:
            predicted_strategy = 'lastturn'
        
        # Get R@5 for predicted strategy
        strategy_col = f"{predicted_strategy}_R@5"
        if strategy_col not in row.index:
            # Fallback to rewrite if strategy column doesn't exist
            strategy_col = "rewrite_R@5"
        
        predicted_r5 = row[strategy_col]
        oracle_r5 = row['best_R@5']
        oracle_strategy = row['best_strategy']
        rewrite_r5 = row['rewrite_R@5']
        
        matched_results.append({
            'task_id': task_id,
            'predicted_strategy': predicted_strategy,
            'oracle_strategy': oracle_strategy,
            'predicted_r5': predicted_r5,
            'oracle_r5': oracle_r5,
            'rewrite_r5': rewrite_r5,
            'correct': predicted_strategy == oracle_strategy,
            'needs_context': tagger_pred.get('needs_context', True),
            'confidence': tagger_pred.get('confidence', 0.5),
        })
    
    if not matched_results:
        return {'error': 'No matched results', 'tagger_name': tagger_name}
    
    results_df = pd.DataFrame(matched_results)
    
    # Calculate metrics
    n_total = len(results_df)
    n_correct = results_df['correct'].sum()
    accuracy = n_correct / n_total
    
    mean_predicted_r5 = results_df['predicted_r5'].mean()
    mean_oracle_r5 = results_df['oracle_r5'].mean()
    mean_rewrite_r5 = results_df['rewrite_r5'].mean()
    
    # Oracle gap captured
    oracle_gap = mean_oracle_r5 - mean_rewrite_r5
    predicted_gap = mean_predicted_r5 - mean_rewrite_r5
    gap_captured = (predicted_gap / oracle_gap * 100) if oracle_gap > 0 else 0
    
    # Strategy distribution
    strategy_counts = results_df['predicted_strategy'].value_counts().to_dict()
    
    # Breakdown by correctness
    correct_by_strategy = results_df.groupby('oracle_strategy')['correct'].mean().to_dict()
    
    # Confidence analysis (if available)
    mean_confidence = results_df['confidence'].mean()
    
    return {
        'tagger_name': tagger_name,
        'n_total': n_total,
        'n_missing': missing,
        'accuracy': accuracy,
        'n_correct': int(n_correct),
        'mean_predicted_r5': mean_predicted_r5,
        'mean_oracle_r5': mean_oracle_r5,
        'mean_rewrite_r5': mean_rewrite_r5,
        'improvement_vs_rewrite': mean_predicted_r5 - mean_rewrite_r5,
        'improvement_pct': (mean_predicted_r5 - mean_rewrite_r5) / mean_rewrite_r5 * 100,
        'oracle_gap': oracle_gap,
        'gap_captured': gap_captured,
        'strategy_distribution': strategy_counts,
        'accuracy_by_oracle_strategy': correct_by_strategy,
        'mean_confidence': mean_confidence,
    }


def print_evaluation_report(metrics: dict):
    """Print a formatted evaluation report."""
    if 'error' in metrics:
        print(f"\n{metrics['tagger_name']}: {metrics['error']}")
        return
    
    print(f"\n{'=' * 60}")
    print(f"EVALUATION: {metrics['tagger_name']}")
    print(f"{'=' * 60}")
    
    print(f"\n### Coverage ###")
    print(f"  Queries evaluated: {metrics['n_total']}")
    print(f"  Missing from tagger: {metrics['n_missing']}")
    
    print(f"\n### Routing Accuracy ###")
    print(f"  Accuracy (match oracle): {metrics['accuracy']:.1%} ({metrics['n_correct']}/{metrics['n_total']})")
    
    print(f"\n### Retrieval Performance (R@5) ###")
    print(f"  Baseline (always rewrite): {metrics['mean_rewrite_r5']:.4f}")
    print(f"  This tagger:               {metrics['mean_predicted_r5']:.4f}")
    print(f"  Oracle ceiling:            {metrics['mean_oracle_r5']:.4f}")
    
    print(f"\n### Improvement ###")
    improvement = metrics['improvement_vs_rewrite']
    improvement_pct = metrics['improvement_pct']
    gap = metrics['gap_captured']
    
    if improvement >= 0:
        print(f"  vs Rewrite baseline: +{improvement:.4f} (+{improvement_pct:.1f}%)")
    else:
        print(f"  vs Rewrite baseline: {improvement:.4f} ({improvement_pct:.1f}%)")
    
    print(f"  Oracle gap captured: {gap:.1f}%")
    
    print(f"\n### Strategy Distribution ###")
    for strategy, count in sorted(metrics['strategy_distribution'].items()):
        pct = count / metrics['n_total'] * 100
        print(f"  {strategy}: {count} ({pct:.1f}%)")
    
    print(f"\n### Accuracy by Oracle Best Strategy ###")
    for strategy, acc in sorted(metrics['accuracy_by_oracle_strategy'].items()):
        print(f"  When {strategy} is best: {acc:.1%}")
    
    print(f"\n### Confidence ###")
    print(f"  Mean confidence: {metrics['mean_confidence']:.3f}")


def compare_taggers(all_metrics: List[dict]):
    """Print a comparison table of all taggers."""
    # Filter out errors
    valid_metrics = [m for m in all_metrics if 'error' not in m]
    
    if not valid_metrics:
        print("\nNo valid tagger results to compare.")
        return
    
    print(f"\n{'=' * 80}")
    print("COMPARISON SUMMARY")
    print(f"{'=' * 80}")
    
    # Build comparison table
    headers = ['Tagger', 'Accuracy', 'R@5', 'vs Baseline', 'Gap Captured']
    rows = []
    
    for m in valid_metrics:
        rows.append([
            m['tagger_name'],
            f"{m['accuracy']:.1%}",
            f"{m['mean_predicted_r5']:.4f}",
            f"{m['improvement_vs_rewrite']:+.4f} ({m['improvement_pct']:+.1f}%)",
            f"{m['gap_captured']:.1f}%",
        ])
    
    # Add baseline and oracle rows
    if valid_metrics:
        baseline_r5 = valid_metrics[0]['mean_rewrite_r5']
        oracle_r5 = valid_metrics[0]['mean_oracle_r5']
        rows.insert(0, ['Baseline (rewrite)', 'N/A', f"{baseline_r5:.4f}", '+0.0000 (+0.0%)', '0.0%'])
        rows.append(['Oracle (ceiling)', '100.0%', f"{oracle_r5:.4f}", 
                    f"+{oracle_r5 - baseline_r5:.4f} (+{(oracle_r5 - baseline_r5)/baseline_r5*100:.1f}%)", '100.0%'])
    
    # Print table
    col_widths = [max(len(str(row[i])) for row in rows + [headers]) for i in range(len(headers))]
    
    header_line = ' | '.join(h.ljust(w) for h, w in zip(headers, col_widths))
    print(f"\n{header_line}")
    print('-' * len(header_line))
    
    for row in rows:
        row_line = ' | '.join(str(c).ljust(w) for c, w in zip(row, col_widths))
        print(row_line)


def save_detailed_results(
    all_metrics: List[dict],
    output_file: Path,
):
    """Save detailed evaluation results to JSON."""
    with open(output_file, 'w') as f:
        json.dump(all_metrics, f, indent=2, default=str)
    print(f"\nDetailed results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate context dependency taggers")
    parser.add_argument("--oracle-data", type=str, 
                        default="../routing_analysis_results/oracle_best_strategy.csv",
                        help="Path to oracle best strategy CSV")
    parser.add_argument("--output", type=str, default="evaluation_results.json",
                        help="Output file for detailed results")
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent
    tagged_queries_dir = script_dir / "tagged_queries"
    
    # Resolve paths
    oracle_file = script_dir / args.oracle_data
    output_file = script_dir / args.output
    
    print("=" * 60)
    print("Context Dependency Tagger Evaluation")
    print("=" * 60)
    print(f"Tagged queries dir: {tagged_queries_dir}")
    
    # Load oracle data
    if not oracle_file.exists():
        print(f"Error: Oracle data not found at {oracle_file}")
        print("Please run the routing analysis first to generate oracle_best_strategy.csv")
        return
    
    oracle_df = load_oracle_data(oracle_file)
    print(f"Loaded oracle data: {len(oracle_df)} queries")
    
    all_metrics = []
    
    # Check if tagged_queries exists
    if not tagged_queries_dir.exists():
        print(f"\nError: Tagged queries directory not found at {tagged_queries_dir}")
        print("Please run the taggers first:")
        print("  python rule_based_tagger.py")
        print("  python llm_based_tagger.py")
        return
    
    # Evaluate rule-based tagger
    print(f"\nLoading rule-based tags from {tagged_queries_dir}")
    rule_results = load_tagger_results_from_files(tagged_queries_dir, 'rule_based_tags')
    
    if rule_results:
        print(f"  Found {len(rule_results)} tagged queries with rule_based_tags")
        rule_metrics = evaluate_tagger(rule_results, oracle_df, "Rule-Based")
        all_metrics.append(rule_metrics)
        print_evaluation_report(rule_metrics)
    else:
        print("  No rule_based_tags found. Run: python rule_based_tagger.py")
    
    # Evaluate LLM-based tagger
    print(f"\nLoading LLM-based tags from {tagged_queries_dir}")
    llm_results = load_tagger_results_from_files(tagged_queries_dir, 'llm_based_tags')
    
    if llm_results:
        print(f"  Found {len(llm_results)} tagged queries with llm_based_tags")
        llm_metrics = evaluate_tagger(llm_results, oracle_df, "LLM-Based (Sonnet 4.5)")
        all_metrics.append(llm_metrics)
        print_evaluation_report(llm_metrics)
    else:
        print("  No llm_based_tags found. Run: python llm_based_tagger.py")
    
    # Compare all taggers
    if len(all_metrics) > 0:
        compare_taggers(all_metrics)
        save_detailed_results(all_metrics, output_file)


if __name__ == "__main__":
    main()
