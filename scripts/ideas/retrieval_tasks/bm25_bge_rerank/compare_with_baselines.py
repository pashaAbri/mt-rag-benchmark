#!/usr/bin/env python3
"""
Compare two-stage BM25+BGE results with baseline retrievers.

This script:
1. Loads evaluation results from the two-stage approach
2. Loads baseline results (BM25, BGE, ELSER)
3. Computes comparison metrics and generates summary tables

Usage:
    python compare_with_baselines.py
    python compare_with_baselines.py --query_type rewrite  # Focus on specific query type
"""

import argparse
import ast
from pathlib import Path
import pandas as pd

from utils import (
    DOMAINS,
    QUERY_TYPES,
    load_aggregate_metrics,
    BASELINE_RESULTS_DIR
)

# Paths
script_dir = Path(__file__).parent
RESULTS_DIR = script_dir / 'results'

BASELINES = ['bm25', 'bge', 'elser']


def load_two_stage_metrics(domain: str, query_type: str) -> dict:
    """Load metrics from two-stage experiment results."""
    agg_file = RESULTS_DIR / f"bm25_bge_rerank_{domain}_{query_type}_evaluated_aggregate.csv"
    return load_aggregate_metrics(agg_file)


def load_baseline_metrics(baseline: str, domain: str, query_type: str) -> dict:
    """Load metrics from baseline results."""
    agg_file = BASELINE_RESULTS_DIR / baseline / 'results' / f"{baseline}_{domain}_{query_type}_evaluated_aggregate.csv"
    return load_aggregate_metrics(agg_file)


def main():
    parser = argparse.ArgumentParser(description='Compare two-stage results with baselines')
    parser.add_argument(
        '--query_type',
        type=str,
        choices=QUERY_TYPES,
        default=None,
        help='Focus on specific query type (default: all)'
    )
    parser.add_argument(
        '--metric',
        type=str,
        choices=['nDCG@10', 'Recall@10', 'nDCG@5', 'Recall@5'],
        default='nDCG@10',
        help='Primary metric for comparison (default: nDCG@10)'
    )
    
    args = parser.parse_args()
    
    query_types = [args.query_type] if args.query_type else QUERY_TYPES
    primary_metric = args.metric
    
    print("=" * 80)
    print("Comparison: Two-Stage BM25+BGE vs Baselines")
    print("=" * 80)
    
    all_results = []
    
    for query_type in query_types:
        print(f"\n{'='*60}")
        print(f"Query Type: {query_type.upper()}")
        print(f"{'='*60}")
        
        comparison_data = []
        
        for domain in DOMAINS:
            row = {'domain': domain, 'query_type': query_type}
            
            # Load two-stage results
            two_stage_metrics = load_two_stage_metrics(domain, query_type)
            if two_stage_metrics:
                row['bm25_bge_rerank'] = two_stage_metrics.get(primary_metric, None)
            else:
                row['bm25_bge_rerank'] = None
            
            # Load baseline results
            for baseline in BASELINES:
                baseline_metrics = load_baseline_metrics(baseline, domain, query_type)
                if baseline_metrics:
                    row[baseline] = baseline_metrics.get(primary_metric, None)
                else:
                    row[baseline] = None
            
            comparison_data.append(row)
            all_results.append(row)
        
        # Create comparison table for this query type
        df = pd.DataFrame(comparison_data)
        
        # Calculate improvements
        def calc_improvement(row, baseline_name):
            """Calculate percentage improvement over baseline."""
            two_stage = row['bm25_bge_rerank']
            baseline_val = row[baseline_name]
            if pd.notna(two_stage) and pd.notna(baseline_val) and baseline_val > 0:
                return f"+{(two_stage - baseline_val) / baseline_val * 100:.1f}%"
            return "N/A"
        
        if 'bm25_bge_rerank' in df.columns:
            for baseline in BASELINES:
                if baseline in df.columns:
                    col_name = f'vs_{baseline}'
                    df[col_name] = df.apply(
                        lambda r, b=baseline: calc_improvement(r, b),
                        axis=1
                    )
        
        print(f"\n{primary_metric} Comparison:")
        print("-" * 80)
        
        # Format for display
        display_df = df.copy()
        for col in ['bm25_bge_rerank'] + BASELINES:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(
                    lambda x: f"{x:.4f}" if pd.notna(x) else "N/A"
                )
        
        print(display_df.to_string(index=False))
    
    # Aggregate summary across all query types
    print("\n" + "=" * 80)
    print("AGGREGATE SUMMARY (All Query Types)")
    print("=" * 80)
    
    if all_results:
        all_df = pd.DataFrame(all_results)
        
        # Calculate mean per method across all domains and query types
        summary = {}
        for method in ['bm25_bge_rerank'] + BASELINES:
            if method in all_df.columns:
                valid_values = all_df[method].dropna()
                if len(valid_values) > 0:
                    summary[method] = {
                        'mean': valid_values.mean(),
                        'std': valid_values.std(),
                        'count': len(valid_values)
                    }
        
        print(f"\n{primary_metric} Summary:")
        print("-" * 60)
        for method, stats in sorted(summary.items(), key=lambda x: -x[1]['mean']):
            print(f"  {method:20s}: {stats['mean']:.4f} ± {stats['std']:.4f} (n={stats['count']})")
        
        # Calculate improvement over best baseline
        if 'bm25_bge_rerank' in summary:
            two_stage_mean = summary['bm25_bge_rerank']['mean']
            best_baseline = max(
                [(name, stats['mean']) for name, stats in summary.items() if name != 'bm25_bge_rerank'],
                key=lambda x: x[1],
                default=(None, 0)
            )
            
            if best_baseline[0]:
                improvement = (two_stage_mean - best_baseline[1]) / best_baseline[1] * 100
                print(f"\n  Improvement over best baseline ({best_baseline[0]}): {improvement:+.2f}%")
    
    # Save detailed comparison
    if all_results:
        output_file = RESULTS_DIR / "comparison_with_baselines.csv"
        all_df.to_csv(output_file, index=False)
        print(f"\nDetailed comparison saved to: {output_file}")
    
    # Also create a pivot table view
    if all_results:
        print("\n" + "=" * 80)
        print("PIVOT TABLE: Domain x Query Type (BM25+BGE Rerank)")
        print("=" * 80)
        
        pivot_df = all_df.pivot_table(
            values='bm25_bge_rerank',
            index='domain',
            columns='query_type',
            aggfunc='first'
        )
        
        if not pivot_df.empty:
            # Format values
            pivot_display = pivot_df.applymap(
                lambda x: f"{x:.4f}" if pd.notna(x) else "N/A"
            )
            print(pivot_display.to_string())
        
        # Save pivot table
        pivot_file = RESULTS_DIR / "two_stage_pivot_table.csv"
        pivot_df.to_csv(pivot_file)
        print(f"\nPivot table saved to: {pivot_file}")


if __name__ == '__main__':
    main()

