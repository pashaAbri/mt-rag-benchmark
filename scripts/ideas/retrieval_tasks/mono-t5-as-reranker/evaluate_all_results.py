#!/usr/bin/env python3
"""
Unified script to evaluate all reranked results and generate comprehensive summaries.

This script:
1. Evaluates all existing reranked files (runs evaluation if needed)
2. Loads metrics from all combinations
3. Generates comprehensive comparison tables
4. Calculates "ALL" domain metrics by aggregating across domains

Usage:
    python evaluate_all_results.py
"""

import sys
import subprocess
import pandas as pd
import ast
from pathlib import Path
from collections import defaultdict

# Paths
script_dir = Path(__file__).parent
project_root = script_dir.parents[3]
sys.path.insert(0, str(project_root))

INTERMEDIATE_DIR = script_dir / "intermediate"
EVAL_SCRIPT_PATH = project_root / 'scripts' / 'evaluation' / 'run_retrieval_eval.py'

DOMAINS = ['clapnq', 'cloud', 'fiqa', 'govt']

# All possible combinations
ALL_COMBINATIONS = [
    ['lastturn', 'rewrite'],
    ['lastturn', 'questions'],
    ['rewrite', 'questions'],
    ['lastturn', 'rewrite', 'questions'],
]

def get_combination_name(strategies):
    """Generate consistent filename-safe name for the combination."""
    return "_".join(sorted(strategies))

def load_evaluation_results(file_path):
    """Load evaluation results from aggregate CSV file."""
    if not file_path.exists():
        return None
    
    try:
        df = pd.read_csv(file_path)
        all_rows = df[df['collection'] == 'all']
        if len(all_rows) == 0:
            all_row = df.iloc[0]
        else:
            all_row = all_rows.iloc[0]
        
        recall = ast.literal_eval(all_row['Recall'])
        ndcg = ast.literal_eval(all_row['nDCG'])
        count = all_row.get('count', 0)
        
        results = {
            'recall_1': recall[0],
            'recall_3': recall[1],
            'recall_5': recall[2],
            'recall_10': recall[3] if len(recall) > 3 else 0.0,
            'ndcg_cut_1': ndcg[0],
            'ndcg_cut_3': ndcg[1],
            'ndcg_cut_5': ndcg[2],
            'ndcg_cut_10': ndcg[3] if len(ndcg) > 3 else 0.0,
            'count': count
        }
        return results
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def calculate_weighted_average(metrics_list):
    """Calculate weighted average across domains."""
    if not metrics_list:
        return None
    
    total_count = sum(m['count'] for m in metrics_list if m)
    if total_count == 0:
        return None
    
    weighted = {}
    for metric in ['recall_1', 'recall_3', 'recall_5', 'recall_10', 
                   'ndcg_cut_1', 'ndcg_cut_3', 'ndcg_cut_5', 'ndcg_cut_10']:
        weighted_sum = sum(m[metric] * m['count'] for m in metrics_list if m)
        weighted[metric] = weighted_sum / total_count if total_count > 0 else 0.0
    
    weighted['count'] = total_count
    return weighted

def main():
    print("="*80)
    print("Evaluating All Reranked Results")
    print("="*80)
    
    # Collect all metrics
    all_metrics = {}  # {combo_name: {domain: metrics}}
    
    # First, ensure all files are evaluated
    print("\nStep 1: Ensuring all files are evaluated...")
    for domain in DOMAINS:
        print(f"\nProcessing {domain}...")
        for combo in ALL_COMBINATIONS:
            combo_name = get_combination_name(combo)
            input_file = INTERMEDIATE_DIR / f"reranked_{combo_name}_{domain}.jsonl"
            evaluated_file = INTERMEDIATE_DIR / f"reranked_{combo_name}_{domain}_evaluated.jsonl"
            agg_file = INTERMEDIATE_DIR / f"reranked_{combo_name}_{domain}_evaluated_aggregate.csv"
            
            if not input_file.exists():
                print(f"  Missing: {input_file.name}")
                continue
            
            # Run evaluation if needed
            if not agg_file.exists():
                print(f"  Evaluating {combo_name}...")
                cmd = [
                    sys.executable, str(EVAL_SCRIPT_PATH),
                    "--input_file", str(input_file),
                    "--output_file", str(evaluated_file)
                ]
                try:
                    subprocess.run(cmd, check=True, capture_output=True)
                except subprocess.CalledProcessError as e:
                    print(f"  Error evaluating {combo_name}: {e}")
                    continue
            
            # Load metrics
            metrics = load_evaluation_results(agg_file)
            if metrics:
                if combo_name not in all_metrics:
                    all_metrics[combo_name] = {}
                all_metrics[combo_name][domain] = metrics
    
    # Calculate "ALL" domain metrics
    print("\nStep 2: Calculating 'ALL' domain metrics...")
    for combo_name in all_metrics:
        domain_metrics = [all_metrics[combo_name][d] for d in DOMAINS if d in all_metrics[combo_name]]
        if domain_metrics:
            all_metrics[combo_name]['all'] = calculate_weighted_average(domain_metrics)
    
    # Generate comprehensive summary
    print("\nStep 3: Generating summary tables...")
    
    # Summary table: nDCG@10
    summary_data = []
    for combo_name in sorted(all_metrics.keys()):
        row = {'Combination': combo_name}
        for domain in ['all'] + DOMAINS:
            if domain in all_metrics[combo_name]:
                metrics = all_metrics[combo_name][domain]
                row[domain.capitalize()] = f"{metrics['ndcg_cut_10']:.4f}"
            else:
                row[domain.capitalize()] = "N/A"
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    print("\n" + "="*80)
    print("SUMMARY: nDCG@10 by Combination")
    print("="*80)
    print(summary_df.to_string(index=False))
    
    # Detailed metrics table
    print("\n" + "="*80)
    print("DETAILED METRICS TABLE")
    print("="*80)
    
    detailed_rows = []
    for domain in ['all'] + DOMAINS:
        for combo_name in sorted(all_metrics.keys()):
            if domain in all_metrics[combo_name]:
                metrics = all_metrics[combo_name][domain]
                row = {
                    'Domain': domain.capitalize(),
                    'Combination': combo_name,
                    'R@1': f"{metrics['recall_1']:.4f}",
                    'R@3': f"{metrics['recall_3']:.4f}",
                    'R@5': f"{metrics['recall_5']:.4f}",
                    'R@10': f"{metrics['recall_10']:.4f}",
                    'nDCG@1': f"{metrics['ndcg_cut_1']:.4f}",
                    'nDCG@3': f"{metrics['ndcg_cut_3']:.4f}",
                    'nDCG@5': f"{metrics['ndcg_cut_5']:.4f}",
                    'nDCG@10': f"{metrics['ndcg_cut_10']:.4f}",
                }
                detailed_rows.append(row)
    
    detailed_df = pd.DataFrame(detailed_rows)
    print("\nDetailed metrics table:")
    print(detailed_df.to_string(index=False))
    
    print("\n" + "="*80)
    print("Evaluation complete!")
    print("="*80)

if __name__ == '__main__':
    main()
