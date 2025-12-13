#!/usr/bin/env python3
"""
Evaluate targeted rewrite results and compare against baseline.

This script:
1. Runs retrieval evaluation on targeted rewrite queries
2. Compares against baseline rewrite performance
3. Generates summary tables and analysis

Usage:
    python evaluate_and_compare.py
"""

import sys
import json
import subprocess
import pandas as pd
import ast
import argparse
from pathlib import Path
from collections import defaultdict

# Paths
script_dir = Path(__file__).parent
project_root = script_dir.parents[3]
sys.path.insert(0, str(project_root))

INTERMEDIATE_DIR = script_dir / "intermediate"
BASELINE_DIR = project_root / "human" / "retrieval_tasks"
EVAL_SCRIPT_PATH = project_root / 'scripts' / 'evaluation' / 'run_retrieval_eval.py'

DOMAINS = ['clapnq', 'cloud', 'fiqa', 'govt']


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


def run_evaluation(input_file, output_file):
    """Run retrieval evaluation on a file."""
    if not input_file.exists():
        print(f"Input file not found: {input_file}")
        return False
    
    agg_file = Path(str(output_file).replace('.jsonl', '_aggregate.csv'))
    if agg_file.exists():
        print(f"  Already evaluated: {agg_file.name}")
        return True
    
    print(f"  Evaluating: {input_file.name}")
    cmd = [
        sys.executable, str(EVAL_SCRIPT_PATH),
        "--input_file", str(input_file),
        "--output_file", str(output_file)
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"  Evaluation failed: {e}")
        return False


def load_analysis_stats(domain):
    """Load analysis stats from targeted rewrite analysis JSON file."""
    analysis_file = INTERMEDIATE_DIR / f"targeted_rewrite_{domain}_analysis.json"
    if not analysis_file.exists():
        return None
    
    try:
        with open(analysis_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception:
        return None
    
    analyses = data.get("analyses", [])
    if not analyses:
        return None
    
    # Compute stats
    total = len(analyses)
    rewritten = sum(1 for a in analyses if a.get("method") == "targeted_rewrite")
    
    # Average turns selected vs available
    turn_stats = [a for a in analyses if a.get("num_history_turns", 0) > 0]
    if turn_stats:
        avg_available = sum(a["num_history_turns"] for a in turn_stats) / len(turn_stats)
        avg_selected = sum(a["selected_turns"] for a in turn_stats) / len(turn_stats)
        pct_filtered = (1 - avg_selected / avg_available) * 100 if avg_available > 0 else 0
    else:
        avg_available = avg_selected = pct_filtered = 0
    
    return {
        "total": total,
        "rewritten": rewritten,
        "avg_available_turns": round(avg_available, 2),
        "avg_selected_turns": round(avg_selected, 2),
        "pct_turns_filtered": round(pct_filtered, 1)
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate targeted rewrite results')
    parser.add_argument("--run_eval", action="store_true", help="Run evaluation first")
    args = parser.parse_args()
    
    print("="*80)
    print("Evaluating Targeted Rewrite Results")
    print("="*80)
    
    # Check for results
    results_exist = any((INTERMEDIATE_DIR / f"targeted_rewrite_{d}.jsonl").exists() for d in DOMAINS)
    if not results_exist:
        print("No targeted rewrite results found. Run run_targeted_rewrite.py first.")
        return
    
    # Run evaluations if requested
    if args.run_eval:
        print("\nStep 1: Running evaluations...")
        for domain in DOMAINS:
            input_file = INTERMEDIATE_DIR / f"targeted_rewrite_{domain}.jsonl"
            output_file = INTERMEDIATE_DIR / f"targeted_rewrite_{domain}_evaluated.jsonl"
            if input_file.exists():
                run_evaluation(input_file, output_file)
            else:
                print(f"  Skipping {domain}: no results file")
    
    # Load metrics
    print("\nStep 2: Loading metrics...")
    
    metrics = {"targeted": {}, "baseline": {}}
    
    for domain in DOMAINS:
        # Targeted rewrite
        targeted_agg = INTERMEDIATE_DIR / f"targeted_rewrite_{domain}_evaluated_aggregate.csv"
        if targeted_agg.exists():
            metrics["targeted"][domain] = load_evaluation_results(targeted_agg)
        
        # Baseline rewrite (from mono-t5 experiment or baselines)
        baseline_agg = project_root / "scripts" / "ideas" / "retrieval_tasks" / "mono-t5-as-reranker" / "intermediate" / f"baseline_rewrite_{domain}_evaluated_aggregate.csv"
        if baseline_agg.exists():
            metrics["baseline"][domain] = load_evaluation_results(baseline_agg)
    
    # Generate comparison table
    print("\n" + "="*80)
    print("COMPARISON: Targeted Rewrite vs Baseline Rewrite")
    print("="*80)
    
    comparison_data = []
    
    for domain in DOMAINS:
        targeted = metrics["targeted"].get(domain)
        baseline = metrics["baseline"].get(domain)
        analysis = load_analysis_stats(domain)
        
        row = {"Domain": domain.capitalize()}
        
        if baseline:
            row["Baseline nDCG@10"] = f"{baseline['ndcg_cut_10']:.4f}"
            row["Baseline R@10"] = f"{baseline['recall_10']:.4f}"
        else:
            row["Baseline nDCG@10"] = "N/A"
            row["Baseline R@10"] = "N/A"
        
        if targeted:
            row["Targeted nDCG@10"] = f"{targeted['ndcg_cut_10']:.4f}"
            row["Targeted R@10"] = f"{targeted['recall_10']:.4f}"
            
            if baseline:
                delta_ndcg = targeted['ndcg_cut_10'] - baseline['ndcg_cut_10']
                pct_change = (delta_ndcg / baseline['ndcg_cut_10']) * 100 if baseline['ndcg_cut_10'] > 0 else 0
                
                row["Δ nDCG@10"] = f"{delta_ndcg:+.4f}"
                row["Δ %"] = f"{pct_change:+.1f}%"
            else:
                row["Δ nDCG@10"] = "N/A"
                row["Δ %"] = "N/A"
        else:
            row["Targeted nDCG@10"] = "N/A"
            row["Targeted R@10"] = "N/A"
            row["Δ nDCG@10"] = "N/A"
            row["Δ %"] = "N/A"
        
        if analysis:
            row["Avg Turns Selected"] = f"{analysis['avg_selected_turns']:.1f}/{analysis['avg_available_turns']:.1f}"
            row["% Filtered"] = f"{analysis['pct_turns_filtered']:.0f}%"
        else:
            row["Avg Turns Selected"] = "N/A"
            row["% Filtered"] = "N/A"
        
        comparison_data.append(row)
    
    if comparison_data:
        df = pd.DataFrame(comparison_data)
        print("\n" + df.to_string(index=False))
    else:
        print("\nNo results to compare. Run evaluations first.")
    
    # Detailed per-domain analysis
    print("\n" + "="*80)
    print("TURN FILTERING ANALYSIS")
    print("="*80)
    
    for domain in DOMAINS:
        analysis = load_analysis_stats(domain)
        if analysis:
            print(f"\n{domain.upper()}:")
            print(f"  Total queries: {analysis['total']}")
            print(f"  Rewritten with context: {analysis['rewritten']}")
            print(f"  Avg available turns: {analysis['avg_available_turns']}")
            print(f"  Avg selected turns: {analysis['avg_selected_turns']}")
            print(f"  % turns filtered out: {analysis['pct_turns_filtered']}%")
    
    print("\n" + "="*80)
    print("Done!")
    print("="*80)


if __name__ == "__main__":
    main()

