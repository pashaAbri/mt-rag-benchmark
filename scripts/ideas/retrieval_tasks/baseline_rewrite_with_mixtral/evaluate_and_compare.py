#!/usr/bin/env python3
"""
Evaluate baseline rewrite (Mixtral 8x7B, full history) results and compare.

This script:
1. Runs retrieval evaluation on baseline (Mixtral, full history) results
2. Compares against:
   - Paper baseline (Mixtral, pre-computed rewrites)
   - Targeted Mixtral (Mixtral, filtered history)
3. Generates summary tables and analysis

Usage:
    python evaluate_and_compare.py --run_eval --retriever elser
"""

import sys
import json
import subprocess
import pandas as pd
import ast
import argparse
from pathlib import Path

# Paths
script_dir = Path(__file__).parent
project_root = script_dir.parents[3]
sys.path.insert(0, str(project_root))

RESULTS_DIR = script_dir / "retrieval_results"
TARGETED_MIXTRAL_DIR = script_dir.parent / "targeted_rewrite_with_mixtral" / "retrieval_results"
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


def main():
    parser = argparse.ArgumentParser(description='Evaluate baseline rewrite (Mixtral 8x7B, full history)')
    parser.add_argument("--run_eval", action="store_true", help="Run evaluation first")
    parser.add_argument("--retriever", type=str, default="elser", 
                        choices=["bm25", "bge", "elser"], help="Retriever to compare")
    args = parser.parse_args()
    
    print("="*80)
    print("Evaluating Baseline Rewrite (Mixtral 8x7B, Full History)")
    print("="*80)
    print(f"Retriever: {args.retriever}")
    
    # Check for results
    results_exist = any((RESULTS_DIR / f"baseline_rewrite_{d}_{args.retriever}.jsonl").exists() for d in DOMAINS)
    if not results_exist:
        print(f"No baseline results found for {args.retriever}.")
        print("Run run_full_pipeline.py first.")
        return
    
    # Run evaluations if requested
    if args.run_eval:
        print("\nStep 1: Running evaluations...")
        for domain in DOMAINS:
            input_file = RESULTS_DIR / f"baseline_rewrite_{domain}_{args.retriever}.jsonl"
            output_file = RESULTS_DIR / f"baseline_rewrite_{domain}_{args.retriever}_evaluated.jsonl"
            if input_file.exists():
                run_evaluation(input_file, output_file)
            else:
                print(f"  Skipping {domain}: no results file")
    
    # Load metrics
    print("\nStep 2: Loading metrics...")
    
    metrics = {
        "baseline_mixtral": {},    # This experiment (Mixtral, full history, live pipeline)
        "targeted_mixtral": {},    # Targeted Mixtral (filtered history)
        "paper_baseline": {}       # Paper baseline (Mixtral, pre-computed rewrites)
    }
    
    for domain in DOMAINS:
        # Baseline Mixtral (this experiment)
        mixtral_baseline_agg = RESULTS_DIR / f"baseline_rewrite_{domain}_{args.retriever}_evaluated_aggregate.csv"
        if mixtral_baseline_agg.exists():
            metrics["baseline_mixtral"][domain] = load_evaluation_results(mixtral_baseline_agg)
        
        # Targeted Mixtral (from sibling experiment)
        targeted_agg = TARGETED_MIXTRAL_DIR / f"targeted_rewrite_{domain}_{args.retriever}_evaluated_aggregate.csv"
        if targeted_agg.exists():
            metrics["targeted_mixtral"][domain] = load_evaluation_results(targeted_agg)
        
        # Paper baseline (Mixtral, pre-computed rewrites)
        paper_agg = project_root / "scripts" / "baselines" / "retrieval_scripts" / args.retriever / "results" / f"{args.retriever}_{domain}_rewrite_evaluated_aggregate.csv"
        if paper_agg.exists():
            metrics["paper_baseline"][domain] = load_evaluation_results(paper_agg)
    
    # Generate comparison table
    print("\n" + "="*110)
    print("COMPARISON: Mixtral Baseline (Full History) vs Targeted Mixtral (Filtered)")
    print("="*110)
    print(f"Retriever: {args.retriever.upper()}")
    print("-"*110)
    
    comparison_data = []
    
    for domain in DOMAINS:
        baseline_mixtral = metrics["baseline_mixtral"].get(domain)
        targeted_mixtral = metrics["targeted_mixtral"].get(domain)
        paper_baseline = metrics["paper_baseline"].get(domain)
        
        row = {"Domain": domain.capitalize()}
        
        # Paper baseline (pre-computed)
        if paper_baseline:
            row["Paper Baseline"] = f"{paper_baseline['ndcg_cut_10']:.4f}"
        else:
            row["Paper Baseline"] = "N/A"
        
        # Mixtral baseline (this experiment)
        if baseline_mixtral:
            row["Mixtral Full"] = f"{baseline_mixtral['ndcg_cut_10']:.4f}"
            
            if paper_baseline:
                delta = baseline_mixtral['ndcg_cut_10'] - paper_baseline['ndcg_cut_10']
                pct = (delta / paper_baseline['ndcg_cut_10']) * 100 if paper_baseline['ndcg_cut_10'] > 0 else 0
                row["Δ vs Paper"] = f"{delta:+.4f} ({pct:+.1f}%)"
            else:
                row["Δ vs Paper"] = "N/A"
        else:
            row["Mixtral Full"] = "N/A"
            row["Δ vs Paper"] = "N/A"
        
        # Targeted Mixtral (filtered)
        if targeted_mixtral:
            row["Mixtral Filtered"] = f"{targeted_mixtral['ndcg_cut_10']:.4f}"
            
            if baseline_mixtral:
                delta = targeted_mixtral['ndcg_cut_10'] - baseline_mixtral['ndcg_cut_10']
                pct = (delta / baseline_mixtral['ndcg_cut_10']) * 100 if baseline_mixtral['ndcg_cut_10'] > 0 else 0
                row["Δ Filtered vs Full"] = f"{delta:+.4f} ({pct:+.1f}%)"
            else:
                row["Δ Filtered vs Full"] = "N/A"
        else:
            row["Mixtral Filtered"] = "N/A"
            row["Δ Filtered vs Full"] = "N/A"
        
        comparison_data.append(row)
    
    if comparison_data:
        df = pd.DataFrame(comparison_data)
        print("\n" + df.to_string(index=False))
    else:
        print("\nNo results to compare. Run evaluations first.")
    
    # Calculate overall averages
    if metrics["baseline_mixtral"]:
        print("\n" + "-"*110)
        print("OVERALL AVERAGES")
        print("-"*110)
        
        mixtral_full_avg = sum(m['ndcg_cut_10'] for m in metrics["baseline_mixtral"].values()) / len(metrics["baseline_mixtral"])
        print(f"  Mixtral Full History avg nDCG@10:     {mixtral_full_avg:.4f}")
        
        if metrics["paper_baseline"]:
            paper_avg = sum(m['ndcg_cut_10'] for m in metrics["paper_baseline"].values()) / len(metrics["paper_baseline"])
            print(f"  Paper Baseline avg nDCG@10:           {paper_avg:.4f}")
            print(f"  Δ Our Pipeline vs Paper:              {mixtral_full_avg - paper_avg:+.4f} ({(mixtral_full_avg - paper_avg) / paper_avg * 100:+.1f}%)")
        
        if metrics["targeted_mixtral"]:
            targeted_avg = sum(m['ndcg_cut_10'] for m in metrics["targeted_mixtral"].values()) / len(metrics["targeted_mixtral"])
            print(f"  Mixtral Filtered avg nDCG@10:         {targeted_avg:.4f}")
            print(f"  Δ Filtered vs Full (effect of filtering): {targeted_avg - mixtral_full_avg:+.4f} ({(targeted_avg - mixtral_full_avg) / mixtral_full_avg * 100:+.1f}%)")
    
    # Summary interpretation
    print("\n" + "="*110)
    print("INTERPRETATION")
    print("="*110)
    print("""
This experiment establishes a FAIR BASELINE for Mixtral 8x7B with our pipeline:

• Paper Baseline:    Mixtral 8x7B with pre-computed rewrites (from paper)
• Mixtral Full:      Mixtral 8x7B with FULL history (THIS experiment, live pipeline)
• Mixtral Filtered:  Mixtral 8x7B with FILTERED history

KEY COMPARISONS:
  - "Δ vs Paper"          = Difference between live pipeline vs pre-computed
  - "Δ Filtered vs Full"  = Effect of context filtering (same LLM, same pipeline)
  
This helps validate our pipeline against the paper and measures the pure
effect of context filtering on Mixtral.
""")
    
    print("\n" + "="*80)
    print("Done!")
    print("="*80)


if __name__ == "__main__":
    main()

