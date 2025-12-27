#!/usr/bin/env python3
"""
Evaluate baseline rewrite (Claude Sonnet, full history) results and compare.

This script:
1. Runs retrieval evaluation on baseline (Sonnet, full history) results
2. Compares against:
   - Paper baseline (Mixtral, full history)
   - Targeted Sonnet (Sonnet, filtered history)
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
TARGETED_SONNET_DIR = script_dir.parent / "targeted_rewrite_with_sonnet" / "retrieval_results"
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
    parser = argparse.ArgumentParser(description='Evaluate baseline rewrite (Sonnet, full history)')
    parser.add_argument("--run_eval", action="store_true", help="Run evaluation first")
    parser.add_argument("--retriever", type=str, default="elser", 
                        choices=["bm25", "bge", "elser"], help="Retriever to compare")
    args = parser.parse_args()
    
    print("="*80)
    print("Evaluating Baseline Rewrite (Claude Sonnet, Full History)")
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
        "baseline_sonnet": {},    # This experiment (Sonnet, full history)
        "targeted_sonnet": {},    # Targeted Sonnet (filtered history)
        "baseline_mixtral": {}    # Paper baseline (Mixtral, full history)
    }
    
    for domain in DOMAINS:
        # Baseline Sonnet (this experiment)
        sonnet_baseline_agg = RESULTS_DIR / f"baseline_rewrite_{domain}_{args.retriever}_evaluated_aggregate.csv"
        if sonnet_baseline_agg.exists():
            metrics["baseline_sonnet"][domain] = load_evaluation_results(sonnet_baseline_agg)
        
        # Targeted Sonnet (from sibling experiment)
        targeted_agg = TARGETED_SONNET_DIR / f"targeted_rewrite_{domain}_{args.retriever}_evaluated_aggregate.csv"
        if targeted_agg.exists():
            metrics["targeted_sonnet"][domain] = load_evaluation_results(targeted_agg)
        
        # Paper baseline (Mixtral, full history)
        mixtral_agg = project_root / "scripts" / "baselines" / "retrieval_scripts" / args.retriever / "results" / f"{args.retriever}_{domain}_rewrite_evaluated_aggregate.csv"
        if mixtral_agg.exists():
            metrics["baseline_mixtral"][domain] = load_evaluation_results(mixtral_agg)
    
    # Generate comparison table
    print("\n" + "="*110)
    print("COMPARISON: Sonnet Baseline (Full History) vs Targeted Sonnet (Filtered)")
    print("="*110)
    print(f"Retriever: {args.retriever.upper()}")
    print("-"*110)
    
    comparison_data = []
    
    for domain in DOMAINS:
        baseline_sonnet = metrics["baseline_sonnet"].get(domain)
        targeted_sonnet = metrics["targeted_sonnet"].get(domain)
        baseline_mixtral = metrics["baseline_mixtral"].get(domain)
        
        row = {"Domain": domain.capitalize()}
        
        # Mixtral baseline (paper)
        if baseline_mixtral:
            row["Mixtral Full"] = f"{baseline_mixtral['ndcg_cut_10']:.4f}"
        else:
            row["Mixtral Full"] = "N/A"
        
        # Sonnet baseline (this experiment)
        if baseline_sonnet:
            row["Sonnet Full"] = f"{baseline_sonnet['ndcg_cut_10']:.4f}"
            
            if baseline_mixtral:
                delta = baseline_sonnet['ndcg_cut_10'] - baseline_mixtral['ndcg_cut_10']
                pct = (delta / baseline_mixtral['ndcg_cut_10']) * 100 if baseline_mixtral['ndcg_cut_10'] > 0 else 0
                row["Δ vs Mixtral"] = f"{delta:+.4f} ({pct:+.1f}%)"
            else:
                row["Δ vs Mixtral"] = "N/A"
        else:
            row["Sonnet Full"] = "N/A"
            row["Δ vs Mixtral"] = "N/A"
        
        # Targeted Sonnet (filtered)
        if targeted_sonnet:
            row["Sonnet Filtered"] = f"{targeted_sonnet['ndcg_cut_10']:.4f}"
            
            if baseline_sonnet:
                delta = targeted_sonnet['ndcg_cut_10'] - baseline_sonnet['ndcg_cut_10']
                pct = (delta / baseline_sonnet['ndcg_cut_10']) * 100 if baseline_sonnet['ndcg_cut_10'] > 0 else 0
                row["Δ Filtered vs Full"] = f"{delta:+.4f} ({pct:+.1f}%)"
            else:
                row["Δ Filtered vs Full"] = "N/A"
        else:
            row["Sonnet Filtered"] = "N/A"
            row["Δ Filtered vs Full"] = "N/A"
        
        comparison_data.append(row)
    
    if comparison_data:
        df = pd.DataFrame(comparison_data)
        print("\n" + df.to_string(index=False))
    else:
        print("\nNo results to compare. Run evaluations first.")
    
    # Calculate overall averages
    if metrics["baseline_sonnet"]:
        print("\n" + "-"*110)
        print("OVERALL AVERAGES")
        print("-"*110)
        
        sonnet_full_avg = sum(m['ndcg_cut_10'] for m in metrics["baseline_sonnet"].values()) / len(metrics["baseline_sonnet"])
        print(f"  Sonnet Full History avg nDCG@10:     {sonnet_full_avg:.4f}")
        
        if metrics["baseline_mixtral"]:
            mixtral_avg = sum(m['ndcg_cut_10'] for m in metrics["baseline_mixtral"].values()) / len(metrics["baseline_mixtral"])
            print(f"  Mixtral Full History avg nDCG@10:    {mixtral_avg:.4f}")
            print(f"  Δ Sonnet vs Mixtral:                 {sonnet_full_avg - mixtral_avg:+.4f} ({(sonnet_full_avg - mixtral_avg) / mixtral_avg * 100:+.1f}%)")
        
        if metrics["targeted_sonnet"]:
            targeted_avg = sum(m['ndcg_cut_10'] for m in metrics["targeted_sonnet"].values()) / len(metrics["targeted_sonnet"])
            print(f"  Sonnet Filtered avg nDCG@10:         {targeted_avg:.4f}")
            print(f"  Δ Filtered vs Full (effect of filtering): {targeted_avg - sonnet_full_avg:+.4f} ({(targeted_avg - sonnet_full_avg) / sonnet_full_avg * 100:+.1f}%)")
    
    # Summary interpretation
    print("\n" + "="*110)
    print("INTERPRETATION")
    print("="*110)
    print("""
This experiment establishes a FAIR BASELINE for Claude Sonnet:

• Mixtral Full:      Mixtral 8x7B with FULL conversation history (paper's approach)
• Sonnet Full:       Claude Sonnet with FULL conversation history (THIS experiment)
• Sonnet Filtered:   Claude Sonnet with FILTERED conversation history

KEY COMPARISONS:
  - "Δ vs Mixtral"         = Effect of LLM quality (same context strategy)
  - "Δ Filtered vs Full"   = PURE effect of context filtering (same LLM)
  
This isolates the effect of filtering from the effect of LLM quality.
""")
    
    print("\n" + "="*80)
    print("Done!")
    print("="*80)


if __name__ == "__main__":
    main()

