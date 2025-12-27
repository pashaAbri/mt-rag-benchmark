#!/usr/bin/env python3
"""
Evaluate targeted rewrite (Claude Sonnet) results and compare against baselines.

This script:
1. Runs retrieval evaluation on targeted rewrite (Sonnet) queries
2. Compares against:
   - Baseline rewrite (Mixtral, full history) - to see overall improvement
   - Targeted rewrite (Mixtral) - to isolate the effect of LLM quality
3. Generates summary tables and analysis

Usage:
    python evaluate_and_compare.py
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
MIXTRAL_RESULTS_DIR = script_dir.parent / "targeted_rewrite_with_mixtral" / "retrieval_results"
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


def load_analysis_stats(domain, retriever="elser"):
    """Load analysis stats from targeted rewrite analysis JSON file."""
    analysis_file = RESULTS_DIR / f"targeted_rewrite_{domain}_{retriever}_analysis.json"
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
    rewritten = sum(1 for a in analyses if "targeted_rewrite" in a.get("method", ""))
    
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
    parser = argparse.ArgumentParser(description='Evaluate targeted rewrite (Claude Sonnet) results')
    parser.add_argument("--run_eval", action="store_true", help="Run evaluation first")
    parser.add_argument("--retriever", type=str, default="elser", 
                        choices=["bm25", "bge", "elser"], help="Retriever to compare")
    args = parser.parse_args()
    
    print("="*80)
    print("Evaluating Targeted Rewrite (Claude Sonnet) Results")
    print("="*80)
    print(f"Retriever: {args.retriever}")
    
    # Check for results
    results_exist = any((RESULTS_DIR / f"targeted_rewrite_{d}_{args.retriever}.jsonl").exists() for d in DOMAINS)
    if not results_exist:
        print(f"No targeted rewrite results found for {args.retriever}.")
        print("Run run_full_pipeline.py first.")
        return
    
    # Run evaluations if requested
    if args.run_eval:
        print("\nStep 1: Running evaluations...")
        for domain in DOMAINS:
            input_file = RESULTS_DIR / f"targeted_rewrite_{domain}_{args.retriever}.jsonl"
            output_file = RESULTS_DIR / f"targeted_rewrite_{domain}_{args.retriever}_evaluated.jsonl"
            if input_file.exists():
                run_evaluation(input_file, output_file)
            else:
                print(f"  Skipping {domain}: no results file")
    
    # Load metrics
    print("\nStep 2: Loading metrics...")
    
    metrics = {
        "targeted_sonnet": {},   # This experiment (Claude Sonnet)
        "targeted_mixtral": {},  # Mixtral targeted rewrite (for comparison)
        "baseline": {}           # Paper baseline (Mixtral, full history)
    }
    
    for domain in DOMAINS:
        # Targeted rewrite with Sonnet (this experiment)
        sonnet_agg = RESULTS_DIR / f"targeted_rewrite_{domain}_{args.retriever}_evaluated_aggregate.csv"
        if sonnet_agg.exists():
            metrics["targeted_sonnet"][domain] = load_evaluation_results(sonnet_agg)
        
        # Targeted rewrite with Mixtral (from sibling experiment)
        mixtral_agg = MIXTRAL_RESULTS_DIR / f"targeted_rewrite_{domain}_{args.retriever}_evaluated_aggregate.csv"
        if mixtral_agg.exists():
            metrics["targeted_mixtral"][domain] = load_evaluation_results(mixtral_agg)
        
        # Baseline (from baselines)
        baseline_agg = project_root / "scripts" / "baselines" / "retrieval_scripts" / args.retriever / "results" / f"{args.retriever}_{domain}_rewrite_evaluated_aggregate.csv"
        if baseline_agg.exists():
            metrics["baseline"][domain] = load_evaluation_results(baseline_agg)
    
    # Generate comparison table
    print("\n" + "="*100)
    print("COMPARISON: Targeted Rewrite (Sonnet) vs Baseline & Mixtral Targeted")
    print("="*100)
    print(f"Retriever: {args.retriever.upper()}")
    print("-"*100)
    
    comparison_data = []
    
    for domain in DOMAINS:
        sonnet = metrics["targeted_sonnet"].get(domain)
        mixtral = metrics["targeted_mixtral"].get(domain)
        baseline = metrics["baseline"].get(domain)
        analysis = load_analysis_stats(domain, args.retriever)
        
        row = {"Domain": domain.capitalize()}
        
        # Baseline (Mixtral, full history)
        if baseline:
            row["Baseline nDCG@10"] = f"{baseline['ndcg_cut_10']:.4f}"
        else:
            row["Baseline nDCG@10"] = "N/A"
        
        # Targeted Mixtral
        if mixtral:
            row["Mixtral nDCG@10"] = f"{mixtral['ndcg_cut_10']:.4f}"
        else:
            row["Mixtral nDCG@10"] = "N/A"
        
        # Targeted Sonnet (this experiment)
        if sonnet:
            row["Sonnet nDCG@10"] = f"{sonnet['ndcg_cut_10']:.4f}"
            
            if baseline:
                delta_vs_baseline = sonnet['ndcg_cut_10'] - baseline['ndcg_cut_10']
                pct_baseline = (delta_vs_baseline / baseline['ndcg_cut_10']) * 100 if baseline['ndcg_cut_10'] > 0 else 0
                row["Δ vs Baseline"] = f"{delta_vs_baseline:+.4f} ({pct_baseline:+.1f}%)"
            else:
                row["Δ vs Baseline"] = "N/A"
            
            if mixtral:
                delta_vs_mixtral = sonnet['ndcg_cut_10'] - mixtral['ndcg_cut_10']
                pct_mixtral = (delta_vs_mixtral / mixtral['ndcg_cut_10']) * 100 if mixtral['ndcg_cut_10'] > 0 else 0
                row["Δ vs Mixtral"] = f"{delta_vs_mixtral:+.4f} ({pct_mixtral:+.1f}%)"
            else:
                row["Δ vs Mixtral"] = "N/A"
        else:
            row["Sonnet nDCG@10"] = "N/A"
            row["Δ vs Baseline"] = "N/A"
            row["Δ vs Mixtral"] = "N/A"
        
        if analysis:
            row["% Filtered"] = f"{analysis['pct_turns_filtered']:.0f}%"
        else:
            row["% Filtered"] = "N/A"
        
        comparison_data.append(row)
    
    if comparison_data:
        df = pd.DataFrame(comparison_data)
        print("\n" + df.to_string(index=False))
    else:
        print("\nNo results to compare. Run evaluations first.")
    
    # Calculate overall averages
    if metrics["targeted_sonnet"] and metrics["baseline"]:
        print("\n" + "-"*100)
        print("OVERALL AVERAGES")
        print("-"*100)
        
        sonnet_avg = sum(m['ndcg_cut_10'] for m in metrics["targeted_sonnet"].values()) / len(metrics["targeted_sonnet"])
        baseline_avg = sum(m['ndcg_cut_10'] for m in metrics["baseline"].values()) / len(metrics["baseline"]) if metrics["baseline"] else 0
        
        print(f"  Baseline avg nDCG@10:         {baseline_avg:.4f}")
        print(f"  Targeted Sonnet avg nDCG@10:  {sonnet_avg:.4f}")
        print(f"  Δ vs Baseline:                {sonnet_avg - baseline_avg:+.4f} ({(sonnet_avg - baseline_avg) / baseline_avg * 100:+.1f}%)")
        
        if metrics["targeted_mixtral"]:
            mixtral_avg = sum(m['ndcg_cut_10'] for m in metrics["targeted_mixtral"].values()) / len(metrics["targeted_mixtral"])
            print(f"  Targeted Mixtral avg nDCG@10: {mixtral_avg:.4f}")
            print(f"  Δ Sonnet vs Mixtral:          {sonnet_avg - mixtral_avg:+.4f} ({(sonnet_avg - mixtral_avg) / mixtral_avg * 100:+.1f}%)")
    
    # Summary interpretation
    print("\n" + "="*100)
    print("INTERPRETATION")
    print("="*100)
    print("""
This experiment tests the effect of using a STRONGER LLM (Claude Sonnet) for targeted query rewriting:

• Baseline:         Mixtral 8x7B with FULL conversation history (paper's approach)
• Targeted Mixtral: Mixtral 8x7B with FILTERED conversation history 
• Targeted Sonnet:  Claude Sonnet with FILTERED conversation history (this experiment)

FULL PIPELINE MODE:
  - Uses GENERATED responses (not ground truth) for context
  - Simulates a real conversational RAG system

KEY INSIGHTS:
  - "Δ vs Baseline" = Total improvement from using Sonnet + context filtering
  - "Δ vs Mixtral"  = Pure effect of LLM quality (same filtering strategy)
  
HYPOTHESIS:
  - Stronger LLMs can better infer from less context
  - Context filtering should work better with Sonnet than Mixtral
""")
    
    print("\n" + "="*80)
    print("Done!")
    print("="*80)


if __name__ == "__main__":
    main()

