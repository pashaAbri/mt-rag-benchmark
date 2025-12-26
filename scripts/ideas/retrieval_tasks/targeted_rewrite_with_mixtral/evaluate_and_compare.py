#!/usr/bin/env python3
"""
Evaluate targeted rewrite (Mixtral) results and compare against baselines.

This script:
1. Runs retrieval evaluation on targeted rewrite (Mixtral) queries
2. Compares against:
   - Baseline rewrite (Mixtral, full history) - to isolate context filtering effect
   - Targeted rewrite (Claude) - to see the LLM effect
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
RESULTS_DIR = script_dir / "retrieval_results"
BASELINE_DIR = project_root / "human" / "retrieval_tasks"
TARGETED_CLAUDE_DIR = project_root / "scripts" / "ideas" / "retrieval_tasks" / "targeted_rewrite" / "retrieval_results"
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


def load_analysis_stats(domain, file_prefix="targeted_rewrite_mixtral"):
    """Load analysis stats from targeted rewrite analysis JSON file."""
    analysis_file = INTERMEDIATE_DIR / f"{file_prefix}_{domain}_analysis.json"
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
    parser = argparse.ArgumentParser(description='Evaluate targeted rewrite (Mixtral) results')
    parser.add_argument("--run_eval", action="store_true", help="Run evaluation first")
    parser.add_argument("--retriever", type=str, default="elser", 
                        choices=["bm25", "bge", "elser"], help="Retriever to compare")
    args = parser.parse_args()
    
    print("="*80)
    print("Evaluating Targeted Rewrite (Mixtral) Results")
    print("="*80)
    print(f"Retriever: {args.retriever}")
    
    # Check for results
    results_exist = any((RESULTS_DIR / f"targeted_rewrite_mixtral_{d}_{args.retriever}.jsonl").exists() for d in DOMAINS)
    if not results_exist:
        print(f"No targeted rewrite (Mixtral) results found for {args.retriever}.")
        print("Run run_targeted_rewrite.py and run_retrieval.py first.")
        return
    
    # Run evaluations if requested
    if args.run_eval:
        print("\nStep 1: Running evaluations...")
        for domain in DOMAINS:
            input_file = RESULTS_DIR / f"targeted_rewrite_mixtral_{domain}_{args.retriever}.jsonl"
            output_file = RESULTS_DIR / f"targeted_rewrite_mixtral_{domain}_{args.retriever}_evaluated.jsonl"
            if input_file.exists():
                run_evaluation(input_file, output_file)
            else:
                print(f"  Skipping {domain}: no results file")
    
    # Load metrics
    print("\nStep 2: Loading metrics...")
    
    metrics = {
        "targeted_mixtral": {},  # This experiment
        "targeted_claude": {},   # Original targeted rewrite (Claude)
        "baseline": {}           # Paper baseline (Mixtral, full history)
    }
    
    for domain in DOMAINS:
        # Targeted rewrite with Mixtral (this experiment)
        targeted_mixtral_agg = RESULTS_DIR / f"targeted_rewrite_mixtral_{domain}_{args.retriever}_evaluated_aggregate.csv"
        if targeted_mixtral_agg.exists():
            metrics["targeted_mixtral"][domain] = load_evaluation_results(targeted_mixtral_agg)
        
        # Targeted rewrite with Claude (original experiment)
        targeted_claude_agg = TARGETED_CLAUDE_DIR / f"targeted_rewrite_{domain}_{args.retriever}_evaluated_aggregate.csv"
        if targeted_claude_agg.exists():
            metrics["targeted_claude"][domain] = load_evaluation_results(targeted_claude_agg)
        
        # Baseline (from baselines)
        baseline_agg = project_root / "scripts" / "baselines" / "retrieval_scripts" / args.retriever / "results" / f"{args.retriever}_{domain}_rewrite_evaluated_aggregate.csv"
        if baseline_agg.exists():
            metrics["baseline"][domain] = load_evaluation_results(baseline_agg)
    
    # Generate comparison table
    print("\n" + "="*100)
    print("COMPARISON: Targeted Rewrite (Mixtral) vs Baseline vs Targeted (Claude)")
    print("="*100)
    print(f"Retriever: {args.retriever.upper()}")
    print("-"*100)
    
    comparison_data = []
    
    for domain in DOMAINS:
        targeted_mixtral = metrics["targeted_mixtral"].get(domain)
        targeted_claude = metrics["targeted_claude"].get(domain)
        baseline = metrics["baseline"].get(domain)
        analysis = load_analysis_stats(domain)
        
        row = {"Domain": domain.capitalize()}
        
        # Baseline (Mixtral, full history)
        if baseline:
            row["Baseline nDCG@10"] = f"{baseline['ndcg_cut_10']:.4f}"
        else:
            row["Baseline nDCG@10"] = "N/A"
        
        # Targeted Mixtral (this experiment)
        if targeted_mixtral:
            row["Targeted Mixtral nDCG@10"] = f"{targeted_mixtral['ndcg_cut_10']:.4f}"
            
            if baseline:
                delta = targeted_mixtral['ndcg_cut_10'] - baseline['ndcg_cut_10']
                pct = (delta / baseline['ndcg_cut_10']) * 100 if baseline['ndcg_cut_10'] > 0 else 0
                row["Δ vs Baseline"] = f"{delta:+.4f} ({pct:+.1f}%)"
            else:
                row["Δ vs Baseline"] = "N/A"
        else:
            row["Targeted Mixtral nDCG@10"] = "N/A"
            row["Δ vs Baseline"] = "N/A"
        
        # Targeted Claude (for comparison)
        if targeted_claude:
            row["Targeted Claude nDCG@10"] = f"{targeted_claude['ndcg_cut_10']:.4f}"
            
            if targeted_mixtral:
                delta = targeted_claude['ndcg_cut_10'] - targeted_mixtral['ndcg_cut_10']
                row["LLM Effect (Claude-Mixtral)"] = f"{delta:+.4f}"
            else:
                row["LLM Effect (Claude-Mixtral)"] = "N/A"
        else:
            row["Targeted Claude nDCG@10"] = "N/A"
            row["LLM Effect (Claude-Mixtral)"] = "N/A"
        
        if analysis:
            row["% Turns Filtered"] = f"{analysis['pct_turns_filtered']:.0f}%"
        else:
            row["% Turns Filtered"] = "N/A"
        
        comparison_data.append(row)
    
    if comparison_data:
        df = pd.DataFrame(comparison_data)
        print("\n" + df.to_string(index=False))
    else:
        print("\nNo results to compare. Run evaluations first.")
    
    # Summary interpretation
    print("\n" + "="*100)
    print("INTERPRETATION")
    print("="*100)
    print("""
This experiment isolates the effect of CONTEXT FILTERING from the LLM quality:

• Baseline (Mixtral, full history): The original paper's approach
• Targeted Mixtral (filtered history): Same LLM, but with context filtering  
• Targeted Claude (filtered history): Better LLM with context filtering

KEY COMPARISONS:
1. "Δ vs Baseline" = Pure effect of context filtering (same LLM)
   - Positive = Context filtering helps
   - Negative = Context filtering hurts
   
2. "LLM Effect (Claude-Mixtral)" = Improvement from using Claude
   - This shows how much of the original experiment's gains came from the better LLM
""")
    
    print("\n" + "="*80)
    print("Done!")
    print("="*80)


if __name__ == "__main__":
    main()

