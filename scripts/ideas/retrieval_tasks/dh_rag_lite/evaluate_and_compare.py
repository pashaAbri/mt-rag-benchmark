#!/usr/bin/env python3
"""
Evaluate DH-RAG results and compare against baseline and targeted rewrite.

This script:
1. Runs retrieval evaluation on DH-RAG queries
2. Compares against baseline rewrite and targeted rewrite performance
3. Generates summary tables and analysis

Usage:
    python evaluate_and_compare.py --run_eval
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

INTERMEDIATE_DIR = script_dir / "intermediate"
RESULTS_DIR = script_dir / "retrieval_results"
EVAL_SCRIPT_PATH = project_root / 'scripts' / 'evaluation' / 'run_retrieval_eval.py'

# Comparison directories
TARGETED_REWRITE_DIR = script_dir.parent / "targeted_rewrite" / "retrieval_results"

DOMAINS = ['clapnq', 'cloud', 'fiqa', 'govt']
RETRIEVERS = ['bm25', 'bge', 'elser']


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
        print(f"  Input file not found: {input_file}")
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
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"  Evaluation failed: {e.stderr}")
        return False


def load_analysis_stats(domain):
    """Load analysis stats from DH-RAG analysis JSON file."""
    analysis_file = INTERMEDIATE_DIR / f"dh_rag_{domain}_analysis.json"
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
    rewritten = sum(1 for a in analyses if a.get("method") in ["dh_rag_filtered", "dh_rag_all"])
    
    # Average turns selected vs available
    turn_stats = [a for a in analyses if a.get("num_history_turns", 0) > 0]
    if turn_stats:
        avg_available = sum(a["num_history_turns"] for a in turn_stats) / len(turn_stats)
        avg_selected = sum(a["selected_turns"] for a in turn_stats) / len(turn_stats)
        pct_filtered = (1 - avg_selected / avg_available) * 100 if avg_available > 0 else 0
    else:
        avg_available = avg_selected = pct_filtered = 0
    
    # Cluster stats
    cluster_stats = [a.get("num_clusters", 0) for a in analyses if a.get("num_clusters", 0) > 0]
    avg_clusters = sum(cluster_stats) / len(cluster_stats) if cluster_stats else 0
    
    # Chain stats
    chain_stats = [a.get("num_chains", 0) for a in analyses if a.get("num_chains", 0) > 0]
    avg_chains = sum(chain_stats) / len(chain_stats) if chain_stats else 0
    
    return {
        "total": total,
        "rewritten": rewritten,
        "avg_available_turns": round(avg_available, 2),
        "avg_selected_turns": round(avg_selected, 2),
        "pct_turns_filtered": round(pct_filtered, 1),
        "avg_clusters": round(avg_clusters, 2),
        "avg_chains": round(avg_chains, 2)
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate DH-RAG results')
    parser.add_argument("--run_eval", action="store_true", help="Run evaluation first")
    parser.add_argument("--retrievers", nargs="+", default=['bm25'], 
                        choices=RETRIEVERS, help="Retrievers to evaluate")
    args = parser.parse_args()
    
    print("="*80)
    print("Evaluating DH-RAG Results")
    print("="*80)
    
    # Check for results
    results_exist = any((RESULTS_DIR / f"dh_rag_{d}_bm25.jsonl").exists() for d in DOMAINS)
    if not results_exist:
        print("No DH-RAG retrieval results found. Run retrieval first.")
        return
    
    # Run evaluations if requested
    if args.run_eval:
        print("\nStep 1: Running evaluations...")
        for domain in DOMAINS:
            for retriever in args.retrievers:
                input_file = RESULTS_DIR / f"dh_rag_{domain}_{retriever}.jsonl"
                output_file = RESULTS_DIR / f"dh_rag_{domain}_{retriever}_evaluated.jsonl"
                if input_file.exists():
                    run_evaluation(input_file, output_file)
                else:
                    print(f"  Skipping {domain}/{retriever}: no results file")
    
    # Load metrics
    print("\nStep 2: Loading metrics...")
    
    metrics = {"dh_rag": {}, "targeted": {}}
    
    for retriever in args.retrievers:
        metrics["dh_rag"][retriever] = {}
        metrics["targeted"][retriever] = {}
        
        for domain in DOMAINS:
            # DH-RAG
            dh_rag_agg = RESULTS_DIR / f"dh_rag_{domain}_{retriever}_evaluated_aggregate.csv"
            if dh_rag_agg.exists():
                metrics["dh_rag"][retriever][domain] = load_evaluation_results(dh_rag_agg)
            
            # Targeted rewrite for comparison
            targeted_agg = TARGETED_REWRITE_DIR / f"targeted_rewrite_{domain}_{retriever}_evaluated_aggregate.csv"
            if targeted_agg.exists():
                metrics["targeted"][retriever][domain] = load_evaluation_results(targeted_agg)
    
    # Generate comparison table
    for retriever in args.retrievers:
        print(f"\n{'='*80}")
        print(f"COMPARISON: DH-RAG vs Targeted Rewrite ({retriever.upper()})")
        print("="*80)
        
        comparison_data = []
        
        for domain in DOMAINS:
            dh_rag = metrics["dh_rag"][retriever].get(domain)
            targeted = metrics["targeted"][retriever].get(domain)
            analysis = load_analysis_stats(domain)
            
            row = {"Domain": domain.capitalize()}
            
            if targeted:
                row["Targeted nDCG@10"] = f"{targeted['ndcg_cut_10']:.4f}"
                row["Targeted R@10"] = f"{targeted['recall_10']:.4f}"
            else:
                row["Targeted nDCG@10"] = "N/A"
                row["Targeted R@10"] = "N/A"
            
            if dh_rag:
                row["DH-RAG nDCG@10"] = f"{dh_rag['ndcg_cut_10']:.4f}"
                row["DH-RAG R@10"] = f"{dh_rag['recall_10']:.4f}"
                
                if targeted:
                    delta_ndcg = dh_rag['ndcg_cut_10'] - targeted['ndcg_cut_10']
                    pct_change = (delta_ndcg / targeted['ndcg_cut_10']) * 100 if targeted['ndcg_cut_10'] > 0 else 0
                    
                    row["Δ nDCG@10"] = f"{delta_ndcg:+.4f}"
                    row["Δ %"] = f"{pct_change:+.1f}%"
                else:
                    row["Δ nDCG@10"] = "N/A"
                    row["Δ %"] = "N/A"
            else:
                row["DH-RAG nDCG@10"] = "N/A"
                row["DH-RAG R@10"] = "N/A"
                row["Δ nDCG@10"] = "N/A"
                row["Δ %"] = "N/A"
            
            comparison_data.append(row)
        
        if comparison_data:
            df = pd.DataFrame(comparison_data)
            print("\n" + df.to_string(index=False))
        else:
            print("\nNo results to compare. Run evaluations first.")
    
    # Detailed DH-RAG analysis
    print("\n" + "="*80)
    print("DH-RAG TURN FILTERING & CLUSTERING ANALYSIS")
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
            print(f"  Avg clusters: {analysis['avg_clusters']}")
            print(f"  Avg chains: {analysis['avg_chains']}")
    
    print("\n" + "="*80)
    print("Done!")
    print("="*80)


if __name__ == "__main__":
    main()

