#!/usr/bin/env python3
"""
Evaluate context summarization retrieval results and compare with baselines.

This script:
1. Evaluates retrieval results using the standard evaluation script
2. Compares performance against baseline methods (last_turn, rewrite)
3. Generates comparison tables and analysis

Usage:
    python evaluate_and_compare.py --domains clapnq --retrievers bm25 bge
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from collections import defaultdict

script_dir = Path(__file__).parent
project_root = script_dir.parents[3]

# Paths
RESULTS_DIR = script_dir / "retrieval_results"
EVAL_SCRIPT = project_root / "scripts" / "evaluation" / "evaluate_retrieval.py"
QRELS_DIR = project_root / "human" / "retrieval_tasks"
BASELINE_RESULTS_DIR = project_root / "scripts" / "baselines" / "retrieval_results"

DOMAINS = ['clapnq', 'cloud', 'fiqa', 'govt']
RETRIEVERS = ['bm25', 'bge', 'elser']


def evaluate_results(domain: str, retriever: str, results_file: Path) -> dict:
    """
    Evaluate retrieval results using the standard evaluation script.
    
    Returns dict with metrics: R@1, R@3, R@5, R@10, nDCG@1, nDCG@3, nDCG@5, nDCG@10
    """
    qrels_file = QRELS_DIR / domain / "qrels" / "dev.tsv"
    
    if not qrels_file.exists():
        print(f"  Warning: Qrels file not found: {qrels_file}")
        return {}
    
    if not results_file.exists():
        print(f"  Warning: Results file not found: {results_file}")
        return {}
    
    # Run evaluation script
    cmd = [
        sys.executable,
        str(EVAL_SCRIPT),
        "--results_file", str(results_file),
        "--qrels_file", str(qrels_file),
        "--output_format", "json"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            # Parse JSON output
            metrics = json.loads(result.stdout)
            return metrics
        else:
            print(f"  Evaluation error: {result.stderr}")
            return {}
    except Exception as e:
        print(f"  Error running evaluation: {e}")
        return {}


def load_baseline_metrics(domain: str, retriever: str, method: str) -> dict:
    """Load baseline metrics from existing results."""
    # Try to find baseline results file
    baseline_file = BASELINE_RESULTS_DIR / f"{method}_{domain}_{retriever}_evaluated_aggregate.csv"
    
    if not baseline_file.exists():
        # Try alternative location
        baseline_file = BASELINE_RESULTS_DIR / f"{domain}_{method}_{retriever}_evaluated.csv"
    
    if not baseline_file.exists():
        return {}
    
    # Parse CSV (simple format: metric,value)
    metrics = {}
    try:
        with open(baseline_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    metrics[parts[0]] = float(parts[1])
    except Exception:
        pass
    
    return metrics


def format_comparison_table(results: dict) -> str:
    """Format results as a comparison table."""
    lines = []
    
    # Header
    metrics = ['R@1', 'R@3', 'R@5', 'R@10', 'nDCG@1', 'nDCG@3', 'nDCG@5', 'nDCG@10']
    header = "| Method | " + " | ".join(metrics) + " |"
    separator = "|" + "|".join(["-" * (len(m) + 2) for m in ["Method"] + metrics]) + "|"
    
    lines.append(header)
    lines.append(separator)
    
    # Rows
    for method, method_results in results.items():
        values = []
        for m in metrics:
            val = method_results.get(m, method_results.get(m.lower(), 0))
            values.append(f"{val:.3f}" if isinstance(val, float) else str(val))
        
        row = f"| {method} | " + " | ".join(values) + " |"
        lines.append(row)
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description='Evaluate and compare context summarization')
    parser.add_argument('--domains', nargs='+', default=['clapnq'],
                        choices=DOMAINS, help='Domains to evaluate')
    parser.add_argument('--retrievers', nargs='+', default=['bm25'],
                        choices=RETRIEVERS, help='Retrievers to evaluate')
    parser.add_argument('--input_suffix', type=str, default='',
                        help='Suffix for input files (e.g., "_test")')
    parser.add_argument('--compare_baselines', action='store_true',
                        help='Compare with baseline methods')
    args = parser.parse_args()
    
    print("="*70)
    print("Evaluating Context Summarization Results")
    print("="*70)
    
    all_results = defaultdict(dict)
    
    for domain in args.domains:
        print(f"\n{'='*60}")
        print(f"Domain: {domain.upper()}")
        print(f"{'='*60}")
        
        for retriever in args.retrievers:
            print(f"\n[{retriever.upper()}]")
            
            suffix = args.input_suffix
            results_file = RESULTS_DIR / f"context_summary_{domain}_{retriever}{suffix}.jsonl"
            
            if not results_file.exists():
                print(f"  Results file not found: {results_file}")
                continue
            
            # Evaluate context summarization
            metrics = evaluate_results(domain, retriever, results_file)
            
            if metrics:
                key = f"{domain}/{retriever}"
                all_results[key]["Context Summary"] = metrics
                
                print(f"  Context Summary:")
                for m, v in metrics.items():
                    print(f"    {m}: {v:.3f}")
            
            # Compare with baselines if requested
            if args.compare_baselines:
                for baseline in ['lastturn', 'rewrite']:
                    baseline_metrics = load_baseline_metrics(domain, retriever, baseline)
                    if baseline_metrics:
                        all_results[key][baseline.capitalize()] = baseline_metrics
                        
                        print(f"  {baseline.capitalize()}:")
                        for m, v in baseline_metrics.items():
                            print(f"    {m}: {v:.3f}")
    
    # Summary comparison
    if args.compare_baselines and all_results:
        print("\n" + "="*70)
        print("COMPARISON SUMMARY")
        print("="*70)
        
        for key, methods in all_results.items():
            print(f"\n{key}:")
            print(format_comparison_table(methods))
            
            # Calculate improvement over baselines
            if "Context Summary" in methods:
                summary_r5 = methods["Context Summary"].get("R@5", 0)
                
                for baseline in ["Lastturn", "Rewrite"]:
                    if baseline in methods:
                        baseline_r5 = methods[baseline].get("R@5", 0)
                        if baseline_r5 > 0:
                            improvement = ((summary_r5 - baseline_r5) / baseline_r5) * 100
                            print(f"  → R@5 improvement over {baseline}: {improvement:+.1f}%")
    
    print("\nDone!")


if __name__ == '__main__':
    main()

