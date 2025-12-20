#!/usr/bin/env python3
"""
Run retrieval using context summarization queries with BM25, BGE, and ELSER.

This script takes the summarized queries from intermediate/ and runs them
through the baseline retrieval systems.

Usage:
    python run_retrieval.py --domains clapnq --retrievers bm25 bge
"""

import argparse
import subprocess
import sys
from pathlib import Path

script_dir = Path(__file__).parent
project_root = script_dir.parents[3]

# Paths
INTERMEDIATE_DIR = script_dir / "intermediate"
RESULTS_DIR = script_dir / "retrieval_results"
BASELINE_DIR = project_root / "scripts" / "baselines" / "retrieval_scripts"
CORPUS_DIR = project_root / "corpora" / "passage_level"

DOMAINS = ['clapnq', 'cloud', 'fiqa', 'govt']
RETRIEVERS = ['bm25', 'bge', 'elser']


def run_bm25_retrieval(domain: str, query_file: Path, output_file: Path, top_k: int = 10):
    """Run BM25 retrieval using PyTerrier."""
    corpus_file = CORPUS_DIR / f"{domain}.jsonl"
    
    # Note: --query_type is required by baseline but not actually used when --query_file is provided
    # We pass "rewrite" as a valid placeholder
    cmd = [
        sys.executable,
        str(BASELINE_DIR / "bm25" / "bm25_retrieval.py"),
        "--domain", domain,
        "--query_type", "rewrite",
        "--corpus_file", str(corpus_file),
        "--query_file", str(query_file),
        "--output_file", str(output_file),
        "--top_k", str(top_k)
    ]
    
    print(f"  Running: {' '.join(cmd[:4])}...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr}")
        return False
    return True


def run_bge_retrieval(domain: str, query_file: Path, output_file: Path, top_k: int = 10):
    """Run BGE dense retrieval."""
    corpus_file = CORPUS_DIR / f"{domain}.jsonl"
    model_path = BASELINE_DIR / "bge" / "models" / "bge-base-en-v1.5"
    
    # Note: --query_type is required by baseline but not actually used when --query_file is provided
    cmd = [
        sys.executable,
        str(BASELINE_DIR / "bge" / "bge_retrieval.py"),
        "--domain", domain,
        "--query_type", "rewrite",
        "--corpus_file", str(corpus_file),
        "--query_file", str(query_file),
        "--output_file", str(output_file),
        "--top_k", str(top_k),
        "--model_path", str(model_path)
    ]
    
    print(f"  Running: {' '.join(cmd[:4])}...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr}")
        return False
    return True


def run_elser_retrieval(domain: str, query_file: Path, output_file: Path, top_k: int = 10):
    """Run ELSER retrieval via Elasticsearch."""
    # Note: --query_type is required by baseline but not actually used when --query_file is provided
    cmd = [
        sys.executable,
        str(BASELINE_DIR / "elser" / "elser_retrieval.py"),
        "--domain", domain,
        "--query_type", "rewrite",
        "--query_file", str(query_file),
        "--output_file", str(output_file),
        "--top_k", str(top_k),
        "--delay", "0.3"
    ]
    
    print(f"  Running: {' '.join(cmd[:4])}...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr}")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description='Run retrieval with context summary queries')
    parser.add_argument('--domains', nargs='+', default=['clapnq'],
                        choices=DOMAINS, help='Domains to process')
    parser.add_argument('--retrievers', nargs='+', default=RETRIEVERS,
                        choices=RETRIEVERS, help='Retrievers to use')
    parser.add_argument('--top_k', type=int, default=10,
                        help='Number of results to retrieve')
    parser.add_argument('--skip_existing', action='store_true',
                        help='Skip if output file already exists')
    parser.add_argument('--input_suffix', type=str, default='',
                        help='Suffix for input files (e.g., "_test")')
    args = parser.parse_args()
    
    print("="*70)
    print("Running Retrieval with Context Summary Queries")
    print("="*70)
    print(f"Domains: {args.domains}")
    print(f"Retrievers: {args.retrievers}")
    print(f"Top-k: {args.top_k}")
    print()
    
    # Create results directory
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    retriever_funcs = {
        'bm25': run_bm25_retrieval,
        'bge': run_bge_retrieval,
        'elser': run_elser_retrieval
    }
    
    results_summary = []
    
    for domain in args.domains:
        suffix = args.input_suffix
        query_file = INTERMEDIATE_DIR / f"context_summary_{domain}{suffix}.jsonl"
        
        if not query_file.exists():
            print(f"\n[{domain}] Query file not found: {query_file}")
            print(f"  Run run_context_summarization.py first")
            continue
        
        print(f"\n{'='*70}")
        print(f"Domain: {domain.upper()}")
        print(f"{'='*70}")
        
        for retriever in args.retrievers:
            output_file = RESULTS_DIR / f"context_summary_{domain}_{retriever}{suffix}.jsonl"
            
            if args.skip_existing and output_file.exists():
                print(f"\n[{retriever.upper()}] Skipping (already exists): {output_file.name}")
                results_summary.append((domain, retriever, "skipped"))
                continue
            
            print(f"\n[{retriever.upper()}] Running retrieval...")
            
            retriever_func = retriever_funcs[retriever]
            success = retriever_func(domain, query_file, output_file, args.top_k)
            
            if success:
                print(f"  ✓ Saved to: {output_file.name}")
                results_summary.append((domain, retriever, "success"))
            else:
                print(f"  ✗ Failed")
                results_summary.append((domain, retriever, "failed"))
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for domain, retriever, status in results_summary:
        emoji = "✓" if status == "success" else ("⊘" if status == "skipped" else "✗")
        print(f"  {emoji} {domain}/{retriever}: {status}")
    
    print(f"\nResults saved to: {RESULTS_DIR}/")


if __name__ == '__main__':
    main()

