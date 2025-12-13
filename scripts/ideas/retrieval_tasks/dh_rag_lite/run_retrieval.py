#!/usr/bin/env python3
"""
Run retrieval using DH-RAG rewritten queries with BM25, BGE, and ELSER.

This script takes the rewritten queries from intermediate/ and runs them
through the baseline retrieval systems.
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
    """Run BGE retrieval."""
    corpus_file = CORPUS_DIR / f"{domain}.jsonl"
    
    cmd = [
        sys.executable,
        str(BASELINE_DIR / "bge" / "bge_retrieval.py"),
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


def run_elser_retrieval(domain: str, query_file: Path, output_file: Path, top_k: int = 10):
    """Run ELSER retrieval using Elasticsearch."""
    cmd = [
        sys.executable,
        str(BASELINE_DIR / "elser" / "elser_retrieval.py"),
        "--domain", domain,
        "--query_type", "rewrite",
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


def main():
    parser = argparse.ArgumentParser(description="Run retrieval for DH-RAG rewritten queries")
    parser.add_argument("--domains", nargs="+", default=DOMAINS, help="Domains to process")
    parser.add_argument("--retrievers", nargs="+", default=RETRIEVERS, 
                        choices=RETRIEVERS, help="Retrievers to use")
    parser.add_argument("--top_k", type=int, default=10, help="Number of results to retrieve")
    parser.add_argument("--skip_existing", action="store_true", help="Skip if output exists")
    args = parser.parse_args()
    
    print("=" * 80)
    print("DH-RAG Retrieval")
    print("=" * 80)
    print(f"Domains: {args.domains}")
    print(f"Retrievers: {args.retrievers}")
    print(f"Top-K: {args.top_k}")
    print()
    
    # Create results directory
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    retriever_funcs = {
        'bm25': run_bm25_retrieval,
        'bge': run_bge_retrieval,
        'elser': run_elser_retrieval
    }
    
    for domain in args.domains:
        print(f"\n{'=' * 40}")
        print(f"Domain: {domain}")
        print("=" * 40)
        
        query_file = INTERMEDIATE_DIR / f"dh_rag_{domain}.jsonl"
        
        if not query_file.exists():
            print(f"  Warning: Query file not found: {query_file}")
            continue
        
        for retriever in args.retrievers:
            output_file = RESULTS_DIR / f"dh_rag_{domain}_{retriever}.jsonl"
            
            if args.skip_existing and output_file.exists():
                print(f"  Skipping {retriever} - output exists")
                continue
            
            print(f"\n  Running {retriever}...")
            success = retriever_funcs[retriever](domain, query_file, output_file, args.top_k)
            
            if success:
                print(f"  ✓ Saved to {output_file.name}")
            else:
                print(f"  ✗ Failed")
    
    print("\n" + "=" * 80)
    print("Retrieval complete!")
    print("=" * 80)
    print(f"\nResults saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()

