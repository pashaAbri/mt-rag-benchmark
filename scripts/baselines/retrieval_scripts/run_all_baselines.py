"""
Run all retrieval baselines on MT-RAG benchmark

This script runs BM25, BGE-base 1.5, and Elser retrieval on all domains
and query types, then evaluates the results.
"""
import argparse
import subprocess
from pathlib import Path


DOMAINS = ['clapnq', 'fiqa', 'govt', 'cloud']
QUERY_TYPES = ['lastturn', 'rewrite']


def run_baseline(retriever: str, domain: str, query_type: str, 
                 corpus_file: str, query_file: str, output_dir: str):
    """
    Run a single retrieval baseline.
    
    Args:
        retriever: Type of retriever (bm25, bge, elser)
        domain: Domain name
        query_type: Query type (lastturn, rewrite)
        corpus_file: Path to corpus file
        query_file: Path to query file
        output_dir: Output directory for results
    """
    output_file = Path(output_dir) / f"{retriever}_{domain}_{query_type}.jsonl"
    
    script_map = {
        'bm25': 'bm25_retrieval.py',
        'bge': 'bge_retrieval.py',
        'elser': 'elser_retrieval.py'
    }
    
    script = script_map[retriever]
    
    cmd = [
        'python', f'scripts/baselines/retrieval_scripts/{script}',
        '--domain', domain,
        '--query_type', query_type,
        '--corpus_file', corpus_file,
        '--query_file', query_file,
        '--output_file', str(output_file),
        '--top_k', '10'
    ]
    
    print(f"\n{'='*80}")
    print(f"Running {retriever.upper()} on {domain} with {query_type} queries")
    print(f"{'='*80}")
    
    try:
        subprocess.run(cmd, check=True)
        print(f"✓ Completed: {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed: {e}")
    except NotImplementedError:
        print(f"⚠ Not yet implemented: {retriever}")


def process_domain_queries(retriever, domain, query_types, args):
    """Process all query types for a given retriever and domain."""
    corpus_file = Path(args.corpus_dir) / f"{domain}.jsonl"
    
    if not corpus_file.exists():
        print(f"⚠ Corpus file not found: {corpus_file}")
        return
    
    for query_type in query_types:
        query_file = Path(args.query_dir) / domain / f"{domain}_{query_type}.jsonl"
        
        if not query_file.exists():
            print(f"⚠ Query file not found: {query_file}")
            continue
        
        run_baseline(retriever, domain, query_type, 
                   str(corpus_file), str(query_file), args.output_dir)


def main():
    parser = argparse.ArgumentParser(description='Run all retrieval baselines')
    parser.add_argument('--retrievers', nargs='+', 
                        choices=['bm25', 'bge', 'elser', 'all'],
                        default=['all'],
                        help='Which retrievers to run')
    parser.add_argument('--domains', nargs='+',
                        choices=DOMAINS + ['all'],
                        default=['all'],
                        help='Which domains to run on')
    parser.add_argument('--query_types', nargs='+',
                        choices=QUERY_TYPES + ['all'],
                        default=['all'],
                        help='Which query types to use')
    parser.add_argument('--corpus_dir', type=str,
                        default='corpora/passage_level',
                        help='Directory containing corpus files')
    parser.add_argument('--query_dir', type=str,
                        default='human/retrieval_tasks',
                        help='Directory containing query files')
    parser.add_argument('--output_dir', type=str,
                        default='scripts/baselines/retrieval_scripts/bm25/results',
                        help='Output directory for results (will be updated per retriever)')
    
    args = parser.parse_args()
    
    # Expand 'all' options
    retrievers = ['bm25', 'bge', 'elser'] if 'all' in args.retrievers else args.retrievers
    domains = DOMAINS if 'all' in args.domains else args.domains
    query_types = QUERY_TYPES if 'all' in args.query_types else args.query_types
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    print("Running retrieval baselines:")
    print(f"  Retrievers: {retrievers}")
    print(f"  Domains: {domains}")
    print(f"  Query types: {query_types}")
    
    # Run all combinations
    for retriever in retrievers:
        for domain in domains:
            process_domain_queries(retriever, domain, query_types, args)
    
    print(f"\n{'='*80}")
    print(f"All baselines complete! Results saved to {args.output_dir}")
    print('='*80)


if __name__ == '__main__':
    main()

