#!/usr/bin/env python3
"""
Two-Stage BM25 + BGE Reranking Retrieval

This script implements a two-stage retrieval approach:
1. Stage 1: BM25 retrieves top 50 candidates (high recall)
2. Stage 2: BGE reranks candidates to get top 10 (high precision)

Usage:
    python run_two_stage_retrieval.py                           # Run all domains and query types
    python run_two_stage_retrieval.py --domain clapnq           # Run specific domain
    python run_two_stage_retrieval.py --query_type rewrite      # Run specific query type
    python run_two_stage_retrieval.py --skip-existing           # Skip if output exists
"""

import sys
import argparse
import subprocess
from pathlib import Path
from tqdm import tqdm

import pyterrier as pt
import pandas as pd

from utils import (
    load_corpus,
    load_queries,
    save_results,
    format_results_for_eval,
    BGEReranker,
    DOMAINS,
    QUERY_TYPES,
    EVAL_SCRIPT_PATH
)

# Configuration
BM25_TOP_K = 50  # Number of candidates from BM25 (Stage 1)
FINAL_TOP_K = 10  # Number of results after reranking (Stage 2)
BGE_BATCH_SIZE = 32

# Paths
script_dir = Path(__file__).parent
INTERMEDIATE_DIR = script_dir / 'intermediate'
RESULTS_DIR = script_dir / 'results'


def escape_query(query: str) -> str:
    """Clean special characters in PyTerrier query to avoid parser errors."""
    special_chars = ['+', '-', '=', '&&', '||', '>', '<', '!', '(', ')', 
                    '{', '}', '[', ']', '^', '"', '~', '*', '?', ':', '\\', '/', '\'']
    
    for char in special_chars:
        query = query.replace(char, ' ')
    
    return ' '.join(query.split())


def run_bm25_stage(corpus: dict, queries: dict, top_k: int = 50) -> dict:
    """
    Stage 1: Run BM25 retrieval to get initial candidates.
    
    Args:
        corpus: Dictionary of documents
        queries: Dictionary of query_id -> query_text
        top_k: Number of candidates to retrieve
        
    Returns:
        Dictionary mapping query_id to list of {doc_id, text, score, ...}
    """
    # Initialize PyTerrier if not already initialized
    if not pt.started():
        pt.init()
    
    print(f"Building BM25 index for {len(corpus)} documents...")
    
    # Convert corpus to PyTerrier format
    corpus_list = []
    for doc_id, doc in corpus.items():
        corpus_list.append({
            'docno': doc_id,
            'text': doc.get('text', '')
        })
    corpus_df = pd.DataFrame(corpus_list)
    
    # Create index
    index_path = str(INTERMEDIATE_DIR / "pyterrier_index")
    Path(index_path).mkdir(parents=True, exist_ok=True)
    indexer = pt.IterDictIndexer(index_path, overwrite=True, meta={'docno': 50})
    index_ref = indexer.index(corpus_df.to_dict('records'))
    
    print(f"Running BM25 retrieval for {len(queries)} queries...")
    
    # Convert queries to PyTerrier format
    queries_list = []
    for qid, text in queries.items():
        clean_text = escape_query(text)
        if not clean_text.strip():
            clean_text = "empty_query_placeholder"
        queries_list.append({'qid': qid, 'query': clean_text})
    queries_df = pd.DataFrame(queries_list)
    
    # Create BM25 retriever
    bm25 = pt.BatchRetrieve(index_ref, wmodel="BM25", num_results=top_k)
    
    # Run retrieval
    try:
        results_df = bm25.transform(queries_df)
    except Exception as e:
        print(f"Error during retrieval: {e}")
        results_df = pd.DataFrame(columns=['qid', 'docno', 'score'])
    
    # Format results as list of documents per query
    results_dict = {qid: [] for qid in queries.keys()}
    for _, row in results_df.iterrows():
        qid = row['qid']
        docno = row['docno']
        score = row['score']
        doc = corpus.get(docno, {})
        
        results_dict[qid].append({
            'document_id': docno,
            'text': doc.get('text', ''),
            'title': doc.get('title', ''),
            'source': doc.get('url', ''),
            'bm25_score': score
        })
    
    return results_dict


def run_bge_rerank_stage(
    reranker: BGEReranker,
    queries: dict,
    bm25_results: dict,
    top_k: int = 10
) -> dict:
    """
    Stage 2: Rerank BM25 candidates using BGE.
    
    Args:
        reranker: BGEReranker instance
        queries: Dictionary of query_id -> query_text
        bm25_results: Dictionary mapping query_id to list of candidate docs
        top_k: Number of final results
        
    Returns:
        Dictionary mapping query_id to {doc_id: score}
    """
    print(f"Reranking {len(bm25_results)} queries with BGE...")
    
    results_dict = {}
    
    for qid in tqdm(queries.keys(), desc="BGE Reranking"):
        query_text = queries[qid]
        candidates = bm25_results.get(qid, [])
        
        if not candidates:
            results_dict[qid] = {}
            continue
        
        # Rerank with BGE
        reranked = reranker.rerank(query_text, candidates, top_k=top_k)
        
        # Format as {doc_id: score}
        results_dict[qid] = {
            doc['document_id']: score
            for doc, score in reranked
        }
    
    return results_dict


def run_evaluation(output_file: Path, evaluated_file: Path):
    """Run the standard evaluation script on results."""
    print("Running evaluation...")
    
    cmd = [
        sys.executable, str(EVAL_SCRIPT_PATH),
        "--input_file", str(output_file),
        "--output_file", str(evaluated_file)
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        # Print aggregate results
        agg_file = Path(str(evaluated_file.with_suffix('')) + "_aggregate.csv")
        if agg_file.exists():
            import ast
            df = pd.read_csv(agg_file)
            all_rows = df[df['collection'] == 'all']
            if len(all_rows) > 0:
                all_row = all_rows.iloc[0]
                ndcg = ast.literal_eval(all_row['nDCG'])
                recall = ast.literal_eval(all_row['Recall'])
                print(f"  nDCG@10: {ndcg[3]:.4f}, Recall@10: {recall[3]:.4f}")
                return {'nDCG@10': ndcg[3], 'Recall@10': recall[3]}
    except subprocess.CalledProcessError as e:
        print(f"Error running evaluation: {e}")
        if e.stderr:
            print(f"Stderr: {e.stderr}")
    
    return None


def main():
    parser = argparse.ArgumentParser(
        description='Two-Stage BM25 + BGE Reranking Retrieval'
    )
    parser.add_argument(
        '--domain',
        type=str,
        choices=DOMAINS,
        default=None,
        help='Domain to process (default: all domains)'
    )
    parser.add_argument(
        '--query_type',
        type=str,
        choices=QUERY_TYPES,
        default=None,
        help='Query type to process (default: all query types)'
    )
    parser.add_argument(
        '--bm25_top_k',
        type=int,
        default=BM25_TOP_K,
        help=f'Number of BM25 candidates (default: {BM25_TOP_K})'
    )
    parser.add_argument(
        '--final_top_k',
        type=int,
        default=FINAL_TOP_K,
        help=f'Number of final results (default: {FINAL_TOP_K})'
    )
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='Skip if output file already exists'
    )
    
    args = parser.parse_args()
    
    # Determine what to process
    domains = [args.domain] if args.domain else DOMAINS
    query_types = [args.query_type] if args.query_type else QUERY_TYPES
    
    # Create output directories
    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Two-Stage BM25 + BGE Reranking Retrieval")
    print("=" * 80)
    print(f"Domains: {domains}")
    print(f"Query types: {query_types}")
    print(f"BM25 top-k: {args.bm25_top_k}")
    print(f"Final top-k: {args.final_top_k}")
    print("=" * 80)
    
    # Initialize BGE reranker (once for all experiments)
    reranker = BGEReranker()
    
    # Track results summary
    results_summary = []
    
    for domain in domains:
        print(f"\n{'='*60}")
        print(f"Processing domain: {domain.upper()}")
        print(f"{'='*60}")
        
        # Load corpus once per domain
        print(f"Loading corpus for {domain}...")
        corpus = load_corpus(domain)
        print(f"Loaded {len(corpus)} documents")
        
        for query_type in query_types:
            print(f"\n--- Query type: {query_type} ---")
            
            output_file = RESULTS_DIR / f"bm25_bge_rerank_{domain}_{query_type}.jsonl"
            evaluated_file = RESULTS_DIR / f"bm25_bge_rerank_{domain}_{query_type}_evaluated.jsonl"
            
            # Check if we should skip
            if args.skip_existing and output_file.exists():
                print(f"Output file exists, skipping: {output_file.name}")
                continue
            
            # Load queries
            queries = load_queries(domain, query_type)
            print(f"Loaded {len(queries)} queries")
            
            # Stage 1: BM25 retrieval
            print("\n[Stage 1] BM25 retrieval...")
            bm25_results = run_bm25_stage(corpus, queries, top_k=args.bm25_top_k)
            
            # Save intermediate BM25 results
            bm25_intermediate_file = INTERMEDIATE_DIR / f"bm25_top{args.bm25_top_k}_{domain}_{query_type}.jsonl"
            bm25_formatted = format_results_for_eval(
                {qid: {doc['document_id']: doc['bm25_score'] for doc in docs}
                 for qid, docs in bm25_results.items()},
                domain,
                corpus
            )
            save_results(bm25_formatted, bm25_intermediate_file)
            
            # Stage 2: BGE reranking
            print("\n[Stage 2] BGE reranking...")
            reranked_results = run_bge_rerank_stage(
                reranker, queries, bm25_results, top_k=args.final_top_k
            )
            
            # Save final results
            final_results = format_results_for_eval(reranked_results, domain, corpus)
            save_results(final_results, output_file)
            
            # Evaluate
            metrics = run_evaluation(output_file, evaluated_file)
            
            if metrics:
                results_summary.append({
                    'domain': domain,
                    'query_type': query_type,
                    **metrics
                })
    
    # Print summary
    if results_summary:
        print("\n" + "=" * 80)
        print("RESULTS SUMMARY")
        print("=" * 80)
        summary_df = pd.DataFrame(results_summary)
        print(summary_df.to_string(index=False))
        
        # Save summary
        summary_file = RESULTS_DIR / "experiment_summary.csv"
        summary_df.to_csv(summary_file, index=False)
        print(f"\nSummary saved to: {summary_file}")
    
    print("\n" + "=" * 80)
    print("Two-stage retrieval experiment complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()

