#!/usr/bin/env python3
"""
Two-Stage BM25 + ELSER Reranking Retrieval

This script implements a two-stage retrieval approach:
1. Stage 1: BM25 retrieves top 50 candidates locally (high recall)
2. Stage 2: ELSER reranks candidates via Elasticsearch filter query (high precision)

Usage:
    python run_bm25_elser_rerank.py                           # Run all domains and query types
    python run_bm25_elser_rerank.py --domain clapnq           # Run specific domain
    python run_bm25_elser_rerank.py --query_type rewrite      # Run specific query type
    python run_bm25_elser_rerank.py --skip-existing           # Skip if output exists

Requires:
    - ES_URL and ES_API_KEY environment variables (or .env file)
"""

import os
import sys
import argparse
import subprocess
import time
import warnings
from pathlib import Path
from tqdm import tqdm

import pyterrier as pt
import pandas as pd
from elasticsearch import Elasticsearch

from utils import (
    load_corpus,
    load_queries,
    save_results,
    format_results_for_eval,
    DOMAINS,
    QUERY_TYPES,
    EVAL_SCRIPT_PATH,
    COLLECTION_NAMES
)

# Suppress deprecation warnings for text_expansion
warnings.filterwarnings('ignore', category=Warning)

# Load environment variables
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parents[4] / '.env'
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass

# Configuration
BM25_TOP_K = 50  # Number of candidates from BM25 (Stage 1)
FINAL_TOP_K = 10  # Number of results after reranking (Stage 2)
ES_DELAY = 0.3  # Delay between ES queries to avoid rate limiting

# Paths
script_dir = Path(__file__).parent
INTERMEDIATE_DIR = script_dir / 'intermediate'
RESULTS_DIR = script_dir / 'results'

# Elasticsearch index names
ES_INDEX_NAMES = {
    'clapnq': 'mtrag-clapnq-elser-512-100-reindexed',
    'cloud': 'mtrag-cloud-elser-512-100',
    'fiqa': 'mtrag-fiqa-elser-512-100',
    'govt': 'mtrag-govt-elser-512-100'
}


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
    
    Returns:
        Dictionary mapping query_id to list of doc_ids
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
    index_path = str(INTERMEDIATE_DIR / "pyterrier_index_elser")
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
    
    # Format results as list of doc_ids per query
    results_dict = {qid: [] for qid in queries.keys()}
    for _, row in results_df.iterrows():
        qid = row['qid']
        docno = row['docno']
        results_dict[qid].append(docno)
    
    return results_dict


def run_elser_rerank_stage(
    es: Elasticsearch,
    index_name: str,
    queries: dict,
    bm25_candidates: dict,
    corpus: dict,
    top_k: int = 10,
    delay: float = 0.3
) -> dict:
    """
    Stage 2: Rerank BM25 candidates using ELSER via Elasticsearch.
    
    Args:
        es: Elasticsearch client
        index_name: ELSER index name
        queries: Dictionary of query_id -> query_text
        bm25_candidates: Dictionary mapping query_id to list of candidate doc_ids
        corpus: Full corpus for getting document content
        top_k: Number of final results
        delay: Delay between queries for rate limiting
        
    Returns:
        Dictionary mapping query_id to {doc_id: score}
    """
    print(f"Reranking {len(bm25_candidates)} queries with ELSER...")
    
    results_dict = {}
    
    for i, qid in enumerate(tqdm(queries.keys(), desc="ELSER Reranking")):
        query_text = queries[qid]
        candidate_ids = bm25_candidates.get(qid, [])
        
        if not candidate_ids:
            results_dict[qid] = {}
            continue
        
        # Build ELSER query with filter on candidate doc IDs
        query_body = {
            "query": {
                "bool": {
                    "must": {
                        "text_expansion": {
                            "ml.tokens": {
                                "model_id": ".elser-2-elastic",
                                "model_text": query_text
                            }
                        }
                    },
                    "filter": {
                        "ids": {
                            "values": candidate_ids
                        }
                    }
                }
            },
            "size": top_k,
            "_source": ["text", "title", "url"]
        }
        
        # Retry logic for rate limits
        max_retries = 3
        retry_delay = 2.0
        
        for attempt in range(max_retries):
            try:
                response = es.search(index=index_name, body=query_body)
                
                # Extract scores
                doc_scores = {}
                for hit in response['hits']['hits']:
                    doc_id = hit['_id']
                    score = float(hit['_score'])
                    doc_scores[doc_id] = score
                
                results_dict[qid] = doc_scores
                break
                
            except Exception as e:
                error_msg = str(e)
                
                if '429' in error_msg or 'rate limit' in error_msg.lower():
                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (attempt + 1)
                        print(f"\n  Rate limit hit, waiting {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    else:
                        print(f"\n  Warning: Query {qid} failed after {max_retries} retries")
                        results_dict[qid] = {}
                else:
                    print(f"\n  Warning: Query {qid} failed: {e}")
                    results_dict[qid] = {}
                    break
        
        # Rate limiting delay
        if i < len(queries) - 1:
            time.sleep(delay)
    
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
        description='Two-Stage BM25 + ELSER Reranking Retrieval'
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
        '--delay',
        type=float,
        default=ES_DELAY,
        help=f'Delay between ES queries in seconds (default: {ES_DELAY})'
    )
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='Skip if output file already exists'
    )
    
    args = parser.parse_args()
    
    # Check Elasticsearch credentials
    es_url = os.getenv('ES_URL')
    api_key = os.getenv('ES_API_KEY')
    
    if not es_url or not api_key:
        print("❌ ES_URL and ES_API_KEY environment variables must be set")
        print("   Make sure they're in your .env file")
        sys.exit(1)
    
    # Connect to Elasticsearch
    print("Connecting to Elasticsearch...")
    es = Elasticsearch(es_url, api_key=api_key, request_timeout=60)
    
    if not es.ping():
        print("❌ Failed to connect to Elasticsearch")
        sys.exit(1)
    
    print("✅ Connected to Elasticsearch\n")
    
    # Determine what to process
    domains = [args.domain] if args.domain else DOMAINS
    query_types = [args.query_type] if args.query_type else QUERY_TYPES
    
    # Create output directories
    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Two-Stage BM25 + ELSER Reranking Retrieval")
    print("=" * 80)
    print(f"Domains: {domains}")
    print(f"Query types: {query_types}")
    print(f"BM25 top-k: {args.bm25_top_k}")
    print(f"Final top-k: {args.final_top_k}")
    print(f"ES delay: {args.delay}s")
    print("=" * 80)
    
    # Track results summary
    results_summary = []
    
    for domain in domains:
        print(f"\n{'='*60}")
        print(f"Processing domain: {domain.upper()}")
        print(f"{'='*60}")
        
        # Get ES index name
        index_name = ES_INDEX_NAMES.get(domain)
        if not index_name:
            print(f"  ❌ No ES index configured for domain: {domain}")
            continue
        
        print(f"Using ES index: {index_name}")
        
        # Load corpus once per domain
        print(f"Loading corpus for {domain}...")
        corpus = load_corpus(domain)
        print(f"Loaded {len(corpus)} documents")
        
        for query_type in query_types:
            print(f"\n--- Query type: {query_type} ---")
            
            output_file = RESULTS_DIR / f"bm25_elser_rerank_{domain}_{query_type}.jsonl"
            evaluated_file = RESULTS_DIR / f"bm25_elser_rerank_{domain}_{query_type}_evaluated.jsonl"
            
            # Check if we should skip
            if args.skip_existing and output_file.exists():
                print(f"Output file exists, skipping: {output_file.name}")
                continue
            
            # Load queries
            queries = load_queries(domain, query_type)
            print(f"Loaded {len(queries)} queries")
            
            # Stage 1: BM25 retrieval
            print("\n[Stage 1] BM25 retrieval...")
            bm25_candidates = run_bm25_stage(corpus, queries, top_k=args.bm25_top_k)
            
            # Stage 2: ELSER reranking
            print("\n[Stage 2] ELSER reranking...")
            reranked_results = run_elser_rerank_stage(
                es, index_name, queries, bm25_candidates, corpus,
                top_k=args.final_top_k, delay=args.delay
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
        summary_file = RESULTS_DIR / "bm25_elser_experiment_summary.csv"
        summary_df.to_csv(summary_file, index=False)
        print(f"\nSummary saved to: {summary_file}")
    
    print("\n" + "=" * 80)
    print("Two-stage BM25+ELSER retrieval experiment complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()

