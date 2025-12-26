#!/usr/bin/env python3
"""
Two-Stage BM25 + ELSER Reranking with Targeted Rewrite Queries

This script runs the two-stage retrieval using the targeted rewrite methodology:
1. Stage 1: BM25 retrieves top-k candidates locally
2. Stage 2: ELSER reranks candidates via Elasticsearch

Usage:
    python run_bm25_elser_targeted.py                    # Run all domains
    python run_bm25_elser_targeted.py --domain clapnq    # Run specific domain
    python run_bm25_elser_targeted.py --bm25_top_k 500   # Custom pool size
"""

import os
import sys
import json
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
    save_results,
    format_results_for_eval,
    DOMAINS,
    EVAL_SCRIPT_PATH,
    COLLECTION_NAMES
)

# Suppress deprecation warnings
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
BM25_TOP_K = 500  # Use 500 as default based on previous experiments
FINAL_TOP_K = 10
ES_DELAY = 0.3

# Paths
script_dir = Path(__file__).parent
project_root = script_dir.parents[3]
TARGETED_QUERIES_DIR = project_root / 'scripts' / 'ideas' / 'retrieval_tasks' / 'targeted_rewrite' / 'intermediate'
INTERMEDIATE_DIR = script_dir / 'intermediate'
RESULTS_DIR = script_dir / 'results'

# Elasticsearch index names
ES_INDEX_NAMES = {
    'clapnq': 'mtrag-clapnq-elser-512-100-reindexed',
    'cloud': 'mtrag-cloud-elser-512-100',
    'fiqa': 'mtrag-fiqa-elser-512-100',
    'govt': 'mtrag-govt-elser-512-100'
}


def load_targeted_queries(domain: str) -> dict:
    """Load targeted rewrite queries for a domain."""
    query_file = TARGETED_QUERIES_DIR / f'targeted_rewrite_{domain}.jsonl'
    
    if not query_file.exists():
        raise FileNotFoundError(f"Targeted queries not found: {query_file}")
    
    queries = {}
    with open(query_file, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            query_id = data.get('_id')
            query_text = data.get('text', '')
            if query_id:
                # Clean query text
                query_text = query_text.replace('|user|: ', '').replace('|user|:', '').strip()
                queries[query_id] = query_text
    
    return queries


def escape_query(query: str) -> str:
    """Clean special characters in PyTerrier query."""
    special_chars = ['+', '-', '=', '&&', '||', '>', '<', '!', '(', ')', 
                    '{', '}', '[', ']', '^', '"', '~', '*', '?', ':', '\\', '/', '\'']
    for char in special_chars:
        query = query.replace(char, ' ')
    return ' '.join(query.split())


def run_bm25_stage(corpus: dict, queries: dict, top_k: int = 500) -> dict:
    """Stage 1: BM25 retrieval."""
    if not pt.started():
        pt.init()
    
    print(f"Building BM25 index for {len(corpus)} documents...")
    
    corpus_list = [{'docno': doc_id, 'text': doc.get('text', '')} 
                   for doc_id, doc in corpus.items()]
    corpus_df = pd.DataFrame(corpus_list)
    
    index_path = str(INTERMEDIATE_DIR / "pyterrier_index_targeted")
    Path(index_path).mkdir(parents=True, exist_ok=True)
    indexer = pt.IterDictIndexer(index_path, overwrite=True, meta={'docno': 50})
    index_ref = indexer.index(corpus_df.to_dict('records'))
    
    print(f"Running BM25 retrieval for {len(queries)} queries...")
    
    queries_list = []
    for qid, text in queries.items():
        clean_text = escape_query(text)
        if not clean_text.strip():
            clean_text = "empty_query_placeholder"
        queries_list.append({'qid': qid, 'query': clean_text})
    queries_df = pd.DataFrame(queries_list)
    
    bm25 = pt.BatchRetrieve(index_ref, wmodel="BM25", num_results=top_k)
    
    try:
        results_df = bm25.transform(queries_df)
    except Exception as e:
        print(f"Error during retrieval: {e}")
        results_df = pd.DataFrame(columns=['qid', 'docno', 'score'])
    
    results_dict = {qid: [] for qid in queries.keys()}
    for _, row in results_df.iterrows():
        results_dict[row['qid']].append(row['docno'])
    
    return results_dict


def run_elser_rerank_stage(es, index_name, queries, bm25_candidates, top_k=10, delay=0.3):
    """Stage 2: ELSER reranking."""
    print(f"Reranking {len(bm25_candidates)} queries with ELSER...")
    
    results_dict = {}
    
    for i, qid in enumerate(tqdm(queries.keys(), desc="ELSER Reranking")):
        query_text = queries[qid]
        candidate_ids = bm25_candidates.get(qid, [])
        
        if not candidate_ids:
            results_dict[qid] = {}
            continue
        
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
                        "ids": {"values": candidate_ids}
                    }
                }
            },
            "size": top_k,
            "_source": ["text", "title", "url"]
        }
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = es.search(index=index_name, body=query_body)
                doc_scores = {hit['_id']: float(hit['_score']) 
                             for hit in response['hits']['hits']}
                results_dict[qid] = doc_scores
                break
            except Exception as e:
                if '429' in str(e) and attempt < max_retries - 1:
                    time.sleep(2.0 * (attempt + 1))
                    continue
                print(f"\n  Warning: Query {qid} failed: {e}")
                results_dict[qid] = {}
                break
        
        if i < len(queries) - 1:
            time.sleep(delay)
    
    return results_dict


def run_evaluation(output_file, evaluated_file):
    """Run evaluation."""
    print("Running evaluation...")
    
    cmd = [sys.executable, str(EVAL_SCRIPT_PATH),
           "--input_file", str(output_file),
           "--output_file", str(evaluated_file)]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        
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
    
    return None


def main():
    parser = argparse.ArgumentParser(description='Two-Stage BM25+ELSER with Targeted Rewrite')
    parser.add_argument('--domain', type=str, choices=DOMAINS, default=None)
    parser.add_argument('--bm25_top_k', type=int, default=BM25_TOP_K)
    parser.add_argument('--delay', type=float, default=ES_DELAY)
    parser.add_argument('--skip-existing', action='store_true')
    
    args = parser.parse_args()
    
    # Check ES credentials
    es_url = os.getenv('ES_URL')
    api_key = os.getenv('ES_API_KEY')
    
    if not es_url or not api_key:
        print("❌ ES_URL and ES_API_KEY must be set")
        sys.exit(1)
    
    print("Connecting to Elasticsearch...")
    es = Elasticsearch(es_url, api_key=api_key, request_timeout=60)
    
    if not es.ping():
        print("❌ Failed to connect to Elasticsearch")
        sys.exit(1)
    
    print("✅ Connected to Elasticsearch\n")
    
    domains = [args.domain] if args.domain else DOMAINS
    
    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Two-Stage BM25 + ELSER with TARGETED REWRITE Queries")
    print("=" * 80)
    print(f"Domains: {domains}")
    print(f"BM25 top-k: {args.bm25_top_k}")
    print("=" * 80)
    
    results_summary = []
    
    for domain in domains:
        print(f"\n{'='*60}")
        print(f"Processing domain: {domain.upper()}")
        print(f"{'='*60}")
        
        index_name = ES_INDEX_NAMES.get(domain)
        if not index_name:
            print(f"  ❌ No ES index for domain: {domain}")
            continue
        
        output_file = RESULTS_DIR / f"bm25_elser_targeted_{domain}.jsonl"
        evaluated_file = RESULTS_DIR / f"bm25_elser_targeted_{domain}_evaluated.jsonl"
        
        if args.skip_existing and output_file.exists():
            print(f"Output exists, skipping: {output_file.name}")
            continue
        
        # Load corpus
        print(f"Loading corpus...")
        corpus = load_corpus(domain)
        print(f"Loaded {len(corpus)} documents")
        
        # Load targeted queries
        print(f"Loading targeted rewrite queries...")
        queries = load_targeted_queries(domain)
        print(f"Loaded {len(queries)} queries")
        
        # Stage 1: BM25
        print("\n[Stage 1] BM25 retrieval...")
        bm25_candidates = run_bm25_stage(corpus, queries, top_k=args.bm25_top_k)
        
        # Stage 2: ELSER
        print("\n[Stage 2] ELSER reranking...")
        reranked_results = run_elser_rerank_stage(
            es, index_name, queries, bm25_candidates,
            top_k=FINAL_TOP_K, delay=args.delay
        )
        
        # Save and evaluate
        final_results = format_results_for_eval(reranked_results, domain, corpus)
        save_results(final_results, output_file)
        
        metrics = run_evaluation(output_file, evaluated_file)
        
        if metrics:
            results_summary.append({'domain': domain, **metrics})
    
    # Summary
    if results_summary:
        print("\n" + "=" * 80)
        print("RESULTS SUMMARY: BM25+ELSER with Targeted Rewrite")
        print("=" * 80)
        summary_df = pd.DataFrame(results_summary)
        print(summary_df.to_string(index=False))
        
        summary_file = RESULTS_DIR / "bm25_elser_targeted_summary.csv"
        summary_df.to_csv(summary_file, index=False)
        print(f"\nSummary saved to: {summary_file}")
    
    print("\n" + "=" * 80)
    print("Complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()

