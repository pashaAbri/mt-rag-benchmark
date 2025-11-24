"""
BM25 Lexical Retrieval Implementation

This script implements BM25 retrieval for the MT-RAG benchmark using PyTerrier.

NOTE on Query Processing:
- Special characters in queries (like '?', '!', ':', '\\', ''') are replaced with spaces.
  This is done to prevent parser errors in PyTerrier's TerrierQLParser, which can crash
  on unescaped special characters.
- We intentionally use space replacement instead of escaping because:
  1. Escaping (e.g. '\?') can still cause lexical errors in some PyTerrier versions.
  2. Removing them allows for a robust "bag-of-words" retrieval without syntax errors.
- Stopwords are removed by default by the underlying Terrier indexer (IterDictIndexer).
  This means queries consisting entirely of stopwords (e.g. "What is it?") will 
  correctly return 0 results.
- Results are initialized with all input query IDs to ensure that queries returning
  0 results are still present in the output file with empty contexts, rather than being missing.
"""
import argparse
from typing import Dict, List
from pathlib import Path
import sys

# Add parent directory to path to import utils
sys.path.append(str(Path(__file__).parent.parent))
from utils import load_corpus, load_queries, save_results, get_collection_name

import pyterrier as pt
import pandas as pd
from tqdm import tqdm
import re


def run_bm25_retrieval(corpus: Dict, queries: Dict[str, str], top_k: int = 10) -> Dict[str, Dict[str, float]]:
    """
    Run BM25 retrieval on the corpus using PyTerrier.
    
    Args:
        corpus: Dictionary of documents
        queries: Dictionary of queries
        top_k: Number of top results to return
        
    Returns:
        Dictionary mapping query_id to {doc_id: score}
    """
    # Initialize PyTerrier if not already initialized
    if not pt.started():
        pt.init()
    
    print("Preparing corpus for indexing...")
    # Convert corpus to PyTerrier format
    corpus_list = []
    for doc_id, doc in tqdm(corpus.items(), desc="Processing corpus"):
        corpus_list.append({
            'docno': doc_id,
            'text': doc.get('text', '')
        })
    
    corpus_df = pd.DataFrame(corpus_list)
    
    print(f"Building BM25 index for {len(corpus_df)} documents...")
    # Create index in the bm25/results directory
    # Note: docno length set to 50 to accommodate long document IDs
    script_dir = Path(__file__).parent
    index_path = str(script_dir / "results" / "pyterrier_index")
    Path(index_path).parent.mkdir(parents=True, exist_ok=True)
    indexer = pt.IterDictIndexer(index_path, overwrite=True, meta={'docno': 50})
    index_ref = indexer.index(corpus_df.to_dict('records'))
    
    print("Index built successfully!")
    
    print(f"Running retrieval on {len(queries)} queries...")
    
    def escape_query(query: str) -> str:
        """Clean special characters in PyTerrier query to avoid parser errors"""
        # Characters that have special meaning in Terrier query language
        # We replace them with spaces to treat the query as a simple bag-of-words
        # Added single quote (') as it caused lexical errors
        special_chars = ['+', '-', '=', '&&', '||', '>', '<', '!', '(', ')', 
                        '{', '}', '[', ']', '^', '"', '~', '*', '?', ':', '\\', '/', '\'']
        
        for char in special_chars:
            query = query.replace(char, ' ')
            
        # Collapse multiple spaces
        return ' '.join(query.split())

    # Convert queries to PyTerrier format
    # We are using the standard PyTerrier parser with escaping
    queries_list = []
    for qid, text in queries.items():
        # Remove |user|: prefix if it exists
        clean_text = text.replace('|user|: ', '').replace('|user|:', '').strip()
        
        # Escape special characters
        clean_text = escape_query(clean_text)
        
        # Handle empty queries
        if not clean_text.strip():
            clean_text = "empty_query_placeholder"
            
        queries_list.append({'qid': qid, 'query': clean_text})
    queries_df = pd.DataFrame(queries_list)
    
    # Create BM25 retriever using the standard parser
    bm25 = pt.BatchRetrieve(index_ref, wmodel="BM25", num_results=top_k)
    
    # Run retrieval
    try:
        results_df = bm25.transform(queries_df)
    except Exception as e:
        print(f"Error during retrieval: {e}")
        print("Attempting to identify problematic queries...")
        # Fallback: try one by one to identify issues
        results_dfs = []
        for _, row in queries_df.iterrows():
            try:
                single_res = bm25.transform(pd.DataFrame([row]))
                results_dfs.append(single_res)
            except Exception as inner_e:
                print(f"Failed query {row['qid']}: {row['query']} - {inner_e}")
        
        if results_dfs:
            results_df = pd.concat(results_dfs)
        else:
            results_df = pd.DataFrame(columns=['qid', 'docno', 'score'])
    
    print("Retrieval complete! Formatting results...")
    # Convert results to expected format
    results_dict = {qid: {} for qid in queries.keys()}
    for _, row in results_df.iterrows():
        qid = row['qid']
        docno = row['docno']
        score = row['score']
        
        results_dict[qid][docno] = score
    
    return results_dict


def main():
    parser = argparse.ArgumentParser(description='Run BM25 retrieval on MT-RAG benchmark')
    parser.add_argument('--domain', type=str, required=True, 
                        choices=['clapnq', 'fiqa', 'govt', 'cloud'],
                        help='Domain to run retrieval on')
    parser.add_argument('--query_type', type=str, required=True,
                        choices=['lastturn', 'rewrite', 'questions'],
                        help='Type of queries to use')
    parser.add_argument('--corpus_file', type=str, required=True,
                        help='Path to corpus JSONL file')
    parser.add_argument('--query_file', type=str, required=True,
                        help='Path to queries JSONL file')
    parser.add_argument('--output_file', type=str, required=True,
                        help='Path to output results file')
    parser.add_argument('--top_k', type=int, default=10,
                        help='Number of top results to retrieve')
    
    args = parser.parse_args()
    
    print(f"Loading corpus from {args.corpus_file}...")
    corpus = load_corpus(args.corpus_file)
    print(f"Loaded {len(corpus)} documents")
    
    print(f"Loading queries from {args.query_file}...")
    queries = load_queries(args.query_file)
    print(f"Loaded {len(queries)} queries")
    
    print("Running BM25 retrieval...")
    results_dict = run_bm25_retrieval(corpus, queries, args.top_k)
    
    # Format results for evaluation script
    collection_name = get_collection_name(args.domain)
    results = []
    for query_id, doc_scores in results_dict.items():
        contexts = []
        for doc_id, score in sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:args.top_k]:
            contexts.append({
                'document_id': doc_id,
                'score': score,
                'text': corpus[doc_id].get('text', ''),
                'title': corpus[doc_id].get('title', ''),
                'source': corpus[doc_id].get('url', '')
            })
        
        results.append({
            'task_id': query_id,
            'Collection': collection_name,
            'contexts': contexts
        })
    
    save_results(results, args.output_file)
    print("BM25 retrieval complete!")


if __name__ == '__main__':
    main()

