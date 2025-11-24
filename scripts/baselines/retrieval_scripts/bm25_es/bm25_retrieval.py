"""
BM25 Lexical Retrieval Implementation using Elasticsearch

This script implements BM25 retrieval for the MT-RAG benchmark using Elasticsearch.
"""
import argparse
from typing import Dict, List
from pathlib import Path
import sys

# Add parent directory to path to import utils
sys.path.append(str(Path(__file__).parent.parent))
from utils import load_corpus, load_queries, save_results, get_collection_name

from elasticsearch import Elasticsearch
from tqdm import tqdm
import time


def create_index(es_client: Elasticsearch, index_name: str) -> None:
    """
    Create an Elasticsearch index with BM25 similarity.
    
    Args:
        es_client: Elasticsearch client
        index_name: Name of the index to create
    """
    # Delete index if it exists
    if es_client.indices.exists(index=index_name):
        print(f"Deleting existing index: {index_name}")
        es_client.indices.delete(index=index_name)
    
    # Create index with BM25 similarity settings
    index_settings = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
            "similarity": {
                "default": {
                    "type": "BM25"
                }
            }
        },
        "mappings": {
            "properties": {
                "text": {
                    "type": "text",
                    "similarity": "default"
                },
                "title": {
                    "type": "text"
                },
                "url": {
                    "type": "keyword"
                }
            }
        }
    }
    
    print(f"Creating index: {index_name}")
    es_client.indices.create(index=index_name, body=index_settings)


def index_documents(es_client: Elasticsearch, index_name: str, corpus: Dict) -> None:
    """
    Index documents into Elasticsearch.
    
    Args:
        es_client: Elasticsearch client
        index_name: Name of the index
        corpus: Dictionary of documents
    """
    print(f"Indexing {len(corpus)} documents...")
    
    # Bulk indexing for efficiency
    from elasticsearch.helpers import bulk
    
    def generate_actions():
        for doc_id, doc in tqdm(corpus.items(), desc="Preparing documents"):
            yield {
                "_index": index_name,
                "_id": doc_id,
                "_source": {
                    "text": doc.get('text', ''),
                    "title": doc.get('title', ''),
                    "url": doc.get('url', '')
                }
            }
    
    # Bulk index with progress
    success, failed = bulk(es_client, generate_actions(), raise_on_error=False)
    print(f"Successfully indexed {success} documents")
    if failed:
        print(f"Failed to index {len(failed)} documents")
    
    # Refresh index to make documents searchable
    es_client.indices.refresh(index=index_name)
    print("Index refreshed and ready for searching")


def run_bm25_retrieval(es_client: Elasticsearch, index_name: str, queries: Dict[str, str], top_k: int = 10) -> Dict[str, Dict[str, float]]:
    """
    Run BM25 retrieval using Elasticsearch.
    
    Args:
        es_client: Elasticsearch client
        index_name: Name of the index to search
        queries: Dictionary of queries
        top_k: Number of top results to return
        
    Returns:
        Dictionary mapping query_id to {doc_id: score}
    """
    results_dict = {}
    
    print(f"Running retrieval on {len(queries)} queries...")
    for qid, query_text in tqdm(queries.items(), desc="Searching"):
        # Clean query text (remove |user|: prefix if present)
        clean_text = query_text.replace('|user|: ', '').replace('|user|:', '').strip()
        
        # Elasticsearch query using BM25
        search_body = {
            "query": {
                "match": {
                    "text": clean_text
                }
            },
            "size": top_k
        }
        
        try:
            response = es_client.search(index=index_name, body=search_body)
            
            # Extract results
            results_dict[qid] = {}
            for hit in response['hits']['hits']:
                doc_id = hit['_id']
                score = hit['_score']
                results_dict[qid][doc_id] = score
        
        except Exception as e:
            print(f"Error searching for query {qid}: {e}")
            results_dict[qid] = {}
    
    return results_dict


def main():
    parser = argparse.ArgumentParser(description='Run BM25 retrieval on MT-RAG benchmark using Elasticsearch')
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
    parser.add_argument('--es_host', type=str, default='localhost',
                        help='Elasticsearch host')
    parser.add_argument('--es_port', type=int, default=9200,
                        help='Elasticsearch port')
    parser.add_argument('--index_name', type=str, default=None,
                        help='Elasticsearch index name (default: mtrag_{domain})')
    
    args = parser.parse_args()
    
    # Set index name
    if args.index_name is None:
        args.index_name = f"mtrag_{args.domain}"
    
    print(f"Connecting to Elasticsearch at {args.es_host}:{args.es_port}...")
    # Initialize Elasticsearch client
    es_client = Elasticsearch(
        hosts=[f"http://{args.es_host}:{args.es_port}"],
        request_timeout=300,
        retry_on_timeout=True,
        max_retries=3
    )
    
    # Check connection
    if not es_client.ping():
        raise ConnectionError(f"Could not connect to Elasticsearch at {args.es_host}:{args.es_port}")
    
    print(f"Connected to Elasticsearch cluster: {es_client.info()['cluster_name']}")
    
    print(f"Loading corpus from {args.corpus_file}...")
    corpus = load_corpus(args.corpus_file)
    print(f"Loaded {len(corpus)} documents")
    
    # Create index and index documents
    create_index(es_client, args.index_name)
    index_documents(es_client, args.index_name, corpus)
    
    print(f"Loading queries from {args.query_file}...")
    queries = load_queries(args.query_file)
    print(f"Loaded {len(queries)} queries")
    
    print("Running BM25 retrieval...")
    results_dict = run_bm25_retrieval(es_client, args.index_name, queries, args.top_k)
    
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
    
    # Close Elasticsearch connection
    es_client.close()
    print("BM25 retrieval complete!")


if __name__ == '__main__':
    main()

