"""
Working ELSER Retrieval Implementation

This script implements ELSER retrieval using the .elser-2-elastic inference endpoint.
Even though text_expansion is deprecated, it's the working method for querying
pre-indexed ELSER tokens.

Includes rate limiting and retry logic to handle Elasticsearch Cloud limits.
"""
import argparse
import os
import sys
from pathlib import Path
from typing import Dict
from elasticsearch import Elasticsearch
import warnings
import time

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from utils import load_queries, save_results, get_collection_name

# Load environment variables
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent.parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass

# Suppress the text_expansion deprecation warning since it's the only working method
warnings.filterwarnings('ignore', category=Warning)

def run_elser_retrieval(
    es: Elasticsearch,
    index_name: str,
    queries: Dict[str, str],
    top_k: int = 10,
    delay: float = 0.5
) -> Dict:
    """
    Run ELSER retrieval using text_expansion (deprecated but working).
    
    Args:
        es: Elasticsearch client
        index_name: Name of the index
        queries: Dictionary of {query_id: query_text}
        top_k: Number of results to return
        delay: Delay between queries in seconds (to avoid rate limits)
        
    Returns:
        List of results in the expected format
    """
    results = []
    
    print(f"Running ELSER retrieval on {len(queries)} queries...")
    print(f"Rate limiting: {delay}s delay between queries")
    
    for i, (query_id, query_text) in enumerate(queries.items(), 1):
        # Clean query text (remove |user|: prefix if present)
        clean_text = query_text.replace('|user|: ', '').replace('|user|:', '').strip()
        
        # Build ELSER query using text_expansion
        query_body = {
            "query": {
                "text_expansion": {
                    "ml.tokens": {
                        "model_id": ".elser-2-elastic",
                        "model_text": clean_text
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
                
                # Format contexts
                contexts = []
                for hit in response['hits']['hits']:
                    contexts.append({
                        'document_id': hit['_id'],
                        'score': float(hit['_score']),
                        'text': hit['_source'].get('text', ''),
                        'title': hit['_source'].get('title', ''),
                        'source': hit['_source'].get('url', '')
                    })
                
                results.append({
                    'task_id': query_id,
                    'Collection': index_name,
                    'contexts': contexts
                })
                
                # Success - break retry loop
                break
                    
            except Exception as e:
                error_msg = str(e)
                
                # Check if it's a rate limit error
                if '429' in error_msg or 'rate limit' in error_msg.lower():
                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (attempt + 1)
                        print(f"  Rate limit hit on query {i}/{len(queries)}, waiting {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    else:
                        print(f"  Warning: Query {query_id} failed after {max_retries} retries (rate limit)")
                        results.append({
                            'task_id': query_id,
                            'Collection': index_name,
                            'contexts': []
                        })
                else:
                    print(f"  Warning: Query {query_id} failed: {e}")
                    results.append({
                        'task_id': query_id,
                        'Collection': index_name,
                        'contexts': []
                    })
                    break
        
        # Progress update
        if i % 50 == 0:
            print(f"  Processed {i}/{len(queries)} queries...")
        
        # Rate limiting delay
        if i < len(queries):  # Don't sleep after last query
            time.sleep(delay)
    
    print(f"Completed {len(results)} queries")
    return results

def main():
    parser = argparse.ArgumentParser(description='Run ELSER retrieval on MT-RAG benchmark')
    parser.add_argument('--domain', type=str, required=True,
                        choices=['clapnq', 'fiqa', 'govt', 'cloud'],
                        help='Domain to run retrieval on')
    parser.add_argument('--query_type', type=str, required=True,
                        choices=['lastturn', 'rewrite', 'questions'],
                        help='Type of queries to use')
    parser.add_argument('--query_file', type=str, required=True,
                        help='Path to queries JSONL file')
    parser.add_argument('--output_file', type=str, required=True,
                        help='Path to output results file')
    parser.add_argument('--top_k', type=int, default=10,
                        help='Number of top results to retrieve')
    parser.add_argument('--delay', type=float, default=0.5,
                        help='Delay between queries in seconds (default: 0.5)')
    
    args = parser.parse_args()
    
    # Get Elasticsearch credentials
    es_url = os.getenv('ES_URL')
    api_key = os.getenv('ES_API_KEY')
    
    if not es_url or not api_key:
        print("❌ ES_URL and ES_API_KEY environment variables must be set")
        print("   Make sure they're in your .env file")
        sys.exit(1)
    
    # Connect to Elasticsearch
    print(f"Connecting to Elasticsearch...")
    es = Elasticsearch(es_url, api_key=api_key, request_timeout=60)
    
    if not es.ping():
        print("❌ Failed to connect to Elasticsearch")
        sys.exit(1)
    
    print(f"✅ Connected successfully\n")
    
    # Determine index name (for ES query)
    index_name = f"mtrag-{args.domain}-elser-512-100"
    
    # Collection name for results (must match what evaluation script expects)
    collection_mapping = {
        'clapnq': 'mt-rag-clapnq-elser-512-100-20240503',
        'fiqa': 'mt-rag-fiqa-beir-elser-512-100-20240501',
        'govt': 'mt-rag-govt-elser-512-100-20240611',
        'cloud': 'mt-rag-ibmcloud-elser-512-100-20240502'
    }
    collection_name = collection_mapping.get(args.domain, index_name)
    
    # Load queries
    print(f"Loading queries from {args.query_file}...")
    queries = load_queries(args.query_file)
    print(f"Loaded {len(queries)} queries\n")
    
    # Run retrieval
    print(f"Running ELSER retrieval on index: {index_name}")
    print(f"Using inference endpoint: .elser-2-elastic")
    print(f"Top-k: {args.top_k}\n")
    
    results = run_elser_retrieval(es, index_name, queries, args.top_k, args.delay)
    
    # Update collection names in results to match evaluation script expectations
    for result in results:
        result['Collection'] = collection_name
    
    # Save results
    print(f"\nSaving results to {args.output_file}...")
    save_results(results, args.output_file)
    
    print(f"✅ ELSER retrieval complete!")
    print(f"   Results saved to: {args.output_file}")

if __name__ == '__main__':
    main()

