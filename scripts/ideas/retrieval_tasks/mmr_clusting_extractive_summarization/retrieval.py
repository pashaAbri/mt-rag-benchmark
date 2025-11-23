import os
import sys
from typing import List, Dict, Any
from elasticsearch import Elasticsearch
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class ElserRetriever:
    def __init__(self, domain: str):
        """
        Initialize ELSER v2 Retriever.
        
        Args:
            domain: Domain name (clapnq, fiqa, govt, cloud)
        """
        self.domain = domain
        self.es_client = self._connect_elasticsearch()
        self.index_name = self._get_index_name(domain)
        
    def _connect_elasticsearch(self) -> Elasticsearch:
        """Connect to Elasticsearch Cloud."""
        es_url = os.getenv('ES_URL')
        api_key = os.getenv('ES_API_KEY')
        
        if not es_url or not api_key:
            raise ValueError("ES_URL and ES_API_KEY environment variables must be set.")
            
        print(f"Connecting to Elasticsearch at {es_url}...")
        es = Elasticsearch(es_url, api_key=api_key, request_timeout=60)
        
        if not es.ping():
            raise ConnectionError("Failed to ping Elasticsearch.")
            
        return es

    def _get_index_name(self, domain: str) -> str:
        """Get the ELSER index name for the domain."""
        # Index naming logic from scripts/baselines/retrieval_scripts/elser/elser_retrieval.py
        if domain == 'clapnq':
            return "mtrag-clapnq-elser-512-100-reindexed"
        else:
            return f"mtrag-{domain}-elser-512-100"

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve documents using ELSER v2 (text_expansion).
        
        Args:
            query: Query string
            top_k: Number of documents to return
            
        Returns:
            List of document dicts with 'score', 'text', 'title', 'document_id'
        """
        # Clean query text
        clean_text = query.replace('|user|: ', '').replace('|user|:', '').strip()
        
        # Build ELSER query
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
        
        try:
            response = self.es_client.search(index=self.index_name, body=query_body)
            
            results = []
            for hit in response['hits']['hits']:
                source = hit['_source']
                results.append({
                    'document_id': hit['_id'],
                    'score': float(hit['_score']),
                    'text': source.get('text', ''),
                    'title': source.get('title', ''),
                    'url': source.get('url', '')
                })
            
            return results
            
        except Exception as e:
            print(f"Error during retrieval from index {self.index_name}: {e}")
            # Fallback or re-raise? For now, return empty list to not crash pipeline
            return []
