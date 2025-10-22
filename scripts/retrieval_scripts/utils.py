"""
Common utilities for retrieval scripts
"""
import json
from typing import Dict, List, Tuple
from pathlib import Path


def load_corpus(corpus_file: str) -> Dict[str, Dict]:
    """
    Load corpus from JSONL file in BEIR format.
    
    Args:
        corpus_file: Path to corpus JSONL file
        
    Returns:
        Dictionary mapping document_id to document data
    """
    corpus = {}
    with open(corpus_file, 'r', encoding='utf-8') as f:
        for line in f:
            doc = json.loads(line)
            corpus[doc['_id']] = doc
    return corpus


def load_queries(query_file: str) -> Dict[str, str]:
    """
    Load queries from JSONL file in BEIR format.
    
    Args:
        query_file: Path to queries JSONL file
        
    Returns:
        Dictionary mapping query_id to query text
    """
    queries = {}
    with open(query_file, 'r', encoding='utf-8') as f:
        for line in f:
            query = json.loads(line)
            queries[query['_id']] = query['text']
    return queries


def load_qrels(qrels_file: str) -> Dict[str, Dict[str, int]]:
    """
    Load qrels (relevance judgments) from TSV file.
    
    Args:
        qrels_file: Path to qrels TSV file
        
    Returns:
        Dictionary mapping query_id to dict of {doc_id: relevance_score}
    """
    import csv
    
    qrels = {}
    with open(qrels_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)  # Skip header
        
        for row in reader:
            query_id, doc_id, score = row[0], row[1], int(row[2])
            if query_id not in qrels:
                qrels[query_id] = {}
            qrels[query_id][doc_id] = score
    
    return qrels


def save_results(results: List[Dict], output_file: str):
    """
    Save retrieval results to JSONL file.
    
    Args:
        results: List of result dictionaries
        output_file: Path to output JSONL file
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    print(f"Results saved to {output_file}")


def get_collection_name(domain: str) -> str:
    """
    Map domain name to collection name used in the benchmark.
    
    Args:
        domain: Domain name (clapnq, fiqa, govt, cloud)
        
    Returns:
        Collection name string
    """
    collection_names = {
        'clapnq': 'mt-rag-clapnq-elser-512-100-20240503',
        'govt': 'mt-rag-govt-elser-512-100-20240611',
        'fiqa': 'mt-rag-fiqa-beir-elser-512-100-20240501',
        'cloud': 'mt-rag-ibmcloud-elser-512-100-20240502'
    }
    return collection_names.get(domain.lower(), domain)

