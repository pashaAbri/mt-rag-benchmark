"""
Utility functions for the BM25 + BGE two-stage retrieval experiment.
"""

import sys
import json
import csv
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

# Determine project root
# File path: scripts/ideas/retrieval_tasks/bm25_bge_rerank/utils.py
# Root path: . (4 levels up)
script_dir = Path(__file__).parent
project_root = script_dir.parents[3]

# Add baselines to path
sys.path.insert(0, str(project_root / 'scripts' / 'baselines' / 'retrieval_scripts'))

# Paths
CORPUS_DIR = project_root / 'corpora' / 'passage_level'
QUERIES_DIR = project_root / 'human' / 'retrieval_tasks'
QRELS_DIR = project_root / 'human' / 'retrieval_tasks'
EVAL_SCRIPT_PATH = project_root / 'scripts' / 'evaluation' / 'run_retrieval_eval.py'
BGE_MODEL_PATH = project_root / 'scripts' / 'baselines' / 'retrieval_scripts' / 'bge' / 'models' / 'bge-base-en-v1.5'
BGE_EMBEDDINGS_DIR = project_root / 'scripts' / 'baselines' / 'retrieval_scripts' / 'bge' / 'embeddings'

# Baseline results paths
BASELINE_RESULTS_DIR = project_root / 'scripts' / 'baselines' / 'retrieval_scripts'

DOMAINS = ['clapnq', 'cloud', 'fiqa', 'govt']
QUERY_TYPES = ['lastturn', 'rewrite', 'questions']

# Collection names mapping for evaluation script
COLLECTION_NAMES = {
    'clapnq': 'mt-rag-clapnq-elser-512-100-20240503',
    'govt': 'mt-rag-govt-elser-512-100-20240611',
    'fiqa': 'mt-rag-fiqa-beir-elser-512-100-20240501',
    'cloud': 'mt-rag-ibmcloud-elser-512-100-20240502'
}


def load_corpus(domain: str) -> Dict[str, Dict]:
    """
    Load corpus from JSONL file in BEIR format.
    
    Args:
        domain: Domain name (clapnq, fiqa, govt, cloud)
        
    Returns:
        Dictionary mapping document_id to document data
    """
    corpus_file = CORPUS_DIR / f'{domain}.jsonl'
    corpus = {}
    with open(corpus_file, 'r', encoding='utf-8') as f:
        for line in f:
            doc = json.loads(line)
            corpus[doc['_id']] = doc
    return corpus


def load_queries(domain: str, query_type: str) -> Dict[str, str]:
    """
    Load queries for a domain and query type.
    
    Args:
        domain: Domain name
        query_type: Query type (lastturn, rewrite, questions)
        
    Returns:
        Dictionary mapping query_id to query text
    """
    query_file = QUERIES_DIR / domain / f'{domain}_{query_type}.jsonl'
    
    if not query_file.exists():
        raise FileNotFoundError(f"Query file not found: {query_file}")
    
    queries = {}
    with open(query_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            query_id = data.get('_id')
            query_text = data.get('text', '')
            if query_id:
                # Clean query text (remove |user|: prefix if present)
                query_text = query_text.replace('|user|: ', '').replace('|user|:', '').strip()
                queries[query_id] = query_text
    
    return queries


def load_qrels(domain: str) -> Dict[str, Dict[str, int]]:
    """Load qrels for a domain."""
    qrels_path = QRELS_DIR / domain / 'qrels' / 'dev.tsv'
    
    if not qrels_path.exists():
        raise FileNotFoundError(f"Qrels file not found: {qrels_path}")
    
    qrels = {}
    with open(qrels_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)  # Skip header
        
        for row in reader:
            if len(row) < 3:
                continue
            query_id, doc_id, score = row[0], row[1], int(row[2])
            if query_id not in qrels:
                qrels[query_id] = {}
            qrels[query_id][doc_id] = score
    
    return qrels


def save_results(results: List[Dict], output_file: Path):
    """
    Save retrieval results to JSONL file.
    
    Args:
        results: List of result dictionaries
        output_file: Path to output JSONL file
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    print(f"Results saved to {output_file}")


def format_results_for_eval(
    results_dict: Dict[str, Dict[str, float]], 
    domain: str,
    corpus: Dict[str, Dict]
) -> List[Dict]:
    """
    Format results dictionary into the JSONL format expected by evaluation script.
    
    Args:
        results_dict: Dictionary mapping query_id to {doc_id: score}
        domain: Domain name
        corpus: Corpus dictionary for getting document text
        
    Returns:
        List of result dictionaries
    """
    collection_name = COLLECTION_NAMES.get(domain, domain)
    results = []
    
    for query_id, doc_scores in results_dict.items():
        contexts = []
        for doc_id, score in sorted(doc_scores.items(), key=lambda x: x[1], reverse=True):
            doc = corpus.get(doc_id, {})
            contexts.append({
                'document_id': doc_id,
                'score': float(score),
                'text': doc.get('text', ''),
                'title': doc.get('title', ''),
                'source': doc.get('url', '')
            })
        
        results.append({
            'task_id': query_id,
            'Collection': collection_name,
            'contexts': contexts
        })
    
    return results


def load_baseline_results(baseline: str, domain: str, query_type: str) -> Dict[str, Dict[str, float]]:
    """
    Load baseline retrieval results.
    
    Args:
        baseline: Baseline name (bm25, bge, elser)
        domain: Domain name
        query_type: Query type
        
    Returns:
        Dictionary mapping query_id to {doc_id: score}
    """
    if baseline == 'bm25':
        results_file = BASELINE_RESULTS_DIR / 'bm25' / 'results' / f'bm25_{domain}_{query_type}.jsonl'
    elif baseline == 'bge':
        results_file = BASELINE_RESULTS_DIR / 'bge' / 'results' / f'bge_{domain}_{query_type}.jsonl'
    elif baseline == 'elser':
        results_file = BASELINE_RESULTS_DIR / 'elser' / 'results' / f'elser_{domain}_{query_type}.jsonl'
    else:
        raise ValueError(f"Unknown baseline: {baseline}")
    
    if not results_file.exists():
        return {}
    
    results = {}
    with open(results_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            task_id = data.get('task_id')
            contexts = data.get('contexts', [])
            if task_id:
                results[task_id] = {
                    ctx['document_id']: ctx['score'] 
                    for ctx in contexts
                }
    
    return results


def load_aggregate_metrics(csv_file: Path) -> Dict[str, float]:
    """
    Load aggregate metrics from evaluation CSV file.
    
    Args:
        csv_file: Path to aggregate CSV file
        
    Returns:
        Dictionary with nDCG@10 and Recall@10 for 'all' collection
    """
    import ast
    import pandas as pd
    
    if not csv_file.exists():
        return {}
    
    df = pd.read_csv(csv_file)
    all_rows = df[df['collection'] == 'all']
    
    if len(all_rows) == 0:
        return {}
    
    all_row = all_rows.iloc[0]
    ndcg = ast.literal_eval(all_row['nDCG'])
    recall = ast.literal_eval(all_row['Recall'])
    
    return {
        'nDCG@1': ndcg[0],
        'nDCG@3': ndcg[1],
        'nDCG@5': ndcg[2],
        'nDCG@10': ndcg[3],
        'Recall@1': recall[0],
        'Recall@3': recall[1],
        'Recall@5': recall[2],
        'Recall@10': recall[3]
    }


class BGEReranker:
    """BGE-based document reranker using sentence transformers."""
    
    def __init__(self, model_path: str = None, device: str = None):
        import torch
        from sentence_transformers import SentenceTransformer
        
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        
        self.device = device
        
        if model_path is None:
            model_path = str(BGE_MODEL_PATH)
        
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"BGE model not found at {model_path}\n"
                f"Please run: python scripts/baselines/retrieval_scripts/bge/download_model.py"
            )
        
        print(f"Loading BGE model from {model_path} on {device}...")
        self.model = SentenceTransformer(model_path, device=device)
        print("BGE model loaded successfully")
    
    def rerank(
        self, 
        query: str, 
        documents: List[Dict], 
        top_k: int = 10,
        batch_size: int = 32
    ) -> List[Tuple[Dict, float]]:
        """
        Rerank documents using BGE cosine similarity.
        
        Args:
            query: Query text
            documents: List of document dictionaries with 'text' field
            top_k: Number of top results to return
            batch_size: Batch size for encoding
            
        Returns:
            List of (document, score) tuples sorted by score descending
        """
        if not documents:
            return []
        
        # Encode query
        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False
        )[0]
        
        # Encode documents
        doc_texts = [doc.get('text', '') for doc in documents]
        doc_embeddings = self.model.encode(
            doc_texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        
        # Compute cosine similarities
        similarities = np.dot(doc_embeddings, query_embedding)
        
        # Create document-score pairs and sort
        doc_score_pairs = list(zip(documents, similarities.tolist()))
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
        
        return doc_score_pairs[:top_k]

