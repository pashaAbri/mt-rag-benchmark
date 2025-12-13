"""
Utility functions and classes for the MonoT5 reranking evaluation (Targeted Rewrite Edition).
"""

import sys
import json
import csv
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple

try:
    import torch
    from transformers import T5ForConditionalGeneration, T5Tokenizer
except ImportError as e:
    # We allow importing utils without torch/transformers if not using the Scorer
    pass

import pytrec_eval

# Determine project root
# File path: scripts/ideas/retrieval_tasks/mono-t5-as-reranker-targeted/utils.py
# Root path: . (4 levels up)
script_dir = Path(__file__).parent
project_root = script_dir.parents[3]

# Configuration
BATCH_SIZE = 32  # Batch size for MonoT5 scoring

# Paths
BASELINE_RESULTS_DIR = project_root / 'scripts' / 'baselines' / 'retrieval_scripts' / 'elser' / 'results'
TARGETED_RESULTS_DIR = project_root / 'scripts' / 'ideas' / 'retrieval_tasks' / 'targeted_rewrite' / 'retrieval_results'
TARGETED_QUERIES_DIR = project_root / 'scripts' / 'ideas' / 'retrieval_tasks' / 'targeted_rewrite' / 'intermediate'

QRELS_DIR = project_root / 'human' / 'retrieval_tasks'
BASELINE_QUERIES_DIR = project_root / 'human' / 'retrieval_tasks'
EVAL_SCRIPT_PATH = project_root / 'scripts' / 'evaluation' / 'run_retrieval_eval.py'

DOMAINS = ['clapnq', 'cloud', 'fiqa', 'govt']
STRATEGIES = ['lastturn', 'targeted_rewrite', 'questions']

# Collection names mapping for evaluation script
COLLECTION_NAMES = {
    'clapnq': 'mt-rag-clapnq-elser-512-100-20240503',
    'govt': 'mt-rag-govt-elser-512-100-20240611',
    'fiqa': 'mt-rag-fiqa-beir-elser-512-100-20240501',
    'cloud': 'mt-rag-ibmcloud-elser-512-100-20240502'
}


class MonoT5Scorer:
    def __init__(self, model_name: str, device: str):
        self.device = device
        print(f"Loading MonoT5 model: {model_name} on {device}")
        
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()

    def score_batch(self, query: str, documents: List[str]) -> List[float]:
        """Score a batch of documents against a query."""
        if not documents:
            return []
            
        inputs = [f"Query: {query} Document: {doc} Relevant:" for doc in documents]
        
        # Tokenize
        encoded = self.tokenizer(
            inputs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **encoded,
                max_new_tokens=1,
                return_dict_in_generate=True,
                output_scores=True
            )
            
        # Get scores for "true" and "false"
        true_id = self.tokenizer.encode("true", add_special_tokens=False)[0]
        false_id = self.tokenizer.encode("false", add_special_tokens=False)[0]
        
        batch_scores = []
        for scores in outputs.scores[0]:
            true_score = scores[true_id].item()
            false_score = scores[false_id].item()
            
            # Softmax
            true_prob = torch.exp(torch.tensor(true_score))
            false_prob = torch.exp(torch.tensor(false_score))
            prob = true_prob / (true_prob + false_prob)
            batch_scores.append(prob.item())
        
        return batch_scores


def load_qrels(domain: str) -> Dict[str, Dict[str, int]]:
    """Load qrels for a domain."""
    qrels_path = QRELS_DIR / domain / 'qrels' / 'dev.tsv'
    
    if not qrels_path.exists():
        return {}
    
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


def load_queries(domain: str, strategy: str) -> Dict[str, str]:
    """Load queries for a domain and strategy."""
    if strategy == 'targeted_rewrite':
        query_file = TARGETED_QUERIES_DIR / f"targeted_rewrite_{domain}.jsonl"
    else:
        # Fallback to standard baseline locations for other strategies
        # e.g. lastturn, questions
        query_file = BASELINE_QUERIES_DIR / domain / f"{domain}_{strategy}.jsonl"
    
    if not query_file.exists():
        print(f"Warning: Query file not found: {query_file}")
        return {}
    
    queries = {}
    with open(query_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            query_id = data.get('_id')
            query_text = data.get('text', '')
            if query_id:
                queries[query_id] = query_text
    
    return queries


def load_retrieval_results(domain: str, strategy: str) -> Dict[str, List[Dict]]:
    """Load retrieval results for a domain and strategy."""
    if strategy == 'targeted_rewrite':
        # Load from targeted results
        filename = f"targeted_rewrite_{domain}_elser.jsonl"
        filepath = TARGETED_RESULTS_DIR / filename
    else:
        # Load from baseline results
        filename = f"elser_{domain}_{strategy}.jsonl"
        filepath = BASELINE_RESULTS_DIR / filename
    
    if not filepath.exists():
        print(f"Warning: Retrieval results not found: {filepath}")
        return {}
    
    results = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                task_id = data.get('task_id')
                contexts = data.get('contexts', [])
                if task_id:
                    results[task_id] = contexts
            except json.JSONDecodeError:
                continue
    
    return results


def combine_and_deduplicate(
    results_by_strategy: Dict[str, Dict[str, List[Dict]]]
) -> Dict[str, List[Dict]]:
    """
    Combine retrieval results from all strategies and deduplicate by document_id.
    
    Returns:
        Dictionary mapping task_id to list of unique documents
    """
    combined = {}
    
    # Find common task_ids
    common_task_ids = None
    strategies_present = list(results_by_strategy.keys())
    
    for strategy in strategies_present:
        if strategy in results_by_strategy:
            task_ids = set(results_by_strategy[strategy].keys())
            if common_task_ids is None:
                common_task_ids = task_ids
            else:
                # We typically want intersection to be safe, but union might be better if some strategies fail on some queries?
                # Sticking to intersection to match baseline logic for fair comparison.
                common_task_ids &= task_ids
    
    if common_task_ids is None:
        return {}
    
    for task_id in common_task_ids:
        # Collect all documents from all strategies
        doc_dict = {}  # document_id -> document info
        
        for strategy in strategies_present:
            if strategy in results_by_strategy and task_id in results_by_strategy[strategy]:
                contexts = results_by_strategy[strategy][task_id]
                for ctx in contexts:
                    doc_id = ctx.get('document_id')
                    if doc_id:
                        # Keep the first occurrence (or could keep highest score)
                        if doc_id not in doc_dict:
                            doc_dict[doc_id] = {
                                'document_id': doc_id,
                                'text': ctx.get('text', ''),
                                'title': ctx.get('title', ''),
                                'source': ctx.get('source', ''),
                                'original_score': ctx.get('score', 0.0),
                                'strategies': [strategy]
                            }
                        else:
                            # Track which strategies retrieved this doc
                            if strategy not in doc_dict[doc_id]['strategies']:
                                doc_dict[doc_id]['strategies'].append(strategy)
        
        combined[task_id] = list(doc_dict.values())
    
    return combined


def rerank_with_monot5(
    scorer: MonoT5Scorer,
    query: str,
    documents: List[Dict],
    top_k: int = 100
) -> List[Tuple[Dict, float]]:
    """
    Rerank documents using MonoT5.
    
    Returns:
        List of (document, score) tuples sorted by score descending
    """
    if not documents:
        return []
    
    # Extract document texts
    doc_texts = [doc.get('text', '') for doc in documents]
    
    # Score in batches
    all_scores = []
    for i in range(0, len(doc_texts), BATCH_SIZE):
        batch_texts = doc_texts[i:i+BATCH_SIZE]
        batch_scores = scorer.score_batch(query, batch_texts)
        all_scores.extend(batch_scores)
    
    # Combine documents with scores and sort
    doc_score_pairs = list(zip(documents, all_scores))
    doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
    
    return doc_score_pairs[:top_k]


def save_results_for_evaluation(
    results: Dict[str, Dict[str, float]], 
    domain: str, 
    output_path: Path
):
    """
    Save reranked results in the JSONL format expected by run_retrieval_eval.py.
    """
    collection_name = COLLECTION_NAMES.get(domain, domain)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for task_id, doc_scores in results.items():
            contexts = [
                {"document_id": doc_id, "score": score} 
                for doc_id, score in doc_scores.items()
            ]
            
            entry = {
                "task_id": task_id,
                "Collection": collection_name,
                "contexts": contexts
            }
            f.write(json.dumps(entry) + '\n')
            
    print(f"Saved {len(results)} results to {output_path}")

