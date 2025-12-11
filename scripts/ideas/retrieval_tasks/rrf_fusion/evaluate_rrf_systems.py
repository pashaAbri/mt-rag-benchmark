#!/usr/bin/env python3
"""
Evaluate Reciprocal Rank Fusion (RRF) for System Fusion (Ensemble).

Hypothesis:
Combining orthogonal retrieval systems (BM25, BGE, ELSER) using RRF
outperforms any single system.

Method:
1. Load retrieval results for 'rewrite' strategy from BM25, BGE, ELSER.
2. Calculate RRF score for each document: Score(d) = sum(1 / (k + rank(d)))
3. Sort and take top 100.
4. Evaluate.
"""

import sys
import json
import subprocess
from pathlib import Path
from collections import defaultdict

# Add project root to path
script_dir = Path(__file__).parent
project_root = script_dir.parents[3]
sys.path.insert(0, str(project_root))

# Paths
BASELINES_DIR = project_root / 'scripts' / 'baselines' / 'retrieval_scripts'
EVAL_SCRIPT_PATH = project_root / 'scripts' / 'evaluation' / 'run_retrieval_eval.py'
INTERMEDIATE_DIR = script_dir / "intermediate"

# Config
DOMAINS = ['clapnq', 'cloud', 'fiqa', 'govt']
SYSTEMS = ['bm25', 'bge', 'elser']
# Weights based on baseline performance (ELSER >> BGE > BM25)
SYSTEM_WEIGHTS = {'elser': 3.0, 'bge': 1.0, 'bm25': 0.5}
STRATEGY = 'rewrite' # Use the best query strategy
RRF_K = 60

# Collection names mapping
COLLECTION_NAMES = {
    'clapnq': 'mt-rag-clapnq-elser-512-100-20240503',
    'govt': 'mt-rag-govt-elser-512-100-20240611',
    'fiqa': 'mt-rag-fiqa-beir-elser-512-100-20240501',
    'cloud': 'mt-rag-ibmcloud-elser-512-100-20240502'
}

def load_retrieval_results(domain: str, system: str, strategy: str):
    """Load retrieval results for a specific system."""
    filename = f"{system}_{domain}_{strategy}.jsonl"
    filepath = BASELINES_DIR / system / 'results' / filename
    
    if not filepath.exists():
        print(f"Warning: File not found {filepath}")
        return {}
    
    results = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            try:
                data = json.loads(line)
                task_id = data.get('task_id')
                contexts = data.get('contexts', [])
                if task_id:
                    results[task_id] = contexts
            except json.JSONDecodeError:
                continue
    return results

def compute_rrf(results_by_system):
    """
    Compute RRF scores for combined results.
    """
    fused_results = {}
    
    # Get all task_ids
    all_task_ids = set()
    for sys_name in results_by_system:
        all_task_ids.update(results_by_system[sys_name].keys())
        
    for task_id in all_task_ids:
        doc_scores = defaultdict(float)
        doc_info = {}
        
        for system in SYSTEMS:
            if system not in results_by_system or task_id not in results_by_system[system]:
                continue
                
            ranked_docs = results_by_system[system][task_id]
            
            for rank, doc in enumerate(ranked_docs):
                doc_id = doc.get('document_id')
                if not doc_id: continue
                
                # RRF Formula with Weights: weight * (1 / (k + rank))
                weight = SYSTEM_WEIGHTS.get(system, 1.0)
                score = weight * (1.0 / (RRF_K + (rank + 1)))
                doc_scores[doc_id] += score
                
                if doc_id not in doc_info:
                    doc_info[doc_id] = doc
        
        # Sort
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Format
        formatted_docs = []
        for doc_id, score in sorted_docs:
            original_doc = doc_info[doc_id]
            new_doc = {
                'document_id': doc_id,
                'score': score,
                'text': original_doc.get('text', ''),
                'title': original_doc.get('title', ''),
                'source': original_doc.get('source', '')
            }
            formatted_docs.append(new_doc)
            
        fused_results[task_id] = formatted_docs
        
    return fused_results

def save_and_evaluate(results, domain):
    """Save results and run evaluation."""
    output_file = INTERMEDIATE_DIR / f"weighted_rrf_systems_{domain}.jsonl"
    
    collection_name = COLLECTION_NAMES.get(domain, domain)
    
    # Save
    with open(output_file, 'w', encoding='utf-8') as f:
        for task_id, contexts in results.items():
            entry = {
                "task_id": task_id,
                "Collection": collection_name,
                "contexts": contexts[:100]
            }
            f.write(json.dumps(entry) + '\n')
            
    print(f"Saved Weighted RRF results to {output_file}")
    # We skip running the official evaluation script because it fails with dependency errors.
    # We will use simple_evaluate.py instead.

def main():
    print("="*80)
    print("RRF System Fusion Evaluation (Ensemble)")
    print("="*80)
    
    INTERMEDIATE_DIR.mkdir(exist_ok=True)
    
    for domain in DOMAINS:
        print(f"\nProcessing {domain.upper()}...")
        
        results_by_system = {}
        for system in SYSTEMS:
            print(f"  Loading {system}...")
            results_by_system[system] = load_retrieval_results(domain, system, STRATEGY)
            
        print("  Computing RRF...")
        fused_results = compute_rrf(results_by_system)
        print(f"  Fused {len(fused_results)} tasks")
        
        save_and_evaluate(fused_results, domain)

if __name__ == "__main__":
    main()
