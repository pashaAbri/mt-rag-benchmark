#!/usr/bin/env python3
"""
Multi-Query Fusion + MonoT5 Reranking

Hypothesis: Different query representations capture different signals:
- Last turn: Direct user intent, minimal noise
- Query rewrite: Contextually enriched, balanced
- Context summary: Rich context, potentially noisy for lexical but good for reranking

By combining retrieval results from all three and reranking with MonoT5 (using
the rewrite query as reference), we get the best of all worlds:
- Higher recall from diverse retrieval
- Better precision from semantic reranking

Usage:
    python run_fusion_rerank.py                           # Run with BM25 on all domains
    python run_fusion_rerank.py --retriever bge           # Run with BGE
    python run_fusion_rerank.py --retriever elser         # Run with ELSER  
    python run_fusion_rerank.py --domains clapnq          # Run on specific domain
    python run_fusion_rerank.py --skip-existing           # Skip if output exists
"""

import sys
import json
import torch
import subprocess
import argparse
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

# Add project root to path
script_dir = Path(__file__).parent
project_root = script_dir.parents[3]
sys.path.insert(0, str(project_root))

# Import MonoT5 utilities from existing reranker
sys.path.insert(0, str(project_root / "scripts/ideas/retrieval_tasks/mono-t5-as-reranker"))
from utils import MonoT5Scorer, load_qrels, load_queries, DOMAINS, EVAL_SCRIPT_PATH

# Configuration
MODEL_PATH = project_root / "scripts/ideas/retrieval_tasks/mono_t5_oracle_selection/.cache"
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# Paths
BASELINE_RESULTS_DIR = {
    'bm25': project_root / "scripts/baselines/retrieval_scripts/bm25/results",
    'bge': project_root / "scripts/baselines/retrieval_scripts/bge/results",
    'elser': project_root / "scripts/baselines/retrieval_scripts/elser/results",
}
CONTEXT_SUMMARY_RESULTS_DIR = script_dir / "retrieval_results"
OUTPUT_DIR = script_dir / "fusion_rerank_results"
QUERIES_DIR = project_root / "human/retrieval_tasks"

# Collection names for evaluation
COLLECTION_NAMES = {
    'clapnq': 'mt-rag-clapnq-bm25-512-100',
    'govt': 'mt-rag-govt-bm25-512-100',
    'fiqa': 'mt-rag-fiqa-beir-bm25-512-100',
    'cloud': 'mt-rag-ibmcloud-bm25-512-100'
}


def load_baseline_results(domain: str, strategy: str, retriever: str) -> dict:
    """Load baseline retrieval results for a domain, strategy, and retriever."""
    results_dir = BASELINE_RESULTS_DIR[retriever]
    filepath = results_dir / f"{retriever}_{domain}_{strategy}.jsonl"
    
    if not filepath.exists():
        print(f"Warning: {filepath} does not exist")
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


def load_context_summary_results(domain: str, retriever: str) -> dict:
    """Load context summary retrieval results for a specific retriever."""
    filepath = CONTEXT_SUMMARY_RESULTS_DIR / f"context_summary_{domain}_{retriever}.jsonl"
    
    if not filepath.exists():
        print(f"Warning: {filepath} does not exist")
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


def combine_and_deduplicate(results_dict: dict) -> dict:
    """
    Combine retrieval results from multiple strategies and deduplicate by document_id.
    
    Args:
        results_dict: {strategy_name: {task_id: [contexts]}}
    
    Returns:
        {task_id: [unique_documents]}
    """
    # Find common task_ids across all strategies
    all_task_ids = None
    for strategy, results in results_dict.items():
        if results:
            task_ids = set(results.keys())
            if all_task_ids is None:
                all_task_ids = task_ids
            else:
                all_task_ids &= task_ids
    
    if all_task_ids is None:
        return {}
    
    combined = {}
    for task_id in all_task_ids:
        doc_dict = {}  # document_id -> document info
        
        for strategy, results in results_dict.items():
            if task_id in results:
                for ctx in results[task_id]:
                    doc_id = ctx.get('document_id')
                    if doc_id and doc_id not in doc_dict:
                        doc_dict[doc_id] = {
                            'document_id': doc_id,
                            'text': ctx.get('text', ''),
                            'title': ctx.get('title', ''),
                            'source': ctx.get('source', ''),
                            'original_score': ctx.get('score', 0.0),
                            'strategy': strategy
                        }
        
        combined[task_id] = list(doc_dict.values())
    
    return combined


def rerank_with_monot5(scorer: MonoT5Scorer, query: str, documents: list, top_k: int = 100) -> list:
    """Rerank documents using MonoT5."""
    if not documents:
        return []
    
    doc_texts = [doc.get('text', '') for doc in documents]
    
    # Score in batches
    batch_size = 32
    all_scores = []
    for i in range(0, len(doc_texts), batch_size):
        batch_texts = doc_texts[i:i+batch_size]
        batch_scores = scorer.score_batch(query, batch_texts)
        all_scores.extend(batch_scores)
    
    # Combine and sort
    doc_score_pairs = list(zip(documents, all_scores))
    doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
    
    return doc_score_pairs[:top_k]


def save_results(results: dict, domain: str, output_path: Path):
    """Save reranked results in evaluation format."""
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


def run_evaluation(input_file: Path, output_file: Path):
    """Run the standard evaluation script."""
    cmd = [
        sys.executable, str(EVAL_SCRIPT_PATH),
        "--input_file", str(input_file),
        "--output_file", str(output_file)
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running evaluation: {e}")
        if e.stderr:
            print(f"Stderr: {e.stderr}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Multi-Query Fusion + MonoT5 Reranking')
    parser.add_argument(
        '--domains',
        nargs='+',
        choices=DOMAINS,
        default=DOMAINS,
        help='Domains to process'
    )
    parser.add_argument(
        '--retriever',
        type=str,
        default='bm25',
        choices=['bm25', 'bge', 'elser'],
        help='Retriever to use (default: bm25)'
    )
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='Skip if output file already exists'
    )
    parser.add_argument(
        '--rerank-query',
        type=str,
        default='rewrite',
        choices=['rewrite', 'lastturn'],
        help='Query type to use for reranking (default: rewrite)'
    )
    args = parser.parse_args()
    
    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    
    # Check model exists
    if not MODEL_PATH.exists():
        print(f"Error: MonoT5 model not found at {MODEL_PATH}")
        print("Run scripts/ideas/retrieval_tasks/mono_t5_oracle_selection/download_model.py first")
        sys.exit(1)
    
    print("="*80)
    print("Multi-Query Fusion + MonoT5 Reranking")
    print("="*80)
    print(f"Retriever: {args.retriever.upper()}")
    print(f"Strategy: lastturn + rewrite + context_summary → MonoT5 rerank")
    print(f"Reranking query: {args.rerank_query}")
    print(f"Device: {DEVICE}")
    print(f"Domains: {args.domains}")
    print("="*80)
    
    # Initialize MonoT5 scorer
    scorer = MonoT5Scorer(str(MODEL_PATH), DEVICE)
    
    all_results = {}
    
    for domain in args.domains:
        print(f"\n{'='*60}")
        print(f"Processing domain: {domain.upper()}")
        print(f"{'='*60}")
        
        output_file = OUTPUT_DIR / f"fusion_reranked_{domain}_{args.retriever}.jsonl"
        evaluated_file = OUTPUT_DIR / f"fusion_reranked_{domain}_{args.retriever}_evaluated.jsonl"
        
        if args.skip_existing and output_file.exists():
            print(f"Output file exists, skipping: {output_file.name}")
            continue
        
        # Load reranking queries
        queries = load_queries(domain, args.rerank_query)
        print(f"Loaded {len(queries)} {args.rerank_query} queries for reranking")
        
        # Load retrieval results from all three sources
        print("Loading retrieval results...")
        results_dict = {
            'lastturn': load_baseline_results(domain, 'lastturn', args.retriever),
            'rewrite': load_baseline_results(domain, 'rewrite', args.retriever),
            'context_summary': load_context_summary_results(domain, args.retriever)
        }
        
        for strategy, results in results_dict.items():
            print(f"  {strategy}: {len(results)} tasks")
        
        # Combine and deduplicate
        combined = combine_and_deduplicate(results_dict)
        print(f"Combined: {len(combined)} tasks with unique documents")
        
        # Calculate stats
        total_docs = sum(len(docs) for docs in combined.values())
        avg_docs = total_docs / len(combined) if combined else 0
        print(f"Average docs per task: {avg_docs:.1f}")
        
        # Rerank with MonoT5
        reranked_results = {}
        tasks_to_process = [tid for tid in combined.keys() if tid in queries]
        
        for task_id in tqdm(tasks_to_process, desc=f"Reranking {domain}"):
            documents = combined[task_id]
            query = queries[task_id]
            
            reranked = rerank_with_monot5(scorer, query, documents, top_k=100)
            
            doc_scores = {doc['document_id']: score for doc, score in reranked}
            if doc_scores:
                reranked_results[task_id] = doc_scores
        
        # Save results
        save_results(reranked_results, domain, output_file)
        
        # Evaluate
        print(f"Running evaluation...")
        if run_evaluation(output_file, evaluated_file):
            # Print aggregate results
            agg_file = Path(str(evaluated_file).replace('.jsonl', '_aggregate.csv'))
            if agg_file.exists():
                import pandas as pd
                import ast
                df = pd.read_csv(agg_file)
                all_rows = df[df['collection'] == 'all']
                if len(all_rows) > 0:
                    all_row = all_rows.iloc[0]
                    ndcg = ast.literal_eval(all_row['nDCG'])
                    recall = ast.literal_eval(all_row['Recall'])
                    print(f"  Results: nDCG@10={ndcg[3]:.4f}, Recall@10={recall[3]:.4f}")
                    all_results[domain] = {'nDCG@10': ndcg[3], 'Recall@10': recall[3]}
    
    # Print summary
    print("\n" + "="*80)
    print(f"SUMMARY: Multi-Query Fusion + MonoT5 Reranking ({args.retriever.upper()})")
    print("="*80)
    
    if all_results:
        print(f"\n{'Domain':<10} {'nDCG@10':>10} {'Recall@10':>12}")
        print("-"*35)
        for domain, metrics in all_results.items():
            print(f"{domain:<10} {metrics['nDCG@10']:>10.4f} {metrics['Recall@10']:>12.4f}")
        
        # Average
        avg_ndcg = sum(m['nDCG@10'] for m in all_results.values()) / len(all_results)
        avg_recall = sum(m['Recall@10'] for m in all_results.values()) / len(all_results)
        print("-"*35)
        print(f"{'Average':<10} {avg_ndcg:>10.4f} {avg_recall:>12.4f}")
    
    print("\n" + "="*80)
    print("Done!")
    print("="*80)


if __name__ == '__main__':
    main()

