"""
Reciprocal Rank Fusion (RRF) for Multi-Strategy Retrieval

Combines results from multiple retrieval strategies using reciprocal rank fusion.
Works with existing MT-RAG retrieval result files.

Reference:
- Cormack et al., "Reciprocal Rank Fusion outperforms Condorcet", SIGIR 2009
- Used in Elasticsearch hybrid search, RAG Fusion, production RAG systems
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List
from collections import defaultdict


def load_retrieval_results(filepath: str) -> Dict[str, List[tuple]]:
    """
    Load retrieval results from MT-RAG format JSONL file.
    
    Format:
    {
        "task_id": "query_id",
        "Collection": "collection_name",
        "contexts": [
            {"document_id": "doc1", "score": 0.95, "text": "...", "title": "..."},
            {"document_id": "doc2", "score": 0.87, ...},
            ...
        ]
    }
    
    Returns:
        {query_id: [(doc_id, rank, score, text, title, source), ...]}
    """
    results = {}
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            query_id = data['task_id']
            contexts = data['contexts']
            
            # Store (doc_id, rank, score, text, title, source) tuples
            # Rank is 1-indexed position in contexts list
            doc_list = []
            for rank, ctx in enumerate(contexts, start=1):
                doc_list.append((
                    ctx['document_id'],
                    rank,
                    ctx.get('score', 0.0),
                    ctx.get('text', ''),
                    ctx.get('title', ''),
                    ctx.get('source', '')
                ))
            
            results[query_id] = doc_list
    
    return results


def reciprocal_rank_fusion(
    strategy_results: List[Dict[str, List[tuple]]],
    k: int = 60
) -> Dict[str, List[tuple]]:
    """
    Apply Reciprocal Rank Fusion to combine multiple retrieval strategies.
    
    RRF formula: score(d) = Σ 1/(k + rank_i(d))
    where rank_i(d) is the rank of document d in strategy i's results
    
    Args:
        strategy_results: List of retrieval results from different strategies
                         Each is {query_id: [(doc_id, rank, score, text, title, source), ...]}
        k: RRF constant (default: 60, as per original paper)
    
    Returns:
        Fused results: {query_id: [(doc_id, rrf_score, text, title, source), ...]}
    """
    # Get all query IDs (should be same across all strategies)
    all_query_ids = set()
    for results in strategy_results:
        all_query_ids.update(results.keys())
    
    fused_results = {}
    
    for query_id in all_query_ids:
        # For this query, collect RRF scores for each document
        doc_data = defaultdict(lambda: {
            'rrf_score': 0.0,
            'text': '',
            'title': '',
            'source': ''
        })
        
        for strategy_idx, results in enumerate(strategy_results):
            if query_id not in results:
                continue
            
            # For each document in this strategy's results
            for doc_id, rank, score, text, title, source in results[query_id]:
                # Add RRF contribution from this strategy
                doc_data[doc_id]['rrf_score'] += 1.0 / (k + rank)
                
                # Keep metadata from first occurrence (or any strategy that has it)
                if not doc_data[doc_id]['text'] and text:
                    doc_data[doc_id]['text'] = text
                if not doc_data[doc_id]['title'] and title:
                    doc_data[doc_id]['title'] = title
                if not doc_data[doc_id]['source'] and source:
                    doc_data[doc_id]['source'] = source
        
        # Convert to sorted list of tuples
        doc_list = [
            (doc_id, data['rrf_score'], data['text'], data['title'], data['source'])
            for doc_id, data in doc_data.items()
        ]
        
        # Sort by RRF score (descending)
        doc_list.sort(key=lambda x: x[1], reverse=True)
        
        fused_results[query_id] = doc_list
    
    return fused_results


def save_results(
    fused_results: Dict[str, List[tuple]],
    collection_name: str,
    output_path: str,
    top_k: int = 10
):
    """
    Save fused results in MT-RAG format.
    
    Args:
        fused_results: {query_id: [(doc_id, rrf_score, text, title, source), ...]}
        collection_name: Name of the collection
        output_path: Path to output JSONL file
        top_k: Number of top documents to keep
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for query_id in sorted(fused_results.keys()):
            doc_list = fused_results[query_id][:top_k]  # Keep only top-k
            
            contexts = []
            for doc_id, rrf_score, text, title, source in doc_list:
                contexts.append({
                    'document_id': doc_id,
                    'score': rrf_score,
                    'text': text,
                    'title': title,
                    'source': source
                })
            
            result = {
                'task_id': query_id,
                'Collection': collection_name,
                'contexts': contexts
            }
            
            f.write(json.dumps(result) + '\n')


def main():
    parser = argparse.ArgumentParser(
        description='Apply Reciprocal Rank Fusion to combine retrieval strategies'
    )
    parser.add_argument(
        '--input_files',
        nargs='+',
        required=True,
        help='Paths to retrieval result files (JSONL format, MT-RAG format)'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        required=True,
        help='Path to save fused results'
    )
    parser.add_argument(
        '--collection_name',
        type=str,
        required=True,
        help='Collection name for output (e.g., mt-rag-clapnq-elser-512-100)'
    )
    parser.add_argument(
        '--top_k',
        type=int,
        default=10,
        help='Number of top documents to keep (default: 10)'
    )
    parser.add_argument(
        '--rrf_k',
        type=int,
        default=60,
        help='RRF constant k (default: 60, as per original paper)'
    )
    
    args = parser.parse_args()
    
    # Load all retrieval results
    print(f"\n{'='*80}")
    print(f"Reciprocal Rank Fusion (RRF) - Multi-Strategy Retrieval")
    print(f"{'='*80}\n")
    
    print(f"Loading {len(args.input_files)} strategy results...")
    strategy_results = []
    for i, filepath in enumerate(args.input_files, 1):
        print(f"  [{i}] {Path(filepath).name}")
        results = load_retrieval_results(filepath)
        strategy_results.append(results)
        print(f"      → {len(results)} queries")
    
    # Apply RRF fusion
    print(f"\nApplying RRF fusion (k={args.rrf_k})...")
    fused_results = reciprocal_rank_fusion(strategy_results, k=args.rrf_k)
    print(f"  → Fused results for {len(fused_results)} queries")
    
    # Save results
    print(f"\nSaving results (top-{args.top_k} per query)...")
    save_results(fused_results, args.collection_name, args.output_file, args.top_k)
    print(f"  → {args.output_file}")
    
    # Print sample
    sample_query = list(fused_results.keys())[0]
    print(f"\nSample fused results for query '{sample_query}':")
    for i, (doc_id, rrf_score, _, _, _) in enumerate(fused_results[sample_query][:5], 1):
        print(f"  {i}. {doc_id}: {rrf_score:.6f}")
    
    print(f"\n{'='*80}")
    print(f"✓ Fusion complete!")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()

