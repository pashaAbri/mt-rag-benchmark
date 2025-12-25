#!/usr/bin/env python3
"""
Debug script comparing ConvDR vs Contriever for retrieval.
Contriever is designed for zero-shot cross-domain transfer.
"""

import json
import os
import sys
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Check if we have the required packages
try:
    import torch
    import numpy as np
    from transformers import AutoTokenizer, AutoModel
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Run: pip install torch transformers numpy")
    sys.exit(1)

# Configuration
CONTRIEVER_CHECKPOINT = "facebook/contriever"  # Zero-shot dense retriever
NUM_QUERIES = 5
NUM_PASSAGES = 100  # Small subset for debugging
MAX_SEQ_LENGTH = 512


def mean_pooling(token_embeddings, attention_mask):
    """Contriever uses mean pooling instead of CLS token."""
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def load_sample_data(domain='clapnq'):
    """Load a small sample of queries and passages."""
    
    # Load queries
    questions_path = PROJECT_ROOT / f'human/retrieval_tasks/{domain}/{domain}_questions.jsonl'
    queries = []
    with open(questions_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= NUM_QUERIES:
                break
            data = json.loads(line)
            queries.append({
                'id': data['_id'],
                'text_raw': data['text'],
            })
    
    # Load qrels
    qrels_path = PROJECT_ROOT / f'human/retrieval_tasks/{domain}/qrels/dev.tsv'
    qrels = {}
    with open(qrels_path, 'r') as f:
        for i, line in enumerate(f):
            if i == 0:  # Skip header
                continue
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                qid, docid, rel = parts[0], parts[1], parts[2]
                if qid not in qrels:
                    qrels[qid] = set()
                qrels[qid].add(docid)
    
    # Load passages (small subset)
    passages_path = PROJECT_ROOT / f'corpora/passage_level/{domain}.jsonl'
    passages = []
    passage_ids = []
    
    # First, get the relevant passage IDs for our queries
    relevant_pids = set()
    for q in queries:
        if q['id'] in qrels:
            relevant_pids.update(qrels[q['id']])
    
    print(f"Relevant passage IDs for {NUM_QUERIES} queries: {len(relevant_pids)}")
    
    # Load passages - prioritize relevant ones
    with open(passages_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            pid = data['id']
            title = data.get('title', '')
            text = data.get('text', '')
            full_text = f"{title} {text}".strip()
            
            # Include if relevant OR if we haven't hit our limit
            if pid in relevant_pids or len(passages) < NUM_PASSAGES:
                passages.append(full_text)
                passage_ids.append(pid)
                
            if len(passages) >= NUM_PASSAGES + len(relevant_pids):
                break
    
    return queries, passages, passage_ids, qrels


def parse_query_text(text, method='last_turn'):
    """Parse query text - strip role markers."""
    
    if method == 'last_turn':
        # Only use the last user turn
        turns = []
        for turn in text.split('\n'):
            turn = turn.strip()
            if turn.startswith('|user|:'):
                turns.append(turn.replace('|user|:', '').strip())
        return turns[-1] if turns else text
    
    elif method == 'all_turns':
        # Use all user turns joined
        turns = []
        for turn in text.split('\n'):
            turn = turn.strip()
            if turn.startswith('|user|:'):
                turns.append(turn.replace('|user|:', '').strip())
            elif turn.startswith('|assistant|:'):
                turns.append(turn.replace('|assistant|:', '').strip())
            elif turn:
                turns.append(turn)
        return ' '.join(turns)
    
    else:
        return text.replace('\n', ' ').strip()


def main():
    print("=" * 60)
    print("Contriever Debug Script")
    print("=" * 60)
    
    # Load data first
    print("\n" + "-" * 40)
    print("Loading sample data...")
    queries, passages, passage_ids, qrels = load_sample_data()
    
    print(f"Loaded {len(queries)} queries")
    print(f"Loaded {len(passages)} passages")
    
    # Show raw queries
    print("\n" + "-" * 40)
    print("QUERIES:")
    for i, q in enumerate(queries):
        print(f"\n[Query {i+1}] ID: {q['id']}")
        last_turn = parse_query_text(q['text_raw'], 'last_turn')
        print(f"  Last turn: {last_turn}")
        
        # Show relevant passages for this query
        if q['id'] in qrels:
            print(f"  Relevant passages: {list(qrels[q['id']])[:3]}...")
    
    # Load Contriever
    print("\n" + "-" * 40)
    print(f"Loading Contriever from {CONTRIEVER_CHECKPOINT}...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(CONTRIEVER_CHECKPOINT)
        model = AutoModel.from_pretrained(CONTRIEVER_CHECKPOINT)
        model.eval()
        print("✓ Contriever loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    def encode_with_contriever(texts):
        """Encode texts with Contriever (uses mean pooling)."""
        inputs = tokenizer(texts, padding=True, truncation=True, 
                          max_length=MAX_SEQ_LENGTH, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            # Contriever uses mean pooling, not CLS
            embeddings = mean_pooling(outputs.last_hidden_state, inputs['attention_mask'])
        return embeddings.numpy()
    
    # Parse queries
    parsed_queries_last = [parse_query_text(q['text_raw'], 'last_turn') for q in queries]
    parsed_queries_all = [parse_query_text(q['text_raw'], 'all_turns') for q in queries]
    
    # Encode
    print("\n" + "-" * 40)
    print("Encoding with Contriever...")
    
    query_embeddings_last = encode_with_contriever(parsed_queries_last)
    query_embeddings_all = encode_with_contriever(parsed_queries_all)
    passage_embeddings = encode_with_contriever(passages)
    
    print(f"  Query embeddings shape: {query_embeddings_last.shape}")
    print(f"  Passage embeddings shape: {passage_embeddings.shape}")
    
    from numpy.linalg import norm
    
    # Test with last turn only
    print("\n" + "-" * 40)
    print("TEST 1: Last turn only")
    print("-" * 40)
    
    # Query-query similarity
    sims = []
    for i in range(len(query_embeddings_last)):
        for j in range(i+1, len(query_embeddings_last)):
            sim = np.dot(query_embeddings_last[i], query_embeddings_last[j]) / (
                norm(query_embeddings_last[i]) * norm(query_embeddings_last[j]))
            sims.append(sim)
    print(f"  Query-query similarity: {np.mean(sims):.4f}")
    print(f"  (Lower = better discrimination between queries)")
    
    # Retrieval
    print(f"\n  Top-5 retrieved per query:")
    hits = 0
    total = 0
    for i, (q, q_emb) in enumerate(zip(queries, query_embeddings_last)):
        scores = np.dot(passage_embeddings, q_emb)
        top_indices = np.argsort(scores)[::-1][:5]
        print(f"\n  Q{i+1}: {parsed_queries_last[i][:50]}...")
        for rank, idx in enumerate(top_indices):
            pid = passage_ids[idx]
            is_rel = "✓ RELEVANT" if q['id'] in qrels and pid in qrels[q['id']] else ""
            if is_rel:
                hits += 1
            print(f"    {rank+1}. {pid[:40]}... ({scores[idx]:.2f}) {is_rel}")
        total += 1
    
    print(f"\n  Hits in top-5: {hits}/{total*5}")
    
    # Test with all turns
    print("\n" + "-" * 40)
    print("TEST 2: All conversation turns")
    print("-" * 40)
    
    # Query-query similarity
    sims = []
    for i in range(len(query_embeddings_all)):
        for j in range(i+1, len(query_embeddings_all)):
            sim = np.dot(query_embeddings_all[i], query_embeddings_all[j]) / (
                norm(query_embeddings_all[i]) * norm(query_embeddings_all[j]))
            sims.append(sim)
    print(f"  Query-query similarity: {np.mean(sims):.4f}")
    
    # Retrieval
    print(f"\n  Top-5 retrieved per query:")
    hits = 0
    for i, (q, q_emb) in enumerate(zip(queries, query_embeddings_all)):
        scores = np.dot(passage_embeddings, q_emb)
        top_indices = np.argsort(scores)[::-1][:5]
        print(f"\n  Q{i+1}: {parsed_queries_all[i][:50]}...")
        for rank, idx in enumerate(top_indices):
            pid = passage_ids[idx]
            is_rel = "✓ RELEVANT" if q['id'] in qrels and pid in qrels[q['id']] else ""
            if is_rel:
                hits += 1
            print(f"    {rank+1}. {pid[:40]}... ({scores[idx]:.2f}) {is_rel}")
    
    print(f"\n  Hits in top-5: {hits}/{total*5}")
    
    print("\n" + "=" * 60)
    print("Debug complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
