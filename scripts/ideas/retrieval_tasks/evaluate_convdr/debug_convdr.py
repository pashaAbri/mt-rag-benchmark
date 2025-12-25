#!/usr/bin/env python3
"""
Debug script for ConvDR evaluation - uses small sample for visibility.
Run locally to diagnose the embedding collapse issue.

KEY INSIGHT: ConvDR uses a teacher-student framework:
- TEACHER (ANCE): Encodes PASSAGES (frozen, from pre-training)
- STUDENT (ConvDR): Encodes QUERIES (trained to match teacher on rewrites)

We must use ANCE for passages and ConvDR for queries!
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
# ConvDR (student) - for encoding QUERIES
CONVDR_CHECKPOINT = str(Path(__file__).parent / ".checkpoints" / "convdr-kd-cast19")

# ANCE (teacher) - for encoding PASSAGES
# This is the model ConvDR was trained with as teacher
ANCE_CHECKPOINT = "castorini/ance-msmarco-passage"  # HuggingFace version of ANCE
NUM_QUERIES = 5
NUM_PASSAGES = 100  # Small subset for debugging
MAX_SEQ_LENGTH = 512

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


def parse_query_text(text, method='raw'):
    """Parse query text using different methods."""
    
    if method == 'raw':
        # Original: just replace newlines with spaces
        return text.replace('\n', ' ').strip()
    
    elif method == 'strip_markers':
        # Strip |user|: and |assistant|: markers, join with space
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
    
    elif method == 'sep_token':
        # Strip markers and join with [SEP]
        turns = []
        for turn in text.split('\n'):
            turn = turn.strip()
            if turn.startswith('|user|:'):
                turns.append(turn.replace('|user|:', '').strip())
            elif turn.startswith('|assistant|:'):
                turns.append(turn.replace('|assistant|:', '').strip())
            elif turn:
                turns.append(turn)
        return ' [SEP] '.join(turns)
    
    elif method == 'last_turn':
        # Only use the last user turn
        turns = []
        for turn in text.split('\n'):
            turn = turn.strip()
            if turn.startswith('|user|:'):
                turns.append(turn.replace('|user|:', '').strip())
        return turns[-1] if turns else text
    
    else:
        return text


def main():
    print("=" * 60)
    print("ConvDR Debug Script")
    print("=" * 60)
    
    # Load data first
    print("\n" + "-" * 40)
    print("Loading sample data...")
    queries, passages, passage_ids, qrels = load_sample_data()
    
    print(f"Loaded {len(queries)} queries")
    print(f"Loaded {len(passages)} passages")
    
    # Show raw queries
    print("\n" + "-" * 40)
    print("RAW QUERIES:")
    for i, q in enumerate(queries):
        print(f"\n[Query {i+1}] ID: {q['id']}")
        print(f"  Raw text:\n    {repr(q['text_raw'][:200])}")
        
        # Show relevant passages for this query
        if q['id'] in qrels:
            print(f"  Relevant passages: {list(qrels[q['id']])[:3]}...")
    
    # Load BOTH models
    print("\n" + "-" * 40)
    print("Loading models...")
    print(f"  ConvDR (query encoder): {CONVDR_CHECKPOINT}")
    print(f"  ANCE (passage encoder): {ANCE_CHECKPOINT}")
    
    try:
        # ConvDR for queries
        convdr_tokenizer = AutoTokenizer.from_pretrained(CONVDR_CHECKPOINT)
        convdr_model = AutoModel.from_pretrained(CONVDR_CHECKPOINT)
        convdr_model.eval()
        print("  ✓ ConvDR loaded")
        
        # ANCE for passages
        ance_tokenizer = AutoTokenizer.from_pretrained(ANCE_CHECKPOINT)
        ance_model = AutoModel.from_pretrained(ANCE_CHECKPOINT)
        ance_model.eval()
        print("  ✓ ANCE loaded")
    except Exception as e:
        print(f"  ❌ Failed to load models: {e}")
        return
    
    def encode_with_model(texts, tokenizer, model):
        """Encode texts with a specific model."""
        inputs = tokenizer(texts, padding=True, truncation=True, 
                          max_length=MAX_SEQ_LENGTH, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :].numpy()
        return embeddings
    
    # First, test the WRONG way (what we were doing before)
    print("\n" + "-" * 40)
    print("TEST 1: WRONG WAY (ConvDR for both queries AND passages)")
    print("-" * 40)
    
    parsed_queries = [parse_query_text(q['text_raw'], 'last_turn') for q in queries]
    query_embeddings_wrong = encode_with_model(parsed_queries, convdr_tokenizer, convdr_model)
    passage_embeddings_wrong = encode_with_model(passages[:20], convdr_tokenizer, convdr_model)
    
    from numpy.linalg import norm
    
    # Query-query similarity
    sims = []
    for i in range(len(query_embeddings_wrong)):
        for j in range(i+1, len(query_embeddings_wrong)):
            sim = np.dot(query_embeddings_wrong[i], query_embeddings_wrong[j]) / (
                norm(query_embeddings_wrong[i]) * norm(query_embeddings_wrong[j]))
            sims.append(sim)
    print(f"  Query-query similarity: {np.mean(sims):.4f} (high = BAD)")
    
    # Retrieval
    print(f"\n  Top-3 retrieved per query:")
    for i, (q, q_emb) in enumerate(zip(queries, query_embeddings_wrong)):
        scores = np.dot(passage_embeddings_wrong, q_emb)
        top_indices = np.argsort(scores)[::-1][:3]
        print(f"\n  Q{i+1}: {parsed_queries[i][:50]}...")
        for rank, idx in enumerate(top_indices):
            pid = passage_ids[idx]
            is_rel = "✓" if q['id'] in qrels and pid in qrels[q['id']] else ""
            print(f"    {rank+1}. {pid[:40]}... ({scores[idx]:.1f}) {is_rel}")
    
    # Now test the CORRECT way
    print("\n" + "-" * 40)
    print("TEST 2: CORRECT WAY (ConvDR for queries, ANCE for passages)")
    print("-" * 40)
    
    query_embeddings_correct = encode_with_model(parsed_queries, convdr_tokenizer, convdr_model)
    passage_embeddings_correct = encode_with_model(passages[:20], ance_tokenizer, ance_model)
    
    # Query-query similarity (should be similar since same query encoder)
    sims = []
    for i in range(len(query_embeddings_correct)):
        for j in range(i+1, len(query_embeddings_correct)):
            sim = np.dot(query_embeddings_correct[i], query_embeddings_correct[j]) / (
                norm(query_embeddings_correct[i]) * norm(query_embeddings_correct[j]))
            sims.append(sim)
    print(f"  Query-query similarity: {np.mean(sims):.4f}")
    
    # Retrieval
    print(f"\n  Top-3 retrieved per query:")
    for i, (q, q_emb) in enumerate(zip(queries, query_embeddings_correct)):
        scores = np.dot(passage_embeddings_correct, q_emb)
        top_indices = np.argsort(scores)[::-1][:3]
        print(f"\n  Q{i+1}: {parsed_queries[i][:50]}...")
        for rank, idx in enumerate(top_indices):
            pid = passage_ids[idx]
            is_rel = "✓" if q['id'] in qrels and pid in qrels[q['id']] else ""
            print(f"    {rank+1}. {pid[:40]}... ({scores[idx]:.1f}) {is_rel}")
    
    print("\n" + "=" * 60)
    print("Debug complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

