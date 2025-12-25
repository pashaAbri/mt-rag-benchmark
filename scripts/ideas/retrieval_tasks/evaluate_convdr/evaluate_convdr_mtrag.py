import os
import json
import torch
import faiss
import numpy as np
import math
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

# Configuration
CHECKPOINT_PATH = os.path.abspath("./checkpoints/convdr-kd-cast19")
BATCH_SIZE_QUERY = 8
BATCH_SIZE_DOC = 32
MAX_SEQ_LENGTH = 512
TOP_K = 10

DOMAINS = ['clapnq', 'fiqa', 'govt', 'cloud']

BASELINES = {
    'clapnq': 0.578,
    'govt': 0.517,
    'cloud': 0.438,
    'fiqa': 0.436
}

def load_qrels(qrels_path):
    qrels = {}
    with open(qrels_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                qid, _, docid, rel = parts
                if qid not in qrels:
                    qrels[qid] = {}
                qrels[qid][docid] = int(rel)
    return qrels

def load_queries(questions_path):
    queries = []
    # Reads queries from jsonl file where text field contains concatenated history
    with open(questions_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            # _id is in format conv_id<::>turn
            qid = data.get('_id')
            text = data.get('text', '')
            # Clean text: remove |user|: and |agent|: tags if present, 
            # though ConvDR might handle them, usually better to have clean text.
            # Based on file inspection, text is like "|user|: ... \n|user|: ..."
            # We will replace newlines with spaces and strip speaker tags for cleaner input
            # although some models might use them. ConvDR paper usually implies concatenated text.
            # We'll just replace newlines with space.
            clean_text = text.replace('\n', ' ').strip()
            queries.append({'id': qid, 'text': clean_text})
    return queries

def load_passages(passages_path):
    passages = []
    ids = []
    with open(passages_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            # MTRAG passage file format: has 'id' and 'text' (and title usually)
            # We concatenate title + text if title exists
            pid = data.get('id')
            title = data.get('title', '')
            text = data.get('text', '')
            full_text = f"{title} {text}".strip()
            passages.append(full_text)
            ids.append(pid)
    return passages, ids

def encode(texts, tokenizer, model, batch_size, device):
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, 
                           max_length=MAX_SEQ_LENGTH, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            # Use [CLS] token embedding (index 0)
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(embeddings)
    return np.vstack(all_embeddings)

def dcg_at_k(r, k):
    r = np.asfarray(r)[:k]
    if r.size:
        return np.sum(r / np.log2(np.arange(2, r.size + 2)))
    return 0.

def ndcg_at_k(r, k):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k) / dcg_max

def evaluate_domain(domain, model, tokenizer, device):
    print(f"\n--- Processing {domain} ---")
    
    # Paths
    base_dir = os.path.abspath(".")
    questions_path = os.path.join(base_dir, f"human/retrieval_tasks/{domain}/{domain}_questions.jsonl")
    qrels_path = os.path.join(base_dir, f"human/retrieval_tasks/{domain}/qrels/dev.tsv")
    passages_path = os.path.join(base_dir, f"corpora/passage_level/{domain}.jsonl")
    
    output_dir = os.path.join(base_dir, "scripts/ideas/retrieval_tasks/evaluate_convdr/results")
    os.makedirs(output_dir, exist_ok=True)
    run_file = os.path.join(output_dir, f"{domain}.trec")

    # Load Data
    print("Loading queries...")
    queries = load_queries(questions_path)
    print(f"Loaded {len(queries)} queries.")
    
    print("Loading qrels...")
    qrels = load_qrels(qrels_path)
    
    print("Loading passages...")
    passage_texts, passage_ids = load_passages(passages_path)
    print(f"Loaded {len(passage_texts)} passages.")

    # Filter queries to those with qrels? Or keep all?
    # Usually we evaluate on all queries in the query file.
    
    # Encode Queries
    print("Encoding queries...")
    query_texts = [q['text'] for q in queries]
    query_embeddings = encode(query_texts, tokenizer, model, BATCH_SIZE_QUERY, device)
    
    # Encode Passages
    print("Encoding passages...")
    passage_embeddings = encode(passage_texts, tokenizer, model, BATCH_SIZE_DOC, device)
    
    # Retrieval with FAISS
    print("Indexing and retrieving...")
    index = faiss.IndexFlatIP(passage_embeddings.shape[1])
    index.add(passage_embeddings)
    
    D, I = index.search(query_embeddings, TOP_K)
    
    # Save Results
    print(f"Saving run file to {run_file}...")
    with open(run_file, 'w') as f:
        for i, (scores, indices) in enumerate(zip(D, I)):
            qid = queries[i]['id']
            for rank, (score, idx) in enumerate(zip(scores, indices)):
                pid = passage_ids[idx]
                f.write(f"{qid} Q0 {pid} {rank+1} {score} ConvDR\n")
    
    # Evaluate
    print("Calculating nDCG@10...")
    ndcg_scores = []
    for i, q in enumerate(queries):
        qid = q['id']
        if qid not in qrels:
            continue
            
        retrieved_indices = I[i]
        relevance_scores = []
        for idx in retrieved_indices:
            pid = passage_ids[idx]
            relevance_scores.append(qrels[qid].get(pid, 0))
            
        ndcg = ndcg_at_k(relevance_scores, 10)
        ndcg_scores.append(ndcg)
        
    avg_ndcg = np.mean(ndcg_scores) if ndcg_scores else 0.0
    return avg_ndcg

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print(f"Loading model from {CHECKPOINT_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_PATH)
    model = AutoModel.from_pretrained(CHECKPOINT_PATH)
    model.to(device)
    model.eval()
    
    results = []
    
    for domain in DOMAINS:
        try:
            ndcg = evaluate_domain(domain, model, tokenizer, device)
            baseline = BASELINES.get(domain, 0.0)
            diff = ndcg - baseline
            results.append({
                'Domain': domain.upper(),
                'ConvDR': ndcg,
                'Baseline': baseline,
                'Difference': diff
            })
        except Exception as e:
            print(f"Error processing {domain}: {e}")
            import traceback
            traceback.print_exc()
            
    # Print Table
    print("\n\n" + "="*60)
    print(f"{'Domain':<10} | {'ConvDR nDCG@10':<15} | {'Baseline nDCG@10':<18} | {'Difference':<10}")
    print("-" * 60)
    for res in results:
        print(f"{res['Domain']:<10} | {res['ConvDR']:<15.4f} | {res['Baseline']:<18.4f} | {res['Difference']:<10.4f}")
    print("="*60 + "\n")
    
    # Simple Analysis
    print("Analysis:")
    for res in results:
        if res['Difference'] > 0:
            print(f"- ConvDR improves on {res['Domain']} by {res['Difference']:.4f}")
        else:
            print(f"- ConvDR hurts on {res['Domain']} by {abs(res['Difference']):.4f}")

if __name__ == "__main__":
    main()

