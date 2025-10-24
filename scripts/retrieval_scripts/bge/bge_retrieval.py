"""
BGE-base 1.5 Dense Retrieval Implementation

This script implements dense retrieval using BGE-base-en-v1.5 for the MT-RAG benchmark.
Model: BAAI/bge-base-en-v1.5
"""
import argparse
from typing import Dict, List, Tuple
from pathlib import Path
import sys
import numpy as np
import torch

# Add parent directory to path to import utils
sys.path.append(str(Path(__file__).parent.parent))
from utils import load_corpus, load_queries, save_results, get_collection_name

from sentence_transformers import SentenceTransformer
from tqdm import tqdm

try:
    import faiss
    FAISS_AVAILABLE = True
except (ImportError, OSError):
    FAISS_AVAILABLE = False
    print("Warning: FAISS not available, using numpy similarity search (slower but more stable)")


def get_device():
    """Detect and return the best available device."""
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'


def numpy_similarity_search(query_embeddings: np.ndarray, corpus_embeddings: np.ndarray, 
                            top_k: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform similarity search using numpy (fallback when FAISS fails).
    
    Args:
        query_embeddings: Query embeddings (n_queries, dim)
        corpus_embeddings: Corpus embeddings (n_docs, dim)
        top_k: Number of top results to return
        
    Returns:
        Tuple of (scores, indices) arrays
    """
    print("Using numpy-based similarity search (stable on ARM Macs)...")
    n_queries = query_embeddings.shape[0]
    
    # Compute cosine similarity (embeddings are already normalized)
    # similarity = query @ corpus.T
    scores_all = np.matmul(query_embeddings, corpus_embeddings.T)
    
    # Get top-k indices and scores for each query
    indices = np.zeros((n_queries, top_k), dtype=np.int64)
    scores = np.zeros((n_queries, top_k), dtype=np.float32)
    
    for i in tqdm(range(n_queries), desc="Finding top-k per query"):
        # Get top-k indices (argsort in descending order)
        top_indices = np.argsort(scores_all[i])[::-1][:top_k]
        indices[i] = top_indices
        scores[i] = scores_all[i][top_indices]
    
    return scores, indices


def load_or_encode_corpus(corpus: Dict, domain: str, model: SentenceTransformer, 
                          batch_size: int = 64) -> Tuple[np.ndarray, List[str]]:
    """
    Load cached embeddings or encode corpus and cache them.
    
    Args:
        corpus: Dictionary of documents
        domain: Domain name for caching
        model: SentenceTransformer model
        batch_size: Batch size for encoding
        
    Returns:
        Tuple of (embeddings array, list of document IDs)
    """
    script_dir = Path(__file__).parent
    embeddings_dir = script_dir / "embeddings"
    embeddings_dir.mkdir(exist_ok=True)
    
    embeddings_file = embeddings_dir / f"{domain}_embeddings.npy"
    doc_ids_file = embeddings_dir / f"{domain}_doc_ids.npy"
    
    # Check if cached embeddings exist
    if embeddings_file.exists() and doc_ids_file.exists():
        print(f"Loading cached embeddings from {embeddings_file}...")
        embeddings = np.load(embeddings_file)
        doc_ids = np.load(doc_ids_file, allow_pickle=True).tolist()
        # Ensure loaded embeddings are contiguous float32 for FAISS
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        print(f"Loaded {len(embeddings)} cached embeddings")
        return embeddings, doc_ids
    
    # Encode corpus
    print(f"Encoding {len(corpus)} documents (this may take several minutes)...")
    doc_ids = list(corpus.keys())
    doc_texts = [corpus[doc_id].get('text', '') for doc_id in doc_ids]
    
    # Encode in batches with progress bar
    embeddings = model.encode(
        doc_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # Normalize for cosine similarity
        device=None  # Use model's device
    )
    
    # Ensure embeddings are numpy arrays on CPU (required for FAISS)
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu().numpy()
    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
    
    # Save embeddings to cache
    print(f"Saving embeddings to {embeddings_file}...")
    np.save(embeddings_file, embeddings)
    np.save(doc_ids_file, np.array(doc_ids, dtype=object))
    print(f"Cached {len(embeddings)} embeddings for future use")
    
    return embeddings, doc_ids


def run_bge_retrieval(corpus: Dict, queries: Dict[str, str], domain: str,
                     model_path: str, top_k: int = 10, batch_size: int = 64) -> Dict[str, Dict[str, float]]:
    """
    Run BGE dense retrieval on the corpus.
    
    Args:
        corpus: Dictionary of documents
        queries: Dictionary of queries
        domain: Domain name for caching
        model_path: Path to BGE model
        top_k: Number of top results to return
        batch_size: Batch size for encoding
        
    Returns:
        Dictionary mapping query_id to {doc_id: score}
    """
    # Detect device
    device = get_device()
    print(f"Using device: {device}")
    
    # Load BGE model
    print(f"Loading BGE model from {model_path}...")
    model = SentenceTransformer(model_path, device=device)
    print(f"Model loaded successfully")
    
    # Load or encode corpus
    corpus_embeddings, doc_ids = load_or_encode_corpus(corpus, domain, model, batch_size)
    
    # Encode queries
    print(f"Encoding {len(queries)} queries...")
    query_ids = list(queries.keys())
    query_texts = []
    for qid in query_ids:
        # Clean query text (remove |user|: prefix if present)
        clean_text = queries[qid].replace('|user|: ', '').replace('|user|:', '').strip()
        query_texts.append(clean_text)
    
    query_embeddings = model.encode(
        query_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
        device=None  # Use model's device
    )
    
    # Ensure query embeddings are numpy arrays on CPU
    if isinstance(query_embeddings, torch.Tensor):
        query_embeddings = query_embeddings.cpu().numpy()
    query_embeddings = np.ascontiguousarray(query_embeddings, dtype=np.float32)
    
    # Perform similarity search
    print(f"Searching for top-{top_k} results...")
    
    # Use numpy-based search (more stable on ARM Macs than FAISS)
    scores, indices = numpy_similarity_search(query_embeddings, corpus_embeddings, top_k)
    
    # Format results
    results_dict = {}
    for i, qid in enumerate(tqdm(query_ids, desc="Formatting results")):
        results_dict[qid] = {}
        for j in range(len(indices[i])):
            doc_idx = indices[i][j]
            score = float(scores[i][j])
            doc_id = doc_ids[doc_idx]
            results_dict[qid][doc_id] = score
    
    return results_dict


def main():
    parser = argparse.ArgumentParser(description='Run BGE-base 1.5 retrieval on MT-RAG benchmark')
    parser.add_argument('--domain', type=str, required=True, 
                        choices=['clapnq', 'fiqa', 'govt', 'cloud'],
                        help='Domain to run retrieval on')
    parser.add_argument('--query_type', type=str, required=True,
                        choices=['lastturn', 'rewrite', 'questions'],
                        help='Type of queries to use')
    parser.add_argument('--corpus_file', type=str, required=True,
                        help='Path to corpus JSONL file')
    parser.add_argument('--query_file', type=str, required=True,
                        help='Path to queries JSONL file')
    parser.add_argument('--output_file', type=str, required=True,
                        help='Path to output results file')
    parser.add_argument('--top_k', type=int, default=10,
                        help='Number of top results to retrieve')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to local BGE model (default: models/bge-base-en-v1.5)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for encoding')
    
    args = parser.parse_args()
    
    # Set default model path if not provided
    if args.model_path is None:
        script_dir = Path(__file__).parent
        args.model_path = str(script_dir / "models" / "bge-base-en-v1.5")
    
    # Check if model exists
    if not Path(args.model_path).exists():
        raise FileNotFoundError(
            f"BGE model not found at {args.model_path}\n"
            f"Please run: python download_model.py"
        )
    
    print(f"Loading corpus from {args.corpus_file}...")
    corpus = load_corpus(args.corpus_file)
    print(f"Loaded {len(corpus)} documents")
    
    print(f"Loading queries from {args.query_file}...")
    queries = load_queries(args.query_file)
    print(f"Loaded {len(queries)} queries")
    
    print(f"Running BGE retrieval...")
    results_dict = run_bge_retrieval(
        corpus, queries, args.domain, args.model_path, args.top_k, args.batch_size
    )
    
    # Format results for evaluation script
    collection_name = get_collection_name(args.domain)
    results = []
    for query_id, doc_scores in results_dict.items():
        contexts = []
        for doc_id, score in sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:args.top_k]:
            contexts.append({
                'document_id': doc_id,
                'score': score,
                'text': corpus[doc_id].get('text', ''),
                'title': corpus[doc_id].get('title', ''),
                'source': corpus[doc_id].get('url', '')
            })
        
        results.append({
            'task_id': query_id,
            'Collection': collection_name,
            'contexts': contexts
        })
    
    save_results(results, args.output_file)
    print("BGE retrieval complete!")


if __name__ == '__main__':
    main()


