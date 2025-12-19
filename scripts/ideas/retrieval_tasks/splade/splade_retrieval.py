"""
SPLADE Sparse Retrieval Implementation

This script implements sparse retrieval using SPLADE for the MT-RAG benchmark.
Model: naver/splade-cocondenser-ensembledistil (default)

SPLADE produces learned sparse representations that combine the interpretability
of sparse methods with the effectiveness of learned representations.
"""
import argparse
from typing import Dict, List, Tuple
from pathlib import Path
import sys
import numpy as np
import torch
from scipy import sparse
from tqdm import tqdm

# Add parent directory to path to import utils
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "baselines" / "retrieval_scripts"))
from utils import load_corpus, load_queries, save_results, get_collection_name


def get_device():
    """Detect and return the best available device."""
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'


class SpladeEncoder:
    """SPLADE encoder for producing sparse representations."""
    
    def __init__(self, model_path: str, device: str = None):
        """
        Initialize the SPLADE encoder.
        
        Args:
            model_path: Path to the SPLADE model
            device: Device to use for encoding
        """
        from transformers import AutoModelForMaskedLM, AutoTokenizer
        
        self.device = device or get_device()
        print(f"Loading SPLADE model from {model_path}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForMaskedLM.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"SPLADE model loaded on {self.device}")
    
    def encode(self, texts: List[str], batch_size: int = 32, 
               show_progress: bool = True) -> sparse.csr_matrix:
        """
        Encode texts into SPLADE sparse representations.
        
        Args:
            texts: List of text strings to encode
            batch_size: Batch size for encoding
            show_progress: Whether to show progress bar
            
        Returns:
            Sparse matrix of shape (n_texts, vocab_size)
        """
        all_vectors = []
        
        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Encoding with SPLADE")
        
        with torch.no_grad():
            for i in iterator:
                batch_texts = texts[i:i + batch_size]
                
                # Tokenize
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                ).to(self.device)
                
                # Get model outputs
                outputs = self.model(**inputs)
                logits = outputs.logits
                
                # Apply SPLADE aggregation: log(1 + ReLU(x)) * attention_mask
                # This creates sparse representations
                attention_mask = inputs["attention_mask"].unsqueeze(-1)
                relu_log = torch.log1p(torch.relu(logits))
                
                # Max pooling over sequence dimension with attention mask
                relu_log = relu_log * attention_mask
                sparse_vecs = torch.max(relu_log, dim=1).values
                
                # Move to CPU and convert to numpy
                sparse_vecs = sparse_vecs.cpu().numpy()
                all_vectors.append(sparse_vecs)
        
        # Stack all vectors
        all_vectors = np.vstack(all_vectors)
        
        # Convert to sparse matrix (keep only non-zero values)
        sparse_matrix = sparse.csr_matrix(all_vectors)
        
        return sparse_matrix


def sparse_dot_product_topk(query_sparse: sparse.csr_matrix, 
                            doc_sparse: sparse.csr_matrix,
                            top_k: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute dot product between query and document sparse matrices and return top-k.
    
    Args:
        query_sparse: Query sparse matrix (n_queries, vocab_size)
        doc_sparse: Document sparse matrix (n_docs, vocab_size)
        top_k: Number of top results to return
        
    Returns:
        Tuple of (scores, indices) arrays
    """
    n_queries = query_sparse.shape[0]
    
    # Compute all scores at once: queries @ docs.T
    # Result is (n_queries, n_docs)
    scores_matrix = query_sparse.dot(doc_sparse.T).toarray()
    
    # Get top-k for each query
    indices = np.zeros((n_queries, top_k), dtype=np.int64)
    scores = np.zeros((n_queries, top_k), dtype=np.float32)
    
    for i in tqdm(range(n_queries), desc="Finding top-k per query"):
        top_indices = np.argsort(scores_matrix[i])[::-1][:top_k]
        indices[i] = top_indices
        scores[i] = scores_matrix[i][top_indices]
    
    return scores, indices


def load_or_encode_corpus(corpus: Dict, domain: str, encoder: SpladeEncoder,
                          batch_size: int = 32) -> Tuple[sparse.csr_matrix, List[str]]:
    """
    Load cached embeddings or encode corpus and cache them.
    
    Args:
        corpus: Dictionary of documents
        domain: Domain name for caching
        encoder: SpladeEncoder instance
        batch_size: Batch size for encoding
        
    Returns:
        Tuple of (sparse embeddings matrix, list of document IDs)
    """
    script_dir = Path(__file__).parent
    embeddings_dir = script_dir / "embeddings"
    embeddings_dir.mkdir(exist_ok=True)
    
    embeddings_file = embeddings_dir / f"{domain}_splade_embeddings.npz"
    doc_ids_file = embeddings_dir / f"{domain}_splade_doc_ids.npy"
    
    # Check if cached embeddings exist
    if embeddings_file.exists() and doc_ids_file.exists():
        print(f"Loading cached SPLADE embeddings from {embeddings_file}...")
        embeddings = sparse.load_npz(embeddings_file)
        doc_ids = np.load(doc_ids_file, allow_pickle=True).tolist()
        print(f"Loaded {embeddings.shape[0]} cached embeddings")
        return embeddings, doc_ids
    
    # Encode corpus
    print(f"Encoding {len(corpus)} documents with SPLADE (this may take a while)...")
    doc_ids = list(corpus.keys())
    doc_texts = [corpus[doc_id].get('text', '') for doc_id in doc_ids]
    
    # Encode in batches
    embeddings = encoder.encode(doc_texts, batch_size=batch_size, show_progress=True)
    
    # Save embeddings to cache
    print(f"Saving embeddings to {embeddings_file}...")
    sparse.save_npz(embeddings_file, embeddings)
    np.save(doc_ids_file, np.array(doc_ids, dtype=object))
    print(f"Cached {embeddings.shape[0]} embeddings for future use")
    
    return embeddings, doc_ids


def run_splade_retrieval(corpus: Dict, queries: Dict[str, str], domain: str,
                         model_path: str, top_k: int = 10, 
                         batch_size: int = 32) -> Dict[str, Dict[str, float]]:
    """
    Run SPLADE sparse retrieval on the corpus.
    
    Args:
        corpus: Dictionary of documents
        queries: Dictionary of queries
        domain: Domain name for caching
        model_path: Path to SPLADE model
        top_k: Number of top results to return
        batch_size: Batch size for encoding
        
    Returns:
        Dictionary mapping query_id to {doc_id: score}
    """
    # Initialize encoder
    encoder = SpladeEncoder(model_path)
    
    # Load or encode corpus
    corpus_embeddings, doc_ids = load_or_encode_corpus(
        corpus, domain, encoder, batch_size
    )
    
    # Encode queries
    print(f"Encoding {len(queries)} queries with SPLADE...")
    query_ids = list(queries.keys())
    query_texts = []
    for qid in query_ids:
        # Clean query text (remove |user|: prefix if present)
        clean_text = queries[qid].replace('|user|: ', '').replace('|user|:', '').strip()
        query_texts.append(clean_text)
    
    query_embeddings = encoder.encode(query_texts, batch_size=batch_size, show_progress=True)
    
    # Perform sparse retrieval
    print(f"Searching for top-{top_k} results...")
    scores, indices = sparse_dot_product_topk(query_embeddings, corpus_embeddings, top_k)
    
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
    parser = argparse.ArgumentParser(description='Run SPLADE retrieval on MT-RAG benchmark')
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
                        help='Path to local SPLADE model (default: models/splade-cocondenser-ensembledistil)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for encoding')
    
    args = parser.parse_args()
    
    # Set default model path if not provided
    if args.model_path is None:
        script_dir = Path(__file__).parent
        args.model_path = str(script_dir / "models" / "naver-splade-cocondenser-ensembledistil")
    
    # Check if model exists
    if not Path(args.model_path).exists():
        raise FileNotFoundError(
            f"SPLADE model not found at {args.model_path}\n"
            f"Please run: python download_model.py"
        )
    
    print(f"Loading corpus from {args.corpus_file}...")
    corpus = load_corpus(args.corpus_file)
    print(f"Loaded {len(corpus)} documents")
    
    print(f"Loading queries from {args.query_file}...")
    queries = load_queries(args.query_file)
    print(f"Loaded {len(queries)} queries")
    
    print("Running SPLADE retrieval...")
    results_dict = run_splade_retrieval(
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
    print("SPLADE retrieval complete!")


if __name__ == '__main__':
    main()
