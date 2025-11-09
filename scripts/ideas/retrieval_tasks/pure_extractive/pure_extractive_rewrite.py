"""
Pure Extractive Query Rewriting using MMR
Approach 1: MMR-Based Term Selection
"""

from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
import nltk
from nltk.corpus import stopwords
from pathlib import Path
import os


class TextPreprocessor:
    """Preprocess text for extractive rewriting."""
    
    def __init__(self):
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            nltk.download('stopwords')
            self.stop_words = set(stopwords.words('english'))
    
    def preprocess(self, text: str) -> List[str]:
        """
        Tokenize and clean text.
        
        Args:
            text: Input text
            
        Returns:
            List of cleaned tokens
        """
        text = text.lower()
        # Remove punctuation but keep spaces
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        # Simple tokenization by splitting on whitespace
        tokens = text.split()
        # Remove extra empty strings
        tokens = [t for t in tokens if t]
        
        # Keep question words even if they're stop words
        question_words = {'what', 'where', 'when', 'who', 'why', 'how', 'which'}
        tokens = [
            t for t in tokens
            if t not in self.stop_words or t in question_words
        ]
        
        return tokens


class PureExtractiveRewriter:
    """
    Pure extractive query rewriting using MMR.
    """
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-base-en-v1.5",
        lambda_param: float = 0.7,
        max_terms: int = 10
    ):
        """
        Initialize rewriter.
        
        Args:
            model_name: Sentence embedding model (default: bge-base-en-v1.5)
            lambda_param: MMR lambda (relevance vs diversity)
            max_terms: Maximum terms to select
        """
        # Set up paths
        script_dir = Path(__file__).parent
        self.root_dir = script_dir.parent.parent.parent.parent  # Workspace root
        models_dir = script_dir.parent / ".models"
        local_model_path = models_dir / "bge-base-en-v1.5"
        
        # Load embedding model
        if local_model_path.exists():
            print(f"Loading local model from {local_model_path}")
            self.encoder = SentenceTransformer(str(local_model_path))
        else:
            print(f"Loading model from HuggingFace: {model_name}")
            self.encoder = SentenceTransformer(model_name)
        
        self.lambda_param = lambda_param
        self.max_terms = max_terms
        self.preprocessor = TextPreprocessor()
    
    def rewrite(
        self,
        current_query: str,
        conversation_history: List[Dict[str, str]]
    ) -> str:
        """
        Rewrite query using pure extractive MMR approach.
        
        Args:
            current_query: User's latest question
            conversation_history: Previous turns
            
        Returns:
            Rewritten query (concatenated keywords)
        """
        query_tokens = self.preprocessor.preprocess(current_query)
        history_text = self.format_history(conversation_history)
        
        candidates = self.extract_candidates(current_query, history_text)
        selected = self.mmr_select(
            query=current_query,
            candidates=candidates,
            k=self.max_terms
        )
        
        rewritten = " ".join(selected)
        return rewritten
    
    def format_history(
        self,
        conversation_history: List[Dict[str, str]]
    ) -> str:
        """
        Format conversation history into text.
        
        Args:
            conversation_history: List of turns
            
        Returns:
            Formatted history string
        """
        history_parts = []
        
        for turn in conversation_history:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            
            if role == "user":
                history_parts.append(content)
        
        return " ".join(history_parts)
    
    def extract_candidates(
        self,
        current_query: str,
        history_text: str
    ) -> List[str]:
        """
        Extract candidate terms/phrases from query and history.
        
        Candidates include:
        1. Unigrams (single words)
        2. Bigrams (two-word phrases)
        3. Trigrams (three-word phrases)
        4. Named entities (if spaCy available)
        """
        candidates = []
        full_text = f"{current_query} {history_text}"
        
        tokens = self.preprocessor.preprocess(full_text)
        
        # Unigrams
        candidates.extend(tokens)
        
        # Bigrams
        candidates.extend([
            f"{tokens[i]} {tokens[i+1]}"
            for i in range(len(tokens)-1)
        ])
        
        # Trigrams
        candidates.extend([
            f"{tokens[i]} {tokens[i+1]} {tokens[i+2]}"
            for i in range(len(tokens)-2)
        ])
        
        # Named entities (optional)
        try:
            import spacy
            nlp = spacy.load("en_core_web_sm")
            doc = nlp(full_text)
            entities = [ent.text.lower() for ent in doc.ents]
            candidates.extend(entities)
        except:
            pass
        
        # Remove duplicates while preserving order
        seen = set()
        unique_candidates = []
        for c in candidates:
            if c not in seen:
                seen.add(c)
                unique_candidates.append(c)
        
        return unique_candidates
    
    def mmr_select(
        self,
        query: str,
        candidates: List[str],
        k: int = 10
    ) -> List[str]:
        """
        Select k terms using Maximal Marginal Relevance.
        
        Args:
            query: Current user query (for relevance)
            candidates: List of candidate terms
            k: Number of terms to select
            
        Returns:
            List of selected terms
        """
        if not candidates:
            return []
        
        query_emb = self.encoder.encode([query])[0]
        candidate_embs = self.encoder.encode(candidates)
        
        selected_indices = []
        selected_embs = []
        
        for _ in range(min(k, len(candidates))):
            mmr_scores = []
            
            for i, cand_emb in enumerate(candidate_embs):
                if i in selected_indices:
                    mmr_scores.append(-np.inf)
                    continue
                
                # Relevance to query
                relevance = cosine_similarity(
                    [query_emb],
                    [cand_emb]
                )[0][0]
                
                # Similarity to already selected (redundancy)
                if not selected_embs:
                    redundancy = 0
                else:
                    similarities = cosine_similarity(
                        [cand_emb],
                        selected_embs
                    )
                    redundancy = np.max(similarities)
                
                # MMR score
                mmr_score = (
                    self.lambda_param * relevance -
                    (1 - self.lambda_param) * redundancy
                )
                mmr_scores.append(mmr_score)
            
            best_idx = np.argmax(mmr_scores)
            selected_indices.append(best_idx)
            selected_embs.append(candidate_embs[best_idx])
        
        return [candidates[i] for i in selected_indices]


def load_mtrag_queries(domain: str, root_dir: Path = None) -> List[Dict]:
    """
    Load MT-RAG queries for a domain.
    
    Args:
        domain: One of ['clapnq', 'cloud', 'fiqa', 'govt']
        root_dir: Workspace root directory (optional, auto-detected if not provided)
        
    Returns:
        List of query dicts with history
    """
    import json
    
    # Get workspace root if not provided (4 levels up from this script)
    if root_dir is None:
        script_dir = Path(__file__).parent
        root_dir = script_dir.parent.parent.parent.parent
    
    questions_path = root_dir / "human" / "retrieval_tasks" / domain / f"{domain}_questions.jsonl"
    
    if not questions_path.exists():
        raise FileNotFoundError(f"Questions file not found: {questions_path}")
    
    with open(questions_path) as f:
        questions = [json.loads(line) for line in f]
    
    results = []
    for item in questions:
        query_id = item['_id']
        full_text = item['text']
        turns = full_text.split('\n')
        
        # Last turn is the current query
        current_query = turns[-1].replace('|user|:', '').strip()
        
        # Previous turns are the conversation history
        conversation_history = []
        for turn in turns[:-1]:
            content = turn.replace('|user|:', '').strip()
            if content:
                conversation_history.append({
                    "role": "user",
                    "content": content
                })
        
        results.append({
            "id": query_id,
            "query": current_query,
            "history": conversation_history
        })
    
    return results


def run_pure_extractive(domain: str, output_format: str = 'mtrag'):
    """
    Run pure extractive rewriting on MT-RAG dataset.
    
    Args:
        domain: Domain to process
        output_format: 'mtrag' (retrieval format, default)
    """
    import json
    
    rewriter = PureExtractiveRewriter(
        lambda_param=0.7,
        max_terms=10
    )
    
    queries = load_mtrag_queries(domain, root_dir=rewriter.root_dir)
    
    results_mtrag = []
    
    for item in queries:
        try:
            rewritten = rewriter.rewrite(
                current_query=item['query'],
                conversation_history=item['history']
            )
            
            # MT-RAG format (for retrieval)
            results_mtrag.append({
                "_id": item['id'],
                "text": f"|user|: {rewritten}"
            })
            
            print(f"[{domain}] {item['id']}")
            print(f"  Original:  {item['query']}")
            print(f"  Rewritten: {rewritten}")
            print()
            
        except Exception as e:
            print(f"Error processing {item['id']}: {e}")
            results_mtrag.append({
                "_id": item['id'],
                "text": f"|user|: {item['query']}"
            })
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Save MT-RAG format
    datasets_dir = os.path.join(script_dir, 'datasets')
    os.makedirs(datasets_dir, exist_ok=True)
    output_path = os.path.join(datasets_dir, f"{domain}_pure_extractive.jsonl")
    
    with open(output_path, 'w') as f:
        for result in results_mtrag:
            f.write(json.dumps(result) + '\n')
    
    print(f"✓ Saved {len(results_mtrag)} queries to {output_path}")
    
    return results_mtrag


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python pure_extractive_rewrite.py <domain>")
        print("Domain: clapnq, cloud, fiqa, or govt")
        sys.exit(1)
    
    domain = sys.argv[1]
    run_pure_extractive(domain)

