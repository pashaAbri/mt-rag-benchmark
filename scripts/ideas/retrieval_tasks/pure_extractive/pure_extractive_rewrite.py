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
from nltk.tokenize import word_tokenize


class TextPreprocessor:
    """Preprocess text for extractive rewriting."""
    
    def __init__(self):
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            nltk.download('stopwords')
            nltk.download('punkt')
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
        text = re.sub(r'[^a-z0-9\s]', '', text)
        tokens = word_tokenize(text)
        
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
        model_name: str = "all-MiniLM-L6-v2",
        lambda_param: float = 0.7,
        max_terms: int = 10
    ):
        """
        Initialize rewriter.
        
        Args:
            model_name: Sentence embedding model
            lambda_param: MMR lambda (relevance vs diversity)
            max_terms: Maximum terms to select
        """
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


def load_mtrag_queries(domain: str) -> List[Dict]:
    """
    Load MT-RAG queries for a domain.
    
    Args:
        domain: One of ['clapnq', 'cloud', 'fiqa', 'govt']
        
    Returns:
        List of query dicts with history
    """
    import json
    
    lastturn_path = f"human/retrieval_tasks/{domain}/{domain}_lastturn.jsonl"
    with open(lastturn_path) as f:
        queries = [json.loads(line) for line in f]
    
    history_path = f"human/retrieval_tasks/{domain}/{domain}_questions.jsonl"
    with open(history_path) as f:
        histories = [json.loads(line) for line in f]
    
    results = []
    for q, h in zip(queries, histories):
        query_id = q['_id']
        current_query = q['text'].replace('|user|:', '').strip()
        
        history_text = h['text']
        turns = history_text.split('\n')
        
        conversation_history = []
        for turn in turns[:-1]:  # Exclude current query
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


def run_pure_extractive(domain: str):
    """
    Run pure extractive rewriting on MT-RAG dataset.
    
    Args:
        domain: Domain to process
    """
    import json
    import os
    
    rewriter = PureExtractiveRewriter(
        lambda_param=0.7,
        max_terms=10
    )
    
    queries = load_mtrag_queries(domain)
    
    results = []
    for item in queries:
        try:
            rewritten = rewriter.rewrite(
                current_query=item['query'],
                conversation_history=item['history']
            )
            
            results.append({
                "id": item['id'],
                "original": item['query'],
                "rewritten": rewritten,
                "history_length": len(item['history'])
            })
            
            print(f"[{domain}] {item['id']}")
            print(f"  Original:  {item['query']}")
            print(f"  Rewritten: {rewritten}")
            print()
            
        except Exception as e:
            print(f"Error processing {item['id']}: {e}")
            results.append({
                "id": item['id'],
                "original": item['query'],
                "rewritten": item['query'],
                "error": str(e)
            })
    
    # Save to results directory within pure_extractive
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    output_path = os.path.join(results_dir, f"{domain}_rewrites.jsonl")
    
    with open(output_path, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    print(f"Saved {len(results)} rewrites to {output_path}")
    
    return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python pure_extractive_rewrite.py <domain>")
        print("Domain: clapnq, cloud, fiqa, or govt")
        sys.exit(1)
    
    domain = sys.argv[1]
    run_pure_extractive(domain)

