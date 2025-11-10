"""
Hybrid Extractive Query Rewriting
Approach 2: MMR + Templates + Entity Recognition
"""

from typing import List, Dict, Optional
import numpy as np
import spacy
from sklearn.metrics.pairwise import cosine_similarity
import sys
import os
from pathlib import Path
from sentence_transformers import SentenceTransformer
import re
import nltk
from nltk.corpus import stopwords


class TextPreprocessor:
    """Preprocess text for extractive rewriting."""
    
    def __init__(self):
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            nltk.download('stopwords')
            self.stop_words = set(stopwords.words('english'))
    
    def preprocess(self, text: str) -> List[str]:
        """Tokenize and clean text."""
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
    """Pure extractive query rewriting using MMR."""
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-base-en-v1.5",
        lambda_param: float = 0.7,
        max_terms: int = 10
    ):
        """Initialize rewriter."""
        # Set up paths
        script_dir = Path(__file__).resolve().parent
        # Go up 5 levels: hybrid_extractive/ -> retrieval_tasks/ -> ideas/ -> scripts/ -> workspace root
        self.root_dir = script_dir.parent.parent.parent.parent
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
    
    def format_history(self, conversation_history: List[Dict[str, str]]) -> str:
        """Format conversation history into text."""
        history_parts = []
        
        for turn in conversation_history:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            
            if role == "user":
                history_parts.append(content)
        
        return " ".join(history_parts)
    
    def extract_candidates(self, current_query: str, history_text: str) -> List[str]:
        """Extract candidate terms/phrases from query and history."""
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
        
        # Remove duplicates while preserving order
        seen = set()
        unique_candidates = []
        for c in candidates:
            if c not in seen:
                seen.add(c)
                unique_candidates.append(c)
        
        return unique_candidates
    
    def mmr_select(self, query: str, candidates: List[str], k: int = 10) -> List[str]:
        """Select k terms using Maximal Marginal Relevance."""
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
                relevance = np.dot(query_emb, cand_emb) / (
                    np.linalg.norm(query_emb) * np.linalg.norm(cand_emb)
                )
                
                # Similarity to already selected (redundancy)
                if not selected_embs:
                    redundancy = 0
                else:
                    similarities = [
                        np.dot(cand_emb, sel_emb) / (
                            np.linalg.norm(cand_emb) * np.linalg.norm(sel_emb)
                        )
                        for sel_emb in selected_embs
                    ]
                    redundancy = max(similarities)
                
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
    """Load MT-RAG queries for a domain."""
    import json
    
    # Get workspace root if not provided
    if root_dir is None:
        script_dir = Path(__file__).resolve().parent
        # Go up 4 levels: hybrid_extractive/ -> retrieval_tasks/ -> ideas/ -> scripts/ -> workspace root
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


class HybridExtractiveRewriter:
    """
    Hybrid extractive: MMR + Templates + Entity Recognition
    """
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-base-en-v1.5",
        lambda_param: float = 0.7,
        max_terms: int = 10,
        entity_boost: float = 1.5
    ):
        """
        Initialize hybrid rewriter.
        
        Args:
            model_name: Sentence embedding model (default: bge-base-en-v1.5)
            lambda_param: MMR lambda parameter
            max_terms: Max terms to select
            entity_boost: Boost factor for named entities
        """
        self.base_rewriter = PureExtractiveRewriter(
            model_name=model_name,
            lambda_param=lambda_param,
            max_terms=max_terms
        )
        
        self.nlp = spacy.load("en_core_web_sm")
        self.entity_boost = entity_boost
    
    def rewrite(
        self,
        current_query: str,
        conversation_history: List[Dict[str, str]]
    ) -> str:
        """
        Rewrite query using hybrid extractive approach.
        """
        q_type = self.classify_question_type(current_query)
        entities = self.extract_entities(current_query, conversation_history)
        pronouns = self.detect_pronouns(current_query)
        
        selected_terms = self.mmr_with_entity_boost(
            current_query,
            conversation_history,
            entities,
            pronouns
        )
        
        rewritten = self.apply_template(
            q_type=q_type,
            terms=selected_terms,
            entities=entities
        )
        
        rewritten = self.post_process(rewritten)
        
        return rewritten
    
    def classify_question_type(self, query: str) -> str:
        """
        Classify question type based on question words.
        
        Returns:
            One of: 'what', 'where', 'when', 'who', 'why', 'how', 
                    'is', 'do', 'keyword', 'statement'
        """
        query_lower = query.lower().strip()
        tokens = query_lower.split()
        
        if not tokens:
            return 'keyword'
        
        first_word = tokens[0]
        
        question_words = {
            'what': 'what', 'where': 'where', 'when': 'when',
            'who': 'who', 'why': 'why', 'how': 'how', 'which': 'which'
        }
        
        if first_word in question_words:
            return question_words[first_word]
        
        if first_word in ['is', 'are', 'was', 'were', 'can', 'could', 'will', 'would', 'should']:
            return 'is'
        
        if first_word in ['do', 'does', 'did', 'have', 'has', 'had']:
            return 'do'
        
        if query.endswith('?'):
            if len(tokens) <= 5:
                return 'keyword'
            else:
                return 'what'
        
        if len(tokens) <= 5:
            return 'keyword'
        
        return 'statement'
    
    def extract_entities(
        self,
        current_query: str,
        conversation_history: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """
        Extract named entities from query and history.
        
        Returns:
            List of entities with type and text
        """
        entities = []
        
        doc = self.nlp(current_query)
        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "type": ent.label_,
                "source": "query"
            })
        
        for turn in conversation_history:
            doc = self.nlp(turn['content'])
            for ent in doc.ents:
                entities.append({
                    "text": ent.text,
                    "type": ent.label_,
                    "source": "history"
                })
        
        # Deduplicate (keep most recent)
        seen = set()
        unique_entities = []
        for ent in reversed(entities):
            key = ent['text'].lower()
            if key not in seen:
                seen.add(key)
                unique_entities.append(ent)
        
        return list(reversed(unique_entities))
    
    def detect_pronouns(self, query: str) -> List[Dict]:
        """
        Detect pronouns in query.
        
        Returns:
            List of detected pronouns with position
        """
        doc = self.nlp(query)
        pronouns = []
        
        for token in doc:
            if token.pos_ == "PRON":
                pronouns.append({
                    "text": token.text,
                    "lemma": token.lemma_,
                    "position": token.i
                })
        
        return pronouns
    
    def mmr_with_entity_boost(
        self,
        current_query: str,
        conversation_history: List[Dict[str, str]],
        entities: List[Dict[str, str]],
        pronouns: List[Dict]
    ) -> List[str]:
        """
        Run MMR but boost scores for named entities.
        """
        history_text = self.base_rewriter.format_history(conversation_history)
        candidates = self.base_rewriter.extract_candidates(current_query, history_text)
        
        entity_texts = {ent['text'].lower() for ent in entities}
        
        selected = []
        query_emb = self.base_rewriter.encoder.encode([current_query])[0]
        candidate_embs = self.base_rewriter.encoder.encode(candidates)
        
        selected_indices = []
        selected_embs = []
        
        for _ in range(min(self.base_rewriter.max_terms, len(candidates))):
            mmr_scores = []
            
            for i, cand_emb in enumerate(candidate_embs):
                if i in selected_indices:
                    mmr_scores.append(-np.inf)
                    continue
                
                relevance = cosine_similarity([query_emb], [cand_emb])[0][0]
                
                # Entity boost
                if candidates[i].lower() in entity_texts:
                    relevance *= self.entity_boost
                
                # Redundancy
                if not selected_embs:
                    redundancy = 0
                else:
                    similarities = cosine_similarity([cand_emb], selected_embs)
                    redundancy = np.max(similarities)
                
                mmr_score = (
                    self.base_rewriter.lambda_param * relevance -
                    (1 - self.base_rewriter.lambda_param) * redundancy
                )
                mmr_scores.append(mmr_score)
            
            best_idx = np.argmax(mmr_scores)
            selected_indices.append(best_idx)
            selected_embs.append(candidate_embs[best_idx])
            selected.append(candidates[best_idx])
        
        return selected
    
    def apply_template(
        self,
        q_type: str,
        terms: List[str],
        entities: List[Dict[str, str]]
    ) -> str:
        """
        Apply question template based on type.
        """
        entity_terms = [ent['text'] for ent in entities if ent['text'] in ' '.join(terms)]
        other_terms = [t for t in terms if t not in entity_terms]
        
        if q_type == 'what':
            return self._apply_what_template(entity_terms, other_terms)
        elif q_type == 'where':
            return self._apply_where_template(entity_terms, other_terms)
        elif q_type == 'when':
            return self._apply_when_template(entity_terms, other_terms)
        elif q_type == 'who':
            return self._apply_who_template(entity_terms, other_terms)
        elif q_type == 'why':
            return self._apply_why_template(entity_terms, other_terms)
        elif q_type == 'how':
            return self._apply_how_template(entity_terms, other_terms)
        elif q_type == 'is':
            return self._apply_is_template(entity_terms, other_terms)
        elif q_type == 'do':
            return self._apply_do_template(entity_terms, other_terms)
        elif q_type == 'keyword':
            return self._apply_keyword_template(entity_terms, other_terms)
        else:
            return self._apply_statement_template(entity_terms, other_terms)
    
    def _apply_what_template(self, entities: List[str], terms: List[str]) -> str:
        """Template: What is/was/are [terms] [entities]?"""
        all_terms = entities + terms
        if not all_terms:
            return "what"
        
        verb = "is"
        if any(t in ['history', 'past', 'before', 'then'] for t in terms):
            verb = "was"
        
        return f"what {verb} {' '.join(all_terms)}"
    
    def _apply_where_template(self, entities: List[str], terms: List[str]) -> str:
        """Template: Where is/are [entities] [terms]?"""
        all_terms = entities + terms
        if not all_terms:
            return "where"
        return f"where {' '.join(all_terms)}"
    
    def _apply_when_template(self, entities: List[str], terms: List[str]) -> str:
        """Template: When did/was [entities] [terms]?"""
        all_terms = entities + terms
        if not all_terms:
            return "when"
        return f"when {' '.join(all_terms)}"
    
    def _apply_who_template(self, entities: List[str], terms: List[str]) -> str:
        """Template: Who is/was [terms] [entities]?"""
        all_terms = entities + terms
        if not all_terms:
            return "who"
        return f"who {' '.join(all_terms)}"
    
    def _apply_how_template(self, entities: List[str], terms: List[str]) -> str:
        """Template: How [terms] [entities]?"""
        all_terms = entities + terms
        if not all_terms:
            return "how"
        return f"how {' '.join(all_terms)}"
    
    def _apply_why_template(self, entities: List[str], terms: List[str]) -> str:
        """Template: Why [terms] [entities]?"""
        all_terms = entities + terms
        if not all_terms:
            return "why"
        return f"why {' '.join(all_terms)}"
    
    def _apply_is_template(self, entities: List[str], terms: List[str]) -> str:
        """Template: Is [entities] [terms]?"""
        all_terms = entities + terms
        if not all_terms:
            return "is"
        return f"is {' '.join(all_terms)}"
    
    def _apply_do_template(self, entities: List[str], terms: List[str]) -> str:
        """Template: Do [entities] [terms]?"""
        all_terms = entities + terms
        if not all_terms:
            return "do"
        return f"do {' '.join(all_terms)}"
    
    def _apply_keyword_template(self, entities: List[str], terms: List[str]) -> str:
        """Template: What about [entities] [terms]?"""
        all_terms = entities + terms
        if not all_terms:
            return "what"
        return f"what about {' '.join(all_terms)}"
    
    def _apply_statement_template(self, entities: List[str], terms: List[str]) -> str:
        """Template: What about [entities] [terms]?"""
        all_terms = entities + terms
        if not all_terms:
            return "what"
        return f"what about {' '.join(all_terms)}"
    
    def post_process(self, query: str) -> str:
        """
        Clean up generated query.
        
        Steps:
        1. Capitalize first letter
        2. Add question mark if missing
        3. Remove duplicate words
        4. Trim whitespace
        """
        query = query.strip()
        
        # Remove duplicate consecutive words
        words = query.split()
        deduped = []
        prev = None
        for word in words:
            if word.lower() != prev:
                deduped.append(word)
            prev = word.lower()
        query = ' '.join(deduped)
        
        # Capitalize first letter
        if query:
            query = query[0].upper() + query[1:]
        
        # Add question mark if missing
        if query and not query.endswith('?'):
            query += '?'
        
        return query


def run_hybrid_extractive(domain: str, output_format: str = 'mtrag'):
    """
    Run hybrid extractive on MT-RAG dataset.
    
    Args:
        domain: Domain to process
        output_format: 'mtrag' (retrieval format, default)
    """
    import json
    import os
    
    rewriter = HybridExtractiveRewriter(
        lambda_param=0.7,
        max_terms=10,
        entity_boost=1.5
    )
    
    queries = load_mtrag_queries(domain, root_dir=rewriter.base_rewriter.root_dir)
    
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
            print(f"Error: {e}")
            results_mtrag.append({
                "_id": item['id'],
                "text": f"|user|: {item['query']}"
            })
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Save MT-RAG format
    datasets_dir = os.path.join(script_dir, 'datasets')
    os.makedirs(datasets_dir, exist_ok=True)
    output_path = os.path.join(datasets_dir, f"{domain}_hybrid_extractive.jsonl")
    
    with open(output_path, 'w') as f:
        for result in results_mtrag:
            f.write(json.dumps(result) + '\n')
    
    print(f"✓ Saved {len(results_mtrag)} queries to {output_path}")
    
    return results_mtrag


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python hybrid_extractive_rewrite.py <domain>")
        print("Domain: clapnq, cloud, fiqa, or govt")
        sys.exit(1)
    
    domain = sys.argv[1]
    run_hybrid_extractive(domain)

