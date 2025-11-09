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
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from pure_extractive.pure_extractive_rewrite import PureExtractiveRewriter, load_mtrag_queries


class HybridExtractiveRewriter:
    """
    Hybrid extractive: MMR + Templates + Entity Recognition
    """
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        lambda_param: float = 0.7,
        max_terms: int = 10,
        entity_boost: float = 1.5
    ):
        """
        Initialize hybrid rewriter.
        
        Args:
            model_name: Sentence embedding model
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


def run_hybrid_extractive(domain: str):
    """
    Run hybrid extractive on MT-RAG dataset.
    """
    import json
    import os
    
    rewriter = HybridExtractiveRewriter(
        lambda_param=0.7,
        max_terms=10,
        entity_boost=1.5
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
            print(f"Error: {e}")
            results.append({
                "id": item['id'],
                "original": item['query'],
                "rewritten": item['query'],
                "error": str(e)
            })
    
    # Save to results directory within hybrid_extractive
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    output_path = os.path.join(results_dir, f"{domain}_rewrites.jsonl")
    
    with open(output_path, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    print(f"Saved to {output_path}")
    
    return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python hybrid_extractive_rewrite.py <domain>")
        print("Domain: clapnq, cloud, fiqa, or govt")
        sys.exit(1)
    
    domain = sys.argv[1]
    run_hybrid_extractive(domain)

