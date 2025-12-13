"""
DH-RAG: Dynamic Historical RAG Implementation

Reference: DH-RAG Paper (2024)

This implementation follows the paper's methodology as closely as possible:
1. Dynamic Historical Information Database H = {(q, p, r), ...}
2. 3-tier Hierarchical Structure: Clusters → Summaries → Triples
3. TF-IDF for hierarchical matching (as specified in paper)
4. Dense embeddings for final relevance scoring
5. Chain-of-Thought Tracking for query sequences
6. Recency + Relevance weighting: W_i = α · Relevance + (1-α) · Recency
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
import time

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import KMeans
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.feature_extraction.text import TfidfVectorizer
except ImportError:
    raise ImportError("Required: sentence-transformers, scikit-learn")


@dataclass
class HistoryTriple:
    """
    Paper's H = {(q_1, p_1, r_1), (q_2, p_2, r_2), ..., (q_{t-1}, p_{t-1}, r_{t-1})}
    
    - q: User query
    - p: Retrieved passage(s) - stored as string for simplicity
    - r: Agent response
    """
    triple_id: int
    query: str
    passage: str  # Could be empty if no retrieval happened
    response: str
    
    # Embeddings (for relevance scoring)
    query_embedding: Optional[np.ndarray] = None
    response_embedding: Optional[np.ndarray] = None
    
    # TF-IDF vectors (for hierarchical matching)
    tfidf_vector: Optional[np.ndarray] = None
    
    # Cluster assignment
    cluster_id: int = -1
    
    # Chain-of-Thought tracking
    chain_id: int = -1  # Which chain this belongs to
    chain_position: int = 0  # Position within the chain
    
    # Timestamp for recency
    timestamp: float = 0.0
    
    def get_combined_text(self) -> str:
        """Returns combined text for TF-IDF vectorization."""
        return f"{self.query} {self.response}"


@dataclass 
class ClusterSummary:
    """
    Paper's Summary Layer - intermediate level between Clusters and Triples.
    Each cluster has one or more summaries that represent sub-topics.
    """
    summary_id: int
    cluster_id: int
    representative_triple_id: int  # The most central triple in this summary
    tfidf_centroid: Optional[np.ndarray] = None
    member_triple_ids: List[int] = field(default_factory=list)


class DHRAG:
    """
    Dynamic Historical RAG - Full Implementation
    
    Paper Components:
    1. Dynamic Historical Information Database (H)
    2. Historical Query Clustering (k clusters)
    3. Hierarchical Matching (TF-IDF tree: Cluster → Summary → Triple)
    4. Chain of Thought Tracking
    5. Information Integration (α-weighted scoring)
    """
    
    def __init__(self,
                 embedding_model: str = 'all-MiniLM-L6-v2',
                 alpha: float = 0.7,
                 max_clusters: int = 10,  # Paper shows k=10 for MobileCS2
                 min_triples_for_clustering: int = 5,
                 summaries_per_cluster: int = 2,  # m parameter (not specified in paper)
                 chain_similarity_threshold: float = 0.5,
                 cluster_bonus: float = 0.1,
                 shared_embedding_model=None):
        """
        Args:
            embedding_model: SentenceTransformer model name for dense embeddings
            alpha: Weight for Relevance vs Recency. Paper: not specified, we default 0.7
            max_clusters: Maximum k clusters. Paper shows k=10 for MobileCS2
            min_triples_for_clustering: Minimum history before clustering activates
            summaries_per_cluster: m summaries per cluster (paper doesn't specify)
            chain_similarity_threshold: Threshold for Chain-of-Thought linking
            cluster_bonus: Bonus for items in matched cluster
            shared_embedding_model: Pre-loaded SentenceTransformer model (for threading)
        """
        # Models - use shared model if provided to avoid threading issues
        if shared_embedding_model is not None:
            self.embedding_model = shared_embedding_model
        else:
            self.embedding_model = SentenceTransformer(embedding_model)
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.tfidf_fitted = False
        
        # Parameters
        self.alpha = alpha
        self.max_clusters = max_clusters
        self.min_triples_for_clustering = min_triples_for_clustering
        self.summaries_per_cluster = summaries_per_cluster
        self.chain_similarity_threshold = chain_similarity_threshold
        self.cluster_bonus = cluster_bonus
        
        # Dynamic Historical Information Database H
        self.history: List[HistoryTriple] = []
        
        # 3-tier Hierarchical Structure
        self.clusters: Dict[int, np.ndarray] = {}  # cluster_id -> TF-IDF centroid
        self.summaries: List[ClusterSummary] = []  # Summary layer
        
        # Chain-of-Thought tracking
        self.chains: Dict[int, List[int]] = defaultdict(list)  # chain_id -> [triple_ids]
        self.next_chain_id = 0
    
    def add_interaction(self, query: str, response: str, passage: str = ""):
        """
        Adds a new (query, passage, response) triple to the history database H.
        
        Paper: H = {(q_1, p_1, r_1), ..., (q_{t-1}, p_{t-1}, r_{t-1})}
        """
        triple_id = len(self.history)
        
        # Compute dense embeddings for relevance scoring
        query_embedding = self.embedding_model.encode(query)
        response_embedding = self.embedding_model.encode(response)
        
        triple = HistoryTriple(
            triple_id=triple_id,
            query=query,
            passage=passage,
            response=response,
            query_embedding=query_embedding,
            response_embedding=response_embedding,
            timestamp=time.time()
        )
        
        self.history.append(triple)
        
        # Update Chain-of-Thought tracking
        self._update_chains(triple)
        
        # Update TF-IDF vectors and hierarchical structure
        self._update_tfidf()
        self._update_clusters()
        self._update_summaries()
    
    def _update_chains(self, new_triple: HistoryTriple):
        """
        Paper: Chain of Thought Tracking
        
        Tracks sequences of related queries. Paper reports:
        - Average chain length: 1.73 steps
        - Distribution: 1-2 steps (32.2%), 2-3 steps (40.2%), 3-4 steps (21.4%), 4-5 steps (6.2%)
        """
        if len(self.history) <= 1:
            # First triple starts its own chain
            new_triple.chain_id = self.next_chain_id
            new_triple.chain_position = 0
            self.chains[self.next_chain_id].append(new_triple.triple_id)
            self.next_chain_id += 1
            return
        
        # Check if this query continues an existing chain
        # Look at the previous triple first (most likely to continue)
        prev_triple = self.history[-2]
        
        # Compute similarity between new query and previous query
        sim = cosine_similarity(
            [new_triple.query_embedding],
            [prev_triple.query_embedding]
        )[0][0]
        
        if sim >= self.chain_similarity_threshold:
            # Continue the chain
            new_triple.chain_id = prev_triple.chain_id
            new_triple.chain_position = prev_triple.chain_position + 1
            self.chains[prev_triple.chain_id].append(new_triple.triple_id)
        else:
            # Start a new chain
            new_triple.chain_id = self.next_chain_id
            new_triple.chain_position = 0
            self.chains[self.next_chain_id].append(new_triple.triple_id)
            self.next_chain_id += 1
    
    def _update_tfidf(self):
        """
        Paper: Uses TF-IDF vectorization for the hierarchical tree structure.
        """
        if len(self.history) < 2:
            return
        
        # Fit/transform TF-IDF on all history
        texts = [t.get_combined_text() for t in self.history]
        
        if not self.tfidf_fitted:
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
            self.tfidf_fitted = True
        else:
            tfidf_matrix = self.tfidf_vectorizer.transform(texts)
        
        # Update each triple's TF-IDF vector
        for i, triple in enumerate(self.history):
            triple.tfidf_vector = tfidf_matrix[i].toarray().flatten()
    
    def _update_clusters(self):
        """
        Paper: Historical Query Clustering
        
        Creates k clusters (paper shows k=10 for MobileCS2).
        Uses centroid-based matching.
        """
        n = len(self.history)
        if n < self.min_triples_for_clustering:
            return
        
        # Use TF-IDF vectors for clustering (as paper specifies)
        if self.history[0].tfidf_vector is None:
            return
        
        tfidf_matrix = np.array([t.tfidf_vector for t in self.history])
        
        # Adaptive k, capped at max_clusters
        k = min(self.max_clusters, max(2, int(np.sqrt(n))))
        
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = kmeans.fit_predict(tfidf_matrix)
        
        # Update cluster assignments and compute centroids
        self.clusters = {}
        for i, triple in enumerate(self.history):
            triple.cluster_id = labels[i]
        
        for cid in range(k):
            mask = labels == cid
            if np.sum(mask) > 0:
                self.clusters[cid] = np.mean(tfidf_matrix[mask], axis=0)
    
    def _update_summaries(self):
        """
        Paper: Summary Layer (intermediate level in 3-tier hierarchy)
        
        Creates m summary nodes per cluster. Each summary represents
        a sub-topic and has a representative triple.
        
        Note: Paper doesn't specify how summaries are created.
        We use sub-clustering within each cluster.
        """
        self.summaries = []
        
        if not self.clusters:
            return
        
        summary_id = 0
        
        for cluster_id, cluster_centroid in self.clusters.items():
            # Get triples in this cluster
            cluster_triples = [t for t in self.history if t.cluster_id == cluster_id]
            
            if len(cluster_triples) == 0:
                continue
            
            # If cluster is small, just use one summary
            if len(cluster_triples) <= self.summaries_per_cluster:
                # Each triple is its own summary
                for triple in cluster_triples:
                    summary = ClusterSummary(
                        summary_id=summary_id,
                        cluster_id=cluster_id,
                        representative_triple_id=triple.triple_id,
                        tfidf_centroid=triple.tfidf_vector,
                        member_triple_ids=[triple.triple_id]
                    )
                    self.summaries.append(summary)
                    summary_id += 1
            else:
                # Sub-cluster to create m summaries
                tfidf_matrix = np.array([t.tfidf_vector for t in cluster_triples])
                m = min(self.summaries_per_cluster, len(cluster_triples))
                
                sub_kmeans = KMeans(n_clusters=m, n_init=5, random_state=42)
                sub_labels = sub_kmeans.fit_predict(tfidf_matrix)
                
                for sub_id in range(m):
                    sub_mask = sub_labels == sub_id
                    sub_triples = [cluster_triples[i] for i in range(len(cluster_triples)) if sub_mask[i]]
                    
                    if len(sub_triples) == 0:
                        continue
                    
                    # Find the most central triple (closest to sub-centroid)
                    sub_centroid = np.mean(tfidf_matrix[sub_mask], axis=0)
                    distances = [
                        np.linalg.norm(t.tfidf_vector - sub_centroid)
                        for t in sub_triples
                    ]
                    representative = sub_triples[np.argmin(distances)]
                    
                    summary = ClusterSummary(
                        summary_id=summary_id,
                        cluster_id=cluster_id,
                        representative_triple_id=representative.triple_id,
                        tfidf_centroid=sub_centroid,
                        member_triple_ids=[t.triple_id for t in sub_triples]
                    )
                    self.summaries.append(summary)
                    summary_id += 1
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Paper Algorithm 1: Main retrieval pipeline
        
        1. Cluster Matching (using TF-IDF)
        2. Hierarchical Matching (Category → Summary → Triple)
        3. Chain of Thought Matching
        4. Information Integration (α-weighted scoring)
        
        Returns:
            List of dicts with retrieved history and scores
        """
        if len(self.history) == 0:
            return []
        
        # Encode query
        query_embedding = self.embedding_model.encode(query)
        
        # Get TF-IDF vector for query (for hierarchical matching)
        if self.tfidf_fitted:
            query_tfidf = self.tfidf_vectorizer.transform([query]).toarray().flatten()
        else:
            query_tfidf = None
        
        n = len(self.history)
        candidates = []
        
        # === Stage 1: Cluster Matching (TF-IDF) ===
        matched_cluster_id = -1
        if self.clusters and query_tfidf is not None:
            cluster_ids = list(self.clusters.keys())
            centroids = np.array([self.clusters[cid] for cid in cluster_ids])
            
            # TF-IDF cosine similarity
            sims = cosine_similarity([query_tfidf], centroids)[0]
            best_idx = np.argmax(sims)
            matched_cluster_id = cluster_ids[best_idx]
        
        # === Stage 2: Summary Matching (Hierarchical) ===
        matched_summary_ids: Set[int] = set()
        if self.summaries and query_tfidf is not None:
            # Filter summaries in matched cluster
            cluster_summaries = [s for s in self.summaries if s.cluster_id == matched_cluster_id]
            
            if cluster_summaries:
                # Match to summary level
                summary_centroids = np.array([s.tfidf_centroid for s in cluster_summaries])
                sims = cosine_similarity([query_tfidf], summary_centroids)[0]
                
                # Take top summaries (could be multiple)
                top_summary_indices = np.argsort(sims)[-2:]  # Top 2 summaries
                for idx in top_summary_indices:
                    matched_summary_ids.add(cluster_summaries[idx].summary_id)
        
        # === Stage 3: Chain of Thought Matching ===
        # Find if query might continue an existing chain
        chain_bonus_triples: Set[int] = set()
        if len(self.history) > 0:
            # Check similarity with recent queries
            recent_triple = self.history[-1]
            sim = cosine_similarity([query_embedding], [recent_triple.query_embedding])[0][0]
            
            if sim >= self.chain_similarity_threshold:
                # Query continues the chain - boost all triples in that chain
                chain_id = recent_triple.chain_id
                chain_bonus_triples = set(self.chains[chain_id])
        
        # === Stage 4: Score All Triples ===
        for triple in self.history:
            # 1. Relevance Score (dense embedding cosine similarity)
            relevance = cosine_similarity([query_embedding], [triple.query_embedding])[0][0]
            relevance = max(0.0, min(1.0, relevance))
            
            # 2. Recency Score (normalized to [0, 1])
            # Paper formula: Recency(t_i) - higher for more recent
            recency = (triple.triple_id + 1) / n
            
            # 3. Cluster Bonus (hierarchical matching reward)
            cluster_match_bonus = self.cluster_bonus if triple.cluster_id == matched_cluster_id else 0.0
            
            # 4. Summary Bonus (deeper hierarchical match)
            summary_match_bonus = 0.0
            for summary in self.summaries:
                if summary.summary_id in matched_summary_ids:
                    if triple.triple_id in summary.member_triple_ids:
                        summary_match_bonus = 0.05  # Extra bonus for summary match
                        break
            
            # 5. Chain Bonus (Chain of Thought tracking)
            chain_match_bonus = 0.05 if triple.triple_id in chain_bonus_triples else 0.0
            
            # Final Score (Paper formula + hierarchical bonuses):
            # W_i = α · Relevance(q_i, q_t) + (1-α) · Recency(t_i) + bonuses
            final_score = (
                (self.alpha * relevance) +
                ((1 - self.alpha) * recency) +
                cluster_match_bonus +
                summary_match_bonus +
                chain_match_bonus
            )
            
            candidates.append({
                'triple_id': triple.triple_id,
                'query': triple.query,
                'response': triple.response,
                'passage': triple.passage,
                'score': final_score,
                'relevance': relevance,
                'recency': recency,
                'cluster_id': triple.cluster_id,
                'chain_id': triple.chain_id,
                'matched_cluster': triple.cluster_id == matched_cluster_id,
                'matched_summary': any(
                    triple.triple_id in s.member_triple_ids 
                    for s in self.summaries if s.summary_id in matched_summary_ids
                ),
                'in_active_chain': triple.triple_id in chain_bonus_triples
            })
        
        # Sort by score and return top_k
        candidates.sort(key=lambda x: x['score'], reverse=True)
        return candidates[:top_k]
    
    def get_chain_stats(self) -> Dict:
        """Returns statistics about Chain-of-Thought tracking (for comparison with paper)."""
        chain_lengths = [len(triples) for triples in self.chains.values()]
        
        if not chain_lengths:
            return {'avg_length': 0, 'distribution': {}}
        
        avg_length = np.mean(chain_lengths)
        
        # Distribution buckets (like paper)
        distribution = {
            '1-2': sum(1 for l in chain_lengths if 1 <= l < 2),
            '2-3': sum(1 for l in chain_lengths if 2 <= l < 3),
            '3-4': sum(1 for l in chain_lengths if 3 <= l < 4),
            '4-5': sum(1 for l in chain_lengths if 4 <= l < 5),
            '5+': sum(1 for l in chain_lengths if l >= 5)
        }
        
        return {
            'avg_length': avg_length,
            'num_chains': len(self.chains),
            'distribution': distribution
        }


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("DH-RAG: Dynamic Historical RAG Demo")
    print("=" * 60)
    
    dh_rag = DHRAG(alpha=0.6, max_clusters=5)
    
    # Simulate a multi-turn conversation with topic switches
    interactions = [
        # Topic 1: American Football
        ("What are the rules of American Football?", 
         "American football is played by two teams of eleven players..."),
        ("How many points is a touchdown?", 
         "A touchdown is worth 6 points. After scoring, the team can attempt..."),
        ("What about field goals?", 
         "A field goal is worth 3 points and is scored by kicking..."),
        
        # Topic 2: Ancient Rome (topic switch)
        ("Tell me about the history of Rome.", 
         "Rome was founded in 753 BC according to tradition..."),
        ("Who was the first emperor?", 
         "Augustus (born Octavian) was the first Roman emperor..."),
        ("How did the empire fall?", 
         "The Western Roman Empire fell in 476 AD when..."),
        
        # Return to Topic 1
        ("Going back to football, what is a safety?",
         "A safety is worth 2 points and occurs when..."),
    ]
    
    print("\n--- Building History Database ---")
    for query, response in interactions[:-1]:
        dh_rag.add_interaction(query, response)
        print(f"Added: '{query[:40]}...'")
    
    print(f"\nHistory size: {len(dh_rag.history)} triples")
    print(f"Clusters: {len(dh_rag.clusters)}")
    print(f"Summaries: {len(dh_rag.summaries)}")
    
    # Chain stats
    chain_stats = dh_rag.get_chain_stats()
    print(f"Chains: {chain_stats['num_chains']}, Avg length: {chain_stats['avg_length']:.2f}")
    
    # Test retrieval
    test_query = interactions[-1][0]  # "Going back to football, what is a safety?"
    print("\n--- Retrieval Test ---")
    print(f"Query: '{test_query}'")
    
    results = dh_rag.retrieve(test_query, top_k=3)
    
    print("\nTop 3 Retrieved:")
    for i, res in enumerate(results):
        print(f"\n{i+1}. Score: {res['score']:.4f}")
        print(f"   Query: '{res['query'][:50]}...'")
        print(f"   Relevance: {res['relevance']:.3f}, Recency: {res['recency']:.3f}")
        print(f"   Cluster Match: {res['matched_cluster']}, Summary Match: {res['matched_summary']}")
        print(f"   In Active Chain: {res['in_active_chain']}")
