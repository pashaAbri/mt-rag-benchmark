#!/usr/bin/env python3
"""
Cluster-Based MMR Query Rewriting with LLM
Implementation of professor's clustering + MMR method for multi-turn query rewriting

Uses incremental context building:
- Turn 1: No rewriting (no context yet)
- Turn 2+: Rewrite using MMR clustering on previous Q+A history + LLM

LLM configuration is handled statically in llm_api.py
"""

import json
import argparse
import re
from pathlib import Path
from typing import List, Dict, Tuple, Any
from collections import defaultdict

import numpy as np
import nltk
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm

import time
from llm_api import call_llm
from context_builder import initialize_dataset, prepare_turn_batch
from prompts import create_rewrite_prompt
from retrieval import ElserRetriever
from generation import Generator

# Download NLTK data if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


# ============================================================================
# MMR Cluster Rewriter Class
# ============================================================================


class MMRClusterRewriter:
    """
    Implements cluster-based MMR query rewriting with LLM.
    
    Steps:
    1. Extract sentences from conversation history
    2. Embed sentences using BGE
    3. Cluster sentences using K-Means
    4. Select representatives from each cluster (closest to centroid)
    5. Apply MMR to select diverse, relevant sentences
    6. Construct rewritten query using LLM
    """
    
    def __init__(
        self,
        embedding_model: str = 'BAAI/bge-base-en-v1.5',
        lambda_param: float = 0.7,
        num_mmr_sentences: int = 5,
        reps_per_cluster: int = 3,
        save_intermediate: bool = True,
        intermediate_dir: str = None
    ):
        """
        Initialize the rewriter.
        
        Args:
            embedding_model: Name of sentence transformer model
            lambda_param: MMR parameter (0-1), higher = more relevance, lower = more diversity
            num_mmr_sentences: Number of sentences to select via MMR
            reps_per_cluster: Number of representatives to select per cluster
            save_intermediate: Whether to save intermediate results
            intermediate_dir: Directory to save intermediate results
        """
        self.lambda_param = lambda_param
        self.num_mmr_sentences = num_mmr_sentences
        self.reps_per_cluster = reps_per_cluster
        self.save_intermediate = save_intermediate
        self.intermediate_dir = intermediate_dir
        
        print(f"Loading embedding model: {embedding_model}")
        self.model = SentenceTransformer(embedding_model)
        
        # Intermediate data storage
        self.intermediate_data = {}
        
    def extract_sentences(
        self, 
        conversation_history: List[Dict[str, str]]
    ) -> List[Dict[str, Any]]:
        """
        Extract candidate sentences from conversation history.
        
        Args:
            conversation_history: List of turns with 'speaker' and 'text'
            
        Returns:
            List of sentence dicts with metadata
        """
        candidate_sentences = []
        
        for turn_idx, turn in enumerate(conversation_history):
            speaker = turn['speaker']
            text = turn['text']
            
            # For user questions, treat as single sentence
            if speaker == 'user':
                # Clean and validate
                text_clean = text.strip()
                if len(text_clean.split()) >= 3:  # At least 3 words
                    candidate_sentences.append({
                        'sentence': text_clean,
                        'speaker': speaker,
                        'turn': turn_idx,
                        'original_text': text
                    })
            else:  # agent response - split into sentences
                sentences = nltk.sent_tokenize(text)
                for sent in sentences:
                    sent_clean = sent.strip()
                    # Filter out very short or conversational sentences
                    if len(sent_clean.split()) >= 3:
                        # Skip purely conversational phrases
                        if not self._is_conversational(sent_clean):
                            candidate_sentences.append({
                                'sentence': sent_clean,
                                'speaker': speaker,
                                'turn': turn_idx,
                                'original_text': text
                            })
        
        # Remove exact duplicates
        seen = set()
        unique_sentences = []
        for sent in candidate_sentences:
            if sent['sentence'] not in seen:
                seen.add(sent['sentence'])
                unique_sentences.append(sent)
        
        return unique_sentences
    
    def _is_conversational(self, sentence: str) -> bool:
        """Check if sentence is purely conversational."""
        conversational_patterns = [
            r'^thank you',
            r'^thanks',
            r"^i'm sorry",
            r"^i don't have",
            r"^i don't know",
            r'^that[\'s]? interesting',
            r'^okay',
            r'^sure',
        ]
        sentence_lower = sentence.lower()
        for pattern in conversational_patterns:
            if re.match(pattern, sentence_lower):
                return True
        return False
    
    def compute_embeddings(
        self, 
        query: str, 
        sentences: List[Dict[str, Any]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute embeddings for query and candidate sentences.
        
        Args:
            query: Current query
            sentences: List of candidate sentence dicts
            
        Returns:
            (query_embedding, sentence_embeddings)
        """
        # Embed query
        query_emb = self.model.encode(query, convert_to_numpy=True)
        
        # Embed sentences
        sentence_texts = [s['sentence'] for s in sentences]
        sentence_embs = self.model.encode(sentence_texts, convert_to_numpy=True)
        
        return query_emb, sentence_embs
    
    def cluster_sentences(
        self,
        sentences: List[Dict[str, Any]],
        embeddings: np.ndarray
    ) -> Tuple[Dict[int, List[Dict]], np.ndarray]:
        """
        Cluster sentences using K-Means.
        
        Args:
            sentences: List of sentence dicts
            embeddings: Sentence embeddings (n_sentences, embedding_dim)
            
        Returns:
            (clusters_dict, centroids)
        """
        n_sentences = len(sentences)
        
        # Handle edge cases
        if n_sentences <= 2:
            # If very few sentences, don't cluster
            clusters = {0: []}
            for i, sent in enumerate(sentences):
                clusters[0].append({
                    **sent,
                    'embedding': embeddings[i],
                    'dist_to_centroid': 0.0
                })
            return clusters, embeddings.mean(axis=0, keepdims=True)
        
        # Determine number of clusters
        # k = sqrt(n), bounded between 2 and 7
        k = max(2, min(7, int(np.sqrt(n_sentences))))
        
        # Apply K-Means
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)
        centroids = kmeans.cluster_centers_
        
        # Group sentences by cluster
        clusters = defaultdict(list)
        for idx, label in enumerate(cluster_labels):
            # Compute cosine distance to centroid
            emb = embeddings[idx]
            centroid = centroids[label]
            
            # Cosine distance = 1 - cosine similarity
            cosine_sim = np.dot(emb, centroid) / (
                np.linalg.norm(emb) * np.linalg.norm(centroid)
            )
            dist = float(1 - cosine_sim)
            
            clusters[int(label)].append({
                **sentences[idx],
                'embedding': emb,
                'dist_to_centroid': dist,
                'cluster_id': int(label)
            })
        
        return dict(clusters), centroids
    
    def select_cluster_representatives(
        self,
        clusters: Dict[int, List[Dict]]
    ) -> List[Dict[str, Any]]:
        """
        Select representative sentences from each cluster (closest to centroid).
        
        Args:
            clusters: Dictionary mapping cluster_id to list of sentences
            
        Returns:
            List of representative sentences
        """
        representatives = []
        
        for cluster_id, cluster_items in clusters.items():
            # Sort by distance to centroid (ascending = most similar)
            sorted_items = sorted(cluster_items, key=lambda x: x['dist_to_centroid'])
            
            # Select top k representatives per cluster
            top_k = min(self.reps_per_cluster, len(sorted_items))
            representatives.extend(sorted_items[:top_k])
        
        return representatives
    
    def mmr_selection(
        self,
        candidates: List[Dict[str, Any]],
        query_emb: np.ndarray,
        k: int = None
    ) -> List[Dict[str, Any]]:
        """
        Select sentences using Maximal Marginal Relevance (MMR).
        
        MMR formula: MR(Di) = λ·sim(Di, Q) - (1-λ)·max sim(Di, Dj) for Dj in S
        
        Args:
            candidates: List of candidate sentence dicts (with 'embedding')
            query_emb: Query embedding
            k: Number of sentences to select (default: self.num_mmr_sentences)
            
        Returns:
            List of selected sentences
        """
        if k is None:
            k = self.num_mmr_sentences
        
        # Edge case: fewer candidates than k
        if len(candidates) <= k:
            return candidates
        
        selected = []
        selected_embs = []
        remaining_indices = list(range(len(candidates)))
        
        for _ in range(k):
            best_score = -float('inf')
            best_idx = None
            
            for idx in remaining_indices:
                emb = candidates[idx]['embedding']
                
                # Relevance: cosine similarity to query
                relevance = np.dot(emb, query_emb) / (
                    np.linalg.norm(emb) * np.linalg.norm(query_emb)
                )
                
                # Redundancy: max similarity to already selected sentences
                if len(selected_embs) > 0:
                    similarities = [
                        np.dot(emb, sel_emb) / (
                            np.linalg.norm(emb) * np.linalg.norm(sel_emb)
                        )
                        for sel_emb in selected_embs
                    ]
                    redundancy = max(similarities)
                else:
                    redundancy = 0.0
                
                # MMR score
                mmr_score = self.lambda_param * relevance - (1 - self.lambda_param) * redundancy
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx
            
            # Add selected sentence
            if best_idx is not None:
                selected.append(candidates[best_idx])
                selected_embs.append(candidates[best_idx]['embedding'])
                remaining_indices.remove(best_idx)
        
        return selected
    
    def construct_query_llm(
        self,
        current_query: str,
        selected_sentences: List[Dict[str, Any]]
    ) -> str:
        """
        Construct query using LLM rewriting.
        
        Args:
            current_query: Original query
            selected_sentences: Sentences selected via MMR
            
        Returns:
            Rewritten query
        """
        # Create prompt using prompt builder
        prompt = create_rewrite_prompt(current_query, selected_sentences)
        
        # Call LLM (always uses mixtral config from llm_api.py)
        response_text = call_llm(
            prompt=prompt,
            llm_config='mixtral',
            max_tokens=200,
            temperature=0.5
        )
        
        # Parse JSON response from the prompt
        try:
            # Simple cleanup to handle potential markdown code blocks
            cleaned_response = response_text.strip()
            if cleaned_response.startswith('```json'):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.endswith('```'):
                cleaned_response = cleaned_response[:-3]
            
            response_json = json.loads(cleaned_response.strip())
            rewritten = response_json.get('reworded version', current_query)
        except json.JSONDecodeError:
            print(f"Warning: Failed to parse JSON response: {response_text[:100]}...")
            # Fallback: try to extract text if it's not valid JSON
            rewritten = response_text.strip()

        return rewritten.strip()
    
    def post_process_query(self, query: str) -> str:
        """
        Post-process the rewritten query.
        
        Args:
            query: Rewritten query
            
        Returns:
            Cleaned query
        """
        # Remove extra whitespace
        query = ' '.join(query.split())
        
        # Capitalize first letter
        if query:
            query = query[0].upper() + query[1:]
        
        # Ensure ends with appropriate punctuation
        if query and query[-1] not in '.?!':
            # If it looks like a question, add ?
            question_words = ['who', 'what', 'when', 'where', 'why', 'how', 'which', 'whose']
            if any(query.lower().startswith(qw) for qw in question_words):
                query += '?'
            else:
                query += '.'
        
        return query
    
    def rewrite_query(
        self,
        task_id: str,
        current_query: str,
        conversation_history: List[Dict[str, str]]
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Rewrite a single query using the full pipeline.
        
        Args:
            task_id: Unique task identifier
            current_query: Query to rewrite
            conversation_history: List of previous turns
            
        Returns:
            (rewritten_query, intermediate_data)
        """
        intermediate = {'task_id': task_id, 'original_query': current_query}
        timing = {}
        
        # Step 1: Extract sentences
        start_time = time.time()
        sentences = self.extract_sentences(conversation_history)
        intermediate['num_extracted_sentences'] = len(sentences)
        intermediate['extracted_sentences'] = [s['sentence'] for s in sentences]
        timing['extract_sentences'] = time.time() - start_time
        
        # Edge case: no sentences extracted
        if len(sentences) == 0:
            intermediate['rewritten_query'] = current_query
            intermediate['method'] = 'no_history'
            intermediate['timing'] = timing
            return current_query, intermediate
        
        # Step 2: Compute embeddings
        start_time = time.time()
        query_emb, sentence_embs = self.compute_embeddings(current_query, sentences)
        timing['compute_embeddings'] = time.time() - start_time
        intermediate['embedding_dim'] = query_emb.shape[0]
        
        # Step 3: Cluster sentences
        start_time = time.time()
        clusters, _ = self.cluster_sentences(sentences, sentence_embs)
        timing['clustering'] = time.time() - start_time
        intermediate['num_clusters'] = len(clusters)
        intermediate['cluster_sizes'] = {k: len(v) for k, v in clusters.items()}
        
        # Step 4: Select cluster representatives
        representatives = self.select_cluster_representatives(clusters)
        intermediate['num_representatives'] = len(representatives)
        intermediate['representative_sentences'] = [r['sentence'] for r in representatives]
        
        # Step 5: Apply MMR
        start_time = time.time()
        selected = self.mmr_selection(representatives, query_emb)
        timing['mmr_selection'] = time.time() - start_time
        intermediate['num_selected'] = len(selected)
        intermediate['selected_sentences'] = [
            {
                'sentence': s['sentence'],
                'speaker': s['speaker'],
                'turn': s['turn'],
                'cluster_id': s.get('cluster_id', -1)
            }
            for s in selected
        ]
        
        # Step 6: Construct rewritten query using LLM
        start_time = time.time()
        rewritten = self.construct_query_llm(current_query, selected)
        timing['llm_rewrite'] = time.time() - start_time
        intermediate['method'] = 'llm'
        
        # Step 7: Post-process
        rewritten = self.post_process_query(rewritten)
        intermediate['rewritten_query'] = rewritten
        
        # Calculate total rewrite latency (sum of all steps)
        # Note: this excludes intermediate logging overhead
        timing['total_rewrite_pipeline'] = sum(timing.values())
        intermediate['timing'] = timing
        
        return rewritten, intermediate
    
    def save_intermediate_data(self, output_path: str):
        """Save intermediate data for analysis."""
        if self.intermediate_data:
            with open(output_path, 'w') as f:
                for task_id, data in self.intermediate_data.items():
                    # Remove embeddings before saving (too large)
                    data_clean = {k: v for k, v in data.items() if 'embedding' not in k.lower()}
                    f.write(json.dumps(data_clean) + '\n')
            print(f"Saved intermediate data to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Cluster-based MMR query rewriting'
    )
    parser.add_argument(
        '--domain',
        type=str,
        required=True,
        choices=['clapnq', 'fiqa', 'govt', 'cloud'],
        help='Domain to process'
    )
    parser.add_argument(
        '--lambda-param',
        type=float,
        default=0.7,
        help='MMR lambda parameter (relevance vs diversity)'
    )
    parser.add_argument(
        '--num-sentences',
        type=int,
        default=5,
        help='Number of sentences to select via MMR'
    )
    parser.add_argument(
        '--reps-per-cluster',
        type=int,
        default=3,
        help='Number of representatives per cluster'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (default: current directory)'
    )
    parser.add_argument(
        '--turn',
        type=int,
        default=None,
        help='Process only specific turn number (if not specified, processes all turns)'
    )
    
    args = parser.parse_args()
    
    # Determine output directory
    if args.output_dir is None:
        args.output_dir = Path(__file__).parent
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    datasets_dir = output_dir / 'datasets'
    intermediate_dir = output_dir / 'intermediate'
    datasets_dir.mkdir(exist_ok=True)
    intermediate_dir.mkdir(exist_ok=True)
    
    # Resolve paths
    
    print(f"\n{'='*60}")
    print("Cluster-Based MMR Query Rewriting (Incremental Context)")
    print(f"{'='*60}")
    print(f"Domain: {args.domain}")
    print(f"Lambda: {args.lambda_param}")
    print(f"MMR sentences: {args.num_sentences}")
    print(f"Reps per cluster: {args.reps_per_cluster}")
    print("Method: LLM-assisted rewriting")
    if args.turn:
        print(f"Processing only turn: {args.turn}")
    print(f"{'='*60}\n")
    
    # Initialize dataset from cleaned_data
    try:
        conversations, _ = initialize_dataset(args.domain)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # Initialize rewriter (always uses LLM with mixtral config from llm_api.py)
    rewriter = MMRClusterRewriter(
        lambda_param=args.lambda_param,
        num_mmr_sentences=args.num_sentences,
        reps_per_cluster=args.reps_per_cluster,
        save_intermediate=True,
        intermediate_dir=str(intermediate_dir)
    )
    
    # Initialize Retriever and Generator
    print(f"Initializing ELSER Retriever for domain: {args.domain}")
    try:
        retriever = ElserRetriever(args.domain)
    except Exception as e:
        print(f"Error initializing retriever: {e}")
        print("Make sure ES_URL and ES_API_KEY are set in .env")
        return
    
    print("Initializing Generator...")
    generator = Generator()
    
    # Prepare output files
    turn_suffix = f'_turn{args.turn}' if args.turn else '_all'
    # Include params in filename to separate experiments
    config_suffix = f'_k{args.num_sentences}_lam{args.lambda_param}'
    
    output_file = datasets_dir / f'{args.domain}_mmr_cluster{config_suffix}{turn_suffix}.jsonl'
    intermediate_file = intermediate_dir / f'{args.domain}_intermediate{config_suffix}{turn_suffix}.jsonl'
    
    # Clear existing files if we are starting fresh
    if output_file.exists():
        print(f"Warning: Output file {output_file} exists. Appending to it.")
    if intermediate_file.exists():
        print(f"Warning: Intermediate file {intermediate_file} exists. Appending to it.")

    processing_stats = {
        'turn1_no_rewrite': 0,
        'turn2plus_rewritten': 0,
        'turn2plus_no_history': 0,
        'errors': 0
    }
    
    total_processed = 0
    
    # Process conversations one by one
    print(f"\n{'='*60}")
    print("Processing Conversations")
    print(f"{'='*60}\n")
    
    for conv_id, turns in tqdm(conversations.items(), desc="Conversations"):
        # Reset history for new conversation
        conversation_history = [] # List of {speaker: str, text: str}
        
        # Sort turns just in case
        turns.sort(key=lambda x: x['turn'])
        
        for turn_data in turns:
            task_id = turn_data['task_id']
            turn_num = turn_data['turn']
            query = turn_data['query']
            
            # Skip if we are only processing specific turns and this isn't one of them
            if args.turn and turn_num != args.turn:
                # If we're skipping this turn, we still need to record it in history 
                # IF we assume the previous turns were processed elsewhere. 
                # But in this "run one conversation" mode, usually we want the flow.
                # If the user asks for specific turn, this logic might be tricky.
                # Assuming "run one conversation" implies running all turns for context.
                # If user strictly wants ONE turn, they shouldn't use this script's new mode efficiently without prior state.
                # For now, if args.turn is set, we only PROCESS that turn, but we won't have history 
                # unless we load it. The prompt implies they want to run the sequence ("turn 1 ... then turn 2").
                # So we will ignore args.turn for the flow or assume they want the whole thing.
                # Let's assume we process ALL turns to build context.
                pass

            # Check if we should process this turn based on args.turn
            # If args.turn is specified, we only SAVE/OUTPUT that turn, 
            # but we might need to process previous turns to build history?
            # The user's request "run one conversation... instead of doing turn 1 for all... then moving to turn 2"
            # strongly implies they want to run the full conversation sequence.
            # So we will process all turns.
            
            try:
                turn_start_time = time.time()
                
                # Prepare intermediate record
                intermediate = {
                    'task_id': task_id, 
                    'original_query': query,
                    'turn': turn_num
                }
                
                # --- Rewriting ---
                # For turn 1, no rewriting needed (no context yet)
                if turn_num == 1:
                    rewritten = query
                    intermediate.update({
                        'rewritten_query': query,
                        'method': 'no_rewriting_turn_1',
                        'num_extracted_sentences': 0,
                        'timing': {'total_rewrite_pipeline': 0.0}
                    })
                    processing_stats['turn1_no_rewrite'] += 1
                else:
                    # For turn 2+, use MMR clustering to rewrite using accumulated history
                    if len(conversation_history) == 0:
                        # No history available (shouldn't happen if we run sequentially)
                        rewritten = query
                        intermediate.update({
                            'rewritten_query': query,
                            'method': 'no_history_available',
                            'num_extracted_sentences': 0,
                            'timing': {'total_rewrite_pipeline': 0.0}
                        })
                        processing_stats['turn2plus_no_history'] += 1
                    else:
                        # Rewrite using MMR clustering
                        rewritten, rewrite_intermediate = rewriter.rewrite_query(task_id, query, conversation_history)
                        intermediate.update(rewrite_intermediate)
                        processing_stats['turn2plus_rewritten'] += 1
                
                # --- Retrieval ---
                # Only print for first few or specific ones to avoid spam
                # print(f"  Retrieving for: {rewritten[:50]}...")
                retrieval_start = time.time()
                contexts = retriever.retrieve(rewritten)
                retrieval_time = time.time() - retrieval_start
                
                intermediate['contexts'] = [
                    {
                        'document_id': ctx.get('document_id'),
                        'title': ctx.get('title'),
                        'score': ctx.get('score'),
                        'url': ctx.get('url')
                    } for ctx in contexts
                ]
                
                # --- Generation ---
                # print(f"  Generating answer...")
                generation_start = time.time()
                agent_response = generator.generate(query, contexts, conversation_history)
                generation_time = time.time() - generation_start
                
                intermediate['agent_response'] = agent_response
                
                # Update timing stats
                if 'timing' not in intermediate:
                    intermediate['timing'] = {}
                intermediate['timing']['retrieval'] = retrieval_time
                intermediate['timing']['generation'] = generation_time
                intermediate['timing']['total_turn'] = time.time() - turn_start_time
                
                # Update History for next turn
                conversation_history.append({'speaker': 'user', 'text': query})
                conversation_history.append({'speaker': 'agent', 'text': agent_response})
                
                # Store result
                result_entry = {
                    '_id': task_id,
                    'text': f'|user|: {rewritten}',
                    'agent_response': agent_response
                }
                
                # Save incrementally
                with open(output_file, 'a') as f:
                    f.write(json.dumps(result_entry) + '\n')
                    
                with open(intermediate_file, 'a') as f:
                    # Remove embeddings before saving
                    data_clean = {k: v for k, v in intermediate.items() if 'embedding' not in k.lower()}
                    f.write(json.dumps(data_clean) + '\n')
                
                total_processed += 1
                
            except Exception as e:
                print(f"\n✗ Error processing {task_id}: {e}")
                # import traceback
                # traceback.print_exc()
                processing_stats['errors'] += 1
    
    # Print statistics
    print(f"\n{'='*60}")
    print("Processing Statistics")
    print(f"{'='*60}")
    print(f"Turn 1 (no rewriting):        {processing_stats['turn1_no_rewrite']}")
    print(f"Turn 2+ (rewritten):          {processing_stats['turn2plus_rewritten']}")
    print(f"Turn 2+ (no history):         {processing_stats['turn2plus_no_history']}")
    print(f"Errors:                       {processing_stats['errors']}")
    print(f"Total processed:              {total_processed}")
    print(f"{'='*60}")
    print(f"✓ Results saved to: {output_file}")
    print(f"✓ Intermediate data saved to: {intermediate_file}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()

