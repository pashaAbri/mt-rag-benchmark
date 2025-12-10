#!/usr/bin/env python3
"""
Mono-T5 Oracle Selection: Using mono-T5 to predict which query strategy is best.

This script:
1. Loads ELSER retrieval results from all three query strategies (lastturn, rewrite, questions)
2. Uses mono-T5 to score each document-query pair
3. Calculates predicted recall@10 based on mono-T5 scores
4. Selects the query strategy with the best predicted recall@10
5. Compares mono-T5 selection to oracle (ground truth) and individual strategies
"""

import sys
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple
import statistics
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from tqdm import tqdm

# Import utilities
from utils import (
    load_retrieval_results_with_texts,
    load_queries,
    calculate_predicted_recall_at_k,
    calculate_performance_with_selection,
    calculate_oracle_performance_strategies,
    save_results_as_json_strategies,
    DOMAINS,
    QUERY_STRATEGIES,
    RETRIEVAL_METHODS,
)

# Add project root to path for imports
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Mono-T5 model configuration
MONO_T5_MODEL = "castorini/monot5-base-msmarco"
CACHE_DIR = script_dir / ".cache"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16  # Adjust based on GPU memory


class MonoT5Scorer:
    """Wrapper for mono-T5 relevance scoring."""
    
    def __init__(self, model_name: str = MONO_T5_MODEL, device: str = DEVICE, cache_dir: Path = None):
        self.device = device
        print(f"Loading mono-T5 model: {model_name} on {device}")
        if cache_dir and cache_dir.exists():
            print(f"Using cached model from: {cache_dir}")
            cache_dir_str = str(cache_dir)
        else:
            cache_dir_str = None
        
        self.tokenizer = T5Tokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir_str
        )
        self.model = T5ForConditionalGeneration.from_pretrained(
            model_name,
            cache_dir=cache_dir_str
        )
        self.model.to(device)
        self.model.eval()
        print("Model loaded successfully")
    
    def score(self, query: str, document: str) -> float:
        """
        Score a query-document pair using mono-T5.
        
        Args:
            query: Query text
            document: Document text
            
        Returns:
            Relevance score (probability of "true" relevance)
        """
        # Format input for mono-T5: "Query: {query} Document: {document} Relevant:"
        input_text = f"Query: {query} Document: {document} Relevant:"
        
        # Tokenize input
        input_ids = self.tokenizer.encode(
            input_text,
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).to(self.device)
        
        # Get logits for "true" and "false" tokens
        with torch.no_grad():
            # Decoder input is just the prompt "Relevant:"
            decoder_input_ids = self.tokenizer.encode("Relevant:", return_tensors="pt").to(self.device)
            
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                decoder_input_ids=decoder_input_ids,
                return_dict=True
            )
            
            # Get logits for the next token position
            logits = outputs.logits[0, -1, :]  # Shape: [vocab_size]
            
            # Find token IDs for "true" and "false"
            true_tokens = self.tokenizer.encode("true", add_special_tokens=False)
            false_tokens = self.tokenizer.encode("false", add_special_tokens=False)
            
            if len(true_tokens) == 0 or len(false_tokens) == 0:
                return 0.5  # Default to neutral if tokens not found
            
            true_id = true_tokens[0]
            false_id = false_tokens[0]
            
            # Get logits
            true_logit = logits[true_id].item()
            false_logit = logits[false_id].item()
            
            # Apply softmax to get probabilities (with numerical stability)
            max_logit = max(true_logit, false_logit)
            exp_true = torch.exp(torch.tensor(true_logit - max_logit))
            exp_false = torch.exp(torch.tensor(false_logit - max_logit))
            prob_true = exp_true / (exp_true + exp_false)
            
            return prob_true.item()
    
    def score_batch(self, query: str, documents: List[str]) -> List[float]:
        """
        Score a query against multiple documents in batch.
        
        Args:
            query: Query text
            documents: List of document texts
            
        Returns:
            List of relevance scores
        """
        scores = []
        for doc in documents:
            scores.append(self.score(query, doc))
        return scores


def select_best_strategy_with_monot5(
    task_id: str,
    query: str,
    lastturn_results: Dict,
    rewrite_results: Dict,
    questions_results: Dict,
    scorer: MonoT5Scorer
) -> Tuple[str, Dict[str, float]]:
    """
    Use mono-T5 to predict which query strategy performs best.
    
    Returns:
        Tuple of (best_strategy_name, predicted_recalls)
    """
    strategy_results = {
        'lastturn': lastturn_results,
        'rewrite': rewrite_results,
        'questions': questions_results
    }
    
    predicted_recalls = {}
    
    for strategy_name, results in strategy_results.items():
        if not results or 'contexts' not in results:
            predicted_recalls[strategy_name] = 0.0
            continue
        
        contexts = results['contexts']
        if len(contexts) == 0:
            predicted_recalls[strategy_name] = 0.0
            continue
        
        # Get document texts
        documents = []
        doc_texts = []
        for ctx in contexts[:10]:  # Top 10 for recall@10
            doc_text = ctx.get('text', '')
            if doc_text:
                documents.append(ctx)
                doc_texts.append(doc_text)
        
        if len(doc_texts) == 0:
            predicted_recalls[strategy_name] = 0.0
            continue
        
        # Score documents with mono-T5
        scores = scorer.score_batch(query, doc_texts)
        
        # Calculate predicted recall@10 with threshold 0.7
        predicted_recall = calculate_predicted_recall_at_k(documents, scores, k=10, threshold=0.7)
        predicted_recalls[strategy_name] = predicted_recall
    
    # Select strategy with highest predicted recall
    # Handle ties by using average mono-T5 score as tiebreaker
    max_recall = max(predicted_recalls.values())
    candidates = [s for s, v in predicted_recalls.items() if v == max_recall]
    
    if len(candidates) > 1:
        # Tie: use average mono-T5 score as tiebreaker
        # Re-score to get average scores for tiebreaking
        avg_scores = {}
        for strategy_name in candidates:
            results = strategy_results[strategy_name]
            contexts = results.get('contexts', [])[:10]
            doc_texts = [ctx.get('text', '') for ctx in contexts if ctx.get('text')]
            if doc_texts:
                scores = scorer.score_batch(query, doc_texts)
                avg_scores[strategy_name] = sum(scores) / len(scores) if scores else 0.0
            else:
                avg_scores[strategy_name] = 0.0
        
        # Select strategy with highest average score
        best_strategy = max(avg_scores.items(), key=lambda x: x[1])[0]
    else:
        best_strategy = candidates[0]
    
    return best_strategy, predicted_recalls


def main():
    """Main analysis function."""
    # Define paths relative to project root
    retrieval_scripts_dir = project_root / "scripts" / "baselines" / "retrieval_scripts"
    
    output_dir = script_dir / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Mono-T5 Oracle Selection Analysis (Query Strategy Selection)")
    print("=" * 80)
    print(f"Retrieval Methods: {', '.join(RETRIEVAL_METHODS)}")
    print(f"Query Strategies: {', '.join(QUERY_STRATEGIES)}")
    print(f"Domains: {', '.join(DOMAINS)}")
    
    # Load queries for all strategies
    print("\nLoading queries for all strategies...")
    queries_by_strategy = {}
    for strategy in QUERY_STRATEGIES:
        queries = load_queries(strategy, project_root)
        queries_by_strategy[strategy] = queries
        print(f"  {strategy}: {len(queries)} queries")
    
    # Load retrieval results for all combinations of RETRIEVAL_METHODS × QUERY_STRATEGIES
    print("\nLoading retrieval results for all combinations...")
    results_by_combination = {}  # (retrieval_method, strategy) -> results
    for retrieval_method in RETRIEVAL_METHODS:
        results_dir = retrieval_scripts_dir / retrieval_method / "results"
        for strategy in QUERY_STRATEGIES:
            results = load_retrieval_results_with_texts(results_dir, retrieval_method, strategy)
            results_by_combination[(retrieval_method, strategy)] = results
            print(f"  {retrieval_method} × {strategy}: {len(results)} tasks")
    
    # For selection, we'll use the first retrieval method (or could extend to select best combination)
    # For now, select between QUERY_STRATEGIES using the first retrieval method
    primary_retrieval_method = RETRIEVAL_METHODS[0]
    print(f"\nUsing {primary_retrieval_method.upper()} for strategy selection...")
    results_by_strategy = {}
    for strategy in QUERY_STRATEGIES:
        results_by_strategy[strategy] = results_by_combination.get((primary_retrieval_method, strategy), {})
    
    # Initialize mono-T5 scorer
    print("\nInitializing mono-T5 scorer...")
    scorer = MonoT5Scorer(cache_dir=CACHE_DIR)
    
    # Get all task IDs (intersection of all strategies)
    all_task_ids = set(results_by_strategy[QUERY_STRATEGIES[0]].keys())
    for strategy in QUERY_STRATEGIES[1:]:
        all_task_ids &= set(results_by_strategy[strategy].keys())
    
    print(f"\nProcessing {len(all_task_ids)} tasks with mono-T5...")
    print(f"(Tasks present in all {len(QUERY_STRATEGIES)} strategies)")
    
    # Use mono-T5 to select best strategy for each task
    monot5_choices = {}
    predicted_recalls = defaultdict(lambda: defaultdict(float))
    
    for task_id in tqdm(all_task_ids, desc="Scoring with mono-T5"):
        # Use rewrite query as the query text (or could use the strategy-specific query)
        # For now, we'll use the rewrite query since it's the most complete
        query = queries_by_strategy['rewrite'].get(task_id, '')
        if not query:
            # Fallback to other strategies
            for strategy in QUERY_STRATEGIES:
                query = queries_by_strategy[strategy].get(task_id, '')
                if query:
                    break
        
        if not query:
            continue
        
        lastturn_data = results_by_strategy['lastturn'].get(task_id, {})
        rewrite_data = results_by_strategy['rewrite'].get(task_id, {})
        questions_data = results_by_strategy['questions'].get(task_id, {})
        
        best_strategy, pred_recalls = select_best_strategy_with_monot5(
            task_id, query, lastturn_data, rewrite_data, questions_data, scorer
        )
        
        monot5_choices[task_id] = best_strategy
        for strategy, recall in pred_recalls.items():
            predicted_recalls[task_id][strategy] = recall
    
    print(f"\nMono-T5 selection complete!")
    choice_counts = Counter(monot5_choices.values())
    print(f"Choices: {dict(choice_counts)}")
    
    # Calculate individual performances
    print("\nCalculating individual performances...")
    individual_perfs = {}
    for strategy in QUERY_STRATEGIES:
        # Create a dict mapping task_id to retriever_scores for this strategy
        strategy_scores = {}
        for task_id in all_task_ids:
            if task_id in results_by_strategy[strategy]:
                strategy_scores[task_id] = results_by_strategy[strategy][task_id].get('retriever_scores', {})
        
        # Calculate average metrics
        metric_scores = defaultdict(list)
        for task_id, scores in strategy_scores.items():
            for m in ['recall_1', 'recall_3', 'recall_5', 'recall_10', 'ndcg_cut_1', 'ndcg_cut_3', 'ndcg_cut_5', 'ndcg_cut_10']:
                if m in scores:
                    metric_scores[m].append(scores[m])
        
        avg_scores = {}
        for m, score_list in metric_scores.items():
            if score_list:
                avg_scores[m] = statistics.mean(score_list)
            else:
                avg_scores[m] = 0.0
        
        individual_perfs[strategy] = avg_scores
    
    # Calculate mono-T5 selection performance
    print("Calculating mono-T5 selection performance...")
    # Collect scores from selected strategies
    metric_scores = defaultdict(list)
    for task_id, selected_strategy in monot5_choices.items():
        if task_id in results_by_strategy[selected_strategy]:
            scores = results_by_strategy[selected_strategy][task_id].get('retriever_scores', {})
            for m in ['recall_1', 'recall_3', 'recall_5', 'recall_10', 'ndcg_cut_1', 'ndcg_cut_3', 'ndcg_cut_5', 'ndcg_cut_10']:
                if m in scores:
                    metric_scores[m].append(scores[m])
    
    monot5_perf = {}
    for m, score_list in metric_scores.items():
        if score_list:
            monot5_perf[m] = statistics.mean(score_list)
        else:
            monot5_perf[m] = 0.0
    
    # Calculate oracle performance (for comparison)
    print("Calculating oracle performance...")
    lastturn_scores = {tid: r.get('retriever_scores', {}) for tid, r in results_by_strategy['lastturn'].items()}
    rewrite_scores = {tid: r.get('retriever_scores', {}) for tid, r in results_by_strategy['rewrite'].items()}
    questions_scores = {tid: r.get('retriever_scores', {}) for tid, r in results_by_strategy['questions'].items()}
    
    oracle_perf, oracle_choices, _ = calculate_oracle_performance_strategies(
        lastturn_scores, rewrite_scores, questions_scores, 'recall_10'
    )
    
    # Save results as JSON files
    print("\nSaving results as JSON files...")
    save_results_as_json_strategies(
        monot5_choices,
        predicted_recalls,
        individual_perfs,
        monot5_perf,
        oracle_perf,
        oracle_choices,
        output_dir
    )
    
    print(f"\n✓ Analysis complete! JSON results saved to: {output_dir}")


if __name__ == "__main__":
    main()
