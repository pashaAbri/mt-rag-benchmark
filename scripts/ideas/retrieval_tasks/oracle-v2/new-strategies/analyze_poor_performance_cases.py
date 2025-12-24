#!/usr/bin/env python3
"""
Analyze queries/cases where neither retriever performs well.

This script identifies tasks where all three strategies (lastturn, rewrite, questions)
perform poorly across retrievers, and analyzes their characteristics to identify
potential opportunities for new query strategies.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import statistics
import pandas as pd


# Strategy names
STRATEGIES = ['lastturn', 'rewrite', 'questions']
RETRIEVERS = ['elser', 'bm25', 'bge']

# Metrics to analyze
METRICS = {
    'recall_1': 'R@1',
    'recall_3': 'R@3',
    'recall_5': 'R@5',
    'recall_10': 'R@10',
    'ndcg_cut_1': 'nDCG@1',
    'ndcg_cut_3': 'nDCG@3',
    'ndcg_cut_5': 'nDCG@5',
    'ndcg_cut_10': 'nDCG@10',
}

# Primary metric for identifying poor performance
PRIMARY_METRIC = 'ndcg_cut_5'  # Focus on nDCG@5


def load_evaluated_results(filepath: Path) -> Dict[str, Dict[str, float]]:
    """
    Load evaluated results from a JSONL file.
    
    Args:
        filepath: Path to the evaluated JSONL file
        
    Returns:
        Dictionary mapping task_id to metrics dictionary
    """
    results = {}
    
    if not filepath.exists():
        print(f"  Warning: File not found: {filepath}")
        return results
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                task_id = data.get('task_id')
                
                if not task_id:
                    continue
                
                # Extract retriever scores
                retriever_scores = data.get('retriever_scores', {})
                
                if retriever_scores:
                    results[task_id] = retriever_scores
                    
            except json.JSONDecodeError as e:
                print(f"  Warning: Failed to parse line {line_num} in {filepath.name}: {e}")
                continue
            except Exception as e:
                print(f"  Warning: Error processing line {line_num} in {filepath.name}: {e}")
                continue
    
    return results


def load_task_data(task_file: Path) -> Optional[Dict]:
    """Load a single task file."""
    try:
        with open(task_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def load_conversation_data(conversations_file: Path) -> Dict[str, Dict]:
    """Load conversation data to get history."""
    conversations = {}
    try:
        with open(conversations_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                for conv in data:
                    conv_id = str(conv.get('conversation_id', ''))
                    if conv_id:
                        conversations[conv_id] = conv
            elif isinstance(data, dict):
                conversations = data
    except (json.JSONDecodeError, IOError):
        pass
    return conversations


def extract_query_features(query_text: str) -> Dict[str, any]:
    """Extract features from query text."""
    if not query_text:
        return {
            'query_length_chars': 0,
            'query_length_words': 0,
            'has_question_mark': False,
            'has_wh_word': False,
            'num_capitalized_words': 0,
            'num_numbers': 0,
        }
    
    import re
    words = query_text.split()
    
    # Check for question mark
    has_question_mark = '?' in query_text
    
    # Check for wh-words (what, who, where, when, why, how, which)
    wh_pattern = r'\b(what|who|where|when|why|how|which|whose|whom)\b'
    has_wh_word = bool(re.search(wh_pattern, query_text.lower()))
    
    # Count capitalized words (excluding first word)
    num_capitalized = sum(1 for w in words[1:] if w and w[0].isupper())
    
    # Count numbers
    num_numbers = len(re.findall(r'\d+', query_text))
    
    return {
        'query_length_chars': len(query_text),
        'query_length_words': len(words),
        'has_question_mark': has_question_mark,
        'has_wh_word': has_wh_word,
        'num_capitalized_words': num_capitalized,
        'num_numbers': num_numbers,
    }


def extract_enrichment_features(enrichments: Dict) -> Dict[str, any]:
    """Extract features from enrichments."""
    features = {
        'answerability': None,
        'question_type': None,
        'multi_turn_type': None,
    }
    
    if enrichments:
        # Answerability (take first if list)
        answerability = enrichments.get('Answerability', [])
        if answerability and isinstance(answerability, list):
            features['answerability'] = answerability[0] if answerability else None
        elif answerability:
            features['answerability'] = answerability
        
        # Question Type (take first if list)
        question_type = enrichments.get('Question Type', [])
        if question_type and isinstance(question_type, list):
            features['question_type'] = question_type[0] if question_type else None
        elif question_type:
            features['question_type'] = question_type
        
        # Multi-Turn Type (take first if list)
        multi_turn = enrichments.get('Multi-Turn', [])
        if multi_turn and isinstance(multi_turn, list):
            features['multi_turn_type'] = multi_turn[0] if multi_turn else None
        elif multi_turn:
            features['multi_turn_type'] = multi_turn
    
    return features


def extract_conversation_features(
    task_data: Dict,
    conversations: Dict[str, Dict]
) -> Dict[str, any]:
    """Extract conversation history features."""
    features = {
        'conversation_length': 0,
        'num_previous_turns': 0,
        'is_first_turn': False,
    }
    
    conversation_id = str(task_data.get('conversation_id', ''))
    turn_id = task_data.get('turn_id', 1)
    features['is_first_turn'] = (turn_id == 1)
    
    if conversation_id in conversations:
        conv = conversations[conversation_id]
        messages = conv.get('messages', [])
        
        # Count total conversation length
        user_messages = [m for m in messages if m.get('speaker') == 'user']
        features['conversation_length'] = len(user_messages)
        features['num_previous_turns'] = turn_id - 1
    
    return features


def find_zero_score_cases(
    all_results: Dict[str, Dict[str, Dict[str, float]]]
) -> List[str]:
    """
    Find cases where all retriever-strategy combinations have recall = 0.
    
    Returns list of task_ids with zero recall across all strategies.
    """
    zero_score_cases = []
    
    for task_id, retriever_results in all_results.items():
        all_zero = True
        
        for retriever in RETRIEVERS:
            strategy_results = retriever_results.get(retriever, {})
            
            for strategy in STRATEGIES:
                scores = strategy_results.get(strategy, {})
                
                # Check all recall metrics - if any is > 0, not a zero case
                recall_scores = [
                    scores.get('recall_1', 0.0),
                    scores.get('recall_3', 0.0),
                    scores.get('recall_5', 0.0),
                    scores.get('recall_10', 0.0),
                ]
                
                # If any recall is > 0, this is not a zero case
                if any(score > 0 for score in recall_scores):
                    all_zero = False
                    break
            
            if not all_zero:
                break
        
        if all_zero:
            zero_score_cases.append(task_id)
    
    return zero_score_cases


def identify_poor_performance_cases(
    all_results: Dict[str, Dict[str, Dict[str, float]]],
    threshold_percentile: float = 0.25,
    exclude_zero_cases: bool = True
) -> Tuple[List[str], Dict[str, float], List[str]]:
    """
    Identify cases where all strategies perform poorly.
    
    Args:
        all_results: Dict[task_id][retriever][strategy] = scores
        threshold_percentile: Percentile threshold (e.g., 0.25 = bottom 25%)
        exclude_zero_cases: If True, exclude zero-score cases from poor performance analysis
        
    Returns:
        Tuple of (poor_performing_task_ids, threshold_values, zero_score_task_ids)
    """
    # First, identify zero-score cases
    zero_score_task_ids = find_zero_score_cases(all_results)
    zero_set = set(zero_score_task_ids)
    
    # Collect all scores for the primary metric across all tasks (excluding zero cases if requested)
    all_scores = []
    
    for task_id, retriever_results in all_results.items():
        # Skip zero cases if excluding them
        if exclude_zero_cases and task_id in zero_set:
            continue
            
        for retriever, strategy_results in retriever_results.items():
            for strategy, scores in strategy_results.items():
                if PRIMARY_METRIC in scores:
                    all_scores.append(scores[PRIMARY_METRIC])
    
    if not all_scores:
        return [], {}, zero_score_task_ids
    
    # Calculate threshold (bottom percentile)
    threshold = statistics.quantiles(all_scores, n=100)[int(threshold_percentile * 100) - 1]
    
    # Find tasks where ALL strategies perform below threshold
    poor_performing_tasks = []
    
    for task_id, retriever_results in all_results.items():
        # Skip zero cases if excluding them (they're analyzed separately)
        if exclude_zero_cases and task_id in zero_set:
            continue
            
        # Check if ALL retriever-strategy combinations are below threshold
        all_poor = True
        max_score = 0.0
        
        for retriever, strategy_results in retriever_results.items():
            for strategy, scores in strategy_results.items():
                score = scores.get(PRIMARY_METRIC, 0.0)
                max_score = max(max_score, score)
                if score >= threshold:
                    all_poor = False
                    break
            if not all_poor:
                break
        
        if all_poor:
            poor_performing_tasks.append((task_id, max_score))
    
    # Sort by worst performance first
    poor_performing_tasks.sort(key=lambda x: x[1])
    task_ids = [task_id for task_id, _ in poor_performing_tasks]
    
    return task_ids, {PRIMARY_METRIC: threshold}, zero_score_task_ids


def analyze_poor_performance_cases(
    poor_task_ids: List[str],
    all_results: Dict[str, Dict[str, Dict[str, float]]],
    tasks_dir: Path,
    conversations_file: Path,
    domain: str = 'all'
) -> pd.DataFrame:
    """
    Analyze characteristics of poor-performing cases.
    
    Returns a DataFrame with task features and performance metrics.
    """
    # Load conversation data
    conversations = {}
    if conversations_file.exists():
        conversations = load_conversation_data(conversations_file)
    
    # Collect task data
    task_features_list = []
    
    for task_id in poor_task_ids:
        # Extract domain and conversation info from task_id
        # Format: conversation_id<::>turn_id
        parts = task_id.split('<::>')
        if len(parts) != 2:
            continue
        
        conv_id, turn_id_str = parts
        turn_id = int(turn_id_str)
        
        # Determine domain from task_id or use provided domain
        task_domain = domain if domain != 'all' else None
        
        # Try to find task file
        task_file = None
        if task_domain:
            task_file = tasks_dir / task_domain / f"{task_id}.json"
        else:
            # Try all domains
            for d in ['clapnq', 'cloud', 'fiqa', 'govt']:
                potential_file = tasks_dir / d / f"{task_id}.json"
                if potential_file.exists():
                    task_file = potential_file
                    task_domain = d
                    break
        
        if not task_file or not task_file.exists():
            continue
        
        # Load task data
        task_data = load_task_data(task_file)
        if not task_data:
            continue
        
        # Extract features
        features = {
            'task_id': task_id,
            'conversation_id': conv_id,
            'turn_id': turn_id,
            'domain': task_domain,
        }
        
        # Query features
        user_data = task_data.get('user', {})
        query_text = user_data.get('text', '')
        query_features = extract_query_features(query_text)
        features.update(query_features)
        features['query_text'] = query_text
        
        # Enrichment features
        enrichments = user_data.get('enrichments', {})
        enrichment_features = extract_enrichment_features(enrichments)
        features.update(enrichment_features)
        
        # Conversation features
        conv_features = extract_conversation_features(task_data, conversations)
        features.update(conv_features)
        
        # Performance metrics for all strategies
        retriever_results = all_results.get(task_id, {})
        for retriever in RETRIEVERS:
            strategy_results = retriever_results.get(retriever, {})
            for strategy in STRATEGIES:
                scores = strategy_results.get(strategy, {})
                for metric_key, metric_name in METRICS.items():
                    if metric_key in scores:
                        features[f'{retriever}_{strategy}_{metric_key}'] = scores[metric_key]
        
        # Best score across all strategies
        best_score = 0.0
        best_combo = None
        for retriever in RETRIEVERS:
            strategy_results = retriever_results.get(retriever, {})
            for strategy in STRATEGIES:
                scores = strategy_results.get(strategy, {})
                score = scores.get(PRIMARY_METRIC, 0.0)
                if score > best_score:
                    best_score = score
                    best_combo = f"{retriever}_{strategy}"
        
        features['best_score'] = best_score
        features['best_combo'] = best_combo
        
        task_features_list.append(features)
    
    return pd.DataFrame(task_features_list)


def generate_analysis_report(
    df: pd.DataFrame,
    output_dir: Path,
    threshold: float,
    zero_score_df: pd.DataFrame = None
):
    """Generate analysis report with insights."""
    
    report = []
    report.append("=" * 100)
    report.append("POOR PERFORMANCE CASES ANALYSIS")
    report.append("=" * 100)
    
    # Zero-score cases summary
    if zero_score_df is not None and len(zero_score_df) > 0:
        report.append(f"\nZERO-SCORE CASES (All retrievers found no relevant documents): {len(zero_score_df)}")
        report.append(f"  These cases are analyzed separately below.")
    
    if len(df) == 0:
        report.append("\nNo poor-performing cases found (excluding zero-score cases).")
        if zero_score_df is None or len(zero_score_df) == 0:
            print("\n".join(report))
            return
    else:
        report.append(f"\nPOOR-PERFORMING CASES (Below {PRIMARY_METRIC} threshold): {len(df)}")
        report.append(f"Threshold ({PRIMARY_METRIC}): {threshold:.4f}")
        report.append(f"Average best score: {df['best_score'].mean():.4f}")
        report.append(f"Median best score: {df['best_score'].median():.4f}")
    
    # Domain distribution
    if len(df) > 0:
        report.append("\n" + "-" * 100)
        report.append("DOMAIN DISTRIBUTION (Poor-Performing Cases)")
        report.append("-" * 100)
        domain_counts = df['domain'].value_counts()
        for domain, count in domain_counts.items():
            pct = (count / len(df)) * 100
            report.append(f"  {domain}: {count} ({pct:.1f}%)")
    
    # Turn distribution
    first_turn_count = 0
    later_turn_count = 0
    if len(df) > 0:
        report.append("\n" + "-" * 100)
        report.append("TURN DISTRIBUTION")
        report.append("-" * 100)
        turn_counts = df['turn_id'].value_counts().sort_index()
        for turn, count in turn_counts.items():
            pct = (count / len(df)) * 100
            report.append(f"  Turn {turn}: {count} ({pct:.1f}%)")
        
        # First turn vs later turns
        first_turn_count = df['is_first_turn'].sum()
        later_turn_count = len(df) - first_turn_count
        report.append(f"\n  First turn: {first_turn_count} ({first_turn_count/len(df)*100:.1f}%)")
        report.append(f"  Later turns: {later_turn_count} ({later_turn_count/len(df)*100:.1f}%)")
    
    # Query length analysis
    if len(df) > 0 and 'query_length_words' in df.columns:
        report.append("\n" + "-" * 100)
        report.append("QUERY LENGTH ANALYSIS")
        report.append("-" * 100)
        report.append(f"  Average query length (words): {df['query_length_words'].mean():.1f}")
        report.append(f"  Median query length (words): {df['query_length_words'].median():.1f}")
        report.append(f"  Average query length (chars): {df['query_length_chars'].mean():.1f}")
    
    # Question type distribution
    if len(df) > 0:
        report.append("\n" + "-" * 100)
        report.append("QUESTION TYPE DISTRIBUTION")
        report.append("-" * 100)
        if 'question_type' in df.columns:
            qtype_counts = df['question_type'].value_counts()
            for qtype, count in qtype_counts.items():
                pct = (count / len(df)) * 100
                report.append(f"  {qtype}: {count} ({pct:.1f}%)")
    
    # Multi-turn type distribution
    if len(df) > 0:
        report.append("\n" + "-" * 100)
        report.append("MULTI-TURN TYPE DISTRIBUTION")
        report.append("-" * 100)
        if 'multi_turn_type' in df.columns:
            mt_counts = df['multi_turn_type'].value_counts()
            for mt_type, count in mt_counts.items():
                pct = (count / len(df)) * 100
                report.append(f"  {mt_type}: {count} ({pct:.1f}%)")
    
    # Answerability distribution
    if len(df) > 0:
        report.append("\n" + "-" * 100)
        report.append("ANSWERABILITY DISTRIBUTION")
        report.append("-" * 100)
        if 'answerability' in df.columns:
            ans_counts = df['answerability'].value_counts()
            for ans, count in ans_counts.items():
                pct = (count / len(df)) * 100
                report.append(f"  {ans}: {count} ({pct:.1f}%)")
    
    # Best retriever-strategy combinations
    if len(df) > 0:
        report.append("\n" + "-" * 100)
        report.append("BEST RETRIEVER-STRATEGY COMBINATIONS")
        report.append("-" * 100)
        if 'best_combo' in df.columns:
            combo_counts = df['best_combo'].value_counts()
            for combo, count in combo_counts.items():
                pct = (count / len(df)) * 100
                report.append(f"  {combo}: {count} ({pct:.1f}%)")
    
    # Sample queries
    if len(df) > 0:
        report.append("\n" + "-" * 100)
        report.append("SAMPLE QUERIES (WORST PERFORMING)")
        report.append("-" * 100)
        worst_queries = df.nsmallest(10, 'best_score')[['query_text', 'best_score', 'domain', 'turn_id', 'question_type', 'multi_turn_type']]
        for idx, row in worst_queries.iterrows():
            report.append(f"\n  Query: {row['query_text']}")
            report.append(f"    Domain: {row['domain']}, Turn: {row['turn_id']}")
            report.append(f"    Best Score: {row['best_score']:.4f}")
            report.append(f"    Question Type: {row.get('question_type', 'N/A')}")
            report.append(f"    Multi-Turn Type: {row.get('multi_turn_type', 'N/A')}")
    
    # Zero-score cases analysis
    if zero_score_df is not None and len(zero_score_df) > 0:
        report.append("\n" + "=" * 100)
        report.append("ZERO-SCORE CASES ANALYSIS")
        report.append("=" * 100)
        report.append(f"\nTotal zero-score cases: {len(zero_score_df)}")
        report.append("These are cases where ALL retrievers found NO relevant documents.")
        
        # Domain distribution for zero cases
        report.append("\n" + "-" * 100)
        report.append("DOMAIN DISTRIBUTION (Zero-Score Cases)")
        report.append("-" * 100)
        zero_domain_counts = zero_score_df['domain'].value_counts()
        for domain, count in zero_domain_counts.items():
            pct = (count / len(zero_score_df)) * 100
            report.append(f"  {domain}: {count} ({pct:.1f}%)")
        
        # Turn distribution for zero cases
        report.append("\n" + "-" * 100)
        report.append("TURN DISTRIBUTION (Zero-Score Cases)")
        report.append("-" * 100)
        zero_first_turn = zero_score_df['is_first_turn'].sum()
        zero_later_turn = len(zero_score_df) - zero_first_turn
        report.append(f"  First turn: {zero_first_turn} ({zero_first_turn/len(zero_score_df)*100:.1f}%)")
        report.append(f"  Later turns: {zero_later_turn} ({zero_later_turn/len(zero_score_df)*100:.1f}%)")
        
        # Answerability for zero cases
        report.append("\n" + "-" * 100)
        report.append("ANSWERABILITY DISTRIBUTION (Zero-Score Cases)")
        report.append("-" * 100)
        if 'answerability' in zero_score_df.columns:
            zero_ans_counts = zero_score_df['answerability'].value_counts()
            for ans, count in zero_ans_counts.items():
                pct = (count / len(zero_score_df)) * 100
                report.append(f"  {ans}: {count} ({pct:.1f}%)")
        
        # Question type for zero cases
        report.append("\n" + "-" * 100)
        report.append("QUESTION TYPE DISTRIBUTION (Zero-Score Cases)")
        report.append("-" * 100)
        if 'question_type' in zero_score_df.columns:
            zero_qtype_counts = zero_score_df['question_type'].value_counts()
            for qtype, count in zero_qtype_counts.items():
                pct = (count / len(zero_score_df)) * 100
                report.append(f"  {qtype}: {count} ({pct:.1f}%)")
        
        # Multi-turn type for zero cases
        report.append("\n" + "-" * 100)
        report.append("MULTI-TURN TYPE DISTRIBUTION (Zero-Score Cases)")
        report.append("-" * 100)
        if 'multi_turn_type' in zero_score_df.columns:
            zero_mt_counts = zero_score_df['multi_turn_type'].value_counts()
            for mt_type, count in zero_mt_counts.items():
                pct = (count / len(zero_score_df)) * 100
                report.append(f"  {mt_type}: {count} ({pct:.1f}%)")
        
        # Sample zero-score queries
        report.append("\n" + "-" * 100)
        report.append("SAMPLE ZERO-SCORE QUERIES")
        report.append("-" * 100)
        sample_zero = zero_score_df[['query_text', 'domain', 'turn_id', 'question_type', 'multi_turn_type']].head(15)
        for idx, row in sample_zero.iterrows():
            report.append(f"\n  Query: {row['query_text']}")
            report.append(f"    Domain: {row['domain']}, Turn: {row['turn_id']}")
            report.append(f"    Question Type: {row.get('question_type', 'N/A')}")
            report.append(f"    Multi-Turn Type: {row.get('multi_turn_type', 'N/A')}")
    
    # Write report
    report_text = "\n".join(report)
    print(report_text)
    
    report_file = output_dir / "poor_performance_analysis.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_text)
    print(f"\nReport saved to: {report_file}")
    
    # Save detailed data
    if len(df) > 0:
        csv_file = output_dir / "poor_performance_cases.csv"
        df.to_csv(csv_file, index=False)
        print(f"Detailed data saved to: {csv_file}")
    
    # Save zero-score cases data
    if zero_score_df is not None and len(zero_score_df) > 0:
        zero_csv_file = output_dir / "zero_score_cases.csv"
        zero_score_df.to_csv(zero_csv_file, index=False)
        print(f"Zero-score cases data saved to: {zero_csv_file}")
    
    # Save JSON summary
    summary = {
        'total_poor_cases': len(df),
        'total_zero_cases': len(zero_score_df) if zero_score_df is not None else 0,
        'threshold': threshold,
        'threshold_metric': PRIMARY_METRIC,
    }
    
    if len(df) > 0:
        summary.update({
            'average_best_score': float(df['best_score'].mean()),
            'median_best_score': float(df['best_score'].median()),
            'domain_distribution': df['domain'].value_counts().to_dict(),
            'turn_distribution': df['turn_id'].value_counts().to_dict(),
            'first_turn_count': int(first_turn_count),
            'later_turn_count': int(later_turn_count),
            'avg_query_length_words': float(df['query_length_words'].mean()),
            'question_type_distribution': df['question_type'].value_counts().to_dict() if 'question_type' in df.columns else {},
            'multi_turn_type_distribution': df['multi_turn_type'].value_counts().to_dict() if 'multi_turn_type' in df.columns else {},
            'answerability_distribution': df['answerability'].value_counts().to_dict() if 'answerability' in df.columns else {},
            'best_combo_distribution': df['best_combo'].value_counts().to_dict() if 'best_combo' in df.columns else {},
        })
    
    if zero_score_df is not None and len(zero_score_df) > 0:
        zero_first_turn = zero_score_df['is_first_turn'].sum()
        zero_later_turn = len(zero_score_df) - zero_first_turn
        summary['zero_score_analysis'] = {
            'domain_distribution': zero_score_df['domain'].value_counts().to_dict(),
            'first_turn_count': int(zero_first_turn),
            'later_turn_count': int(zero_later_turn),
            'avg_query_length_words': float(zero_score_df['query_length_words'].mean()),
            'question_type_distribution': zero_score_df['question_type'].value_counts().to_dict() if 'question_type' in zero_score_df.columns else {},
            'multi_turn_type_distribution': zero_score_df['multi_turn_type'].value_counts().to_dict() if 'multi_turn_type' in zero_score_df.columns else {},
            'answerability_distribution': zero_score_df['answerability'].value_counts().to_dict() if 'answerability' in zero_score_df.columns else {},
        }
    
    json_file = output_dir / "poor_performance_summary.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {json_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze queries where all retrievers perform poorly"
    )
    parser.add_argument(
        "--retriever",
        type=str,
        choices=['elser', 'bm25', 'bge', 'all'],
        default='elser',
        help="Retriever to analyze (default: elser)"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Directory containing evaluated results (relative to project root). If not specified, uses default based on retriever."
    )
    parser.add_argument(
        "--domain",
        type=str,
        choices=['all', 'clapnq', 'cloud', 'fiqa', 'govt'],
        default='all',
        help="Domain to analyze (default: all)"
    )
    parser.add_argument(
        "--threshold-percentile",
        type=float,
        default=0.25,
        help="Percentile threshold for poor performance (default: 0.25 = bottom 25%%)"
    )
    parser.add_argument(
        "--tasks-dir",
        type=str,
        default="cleaned_data/tasks",
        help="Directory containing task JSON files"
    )
    parser.add_argument(
        "--conversations-file",
        type=str,
        default="human/conversations/conversations.json",
        help="Path to conversations JSON file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="new-strategies",
        help="Output directory for analysis results"
    )
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent
    # Go up from new-strategies/oracle-v2/retrieval_tasks/ideas/scripts to project root (5 levels)
    project_root = script_dir.parents[4]
    
    # Resolve paths
    if args.results_dir:
        if Path(args.results_dir).is_absolute():
            results_dir = Path(args.results_dir)
        else:
            results_dir = (project_root / args.results_dir).resolve()
    else:
        results_dir = (project_root / "scripts/baselines/retrieval_scripts").resolve()
    
    tasks_dir = (project_root / args.tasks_dir).resolve()
    conversations_file = project_root / args.conversations_file
    output_dir = script_dir / args.output_dir
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 100)
    print("POOR PERFORMANCE CASES ANALYSIS")
    print("=" * 100)
    print(f"Retriever(s): {args.retriever}")
    print(f"Results directory: {results_dir}")
    print(f"Tasks directory: {tasks_dir}")
    print(f"Domain: {args.domain}")
    print(f"Threshold percentile: {args.threshold_percentile}")
    print(f"Output directory: {output_dir}")
    
    # Determine which retrievers to analyze
    if args.retriever == 'all':
        retrievers_to_analyze = RETRIEVERS
    else:
        retrievers_to_analyze = [args.retriever]
    
    # Load all results
    print("\nLoading evaluated results...")
    all_results = {}  # Dict[task_id][retriever][strategy] = scores
    
    for retriever in retrievers_to_analyze:
        retriever_dir = results_dir / retriever / "results"
        
        for strategy in STRATEGIES:
            if args.domain == 'all':
                filepath = retriever_dir / f"{retriever}_all_{strategy}_evaluated.jsonl"
            else:
                filepath = retriever_dir / f"{retriever}_{args.domain}_{strategy}_evaluated.jsonl"
            
            print(f"  Loading {retriever}/{strategy}...")
            results = load_evaluated_results(filepath)
            
            for task_id, scores in results.items():
                if task_id not in all_results:
                    all_results[task_id] = {}
                if retriever not in all_results[task_id]:
                    all_results[task_id][retriever] = {}
                all_results[task_id][retriever][strategy] = scores
    
    print(f"\nLoaded results for {len(all_results)} tasks")
    
    # Identify poor-performing cases and zero-score cases
    print("\nIdentifying poor-performing cases...")
    poor_task_ids, thresholds, zero_task_ids = identify_poor_performance_cases(
        all_results, args.threshold_percentile, exclude_zero_cases=True
    )
    print(f"Found {len(poor_task_ids)} poor-performing cases")
    print(f"Found {len(zero_task_ids)} zero-score cases (all retrievers found no relevant documents)")
    
    if len(poor_task_ids) == 0 and len(zero_task_ids) == 0:
        print("No poor-performing cases found. Try adjusting the threshold percentile.")
        return
    
    # Analyze poor-performing cases
    df = None
    if len(poor_task_ids) > 0:
        print("\nAnalyzing poor-performing cases...")
        df = analyze_poor_performance_cases(
            poor_task_ids, all_results, tasks_dir, conversations_file, args.domain
        )
        print(f"Analyzed {len(df)} poor-performing cases")
    
    # Analyze zero-score cases
    zero_df = None
    if len(zero_task_ids) > 0:
        print("\nAnalyzing zero-score cases...")
        zero_df = analyze_poor_performance_cases(
            zero_task_ids, all_results, tasks_dir, conversations_file, args.domain
        )
        print(f"Analyzed {len(zero_df)} zero-score cases")
    
    # Generate report
    print("\nGenerating analysis report...")
    threshold_value = thresholds.get(PRIMARY_METRIC, 0.0) if thresholds else 0.0
    generate_analysis_report(df if df is not None else pd.DataFrame(), output_dir, threshold_value, zero_df)
    
    print("\n" + "=" * 100)
    print("ANALYSIS COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()

