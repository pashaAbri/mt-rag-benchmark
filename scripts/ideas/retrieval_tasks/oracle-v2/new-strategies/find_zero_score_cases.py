#!/usr/bin/env python3
"""
Find cases where all retrievers get a score of 0 (no relevant documents found).

This script identifies tasks where all three strategies (lastturn, rewrite, questions)
across all retrievers fail to find any relevant documents.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict


# Strategy names
STRATEGIES = ['lastturn', 'rewrite', 'questions']
RETRIEVERS = ['elser', 'bm25', 'bge']

# Metrics to check for zero scores
METRICS_TO_CHECK = ['recall_1', 'recall_3', 'recall_5', 'recall_10', 'ndcg_cut_5']
PRIMARY_METRIC = 'ndcg_cut_5'


def load_evaluated_results(filepath: Path) -> Dict[str, Dict[str, float]]:
    """Load evaluated results from a JSONL file."""
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


def load_task_data(task_file: Path) -> Dict:
    """Load a single task file."""
    try:
        with open(task_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


def find_zero_score_cases(
    all_results: Dict[str, Dict[str, Dict[str, float]]]
) -> List[Tuple[str, Dict]]:
    """
    Find cases where all retriever-strategy combinations score 0.
    
    A case is considered zero-score if ALL recall metrics are 0 for ALL
    retriever-strategy combinations.
    
    Returns list of (task_id, details_dict) tuples.
    """
    zero_score_cases = []
    
    for task_id, retriever_results in all_results.items():
        all_zero = True
        scores_summary = {}
        
        for retriever in RETRIEVERS:
            strategy_results = retriever_results.get(retriever, {})
            scores_summary[retriever] = {}
            
            for strategy in STRATEGIES:
                scores = strategy_results.get(strategy, {})
                
                # Check all recall metrics - if any is > 0, not a zero case
                recall_scores = {
                    'recall_1': scores.get('recall_1', 0.0),
                    'recall_3': scores.get('recall_3', 0.0),
                    'recall_5': scores.get('recall_5', 0.0),
                    'recall_10': scores.get('recall_10', 0.0),
                }
                
                scores_summary[retriever][strategy] = {
                    **recall_scores,
                    'ndcg_cut_5': scores.get('ndcg_cut_5', 0.0),
                }
                
                # Check if any recall is > 0
                if any(score > 0 for score in recall_scores.values()):
                    all_zero = False
                    break
            
            if not all_zero:
                break
        
        if all_zero:
            zero_score_cases.append((task_id, scores_summary))
    
    return zero_score_cases


def analyze_zero_cases(
    zero_cases: List[Tuple[str, Dict]],
    tasks_dir: Path,
    conversations_file: Path,
    domain: str = 'all'
) -> List[Dict]:
    """Analyze zero-score cases and extract their characteristics."""
    import re
    
    # Load conversation data
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
    except Exception:
        pass
    
    analyzed_cases = []
    
    for task_id, scores_summary in zero_cases:
        # Extract domain and conversation info from task_id
        parts = task_id.split('<::>')
        if len(parts) != 2:
            continue
        
        conv_id, turn_id_str = parts
        turn_id = int(turn_id_str)
        
        # Try to find task file
        task_file = None
        task_domain = domain if domain != 'all' else None
        
        if task_domain:
            task_file = tasks_dir / task_domain / f"{task_id}.json"
        else:
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
        
        # Extract information
        user_data = task_data.get('user', {})
        query_text = user_data.get('text', '')
        enrichments = user_data.get('enrichments', {})
        
        # Extract features
        case_info = {
            'task_id': task_id,
            'conversation_id': conv_id,
            'turn_id': turn_id,
            'domain': task_domain,
            'query_text': query_text,
            'query_length_words': len(query_text.split()),
            'query_length_chars': len(query_text),
            'has_question_mark': '?' in query_text,
            'has_wh_word': bool(re.search(r'\b(what|who|where|when|why|how|which|whose|whom)\b', query_text.lower())),
            'answerability': None,
            'question_type': None,
            'multi_turn_type': None,
            'is_first_turn': (turn_id == 1),
            'conversation_length': 0,
            'num_previous_turns': turn_id - 1,
        }
        
        # Enrichments
        if enrichments:
            answerability = enrichments.get('Answerability', [])
            if answerability and isinstance(answerability, list):
                case_info['answerability'] = answerability[0] if answerability else None
            elif answerability:
                case_info['answerability'] = answerability
            
            question_type = enrichments.get('Question Type', [])
            if question_type and isinstance(question_type, list):
                case_info['question_type'] = question_type[0] if question_type else None
            elif question_type:
                case_info['question_type'] = question_type
            
            multi_turn = enrichments.get('Multi-Turn', [])
            if multi_turn and isinstance(multi_turn, list):
                case_info['multi_turn_type'] = multi_turn[0] if multi_turn else None
            elif multi_turn:
                case_info['multi_turn_type'] = multi_turn
        
        # Conversation features
        if conv_id in conversations:
            conv = conversations[conv_id]
            messages = conv.get('messages', [])
            user_messages = [m for m in messages if m.get('speaker') == 'user']
            case_info['conversation_length'] = len(user_messages)
        
        # Add scores summary
        case_info['scores'] = scores_summary
        
        analyzed_cases.append(case_info)
    
    return analyzed_cases


def print_analysis(analyzed_cases: List[Dict], output_file: Path):
    """Print and save analysis of zero-score cases."""
    if not analyzed_cases:
        print("No zero-score cases found.")
        return
    
    report = []
    report.append("=" * 100)
    report.append("ZERO-SCORE CASES ANALYSIS")
    report.append("=" * 100)
    report.append(f"\nTotal zero-score cases: {len(analyzed_cases)}")
    
    # Domain distribution
    domain_counts = defaultdict(int)
    for case in analyzed_cases:
        domain_counts[case['domain']] += 1
    
    report.append("\n" + "-" * 100)
    report.append("DOMAIN DISTRIBUTION")
    report.append("-" * 100)
    for domain, count in sorted(domain_counts.items()):
        pct = (count / len(analyzed_cases)) * 100
        report.append(f"  {domain}: {count} ({pct:.1f}%)")
    
    # Turn distribution
    turn_counts = defaultdict(int)
    for case in analyzed_cases:
        turn_counts[case['turn_id']] += 1
    
    report.append("\n" + "-" * 100)
    report.append("TURN DISTRIBUTION")
    report.append("-" * 100)
    first_turn_count = sum(1 for c in analyzed_cases if c['is_first_turn'])
    later_turn_count = len(analyzed_cases) - first_turn_count
    report.append(f"  First turn: {first_turn_count} ({first_turn_count/len(analyzed_cases)*100:.1f}%)")
    report.append(f"  Later turns: {later_turn_count} ({later_turn_count/len(analyzed_cases)*100:.1f}%)")
    
    for turn in sorted(turn_counts.keys()):
        count = turn_counts[turn]
        pct = (count / len(analyzed_cases)) * 100
        report.append(f"  Turn {turn}: {count} ({pct:.1f}%)")
    
    # Answerability
    answerability_counts = defaultdict(int)
    for case in analyzed_cases:
        ans = case.get('answerability', 'UNKNOWN')
        answerability_counts[ans] += 1
    
    report.append("\n" + "-" * 100)
    report.append("ANSWERABILITY DISTRIBUTION")
    report.append("-" * 100)
    for ans, count in sorted(answerability_counts.items()):
        pct = (count / len(analyzed_cases)) * 100
        report.append(f"  {ans}: {count} ({pct:.1f}%)")
    
    # Question type
    qtype_counts = defaultdict(int)
    for case in analyzed_cases:
        qtype = case.get('question_type', 'UNKNOWN')
        qtype_counts[qtype] += 1
    
    report.append("\n" + "-" * 100)
    report.append("QUESTION TYPE DISTRIBUTION")
    report.append("-" * 100)
    for qtype, count in sorted(qtype_counts.items()):
        pct = (count / len(analyzed_cases)) * 100
        report.append(f"  {qtype}: {count} ({pct:.1f}%)")
    
    # Multi-turn type
    mt_counts = defaultdict(int)
    for case in analyzed_cases:
        mt = case.get('multi_turn_type', 'UNKNOWN')
        mt_counts[mt] += 1
    
    report.append("\n" + "-" * 100)
    report.append("MULTI-TURN TYPE DISTRIBUTION")
    report.append("-" * 100)
    for mt, count in sorted(mt_counts.items()):
        pct = (count / len(analyzed_cases)) * 100
        report.append(f"  {mt}: {count} ({pct:.1f}%)")
    
    # Query length stats
    avg_length = sum(c['query_length_words'] for c in analyzed_cases) / len(analyzed_cases)
    report.append("\n" + "-" * 100)
    report.append("QUERY CHARACTERISTICS")
    report.append("-" * 100)
    report.append(f"  Average query length (words): {avg_length:.1f}")
    report.append(f"  Queries with question mark: {sum(1 for c in analyzed_cases if c['has_question_mark'])} ({sum(1 for c in analyzed_cases if c['has_question_mark'])/len(analyzed_cases)*100:.1f}%)")
    report.append(f"  Queries with wh-word: {sum(1 for c in analyzed_cases if c['has_wh_word'])} ({sum(1 for c in analyzed_cases if c['has_wh_word'])/len(analyzed_cases)*100:.1f}%)")
    
    # Sample cases
    report.append("\n" + "=" * 100)
    report.append("SAMPLE ZERO-SCORE CASES")
    report.append("=" * 100)
    
    # Show samples from different domains
    samples_shown = set()
    samples_per_domain = 3
    
    for domain in sorted(domain_counts.keys()):
        domain_cases = [c for c in analyzed_cases if c['domain'] == domain]
        report.append(f"\n--- {domain.upper()} Domain ({len(domain_cases)} cases) ---")
        
        for i, case in enumerate(domain_cases[:samples_per_domain]):
            report.append(f"\n  Case {i+1}:")
            report.append(f"    Task ID: {case['task_id']}")
            report.append(f"    Query: {case['query_text']}")
            report.append(f"    Turn: {case['turn_id']} (First turn: {case['is_first_turn']})")
            report.append(f"    Answerability: {case.get('answerability', 'N/A')}")
            report.append(f"    Question Type: {case.get('question_type', 'N/A')}")
            report.append(f"    Multi-Turn Type: {case.get('multi_turn_type', 'N/A')}")
            report.append(f"    Query Length: {case['query_length_words']} words")
    
    report_text = "\n".join(report)
    print(report_text)
    
    # Save report
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report_text)
    print(f"\nReport saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Find cases where all retrievers score 0"
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
        help="Directory containing evaluated results"
    )
    parser.add_argument(
        "--domain",
        type=str,
        choices=['all', 'clapnq', 'cloud', 'fiqa', 'govt'],
        default='all',
        help="Domain to analyze (default: all)"
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
        "--output",
        type=str,
        default="zero_score_cases_analysis.txt",
        help="Output file for analysis"
    )
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent
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
    output_file = script_dir / args.output
    
    print("=" * 100)
    print("FINDING ZERO-SCORE CASES")
    print("=" * 100)
    print(f"Retriever(s): {args.retriever}")
    print(f"Results directory: {results_dir}")
    print(f"Domain: {args.domain}")
    
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
    
    # Find zero-score cases
    print("\nFinding zero-score cases...")
    zero_cases = find_zero_score_cases(all_results)
    print(f"Found {len(zero_cases)} zero-score cases")
    
    if len(zero_cases) == 0:
        print("No zero-score cases found.")
        return
    
    # Analyze zero cases
    print("\nAnalyzing zero-score cases...")
    analyzed_cases = analyze_zero_cases(
        zero_cases, tasks_dir, conversations_file, args.domain
    )
    
    print(f"Analyzed {len(analyzed_cases)} cases")
    
    # Print and save analysis
    print_analysis(analyzed_cases, output_file)
    
    # Save detailed JSON
    json_file = script_dir / "zero_score_cases.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(analyzed_cases, f, indent=2)
    print(f"\nDetailed data saved to: {json_file}")


if __name__ == "__main__":
    main()

