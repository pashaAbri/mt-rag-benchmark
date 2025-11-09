#!/usr/bin/env python3
"""
Calculate and aggregate all metrics from generation results.
Saves raw aggregated data to JSON for later table generation.

This script separates data calculation from presentation.
"""

import json
import os
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any
import statistics


# Model display names mapping
MODEL_NAMES = {
    "command_r_plus": "Command-R+ (104B)",
    "gpt_4o": "GPT-4o",
    "gpt_4o_mini": "GPT-4o-mini",
    "llama_3.1_405b": "Llama 3.1 405B Instruct",
    "llama_3.1_70b": "Llama 3.1 70B Instruct",
    "llama_3.1_8b": "Llama 3.1 8B Instruct",
    "mixtral_8x22b": "Mixtral 8x22B Instruct",
    "qwen_2.5_72b": "Qwen 2.5 (72B)",
    "qwen_2.5_7b": "Qwen 2.5 (7B)",
}

# Domain extraction from collection name
DOMAIN_MAP = {
    "clapnq": "CLAPNQ",
    "fiqa": "FiQA",
    "govt": "Govt",
    "cloud": "Cloud",
}

# All metrics to calculate
METRICS = ['RL_F_idk', 'RB_llm_idk', 'RB_agg_idk']

# Scenarios
SCENARIOS = ['reference', 'reference_rag', 'full_rag']


def extract_domain(collection_name: str) -> str:
    """Extract domain from collection name."""
    collection_lower = collection_name.lower()
    for key, domain in DOMAIN_MAP.items():
        if key in collection_lower:
            return domain
    return "Unknown"


def load_results(file_path: Path) -> List[Dict]:
    """Load results from a JSONL file."""
    results = []
    if not file_path.exists():
        return results
        
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def calculate_answerability_accuracy(results: List[Dict]) -> float:
    """
    Calculate answerability accuracy.
    
    Correct behavior:
    - ANSWERABLE questions: Model should answer (idk_eval=0)
    - UNANSWERABLE/PARTIAL questions: Model should decline (idk_eval=1)
    """
    if not results:
        return None
    
    correct_count = 0
    for r in results:
        idk_eval = r.get('metrics', {}).get('idk_eval', [None])[0]
        if idk_eval is None:
            continue
            
        answerability = r.get('Answerability', [])
        
        # For ANSWERABLE: correct if model answered (idk_eval=0)
        if 'ANSWERABLE' in answerability and 'PARTIAL' not in str(answerability):
            if idk_eval == 0:
                correct_count += 1
        # For UNANSWERABLE or PARTIALLY_ANSWERABLE: correct if model declined (idk_eval=1)
        else:
            if idk_eval == 1:
                correct_count += 1
    
    return correct_count / len(results) if results else None


def get_metric_value(result: Dict, metric_name: str) -> float:
    """Get metric value from result, handling lists."""
    metrics = result.get('metrics', {})
    value = metrics.get(metric_name, [None])
    if isinstance(value, list):
        return value[0] if value else None
    return value


def aggregate_metrics(results: List[Dict], metric_names: List[str]) -> Dict[str, Any]:
    """Aggregate metrics across results."""
    if not results:
        return {
            'count': 0,
            'ans_acc': None,
            **{metric: None for metric in metric_names}
        }
    
    aggregated = {
        'count': len(results),
        'ans_acc': calculate_answerability_accuracy(results)
    }
    
    # Calculate average for each metric
    for metric_name in metric_names:
        values = [get_metric_value(r, metric_name) for r in results]
        valid_values = [v for v in values if v is not None]
        aggregated[metric_name] = statistics.mean(valid_values) if valid_values else None
    
    return aggregated


def filter_by_answerability(results: List[Dict], answerability: str) -> List[Dict]:
    """Filter results by answerability type."""
    filtered = []
    for r in results:
        ans_list = r.get('Answerability', [])
        # Handle both 'PARTIAL' and 'PARTIALLY_ANSWERABLE'
        if answerability.upper() == 'PARTIALLY_ANSWERABLE' or answerability.upper() == 'PARTIAL':
            if 'PARTIAL' in [a.upper() for a in ans_list] or 'PARTIALLY_ANSWERABLE' in [a.upper() for a in ans_list]:
                filtered.append(r)
        elif answerability.upper() in [a.upper() for a in ans_list]:
            filtered.append(r)
    return filtered


def filter_by_turn(results: List[Dict], is_first_turn: bool) -> List[Dict]:
    """Filter results by turn position (turns are 1-indexed)."""
    filtered = []
    for r in results:
        turn = r.get('turn', None)
        if turn is None:
            continue
        if is_first_turn and turn == 1:
            filtered.append(r)
        elif not is_first_turn and turn > 1:
            filtered.append(r)
    return filtered


def filter_by_domain(results: List[Dict], domain: str) -> List[Dict]:
    """Filter results by domain."""
    filtered = []
    for r in results:
        collection = r.get('Collection', '')
        result_domain = extract_domain(collection)
        if result_domain == domain:
            filtered.append(r)
    return filtered


def calculate_all_metrics(base_dir: Path) -> Dict[str, Any]:
    """
    Calculate all metrics for all models across all dimensions.
    
    Returns a comprehensive dictionary with all aggregated results.
    """
    print("=" * 70)
    print("CALCULATING ALL METRICS")
    print("=" * 70)
    
    all_data = {
        'metadata': {
            'base_dir': str(base_dir),
            'models': list(MODEL_NAMES.keys()),
            'scenarios': SCENARIOS,
            'domains': list(DOMAIN_MAP.values()),
            'metrics': METRICS
        },
        'results': {}
    }
    
    # Process each scenario
    for scenario in SCENARIOS:
        print(f"\n{'='*70}")
        print(f"Processing Scenario: {scenario.upper()}")
        print(f"{'='*70}")
        
        scenario_dir = base_dir / scenario / "results"
        all_data['results'][scenario] = {}
        
        # Process each model
        for model_key, model_name in MODEL_NAMES.items():
            print(f"\n  Processing: {model_name}")
            
            file_pattern = f"{model_key}_{scenario}_evaluated.jsonl"
            file_path = scenario_dir / file_pattern
            
            if not file_path.exists():
                print(f"    ⚠️  File not found: {file_path}")
                all_data['results'][scenario][model_key] = None
                continue
            
            # Load all results for this model
            results = load_results(file_path)
            print(f"    ✓ Loaded {len(results)} results")
            
            model_data = {
                'total': aggregate_metrics(results, METRICS)
            }
            
            # Only calculate breakdowns for reference scenario
            if scenario == 'reference':
                # By Answerability
                model_data['by_answerability'] = {}
                for ans_type in ['ANSWERABLE', 'PARTIAL', 'UNANSWERABLE']:
                    filtered = filter_by_answerability(results, ans_type)
                    model_data['by_answerability'][ans_type] = aggregate_metrics(filtered, METRICS)
                    print(f"      - {ans_type}: {len(filtered)} questions")
                
                # By Turn
                model_data['by_turn'] = {}
                turn_1 = filter_by_turn(results, True)
                turn_gt_1 = filter_by_turn(results, False)
                model_data['by_turn']['TURN_1'] = aggregate_metrics(turn_1, METRICS)
                model_data['by_turn']['TURN_GT_1'] = aggregate_metrics(turn_gt_1, METRICS)
                print(f"      - TURN 1: {len(turn_1)} questions")
                print(f"      - TURN > 1: {len(turn_gt_1)} questions")
                
                # By Domain
                model_data['by_domain'] = {}
                for domain in DOMAIN_MAP.values():
                    filtered = filter_by_domain(results, domain)
                    model_data['by_domain'][domain] = aggregate_metrics(filtered, METRICS)
                    print(f"      - {domain}: {len(filtered)} questions")
            
            all_data['results'][scenario][model_key] = model_data
    
    return all_data


def save_results(data: Dict[str, Any], output_path: Path):
    """Save aggregated results to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"\n✓ Results saved to: {output_path}")


def main():
    """Main function to calculate and save all metrics."""
    # Paths - script is in .analysis_generated/, results are in parent directory
    script_dir = Path(__file__).parent  # .analysis_generated/
    base_dir = script_dir.parent  # scripts/generation_scripts/
    output_dir = script_dir  # Save output in same dir as scripts
    
    output_file = output_dir / "aggregated_results.json"
    
    print("\n" + "=" * 70)
    print("MT-RAG METRICS CALCULATOR")
    print("=" * 70)
    print(f"Base directory: {base_dir}")
    print(f"Output file: {output_file}")
    
    # Calculate all metrics
    all_data = calculate_all_metrics(base_dir)
    
    # Save to JSON
    save_results(all_data, output_file)
    
    # Print summary
    print("\n" + "=" * 70)
    print("CALCULATION SUMMARY")
    print("=" * 70)
    
    for scenario in SCENARIOS:
        scenario_data = all_data['results'].get(scenario, {})
        models_with_data = sum(1 for v in scenario_data.values() if v is not None)
        print(f"  {scenario}: {models_with_data}/{len(MODEL_NAMES)} models processed")
    
    print("\n" + "=" * 70)
    print("✓ ALL METRICS CALCULATED SUCCESSFULLY!")
    print("=" * 70)
    print("\nNext step: Run 'python3 generate_tables.py' to create formatted tables")
    print(f"Output will be saved to: {output_dir}/")


if __name__ == "__main__":
    main()

