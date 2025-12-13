#!/usr/bin/env python3
"""
Evaluate query tagging performance: Compare LLM predictions against human annotations.

This script:
1. Loads all tagged queries from `tagged_queries/`
2. Compares LLM predictions against human annotations
3. Computes per-class and overall metrics (precision, recall, F1)
4. Saves results to a JSON file

Usage:
    python tag_queries_performance.py [--output OUTPUT_FILE]
"""

import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# Paths
script_dir = Path(__file__).parent
TAGGED_DIR = script_dir / "tagged_queries"
DEFAULT_OUTPUT = script_dir / "tag_queries_performance.json"

DOMAINS = ['clapnq', 'cloud', 'fiqa', 'govt']

QUESTION_TYPES = [
    "Comparative",
    "Composite",
    "Explanation",
    "Factoid",
    "How-To",
    "Keyword",
    "Non-Question",
    "Opinion",
    "Summarization",
    "Troubleshooting"
]


def extract_tags(data, source):
    """Extract tags from data based on source ('human' or 'llm')."""
    tags = []
    
    if source == 'human':
        if 'user' in data and 'enrichments' in data['user']:
            enrichments = data['user']['enrichments']
            if 'Question Type' in enrichments:
                val = enrichments['Question Type']
                if isinstance(val, list):
                    tags = val
                elif isinstance(val, str):
                    tags = [val]
    elif source == 'llm':
        if 'oracle_metadata' in data:
            val = data['oracle_metadata'].get('predicted_question_type', [])
            if isinstance(val, list):
                tags = val
            elif isinstance(val, str) and val not in ["Unknown", "Error"]:
                tags = [val]
    
    # Normalize tags
    return [t.strip() for t in tags if t and t.strip() in QUESTION_TYPES]


def compute_metrics(tp, fp, fn):
    """Compute precision, recall, and F1 from counts."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4)
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate query tagging performance")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT), help="Output JSON file")
    args = parser.parse_args()
    
    # Per-class statistics
    class_stats = {qt: {"tp": 0, "fp": 0, "fn": 0} for qt in QUESTION_TYPES}
    
    # Domain-level statistics
    domain_stats = {d: {"total": 0, "exact_match": 0, "partial_match": 0, "no_match": 0} for d in DOMAINS}
    
    # Overall statistics
    total_samples = 0
    exact_matches = 0  # All predicted tags match all human tags
    partial_matches = 0  # At least one overlapping tag
    no_matches = 0  # No overlap at all
    
    # Confusion examples
    confusion_examples = []
    
    # Tagging model info
    tagging_model = None
    
    for domain in DOMAINS:
        domain_dir = TAGGED_DIR / domain
        if not domain_dir.exists():
            print(f"Skipping {domain}: directory not found")
            continue
        
        json_files = list(domain_dir.glob("*.json"))
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Extract tags
                human_tags = extract_tags(data, 'human')
                llm_tags = extract_tags(data, 'llm')
                
                # Get tagging model
                if tagging_model is None and 'oracle_metadata' in data:
                    tagging_model = data['oracle_metadata'].get('tagging_model', 'unknown')
                
                # Skip if either is empty
                if not human_tags or not llm_tags:
                    continue
                
                total_samples += 1
                domain_stats[domain]["total"] += 1
                
                human_set = set(human_tags)
                llm_set = set(llm_tags)
                
                # Match statistics
                overlap = human_set & llm_set
                
                if human_set == llm_set:
                    exact_matches += 1
                    domain_stats[domain]["exact_match"] += 1
                elif overlap:
                    partial_matches += 1
                    domain_stats[domain]["partial_match"] += 1
                else:
                    no_matches += 1
                    domain_stats[domain]["no_match"] += 1
                    
                    # Record confusion example
                    query = data.get('user', {}).get('text', "Unknown")
                    confusion_examples.append({
                        "domain": domain,
                        "query": query[:200],  # Truncate long queries
                        "human": list(human_set),
                        "llm": list(llm_set),
                        "file": json_file.name
                    })
                
                # Per-class metrics (multi-label)
                for qt in QUESTION_TYPES:
                    in_human = qt in human_set
                    in_llm = qt in llm_set
                    
                    if in_human and in_llm:
                        class_stats[qt]["tp"] += 1
                    elif in_llm and not in_human:
                        class_stats[qt]["fp"] += 1
                    elif in_human and not in_llm:
                        class_stats[qt]["fn"] += 1
                        
            except Exception as e:
                print(f"Error processing {json_file}: {e}")
                continue
    
    # Compute per-class metrics
    class_metrics = {}
    for qt in QUESTION_TYPES:
        stats = class_stats[qt]
        metrics = compute_metrics(stats["tp"], stats["fp"], stats["fn"])
        metrics["support"] = stats["tp"] + stats["fn"]  # Total human annotations
        metrics["predictions"] = stats["tp"] + stats["fp"]  # Total LLM predictions
        class_metrics[qt] = metrics
    
    # Compute micro and macro averages
    total_tp = sum(class_stats[qt]["tp"] for qt in QUESTION_TYPES)
    total_fp = sum(class_stats[qt]["fp"] for qt in QUESTION_TYPES)
    total_fn = sum(class_stats[qt]["fn"] for qt in QUESTION_TYPES)
    
    micro_metrics = compute_metrics(total_tp, total_fp, total_fn)
    
    # Macro average (average of per-class metrics, excluding zero-support classes)
    active_classes = [qt for qt in QUESTION_TYPES if class_metrics[qt]["support"] > 0]
    macro_precision = sum(class_metrics[qt]["precision"] for qt in active_classes) / len(active_classes) if active_classes else 0
    macro_recall = sum(class_metrics[qt]["recall"] for qt in active_classes) / len(active_classes) if active_classes else 0
    macro_f1 = sum(class_metrics[qt]["f1"] for qt in active_classes) / len(active_classes) if active_classes else 0
    
    macro_metrics = {
        "precision": round(macro_precision, 4),
        "recall": round(macro_recall, 4),
        "f1": round(macro_f1, 4)
    }
    
    # Domain-level metrics
    domain_metrics = {}
    for d in DOMAINS:
        stats = domain_stats[d]
        if stats["total"] > 0:
            domain_metrics[d] = {
                "total": stats["total"],
                "exact_match": stats["exact_match"],
                "exact_match_rate": round(stats["exact_match"] / stats["total"], 4),
                "partial_match": stats["partial_match"],
                "partial_match_rate": round(stats["partial_match"] / stats["total"], 4),
                "no_match": stats["no_match"],
                "no_match_rate": round(stats["no_match"] / stats["total"], 4)
            }
    
    # Compile results
    results = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "tagging_model": tagging_model,
            "domains_evaluated": DOMAINS,
            "question_types": QUESTION_TYPES
        },
        "overall": {
            "total_samples": total_samples,
            "exact_matches": exact_matches,
            "exact_match_rate": round(exact_matches / total_samples, 4) if total_samples > 0 else 0,
            "partial_matches": partial_matches,
            "partial_match_rate": round(partial_matches / total_samples, 4) if total_samples > 0 else 0,
            "no_matches": no_matches,
            "no_match_rate": round(no_matches / total_samples, 4) if total_samples > 0 else 0
        },
        "micro_average": micro_metrics,
        "macro_average": macro_metrics,
        "per_class": class_metrics,
        "per_domain": domain_metrics,
        "confusion_examples": confusion_examples[:50]  # Limit to 50 examples
    }
    
    # Save results
    output_path = Path(args.output)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to: {output_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("TAGGING PERFORMANCE SUMMARY")
    print("=" * 60)
    print(f"Model: {tagging_model}")
    print(f"Total Samples: {total_samples}")
    print(f"\nMatch Rates:")
    print(f"  Exact Match:   {exact_matches:4d} ({results['overall']['exact_match_rate']:.1%})")
    print(f"  Partial Match: {partial_matches:4d} ({results['overall']['partial_match_rate']:.1%})")
    print(f"  No Match:      {no_matches:4d} ({results['overall']['no_match_rate']:.1%})")
    print(f"\nMicro-Average: P={micro_metrics['precision']:.3f} R={micro_metrics['recall']:.3f} F1={micro_metrics['f1']:.3f}")
    print(f"Macro-Average: P={macro_metrics['precision']:.3f} R={macro_metrics['recall']:.3f} F1={macro_metrics['f1']:.3f}")
    print("\nPer-Class F1 Scores:")
    for qt in sorted(QUESTION_TYPES, key=lambda x: class_metrics[x]["f1"], reverse=True):
        m = class_metrics[qt]
        if m["support"] > 0:
            print(f"  {qt:16s}: F1={m['f1']:.3f} (P={m['precision']:.3f}, R={m['recall']:.3f}, support={m['support']})")


if __name__ == "__main__":
    main()

