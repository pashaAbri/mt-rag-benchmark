#!/usr/bin/env python3
"""
Calculate statistics for enrichment subtypes across all tasks in cleaned_data/tasks/
"""

import json
import os
from collections import defaultdict
from pathlib import Path

def calculate_enrichment_stats(tasks_dir):
    """
    Calculate statistics for each enrichment subtype.
    
    Args:
        tasks_dir: Path to the cleaned_data/tasks directory
        
    Returns:
        Dictionary with statistics for each enrichment type
    """
    # Initialize counters
    question_type_counts = defaultdict(int)
    multi_turn_counts = defaultdict(int)
    answerability_counts = defaultdict(int)
    domain_counts = defaultdict(int)
    total_tasks = 0
    
    # Iterate through all domain directories
    tasks_path = Path(tasks_dir)
    for domain_dir in tasks_path.iterdir():
        if not domain_dir.is_dir():
            continue
            
        domain = domain_dir.name
        json_files = list(domain_dir.glob("*.json"))
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    task_data = json.load(f)
                
                total_tasks += 1
                domain_counts[domain] += 1
                
                # Extract enrichments
                if 'user' in task_data and 'enrichments' in task_data['user']:
                    enrichments = task_data['user']['enrichments']
                    
                    # Count Question Types (can be multiple per task)
                    if 'Question Type' in enrichments:
                        for qtype in enrichments['Question Type']:
                            question_type_counts[qtype] += 1
                    
                    # Count Multi-Turn types
                    if 'Multi-Turn' in enrichments:
                        for mtype in enrichments['Multi-Turn']:
                            multi_turn_counts[mtype] += 1
                    
                    # Count Answerability types
                    if 'Answerability' in enrichments:
                        for atype in enrichments['Answerability']:
                            answerability_counts[atype] += 1
                            
            except Exception as e:
                print(f"Error processing {json_file}: {e}")
                continue
    
    return {
        'question_types': dict(question_type_counts),
        'multi_turn': dict(multi_turn_counts),
        'answerability': dict(answerability_counts),
        'domains': dict(domain_counts),
        'total_tasks': total_tasks
    }

def format_stats_table(stats):
    """
    Format statistics as markdown tables.
    
    Args:
        stats: Dictionary with statistics
        
    Returns:
        String with markdown tables
    """
    output = []
    
    # Question Type Statistics
    output.append("### Question Type Statistics\n")
    output.append("| Question Type | Count | Percentage |")
    output.append("|---------------|-------|------------|")
    
    question_types = sorted(stats['question_types'].items(), key=lambda x: x[1], reverse=True)
    total_qtypes = sum(stats['question_types'].values())
    
    for qtype, count in question_types:
        percentage = (count / total_qtypes * 100) if total_qtypes > 0 else 0
        output.append(f"| {qtype} | {count} | {percentage:.1f}% |")
    
    output.append(f"| **Total** | **{total_qtypes}** | **100.0%** |")
    output.append("")
    
    # Multi-Turn Statistics
    output.append("### Multi-Turn Type Statistics\n")
    output.append("| Multi-Turn Type | Count | Percentage |")
    output.append("|-----------------|-------|------------|")
    
    multi_turn = sorted(stats['multi_turn'].items(), key=lambda x: x[1], reverse=True)
    total_multi = sum(stats['multi_turn'].values())
    
    for mtype, count in multi_turn:
        percentage = (count / total_multi * 100) if total_multi > 0 else 0
        output.append(f"| {mtype} | {count} | {percentage:.1f}% |")
    
    output.append(f"| **Total** | **{total_multi}** | **100.0%** |")
    output.append("")
    
    # Answerability Statistics
    output.append("### Answerability Statistics\n")
    output.append("| Answerability Type | Count | Percentage |")
    output.append("|-------------------|-------|------------|")
    
    answerability = sorted(stats['answerability'].items(), key=lambda x: x[1], reverse=True)
    total_ans = sum(stats['answerability'].values())
    
    for atype, count in answerability:
        percentage = (count / total_ans * 100) if total_ans > 0 else 0
        output.append(f"| {atype} | {count} | {percentage:.1f}% |")
    
    output.append(f"| **Total** | **{total_ans}** | **100.0%** |")
    output.append("")
    
    # Domain Statistics
    output.append("### Domain Statistics\n")
    output.append("| Domain | Count | Percentage |")
    output.append("|--------|-------|------------|")
    
    domains = sorted(stats['domains'].items(), key=lambda x: x[1], reverse=True)
    total_domains = sum(stats['domains'].values())
    
    for domain, count in domains:
        percentage = (count / total_domains * 100) if total_domains > 0 else 0
        output.append(f"| {domain.upper()} | {count} | {percentage:.1f}% |")
    
    output.append(f"| **Total** | **{total_domains}** | **100.0%** |")
    output.append("")
    
    # Summary
    output.append("### Summary\n")
    output.append(f"- **Total Tasks**: {stats['total_tasks']}")
    output.append(f"- **Total Question Type Labels**: {total_qtypes}")
    output.append(f"- **Total Multi-Turn Labels**: {total_multi}")
    output.append(f"- **Total Answerability Labels**: {total_ans}")
    output.append("")
    output.append("*Note: Some tasks may have multiple Question Type labels, which is why the total Question Type labels may exceed the total number of tasks.*")
    
    return "\n".join(output)

def main():
    # Get the tasks directory
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    tasks_dir = repo_root / "cleaned_data" / "tasks"
    
    if not tasks_dir.exists():
        print(f"Error: Tasks directory not found at {tasks_dir}")
        return
    
    print(f"Calculating enrichment statistics from {tasks_dir}...")
    stats = calculate_enrichment_stats(tasks_dir)
    
    # Print summary
    print(f"\nTotal tasks processed: {stats['total_tasks']}")
    print(f"\nQuestion Types: {len(stats['question_types'])} unique types")
    print(f"Multi-Turn Types: {len(stats['multi_turn'])} unique types")
    print(f"Answerability Types: {len(stats['answerability'])} unique types")
    print(f"Domains: {len(stats['domains'])} domains")
    
    # Generate markdown tables
    markdown_output = format_stats_table(stats)
    
    # Save to file
    output_file = script_dir / "enrichment_stats.md"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# Enrichment Statistics\n\n")
        f.write(markdown_output)
    
    print(f"\nStatistics saved to {output_file}")
    print("\nMarkdown tables:")
    print("=" * 80)
    print(markdown_output)
    
    return stats

if __name__ == "__main__":
    main()

