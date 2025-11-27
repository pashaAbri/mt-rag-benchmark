#!/usr/bin/env python3
"""
Generate a markdown summary document with tables from enrichment performance CSV files.
"""

import csv
from pathlib import Path
from collections import defaultdict

# Key metrics to display in tables
PRIMARY_METRICS = ['R@5', 'nDCG@5']
SECONDARY_METRICS = ['R@1', 'R@3', 'R@10', 'nDCG@1', 'nDCG@3', 'nDCG@10']
STRATEGIES = ['lastturn', 'rewrite', 'questions']
RETRIEVAL_METHODS = ['elser', 'bm25', 'bge']


def load_csv_data(csv_file: Path) -> dict:
    """Load CSV data into a nested dictionary structure."""
    data = {}
    
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            enrichment_type = row['enrichment_type']
            subtype = row['subtype']
            strategy = row['strategy']
            metric = row['metric']
            
            retrieval_method = row.get('retrieval_method', 'elser')  # Default to elser for backward compatibility
            
            if enrichment_type not in data:
                data[enrichment_type] = {}
            if subtype not in data[enrichment_type]:
                data[enrichment_type][subtype] = {}
            if retrieval_method not in data[enrichment_type][subtype]:
                data[enrichment_type][subtype][retrieval_method] = {}
            if strategy not in data[enrichment_type][subtype][retrieval_method]:
                data[enrichment_type][subtype][retrieval_method][strategy] = {}
            
            data[enrichment_type][subtype][retrieval_method][strategy][metric] = {
                'mean': float(row['mean']),
                'median': float(row['median']),
                'std': float(row['std']),
                'min': float(row['min']),
                'max': float(row['max']),
                'count': int(row['count'])
            }
    
    return data


def format_metric_value(mean: float) -> str:
    """Format metric value to 3 decimal places."""
    return f"{mean:.3f}"


def format_subtype_name(subtype: str) -> str:
    """Format subtype name for display."""
    if subtype == 'N/A':
        return 'N/A'
    elif subtype == 'ANSWERABLE':
        return 'Answerable'
    elif subtype == 'PARTIAL':
        return 'Partial'
    elif subtype == 'UNANSWERABLE':
        return 'Unanswerable'
    elif subtype == 'CONVERSATIONAL':
        return 'Conversational'
    elif subtype == 'Follow-up':
        return 'Follow-up'
    elif subtype == 'Clarification':
        return 'Clarification'
    else:
        return subtype.replace('_', ' ').title()


def create_comprehensive_table(data: dict, enrichment_type: str) -> str:
    """Create a single comprehensive table for an enrichment type with all metrics and retrieval methods."""
    lines = []
    
    # Get all subtypes for this enrichment type
    subtypes = sorted(data[enrichment_type].keys())
    
    if not subtypes:
        return ""
    
    # Table header - similar to master_retrieval_results format
    lines.append(f"## {enrichment_type.replace('_', ' ').title()} Performance")
    lines.append("")
    lines.append("| Subtype | Count | Retrieval Method | Strategy | R@1 | R@3 | R@5 | R@10 | nDCG@1 | nDCG@3 | nDCG@5 | nDCG@10 |")
    lines.append("| :------ | :---- | :--------------- | :------- | :-- | :-- | :-- | :--- | :----- | :----- | :----- | :------ |")
    
    # Table rows - one row per subtype-retrieval_method-strategy combination
    for subtype in subtypes:
        count = 0
        
        # Get count from any method/strategy (they should be the same)
        for retrieval_method in RETRIEVAL_METHODS:
            if retrieval_method in data[enrichment_type][subtype]:
                for strategy in STRATEGIES:
                    if strategy in data[enrichment_type][subtype][retrieval_method]:
                        if 'R@5' in data[enrichment_type][subtype][retrieval_method][strategy]:
                            count = data[enrichment_type][subtype][retrieval_method][strategy]['R@5']['count']
                            break
                if count > 0:
                    break
            if count > 0:
                break
        
        display_subtype = format_subtype_name(subtype)
        
        # Create rows for each retrieval method and strategy combination
        first_row = True
        for retrieval_method in RETRIEVAL_METHODS:
            if retrieval_method not in data[enrichment_type][subtype]:
                # Skip if no data for this retrieval method
                continue
            
            method_display = retrieval_method.upper()
            
            for strategy in STRATEGIES:
                if strategy not in data[enrichment_type][subtype][retrieval_method]:
                    continue
                
                strategy_display = strategy.replace('_', ' ').title()
                
                # Get all metric values for this method-strategy combination
                metrics_row = []
                all_metrics = ['R@1', 'R@3', 'R@5', 'R@10', 'nDCG@1', 'nDCG@3', 'nDCG@5', 'nDCG@10']
                
                for metric in all_metrics:
                    if metric in data[enrichment_type][subtype][retrieval_method][strategy]:
                        metrics_row.append(format_metric_value(
                            data[enrichment_type][subtype][retrieval_method][strategy][metric]['mean']
                        ))
                    else:
                        metrics_row.append("N/A")
                
                # First row shows subtype and count, subsequent rows are empty for subtype
                if first_row:
                    lines.append(f"| {display_subtype} | {count} | {method_display} | {strategy_display} | {' | '.join(metrics_row)} |")
                    first_row = False
                else:
                    lines.append(f"| | | {method_display} | {strategy_display} | {' | '.join(metrics_row)} |")
    
    lines.append("")
    return "\n".join(lines)


def create_detailed_table(data: dict, enrichment_type: str, subtype: str) -> str:
    """Create a detailed table showing all metrics for a specific subtype."""
    lines = []
    
    lines.append(f"#### {subtype.replace('_', ' ').title()} - Detailed Metrics")
    lines.append("")
    lines.append(f"| Metric | Last Turn | Rewrite | Questions |")
    lines.append(f"|--------|-----------|---------|-----------|")
    
    all_metrics = PRIMARY_METRICS + SECONDARY_METRICS
    
    for metric in all_metrics:
        values = {}
        for strategy in STRATEGIES:
            if metric in data[enrichment_type][subtype][strategy]:
                values[strategy] = format_metric_value(
                    data[enrichment_type][subtype][strategy][metric]['mean']
                )
            else:
                values[strategy] = "N/A"
        
        lines.append(f"| {metric} | {values['lastturn']} | {values['rewrite']} | {values['questions']} |")
    
    lines.append("")
    return "\n".join(lines)


def generate_markdown(data: dict) -> str:
    """Generate the complete markdown document."""
    lines = []
    
    lines.append("# Enrichment Performance Analysis Results")
    lines.append("")
    lines.append("This document summarizes retrieval performance across different enrichment subtypes")
    lines.append("and query strategies (Last Turn, Rewrite, Questions) using ELSER retrieval.")
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    lines.append("The analysis compares three query strategies:")
    lines.append("- **Last Turn**: Uses only the current question (no context)")
    lines.append("- **Rewrite**: Uses LLM-rewritten question with full context")
    lines.append("- **Questions**: Uses original question format")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # Process each enrichment type
    enrichment_types = ['question_types', 'multi_turn', 'answerability']
    enrichment_titles = {
        'question_types': 'Question Type',
        'multi_turn': 'Multi-Turn Type',
        'answerability': 'Answerability'
    }
    
    for enrichment_type in enrichment_types:
        if enrichment_type not in data:
            continue
        
        # Create comprehensive table with all metrics
        table = create_comprehensive_table(data, enrichment_type)
        if table:
            lines.append(table)
        
        # Add key insights
        lines.append("### Key Insights")
        lines.append("")
        
        # Find best/worst performing subtypes for rewrite strategy (using elser)
        subtypes = sorted(data[enrichment_type].keys())
        rewrite_r5 = {}
        
        for subtype in subtypes:
            if 'elser' in data[enrichment_type][subtype]:
                if 'rewrite' in data[enrichment_type][subtype]['elser']:
                    if 'R@5' in data[enrichment_type][subtype]['elser']['rewrite']:
                        rewrite_r5[subtype] = data[enrichment_type][subtype]['elser']['rewrite']['R@5']['mean']
        
        if rewrite_r5:
            best_subtype = max(rewrite_r5.items(), key=lambda x: x[1])
            worst_subtype = min(rewrite_r5.items(), key=lambda x: x[1])
            
            best_name = format_subtype_name(best_subtype[0])
            worst_name = format_subtype_name(worst_subtype[0])
            
            lines.append(f"- **Best performing**: {best_name} (R@5 = {best_subtype[1]:.3f}, ELSER-Rewrite)")
            lines.append(f"- **Worst performing**: {worst_name} (R@5 = {worst_subtype[1]:.3f}, ELSER-Rewrite)")
            
            # Calculate improvement from lastturn to rewrite (using elser)
            if 'elser' in data[enrichment_type][best_subtype[0]]:
                if 'lastturn' in data[enrichment_type][best_subtype[0]]['elser']:
                    if 'R@5' in data[enrichment_type][best_subtype[0]]['elser']['lastturn']:
                        lastturn_val = data[enrichment_type][best_subtype[0]]['elser']['lastturn']['R@5']['mean']
                        improvement = ((best_subtype[1] - lastturn_val) / lastturn_val) * 100 if lastturn_val > 0 else 0
                        lines.append(f"- **Rewrite improvement over Last Turn (ELSER)**: {improvement:+.1f}%")
        
        lines.append("")
        lines.append("---")
        lines.append("")
    
    # Add summary section
    lines.append("## Summary")
    lines.append("")
    lines.append("### Strategy Comparison")
    lines.append("")
    lines.append("| Strategy | Average R@5 | Average nDCG@5 |")
    lines.append("|----------|-------------|----------------|")
    
    # Calculate averages by retrieval method and strategy
    method_strategy_stats = defaultdict(lambda: defaultdict(lambda: {'r5': [], 'ndcg5': []}))
    
    for enrichment_type in enrichment_types:
        if enrichment_type not in data:
            continue
        for subtype in data[enrichment_type]:
            for retrieval_method in RETRIEVAL_METHODS:
                if retrieval_method not in data[enrichment_type][subtype]:
                    continue
                for strategy in STRATEGIES:
                    if strategy not in data[enrichment_type][subtype][retrieval_method]:
                        continue
                    if 'R@5' in data[enrichment_type][subtype][retrieval_method][strategy]:
                        method_strategy_stats[retrieval_method][strategy]['r5'].append(
                            data[enrichment_type][subtype][retrieval_method][strategy]['R@5']['mean']
                        )
                    if 'nDCG@5' in data[enrichment_type][subtype][retrieval_method][strategy]:
                        method_strategy_stats[retrieval_method][strategy]['ndcg5'].append(
                            data[enrichment_type][subtype][retrieval_method][strategy]['nDCG@5']['mean']
                        )
    
    # Update table header to include retrieval method
    lines.append("| Retrieval Method | Strategy | Average R@5 | Average nDCG@5 |")
    lines.append("| :--------------- | :------- | :---------- | :------------- |")
    
    for retrieval_method in RETRIEVAL_METHODS:
        if retrieval_method not in method_strategy_stats:
            continue
        method_display = retrieval_method.upper()
        for strategy in STRATEGIES:
            if strategy not in method_strategy_stats[retrieval_method]:
                continue
            stats = method_strategy_stats[retrieval_method][strategy]
            avg_r5 = sum(stats['r5']) / len(stats['r5']) if stats['r5'] else 0
            avg_ndcg5 = sum(stats['ndcg5']) / len(stats['ndcg5']) if stats['ndcg5'] else 0
            
            strategy_name = strategy.replace('_', ' ').title()
            lines.append(f"| {method_display} | {strategy_name} | {avg_r5:.3f} | {avg_ndcg5:.3f} |")
    
    lines.append("")
    lines.append("### Findings")
    lines.append("")
    lines.append("1. **Query Rewrite** generally outperforms **Last Turn** across most enrichment subtypes")
    lines.append("2. **Questions** strategy (original format) performs significantly worse than both rewrite and lastturn")
    lines.append("3. First turn questions (N/A) show identical performance across all strategies (no context to rewrite)")
    lines.append("4. Answerable questions consistently outperform partial answerable questions")
    lines.append("5. Follow-up questions benefit more from query rewriting than clarification questions")
    lines.append("")
    
    return "\n".join(lines)


def main():
    script_dir = Path(__file__).parent
    results_dir = script_dir / "enrichment_analysis_results"
    output_file = script_dir / "enrichment_performance_summary.md"
    
    # Load all CSV files and merge into single data structure
    all_data = {}
    
    csv_files = {
        'question_types': results_dir / 'enrichment_performance_question_types.csv',
        'multi_turn': results_dir / 'enrichment_performance_multi_turn.csv',
        'answerability': results_dir / 'enrichment_performance_answerability.csv'
    }
    
    for key, csv_file in csv_files.items():
        if csv_file.exists():
            print(f"Loading {csv_file.name}...")
            file_data = load_csv_data(csv_file)
            # Merge into all_data (file_data already has enrichment_type as top key)
            for enrichment_type, subtypes in file_data.items():
                if enrichment_type not in all_data:
                    all_data[enrichment_type] = {}
                all_data[enrichment_type].update(subtypes)
        else:
            print(f"Warning: {csv_file.name} not found")
    
    # Generate markdown
    print(f"\nGenerating markdown summary...")
    markdown_content = generate_markdown(all_data)
    
    # Write to file
    with open(output_file, 'w') as f:
        f.write(markdown_content)
    
    print(f"Summary saved to: {output_file}")


if __name__ == '__main__':
    main()

