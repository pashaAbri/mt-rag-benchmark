#!/usr/bin/env python3
"""
Compare performance of multiple fine-tuned MonoT5 checkpoints.
"""

import json
import statistics
from pathlib import Path
from typing import Dict, List

# Load results
results_dir = Path(__file__).parent / "results" / "checkpoints_v2"
checkpoints = ["final_model", "checkpoint-1000", "checkpoint-500"]

# Also load base model for comparison
base_results_path = Path(__file__).parent / "results" / "base_model_results.json"

# Load checkpoint results
checkpoint_results = {}
for checkpoint in checkpoints:
    checkpoint_file = results_dir / f"{checkpoint}_results.json"
    if checkpoint_file.exists():
        with open(checkpoint_file, 'r') as f:
            checkpoint_results[checkpoint] = json.load(f)
    else:
        print(f"Warning: {checkpoint_file} not found")

# Load base model results if available
base_results = None
if base_results_path.exists():
    with open(base_results_path, 'r') as f:
        base_results = json.load(f)

# Extract common tasks
if checkpoint_results:
    common_tasks = set(checkpoint_results[list(checkpoint_results.keys())[0]]['tasks'].keys())
    for checkpoint_name, results in checkpoint_results.items():
        common_tasks &= set(results['tasks'].keys())
    
    if base_results:
        common_tasks &= set(base_results['tasks'].keys())
    
    print(f"Common tasks: {len(common_tasks)}")
    
    # Metrics of interest
    metrics_of_interest = [
        'recall_1', 'recall_3', 'recall_5', 'recall_10',
        'ndcg_cut_1', 'ndcg_cut_3', 'ndcg_cut_5', 'ndcg_cut_10'
    ]
    
    # Calculate performance for each checkpoint
    def calculate_model_perf(tasks_data, task_ids):
        perf = {m: [] for m in metrics_of_interest}
        for task_id in task_ids:
            task = tasks_data[task_id]
            choice = task['selected_strategy']
            metrics = task['actual_metrics'].get(choice, {})
            
            for m in metrics_of_interest:
                perf[m].append(metrics.get(m, 0.0))
                
        return {m: statistics.mean(vals) for m, vals in perf.items()}
    
    # Calculate Oracle (Upper Bound)
    oracle_perf = {m: [] for m in metrics_of_interest}
    strategies = ['lastturn', 'rewrite', 'questions']
    
    # Use first checkpoint's task data for oracle calculation
    first_checkpoint = list(checkpoint_results.keys())[0]
    for task_id in common_tasks:
        task_data = checkpoint_results[first_checkpoint]['tasks'][task_id]
        actual_metrics = task_data['actual_metrics']
        
        best_r10 = -1.0
        best_strat_metrics = {}
        
        for s in strategies:
            m = actual_metrics.get(s, {})
            r10 = m.get('recall_10', 0.0)
            if r10 > best_r10:
                best_r10 = r10
                best_strat_metrics = m
        
        for m in metrics_of_interest:
            oracle_perf[m].append(best_strat_metrics.get(m, 0.0))
    
    oracle_avg = {m: statistics.mean(vals) for m, vals in oracle_perf.items()}
    
    # Calculate pure strategy performance
    pure_strategy_perf = {s: {m: [] for m in metrics_of_interest} for s in strategies}
    
    for task_id in common_tasks:
        task_data = checkpoint_results[first_checkpoint]['tasks'][task_id]
        actual_metrics = task_data['actual_metrics']
        
        for strategy in strategies:
            strat_metrics = actual_metrics.get(strategy, {})
            for metric in metrics_of_interest:
                val = strat_metrics.get(metric, 0.0)
                pure_strategy_perf[strategy][metric].append(val)
    
    # Print Markdown Table
    print("\n# Performance Comparison: Fine-Tuned MonoT5 Checkpoints")
    print()
    print("| Method | R@1 | R@3 | R@5 | R@10 | nDCG@1 | nDCG@3 | nDCG@5 | nDCG@10 |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    print("| **Baselines** | | | | | | | | |")
    
    for s in strategies:
        avgs = {m: statistics.mean(pure_strategy_perf[s][m]) for m in metrics_of_interest}
        print(f"| {s.capitalize()} | {avgs['recall_1']:.4f} | {avgs['recall_3']:.4f} | {avgs['recall_5']:.4f} | {avgs['recall_10']:.4f} | {avgs['ndcg_cut_1']:.4f} | {avgs['ndcg_cut_3']:.4f} | {avgs['ndcg_cut_5']:.4f} | {avgs['ndcg_cut_10']:.4f} |")
    
    print("| **Selection Methods** | | | | | | | | |")
    
    # Base Model (if available)
    if base_results:
        base_perf = calculate_model_perf(base_results['tasks'], common_tasks)
        print(f"| Base MonoT5 | {base_perf['recall_1']:.4f} | {base_perf['recall_3']:.4f} | {base_perf['recall_5']:.4f} | {base_perf['recall_10']:.4f} | {base_perf['ndcg_cut_1']:.4f} | {base_perf['ndcg_cut_3']:.4f} | {base_perf['ndcg_cut_5']:.4f} | {base_perf['ndcg_cut_10']:.4f} |")
    
    # Checkpoint models
    for checkpoint_name in checkpoints:
        if checkpoint_name in checkpoint_results:
            checkpoint_perf = calculate_model_perf(checkpoint_results[checkpoint_name]['tasks'], common_tasks)
            summary = checkpoint_results[checkpoint_name]['summary']
            accuracy = summary.get('accuracy', 0.0)
            print(f"| **{checkpoint_name}** (acc: {accuracy:.3f}) | {checkpoint_perf['recall_1']:.4f} | {checkpoint_perf['recall_3']:.4f} | {checkpoint_perf['recall_5']:.4f} | {checkpoint_perf['recall_10']:.4f} | {checkpoint_perf['ndcg_cut_1']:.4f} | {checkpoint_perf['ndcg_cut_3']:.4f} | {checkpoint_perf['ndcg_cut_5']:.4f} | {checkpoint_perf['ndcg_cut_10']:.4f} |")
    
    # Oracle
    print(f"| **Oracle (Upper Bound)** | {oracle_avg['recall_1']:.4f} | {oracle_avg['recall_3']:.4f} | {oracle_avg['recall_5']:.4f} | {oracle_avg['recall_10']:.4f} | {oracle_avg['ndcg_cut_1']:.4f} | {oracle_avg['ndcg_cut_3']:.4f} | {oracle_avg['ndcg_cut_5']:.4f} | {oracle_avg['ndcg_cut_10']:.4f} |")
    
    # Save comparison table
    output_file = results_dir / "comparison_table.md"
    with open(output_file, 'w') as f:
        f.write("# Performance Comparison: Fine-Tuned MonoT5 Checkpoints\n\n")
        f.write("| Method | R@1 | R@3 | R@5 | R@10 | nDCG@1 | nDCG@3 | nDCG@5 | nDCG@10 |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        f.write("| **Baselines** | | | | | | | | |\n")
        
        for s in strategies:
            avgs = {m: statistics.mean(pure_strategy_perf[s][m]) for m in metrics_of_interest}
            f.write(f"| {s.capitalize()} | {avgs['recall_1']:.4f} | {avgs['recall_3']:.4f} | {avgs['recall_5']:.4f} | {avgs['recall_10']:.4f} | {avgs['ndcg_cut_1']:.4f} | {avgs['ndcg_cut_3']:.4f} | {avgs['ndcg_cut_5']:.4f} | {avgs['ndcg_cut_10']:.4f} |\n")
        
        f.write("| **Selection Methods** | | | | | | | | |\n")
        
        if base_results:
            base_perf = calculate_model_perf(base_results['tasks'], common_tasks)
            f.write(f"| Base MonoT5 | {base_perf['recall_1']:.4f} | {base_perf['recall_3']:.4f} | {base_perf['recall_5']:.4f} | {base_perf['recall_10']:.4f} | {base_perf['ndcg_cut_1']:.4f} | {base_perf['ndcg_cut_3']:.4f} | {base_perf['ndcg_cut_5']:.4f} | {base_perf['ndcg_cut_10']:.4f} |\n")
        
        for checkpoint_name in checkpoints:
            if checkpoint_name in checkpoint_results:
                checkpoint_perf = calculate_model_perf(checkpoint_results[checkpoint_name]['tasks'], common_tasks)
                summary = checkpoint_results[checkpoint_name]['summary']
                accuracy = summary.get('accuracy', 0.0)
                f.write(f"| **{checkpoint_name}** (acc: {accuracy:.3f}) | {checkpoint_perf['recall_1']:.4f} | {checkpoint_perf['recall_3']:.4f} | {checkpoint_perf['recall_5']:.4f} | {checkpoint_perf['recall_10']:.4f} | {checkpoint_perf['ndcg_cut_1']:.4f} | {checkpoint_perf['ndcg_cut_3']:.4f} | {checkpoint_perf['ndcg_cut_5']:.4f} | {checkpoint_perf['ndcg_cut_10']:.4f} |\n")
        
        f.write(f"| **Oracle (Upper Bound)** | {oracle_avg['recall_1']:.4f} | {oracle_avg['recall_3']:.4f} | {oracle_avg['recall_5']:.4f} | {oracle_avg['recall_10']:.4f} | {oracle_avg['ndcg_cut_1']:.4f} | {oracle_avg['ndcg_cut_3']:.4f} | {oracle_avg['ndcg_cut_5']:.4f} | {oracle_avg['ndcg_cut_10']:.4f} |\n")
    
    print(f"\nComparison table saved to {output_file}")
    
else:
    print("No checkpoint results found!")
