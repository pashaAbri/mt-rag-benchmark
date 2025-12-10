
import json
import statistics
from pathlib import Path

# Load results
results_dir = Path("/Users/pastil/Dev/Github/mt-rag-benchmark/scripts/ideas/retrieval_tasks/mono_t5_oracle_selection/results")
full_dataset_path = results_dir / "base_model_full_dataset_results.json"

if not full_dataset_path.exists():
    print(f"Error: {full_dataset_path} not found. Please run run_base_model_full.py first.")
    exit(1)

with open(full_dataset_path, 'r') as f:
    full_data = json.load(f)

# Extract tasks
tasks = full_data['tasks']
task_ids = list(tasks.keys())
print(f"Total tasks processed: {len(task_ids)}")

# Calculate metrics for pure strategies
strategies = ['lastturn', 'rewrite', 'questions']
metrics_of_interest = [
    'recall_1', 'recall_3', 'recall_5', 'recall_10',
    'ndcg_cut_1', 'ndcg_cut_3', 'ndcg_cut_5', 'ndcg_cut_10'
]

pure_strategy_perf = {s: {m: [] for m in metrics_of_interest} for s in strategies}

for task_id in task_ids:
    task_data = tasks[task_id]
    actual_metrics = task_data['actual_metrics']
    
    for strategy in strategies:
        strat_metrics = actual_metrics.get(strategy, {})
        for metric in metrics_of_interest:
            val = strat_metrics.get(metric, 0.0)
            pure_strategy_perf[strategy][metric].append(val)

# Calculate Base MonoT5 Performance
base_perf = {m: [] for m in metrics_of_interest}
for task_id in task_ids:
    task = tasks[task_id]
    choice = task['selected_strategy']
    metrics = task['actual_metrics'].get(choice, {})
    
    for m in metrics_of_interest:
        base_perf[m].append(metrics.get(m, 0.0))

base_avg = {m: statistics.mean(vals) for m, vals in base_perf.items()}

# Calculate Oracle (Upper Bound)
oracle_perf = {m: [] for m in metrics_of_interest}
for task_id in task_ids:
    task_data = tasks[task_id]
    actual_metrics = task_data['actual_metrics']
    
    # Oracle strategy is defined as best Recall@10
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

# Print Markdown Table
print("# Performance Comparison: Full Dataset (Base MonoT5 vs Baselines)")
print()
print("| Method | R@1 | R@3 | R@5 | R@10 | nDCG@1 | nDCG@3 | nDCG@5 | nDCG@10 |")
print("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
print("| **Baselines** | | | | | | | | |")

for s in strategies:
    avgs = {m: statistics.mean(pure_strategy_perf[s][m]) for m in metrics_of_interest}
    print(f"| {s.capitalize()} | {avgs['recall_1']:.4f} | {avgs['recall_3']:.4f} | {avgs['recall_5']:.4f} | {avgs['recall_10']:.4f} | {avgs['ndcg_cut_1']:.4f} | {avgs['ndcg_cut_3']:.4f} | {avgs['ndcg_cut_5']:.4f} | {avgs['ndcg_cut_10']:.4f} |")

print("| **Selection Methods** | | | | | | | | |")

# Base Model
print(f"| Base MonoT5 | {base_avg['recall_1']:.4f} | {base_avg['recall_3']:.4f} | {base_avg['recall_5']:.4f} | {base_avg['recall_10']:.4f} | {base_avg['ndcg_cut_1']:.4f} | {base_avg['ndcg_cut_3']:.4f} | {base_avg['ndcg_cut_5']:.4f} | {base_avg['ndcg_cut_10']:.4f} |")

# Oracle
print(f"| **Oracle (Upper Bound)** | {oracle_avg['recall_1']:.4f} | {oracle_avg['recall_3']:.4f} | {oracle_avg['recall_5']:.4f} | {oracle_avg['recall_10']:.4f} | {oracle_avg['ndcg_cut_1']:.4f} | {oracle_avg['ndcg_cut_3']:.4f} | {oracle_avg['ndcg_cut_5']:.4f} | {oracle_avg['ndcg_cut_10']:.4f} |")
