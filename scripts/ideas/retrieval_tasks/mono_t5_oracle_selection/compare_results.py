
import json
import statistics
from pathlib import Path

# Load results
results_dir = Path("/Users/pastil/Dev/Github/mt-rag-benchmark/scripts/ideas/retrieval_tasks/mono_t5_oracle_selection/results")
finetuned_path = results_dir / "finetuned_model_results.json"
base_model_path = results_dir / "base_model_results.json"

with open(finetuned_path, 'r') as f:
    finetuned_data = json.load(f)

with open(base_model_path, 'r') as f:
    base_data = json.load(f)

# Extract tasks
finetuned_tasks = finetuned_data['tasks']
base_tasks = base_data['tasks']

# Identify the 52 common tasks (should be all of them)
common_tasks = set(finetuned_tasks.keys()) & set(base_tasks.keys())
print(f"Common tasks: {len(common_tasks)}")

# Calculate metrics for pure strategies
strategies = ['lastturn', 'rewrite', 'questions']
metrics_of_interest = [
    'recall_1', 'recall_3', 'recall_5', 'recall_10',
    'ndcg_cut_1', 'ndcg_cut_3', 'ndcg_cut_5', 'ndcg_cut_10'
]

pure_strategy_perf = {s: {m: [] for m in metrics_of_interest} for s in strategies}

for task_id in common_tasks:
    task_data = finetuned_tasks[task_id] # Metric data is the same in both files
    actual_metrics = task_data['actual_metrics']
    
    for strategy in strategies:
        strat_metrics = actual_metrics.get(strategy, {})
        for metric in metrics_of_interest:
            val = strat_metrics.get(metric, 0.0)
            pure_strategy_perf[strategy][metric].append(val)

# Calculate averages
pass


# Calculate MonoT5 Performance (Base vs Fine-tuned)
def calculate_model_perf(tasks_data, task_ids):
    perf = {m: [] for m in metrics_of_interest}
    for task_id in task_ids:
        task = tasks_data[task_id]
        choice = task['selected_strategy']
        metrics = task['actual_metrics'].get(choice, {})
        
        for m in metrics_of_interest:
            perf[m].append(metrics.get(m, 0.0))
            
    return {m: statistics.mean(vals) for m, vals in perf.items()}

base_perf = calculate_model_perf(base_tasks, common_tasks)
finetuned_perf = calculate_model_perf(finetuned_tasks, common_tasks)

# Calculate Oracle (Upper Bound)
oracle_perf = {m: [] for m in metrics_of_interest}
for task_id in common_tasks:
    task_data = finetuned_tasks[task_id]
    # Oracle strategy is defined as best Recall@10 in the json
    # Let's verify that or re-calculate strictly based on R@10
    actual_metrics = task_data['actual_metrics']
    
    best_r10 = -1.0
    best_strat_metrics = {}
    
    for s in strategies:
        m = actual_metrics.get(s, {})
        r10 = m.get('recall_10', 0.0)
        if r10 > best_r10:
            best_r10 = r10
            best_strat_metrics = m
    
    # Oracle Score for this task is the max score achievable
    # Note: Oracle selection is usually based on R@10, so we optimize R@10.
    # The other metrics are taken from the strategy that maximized R@10.
    for m in metrics_of_interest:
        oracle_perf[m].append(best_strat_metrics.get(m, 0.0))

oracle_avg = {m: statistics.mean(vals) for m, vals in oracle_perf.items()}

# Print Markdown Table
print("# Performance Comparison: Retrieval Strategy Selection")
print()
print("| Method | R@1 | R@3 | R@5 | R@10 | nDCG@1 | nDCG@3 | nDCG@5 | nDCG@10 |")
print("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
print("| **Baselines** | | | | | | | | |")

for s in strategies:
    avgs = {m: statistics.mean(pure_strategy_perf[s][m]) for m in metrics_of_interest}
    print(f"| {s.capitalize()} | {avgs['recall_1']:.4f} | {avgs['recall_3']:.4f} | {avgs['recall_5']:.4f} | {avgs['recall_10']:.4f} | {avgs['ndcg_cut_1']:.4f} | {avgs['ndcg_cut_3']:.4f} | {avgs['ndcg_cut_5']:.4f} | {avgs['ndcg_cut_10']:.4f} |")

print("| **Selection Methods** | | | | | | | | |")

# Base Model
print(f"| Base MonoT5 | {base_perf['recall_1']:.4f} | {base_perf['recall_3']:.4f} | {base_perf['recall_5']:.4f} | {base_perf['recall_10']:.4f} | {base_perf['ndcg_cut_1']:.4f} | {base_perf['ndcg_cut_3']:.4f} | {base_perf['ndcg_cut_5']:.4f} | {base_perf['ndcg_cut_10']:.4f} |")

# Fine-Tuned Model
print(f"| Fine-Tuned MonoT5 | {finetuned_perf['recall_1']:.4f} | {finetuned_perf['recall_3']:.4f} | {finetuned_perf['recall_5']:.4f} | {finetuned_perf['recall_10']:.4f} | {finetuned_perf['ndcg_cut_1']:.4f} | {finetuned_perf['ndcg_cut_3']:.4f} | {finetuned_perf['ndcg_cut_5']:.4f} | {finetuned_perf['ndcg_cut_10']:.4f} |")

# Oracle
print(f"| **Oracle (Upper Bound)** | {oracle_avg['recall_1']:.4f} | {oracle_avg['recall_3']:.4f} | {oracle_avg['recall_5']:.4f} | {oracle_avg['recall_10']:.4f} | {oracle_avg['ndcg_cut_1']:.4f} | {oracle_avg['ndcg_cut_3']:.4f} | {oracle_avg['ndcg_cut_5']:.4f} | {oracle_avg['ndcg_cut_10']:.4f} |")

