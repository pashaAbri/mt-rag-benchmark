"""Quick verification of oracle analysis numbers"""
import json
from pathlib import Path

# Ground truth from CSV files
baseline_truth = {
    'elser': {
        'clapnq': {'recall_5': 0.55161, 'ndcg_5': 0.51349},
        'cloud': {'recall_5': 0.42966, 'ndcg_5': 0.39396},
        'fiqa': {'recall_5': 0.40164, 'ndcg_5': 0.37791},
        'govt': {'recall_5': 0.50824, 'ndcg_5': 0.45402},
    },
    'bge': {
        'clapnq': {'recall_5': 0.46189, 'ndcg_5': 0.43761},
        'cloud': {'recall_5': 0.33830, 'ndcg_5': 0.30332},
        'fiqa': {'recall_5': 0.30775, 'ndcg_5': 0.29381},
        'govt': {'recall_5': 0.40388, 'ndcg_5': 0.36767},
    },
    'bm25': {
        'clapnq': {'recall_5': 0.27960, 'ndcg_5': 0.25253},
        'cloud': {'recall_5': 0.23404, 'ndcg_5': 0.21140},
        'fiqa': {'recall_5': 0.18294, 'ndcg_5': 0.15558},
        'govt': {'recall_5': 0.34346, 'ndcg_5': 0.30963},
    }
}

print("="*80)
print("VERIFICATION: Checking Oracle Analysis Numbers")
print("="*80)
print()

# Run oracle analysis and compare
import subprocess
import sys

for retriever in ['elser', 'bge', 'bm25']:
    print(f"\n{retriever.upper()}:")
    print("-" * 40)
    
    for domain in ['clapnq', 'cloud', 'fiqa', 'govt']:
        result = subprocess.run(
            [sys.executable, 'scripts/ideas/retrieval_tasks/multi_strategy_fusion/analyze_per_query_winners.py',
             '--retriever', retriever, '--domain', domain],
            capture_output=True,
            text=True
        )
        
        # Parse output
        lines = result.stdout.split('\n')
        rewrite_recall = None
        oracle_recall = None
        
        for i, line in enumerate(lines):
            if 'Best single strategy:' in line and i+1 < len(lines):
                if 'Recall@5:' in lines[i+1]:
                    rewrite_recall = float(lines[i+1].split(':')[1].strip())
            if 'Oracle Recall@5:' in line:
                oracle_recall = float(line.split(':')[1].strip())
        
        # Compare
        expected = baseline_truth[retriever][domain]['recall_5']
        
        if rewrite_recall is not None:
            diff = abs(rewrite_recall - expected)
            status = "✅" if diff < 0.001 else "❌"
            print(f"  {domain:8s} - Baseline: {expected:.4f} vs Analyzed: {rewrite_recall:.4f} {status}")
            if oracle_recall:
                improvement = ((oracle_recall - rewrite_recall) / rewrite_recall) * 100
                print(f"           - Oracle: {oracle_recall:.4f} (+{improvement:.1f}%)")

print("\n" + "="*80)
print("Verification complete!")
print("="*80)

