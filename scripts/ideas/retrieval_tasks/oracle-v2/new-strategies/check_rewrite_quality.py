#!/usr/bin/env python3
"""
Compare original queries vs rewrites for zero-score cases.
"""
import pandas as pd
import json
from pathlib import Path

# Load zero-score cases
df = pd.read_csv('scripts/ideas/retrieval_tasks/oracle-v2/new-strategies/new-strategies/zero_score_cases.csv')

print(f"Loaded {len(df)} zero-score cases")

# Function to load rewrites
def load_rewrites(domain):
    path = Path(f"human/retrieval_tasks/{domain}/{domain}_rewrite.jsonl")
    rewrites = {}
    if path.exists():
        with open(path, 'r') as f:
            for line in f:
                data = json.loads(line)
                rewrites[data['_id']] = data['text'].replace('|user|: ', '').strip()
    return rewrites

# Load all rewrites
all_rewrites = {}
for domain in df['domain'].unique():
    all_rewrites.update(load_rewrites(domain))

# Compare
print("\nComparing Original vs Rewrite (First 10 Zero-Score Cases):")
print("-" * 100)
print(f"{'Turn':<5} {'Type':<15} {'Original':<50} {'Rewrite'}")
print("-" * 100)

count = 0
for idx, row in df.iterrows():
    task_id = row['task_id']
    original = row['query_text']
    rewrite = all_rewrites.get(task_id, "N/A")
    
    # Check if rewrite failed to expand length (simple heuristic for context addition)
    orig_len = len(original.split())
    rw_len = len(rewrite.split())
    
    if count < 10:
        print(f"{row['turn_id']:<5} {row['multi_turn_type']:<15} {original[:48]:<50} {rewrite}")
    count += 1

# Statistics
print("\n" + "-" * 100)
print("Length Analysis (Zero-Score Cases):")
length_diffs = []
shorter_rewrites = 0
same_length = 0

for idx, row in df.iterrows():
    task_id = row['task_id']
    rewrite = all_rewrites.get(task_id, "")
    if not rewrite: continue
    
    orig_len = len(row['query_text'].split())
    rw_len = len(rewrite.split())
    
    length_diffs.append(rw_len - orig_len)
    if rw_len < orig_len:
        shorter_rewrites += 1
    elif rw_len == orig_len:
        same_length += 1

avg_diff = sum(length_diffs) / len(length_diffs)
print(f"Average words added by rewrite: {avg_diff:.1f}")
print(f"Rewrites shorter than original: {shorter_rewrites} ({shorter_rewrites/len(df)*100:.1f}%)")
print(f"Rewrites same length as original: {same_length} ({same_length/len(df)*100:.1f}%)")

