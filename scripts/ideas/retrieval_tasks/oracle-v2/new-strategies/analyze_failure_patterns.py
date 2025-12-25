#!/usr/bin/env python3
"""
Analyze failure patterns in rewrites for zero-score cases.
Categorizes failures into:
1. Unresolved Pronouns (it, this, that, he, she, they)
2. Generic Terms (the series, the movie, the movement, etc.)
3. Shorter/Same Length (Failure to add context)
4. Question Type Mismatch (e.g. rewrite is a statement)
"""
import pandas as pd
import json
from pathlib import Path
import re

# Load zero-score cases
df = pd.read_csv('scripts/ideas/retrieval_tasks/oracle-v2/new-strategies/new-strategies/zero_score_cases.csv')

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

# Analysis Counters
patterns = {
    'unresolved_pronouns': 0,
    'generic_terms': 0,
    'shorter_or_same': 0,
    'total_analyzed': 0
}

# Regex patterns
pronoun_pattern = r'\b(it|this|that|he|she|they|him|her|them)\b'
generic_pattern = r'\b(the (series|show|movie|film|book|author|writer|movement|company|policy|act|bill))\b'

print(f"{'ID':<10} {'Domain':<10} {'Issue':<20} {'Rewrite'}")
print("-" * 100)

for idx, row in df.iterrows():
    task_id = row['task_id']
    rewrite = all_rewrites.get(task_id, "").lower()
    original = row['query_text'].lower()
    
    if not rewrite: continue
    
    patterns['total_analyzed'] += 1
    issues = []
    
    # Check 1: Length (proxy for context addition)
    if len(rewrite.split()) <= len(original.split()):
        patterns['shorter_or_same'] += 1
        issues.append("NoContextAdded")

    # Check 2: Unresolved Pronouns
    if re.search(pronoun_pattern, rewrite):
        patterns['unresolved_pronouns'] += 1
        issues.append("Pronoun")
        
    # Check 3: Generic Terms
    if re.search(generic_pattern, rewrite):
        patterns['generic_terms'] += 1
        issues.append("GenericRef")
    
    if issues:
        print(f"{row['turn_id']:<10} {row['domain']:<10} {','.join(issues):<20} {rewrite[:60]}...")

print("\n" + "=" * 50)
print("FAILURE PATTERN STATISTICS (98 Cases)")
print("=" * 50)
print(f"Total Analyzed: {patterns['total_analyzed']}")
print(f"1. Failed to add context (Length <= Original): {patterns['shorter_or_same']} ({patterns['shorter_or_same']/patterns['total_analyzed']*100:.1f}%)")
print(f"2. Contains Unresolved Pronouns (it, this...): {patterns['unresolved_pronouns']} ({patterns['unresolved_pronouns']/patterns['total_analyzed']*100:.1f}%)")
print(f"3. Contains Generic Terms (the series...): {patterns['generic_terms']} ({patterns['generic_terms']/patterns['total_analyzed']*100:.1f}%)")

# Overlap analysis
print("\nNote: Categories overlap. A short query might also have pronouns.")

