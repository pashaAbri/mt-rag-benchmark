import pandas as pd
import json
import os

def load_intermediate_data(domain):
    """Load intermediate stats (clusters, timing) from jsonl."""
    data = []
    path = f"../intermediate/{domain}_intermediate_k10_lam0.7_all.jsonl"
    if not os.path.exists(path):
        return pd.DataFrame()
        
    with open(path, 'r') as f:
        for line in f:
            try:
                entry = json.loads(line)
                row = {
                    'task_id': entry.get('task_id'),
                    'turn': entry.get('turn'),
                    'original_query': entry.get('original_query'),
                    'rewritten_query': entry.get('rewritten_query'),
                    'method': entry.get('method', ''),
                    'num_extracted_sentences': entry.get('num_extracted_sentences', 0),
                    'num_selected': entry.get('num_selected', 0),
                    'selected_sentences': entry.get('selected_sentences', []),
                }
                data.append(row)
            except Exception as e:
                continue
                
    return pd.DataFrame(data)

def load_evaluation_data(domain):
    """Load retrieval metrics (Recall, NDCG) from evaluated jsonl."""
    data = []
    path = f"../results_k10/bge_{domain}_mmr_cluster_k10_evaluated.jsonl"
    if not os.path.exists(path):
        return pd.DataFrame()
        
    with open(path, 'r') as f:
        for line in f:
            try:
                entry = json.loads(line)
                scores = entry.get('retriever_scores', {})
                row = {
                    'task_id': entry.get('task_id'),
                    'Recall@10': scores.get('recall_10', 0),
                    'NDCG@10': scores.get('ndcg_cut_10', 0),
                }
                data.append(row)
            except Exception as e:
                continue
                
    return pd.DataFrame(data)

def extract_base_task_id(task_id):
    """Extract base task_id from format 'base_id<::>turn'"""
    if '<::>' in str(task_id):
        return str(task_id).split('<::>')[0]
    return str(task_id)

def find_decline_examples(domain, min_decline=0.1):
    """Find examples where turn 1 has higher recall than turn 2."""
    df_inter = load_intermediate_data(domain)
    df_eval = load_evaluation_data(domain)
    
    if df_inter.empty or df_eval.empty:
        return []
    
    # Merge
    df = pd.merge(df_inter, df_eval, on='task_id', how='inner')
    
    # Extract base task_id for matching
    df['base_task_id'] = df['task_id'].apply(extract_base_task_id)
    
    # Get turn 1 and turn 2 data
    turn1 = df[df['turn'] == 1].set_index('base_task_id')
    turn2 = df[df['turn'] == 2].set_index('base_task_id')
    
    # Find tasks that exist in both turns
    common_tasks = set(turn1.index) & set(turn2.index)
    
    examples = []
    for base_task_id in common_tasks:
        t1_row = turn1.loc[base_task_id]
        t2_row = turn2.loc[base_task_id]
        
        # Handle case where there might be multiple rows (shouldn't happen, but be safe)
        if isinstance(t1_row, pd.DataFrame):
            t1_row = t1_row.iloc[0]
        if isinstance(t2_row, pd.DataFrame):
            t2_row = t2_row.iloc[0]
        
        t1_recall = t1_row['Recall@10']
        t2_recall = t2_row['Recall@10']
        
        # Check if there's a decline
        decline = t1_recall - t2_recall
        if decline >= min_decline:
            selected_sents = t2_row['selected_sentences']
            if not isinstance(selected_sents, list):
                try:
                    if selected_sents is None or (isinstance(selected_sents, float) and pd.isna(selected_sents)):
                        selected_sents = []
                    else:
                        selected_sents = list(selected_sents) if hasattr(selected_sents, '__iter__') else []
                except:
                    selected_sents = []
            
            examples.append({
                'task_id': base_task_id,
                'full_task_id_turn1': str(t1_row['task_id']),
                'full_task_id_turn2': str(t2_row['task_id']),
                'domain': domain,
                'turn1_recall': float(t1_recall),
                'turn2_recall': float(t2_recall),
                'decline': float(decline),
                'turn1_query': str(t1_row['original_query']),
                'turn2_query': str(t2_row['original_query']),
                'turn2_rewritten': str(t2_row['rewritten_query']),
                'turn2_extracted': int(t2_row['num_extracted_sentences']),
                'turn2_selected': int(t2_row['num_selected']),
                'turn2_selected_sentences': selected_sents,
            })
    
    # Sort by decline amount
    examples.sort(key=lambda x: x['decline'], reverse=True)
    return examples

def main():
    domains = ['clapnq', 'cloud', 'fiqa', 'govt']
    all_examples = []
    
    print("Finding examples of sharp decline between Turn 1 and Turn 2...")
    print("="*80)
    
    for domain in domains:
        # Look for any decline of at least 0.10
        examples = find_decline_examples(domain, min_decline=0.10)
        all_examples.extend(examples)
        print(f"\n{domain.upper()}: Found {len(examples)} examples with decline >= 0.10")
    
    # Sort all by decline
    all_examples.sort(key=lambda x: x['decline'], reverse=True)
    
    # Print top 5 examples
    print("\n\n" + "="*80)
    print("TOP 5 EXAMPLES OF SHARP DECLINE (Turn 1 → Turn 2)")
    print("="*80)
    
    for i, ex in enumerate(all_examples[:5], 1):
        print(f"\n{'='*80}")
        print(f"Example {i} - {ex['domain'].upper()}")
        print(f"Task ID: {ex['task_id']}")
        print(f"Decline: {ex['turn1_recall']:.4f} → {ex['turn2_recall']:.4f} (Δ = -{ex['decline']:.4f})")
        print(f"\nTurn 1 Query: {ex['turn1_query']}")
        print(f"\nTurn 2 Original Query: {ex['turn2_query']}")
        print(f"Turn 2 Rewritten Query: {ex['turn2_rewritten']}")
        print(f"\nTurn 2 Context Stats:")
        print(f"  - Extracted sentences: {ex['turn2_extracted']}")
        print(f"  - Selected sentences: {ex['turn2_selected']}")
        print(f"\nTurn 2 Selected Sentences (first 3):")
        for j, sent in enumerate(ex['turn2_selected_sentences'][:3], 1):
            print(f"  {j}. [{sent.get('speaker', '?')}, Turn {sent.get('turn', '?')}] {sent.get('sentence', '')[:100]}...")
    
    # Save to JSON for further analysis
    output_file = "turn1_turn2_decline_examples.json"
    with open(output_file, 'w') as f:
        json.dump(all_examples, f, indent=2)
    print(f"\n\n✓ Saved all {len(all_examples)} examples to: {output_file}")

if __name__ == "__main__":
    main()

