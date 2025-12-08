import pandas as pd
import json
import os
import numpy as np
from datetime import datetime

def load_intermediate_data(domain):
    """Load intermediate stats (clusters, timing) from jsonl."""
    data = []
    path = f"../intermediate/{domain}_intermediate_k10_lam0.7_all.jsonl"
    if not os.path.exists(path):
        print(f"Warning: Intermediate file not found: {path}")
        return pd.DataFrame()
        
    with open(path, 'r') as f:
        for line in f:
            try:
                entry = json.loads(line)
                
                # Extract relevant stats
                row = {
                    'task_id': entry.get('task_id'),
                    'turn': entry.get('turn'),
                    'num_extracted_sentences': entry.get('num_extracted_sentences', 0),
                    'num_clusters': entry.get('num_clusters', 0),
                    'num_representatives': entry.get('num_representatives', 0),
                    'num_selected': entry.get('num_selected', 0),
                    'method': entry.get('method', ''),
                }
                
                # Timing stats
                timing = entry.get('timing', {})
                row['time_total_pipeline'] = timing.get('total_rewrite_pipeline', 0)
                row['time_clustering'] = timing.get('clustering', 0)
                row['time_mmr'] = timing.get('mmr_selection', 0)
                row['time_llm'] = timing.get('llm_rewrite', 0)
                
                data.append(row)
            except Exception as e:
                continue
                
    return pd.DataFrame(data)

def load_evaluation_data(domain):
    """Load retrieval metrics (Recall, NDCG) from evaluated jsonl."""
    data = []
    path = f"../results_k10/bge_{domain}_mmr_cluster_k10_evaluated.jsonl"
    if not os.path.exists(path):
        print(f"Warning: Evaluation file not found: {path}")
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
                    'Recall@1': scores.get('recall_1', 0),
                }
                data.append(row)
            except Exception as e:
                continue
                
    return pd.DataFrame(data)

def dataframe_to_markdown(df, index_name='Turn'):
    """Convert a pandas DataFrame to a markdown table.
    Assumes the dataframe columns are already in the desired order.
    """
    # Reset index to include it as a column
    df_md = df.reset_index()
    
    # Create markdown table header
    md_lines = []
    
    # Header row - index column first, then all other columns in their current order
    headers = [index_name] + list(df_md.columns[1:])
    md_lines.append('| ' + ' | '.join(str(h) for h in headers) + ' |')
    
    # Separator row
    md_lines.append('|' + '|'.join(['---'] * len(headers)) + '|')
    
    # Data rows
    for _, row in df_md.iterrows():
        # Format turn as integer, count as integer, others as-is
        turn_val = str(int(row.iloc[0])) if isinstance(row.iloc[0], (int, float)) and row.iloc[0] == int(row.iloc[0]) else str(row.iloc[0])
        other_vals = []
        for i, v in enumerate(row.iloc[1:], start=1):
            col_name = df_md.columns[i]
            if col_name == 'count':
                other_vals.append(str(int(v)) if pd.notna(v) else '0')
            else:
                other_vals.append(str(v))
        values = [turn_val] + other_vals
        md_lines.append('| ' + ' | '.join(values) + ' |')
    
    return '\n'.join(md_lines)

def analyze_domain(domain, md_lines):
    """Analyze a domain and append markdown output to md_lines list."""
    # Load data
    df_inter = load_intermediate_data(domain)
    df_eval = load_evaluation_data(domain)
    
    if df_inter.empty or df_eval.empty:
        md_lines.append(f"\n### {domain.upper()}\n\n*Missing data for domain.*\n")
        return None

    # Merge
    df = pd.merge(df_inter, df_eval, on='task_id', how='inner')
    
    # Group by turn
    turn_stats = df.groupby('turn').agg({
        'Recall@10': ['mean', 'count'],
        'NDCG@10': 'mean',
        'num_extracted_sentences': 'mean',
        'num_clusters': 'mean',
        'num_selected': 'mean',
        'time_total_pipeline': 'mean',
        'time_clustering': 'mean',
        'time_llm': 'mean'
    }).round(4)
    
    # Flatten columns
    turn_stats.columns = ['_'.join(col).strip() for col in turn_stats.columns.values]
    
    # Rename count and time columns
    turn_stats.rename(columns={
        'Recall@10_count': 'count',
        'time_total_pipeline_mean': 'time_total_pipeline(s)_mean',
        'time_clustering_mean': 'time_clustering(s)_mean',
        'time_llm_mean': 'time_llm(s)_mean'
    }, inplace=True)
    
    # Reorder columns to put count right after Turn (will be second column in table)
    # Get all columns except count and Recall@10_mean
    other_cols = [c for c in turn_stats.columns if c not in ['Recall@10_mean', 'count']]
    # Order: count first (will be second after reset_index adds turn), then Recall@10_mean, then rest
    cols = ['count', 'Recall@10_mean'] + other_cols
    # Only reorder if all columns exist
    if all(c in turn_stats.columns for c in cols):
        turn_stats = turn_stats[cols]
    
    # Add to markdown
    md_lines.append(f"\n### {domain.upper()}\n")
    md_lines.append(dataframe_to_markdown(turn_stats))
    md_lines.append("")
    
    # Return for aggregation
    df['domain'] = domain
    return df

def main():
    domains = ['clapnq', 'cloud', 'fiqa', 'govt']
    all_data = []
    md_lines = []
    
    # Markdown header
    md_lines.append("# Turn-by-Turn Performance Analysis: k10_lambda0.7 Experiment\n")
    md_lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    md_lines.append("This analysis examines the performance of the MMR clustering query rewriting approach ")
    md_lines.append("across conversation turns for the k=10, lambda=0.7 configuration.\n")
    
    # Domain-specific analyses
    md_lines.append("## Domain-Specific Results\n")
    
    for domain in domains:
        df = analyze_domain(domain, md_lines)
        if df is not None:
            all_data.append(df)
            
    # Overall aggregate
    if all_data:
        md_lines.append("\n## Overall Aggregate (All Domains)\n")
        full_df = pd.concat(all_data)
        
        overall_stats = full_df.groupby('turn').agg({
            'Recall@10': ['mean', 'count'],
            'NDCG@10': 'mean',
            'num_extracted_sentences': 'mean',
            'num_clusters': 'mean',
            'num_selected': 'mean',
            'time_total_pipeline': 'mean',
            'time_clustering': 'mean',
            'time_llm': 'mean'
        }).round(4)
        
        overall_stats.columns = ['_'.join(col).strip() for col in overall_stats.columns.values]
        
        # Rename count and time columns
        overall_stats.rename(columns={
            'Recall@10_count': 'count',
            'time_total_pipeline_mean': 'time_total_pipeline(s)_mean',
            'time_clustering_mean': 'time_clustering(s)_mean',
            'time_llm_mean': 'time_llm(s)_mean'
        }, inplace=True)
        
        # Reorder columns to put count right after Turn (will be second column in table)
        cols = ['count', 'Recall@10_mean'] + [c for c in overall_stats.columns if c not in ['Recall@10_mean', 'count']]
        overall_stats = overall_stats[cols]
        
        md_lines.append(dataframe_to_markdown(overall_stats))
        md_lines.append("")
        
        # Write markdown file
        output_path = "k10_turn_analysis.md"
        with open(output_path, 'w') as f:
            f.write('\n'.join(md_lines))
        
        print(f"✓ Markdown report written to: {output_path}")
    else:
        print("No data available for analysis.")

if __name__ == "__main__":
    main()

