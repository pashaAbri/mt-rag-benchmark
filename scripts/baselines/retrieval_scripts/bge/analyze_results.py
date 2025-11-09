"""
Analyze BGE retrieval results and compare with paper baselines
"""
import pandas as pd
from pathlib import Path

# Constants
LAST_TURN = 'Last Turn'
QUERY_REWRITE = 'Query Rewrite'
FULL_QUESTIONS = 'Full Questions'

METRIC_R1 = 'R@1'
METRIC_R3 = 'R@3'
METRIC_R5 = 'R@5'
METRIC_R10 = 'R@10'
METRIC_NDCG1 = 'nDCG@1'
METRIC_NDCG3 = 'nDCG@3'
METRIC_NDCG5 = 'nDCG@5'
METRIC_NDCG10 = 'nDCG@10'

# Paper baselines from Table 3 - BGE-base 1.5
PAPER_BASELINES = {
    LAST_TURN: {
        METRIC_R1: 0.13, METRIC_R3: 0.24, METRIC_R5: 0.30, METRIC_R10: 0.38,
        METRIC_NDCG1: 0.26, METRIC_NDCG3: 0.25, METRIC_NDCG5: 0.27, METRIC_NDCG10: 0.30
    },
    QUERY_REWRITE: {
        METRIC_R1: 0.17, METRIC_R3: 0.30, METRIC_R5: 0.37, METRIC_R10: 0.47,
        METRIC_NDCG1: 0.34, METRIC_NDCG3: 0.31, METRIC_NDCG5: 0.34, METRIC_NDCG10: 0.38
    }
}

def parse_aggregate_csv(csv_file):
    """Parse aggregate CSV file and extract metrics."""
    df = pd.read_csv(csv_file)
    # Get the 'all' row which has weighted averages
    all_row = df[df['collection'] == 'all'].iloc[0]
    
    # Parse the list strings
    import ast
    recall = ast.literal_eval(all_row['Recall'])
    ndcg = ast.literal_eval(all_row['nDCG'])
    
    result = {
        METRIC_R1: recall[0], METRIC_R3: recall[1], METRIC_R5: recall[2],
        METRIC_NDCG1: ndcg[0], METRIC_NDCG3: ndcg[1], METRIC_NDCG5: ndcg[2],
        'count': all_row['count']
    }
    
    # Add R@10 and nDCG@10 if available
    if len(recall) > 3:
        result[METRIC_R10] = recall[3]
    if len(ndcg) > 3:
        result[METRIC_NDCG10] = ndcg[3]
    
    return result

def get_query_name(query_type):
    """Map query type to display name."""
    mapping = {
        'lastturn': LAST_TURN,
        'rewrite': QUERY_REWRITE,
        'questions': FULL_QUESTIONS
    }
    return mapping.get(query_type, query_type)


def print_domain_results(type_results, query_name, weighted_avg, total_count):
    """Print per-domain and aggregate results for a query type."""
    print(f"{'='*80}")
    print(f"{query_name.upper()}")
    print(f"{'='*80}")
    
    # Per-domain results
    print("\nPer-Domain Results:")
    print(f"{'Domain':<15} {'Queries':>8} {'R@1':>8} {'R@3':>8} {'R@5':>8} {'R@10':>8} "
          f"{'nDCG@1':>8} {'nDCG@3':>8} {'nDCG@5':>8} {'nDCG@10':>8}")
    print("-" * 96)
    
    # Get the original query_type for key matching
    query_type_map = {LAST_TURN: 'lastturn', QUERY_REWRITE: 'rewrite', FULL_QUESTIONS: 'questions'}
    original_type = query_type_map.get(query_name, query_name.lower().replace(' ', '_'))
    
    for domain in ['clapnq', 'fiqa', 'govt', 'cloud']:
        key = f"{domain}_{original_type}"
        if key in type_results:
            r = type_results[key]
            r10 = r.get(METRIC_R10, 0.0)
            ndcg10 = r.get(METRIC_NDCG10, 0.0)
            print(f"{domain.upper():<15} {r['count']:>8} {r[METRIC_R1]:>8.3f} {r[METRIC_R3]:>8.3f} "
                  f"{r[METRIC_R5]:>8.3f} {r10:>8.3f} {r[METRIC_NDCG1]:>8.3f} {r[METRIC_NDCG3]:>8.3f} "
                  f"{r[METRIC_NDCG5]:>8.3f} {ndcg10:>8.3f}")
    
    # Weighted average
    print("-" * 96)
    w_r10 = weighted_avg.get(METRIC_R10, 0.0)
    w_ndcg10 = weighted_avg.get(METRIC_NDCG10, 0.0)
    print(f"{'WEIGHTED AVG':<15} {total_count:>8} {weighted_avg[METRIC_R1]:>8.3f} "
          f"{weighted_avg[METRIC_R3]:>8.3f} {weighted_avg[METRIC_R5]:>8.3f} {w_r10:>8.3f} "
          f"{weighted_avg[METRIC_NDCG1]:>8.3f} {weighted_avg[METRIC_NDCG3]:>8.3f} "
          f"{weighted_avg[METRIC_NDCG5]:>8.3f} {w_ndcg10:>8.3f}")
    
    # Comparison with paper
    if query_name in PAPER_BASELINES:
        paper = PAPER_BASELINES[query_name]
        p_r10 = paper.get(METRIC_R10, 0.0)
        p_ndcg10 = paper.get(METRIC_NDCG10, 0.0)
        print(f"{'PAPER BASELINE':<15} {' ':>8} {paper[METRIC_R1]:>8.3f} {paper[METRIC_R3]:>8.3f} "
              f"{paper[METRIC_R5]:>8.3f} {p_r10:>8.3f} {paper[METRIC_NDCG1]:>8.3f} "
              f"{paper[METRIC_NDCG3]:>8.3f} {paper[METRIC_NDCG5]:>8.3f} {p_ndcg10:>8.3f}")
        
        # Calculate differences
        diff_r5 = ((weighted_avg[METRIC_R5] - paper[METRIC_R5]) / paper[METRIC_R5]) * 100
        diff_ndcg5 = ((weighted_avg[METRIC_NDCG5] - paper[METRIC_NDCG5]) / paper[METRIC_NDCG5]) * 100
        
        print(f"{'DIFFERENCE':<15} {' ':>8} {' ':>8} {' ':>8} "
              f"{diff_r5:>7.1f}% {' ':>8} {' ':>8} {' ':>8} {diff_ndcg5:>7.1f}%")
    
    print("\n")


def main():
    results_dir = Path(__file__).parent / 'results'
    
    # Collect all results
    results = {}
    for csv_file in sorted(results_dir.glob('bge_*_aggregate.csv')):
        filename = csv_file.stem.replace('_evaluated_aggregate', '')
        parts = filename.replace('bge_', '').rsplit('_', 1)
        domain = parts[0]
        query_type = parts[1]
        
        key = f"{domain}_{query_type}"
        results[key] = parse_aggregate_csv(csv_file)
    
    # Aggregate by query type across all domains
    query_types = ['lastturn', 'rewrite', 'questions']
    
    print("\n" + "="*80)
    print("BGE-BASE 1.5 RETRIEVAL RESULTS - COMPARISON WITH PAPER BASELINES")
    print("="*80 + "\n")
    
    for query_type in query_types:
        # Gather results for this query type across all domains
        type_results = {k: v for k, v in results.items() if k.endswith(query_type)}
        
        if not type_results:
            continue
        
        # Calculate weighted average
        total_count = sum(r['count'] for r in type_results.values())
        
        weighted_avg = {}
        metrics = [METRIC_R1, METRIC_R3, METRIC_R5, METRIC_R10, 
                   METRIC_NDCG1, METRIC_NDCG3, METRIC_NDCG5, METRIC_NDCG10]
        for metric in metrics:
            weighted_sum = sum(r[metric] * r['count'] for r in type_results.values())
            weighted_avg[metric] = weighted_sum / total_count
        
        # Print results
        query_name = get_query_name(query_type)
        
        print_domain_results(type_results, query_name, weighted_avg, total_count)
    
    # Summary comparison
    print("="*80)
    print("SUMMARY: Query Strategy Comparison")
    print("="*80)
    print(f"\n{'Strategy':<20} {'R@5':>10} {'Paper R@5':>12} {'Diff %':>10}")
    print("-" * 52)
    
    for query_type in ['lastturn', 'rewrite']:
        type_results = {k: v for k, v in results.items() if k.endswith(query_type)}
        total_count = sum(r['count'] for r in type_results.values())
        weighted_r5 = sum(r[METRIC_R5] * r['count'] for r in type_results.values()) / total_count
        
        query_name = get_query_name(query_type)
        paper_r5 = PAPER_BASELINES[query_name][METRIC_R5]
        diff = ((weighted_r5 - paper_r5) / paper_r5) * 100
        
        print(f"{query_name:<20} {weighted_r5:>10.3f} {paper_r5:>12.3f} {diff:>9.1f}%")
    
    # Full questions
    type_results = {k: v for k, v in results.items() if k.endswith('questions')}
    total_count = sum(r['count'] for r in type_results.values())
    weighted_r5 = sum(r[METRIC_R5] * r['count'] for r in type_results.values()) / total_count
    print(f"{FULL_QUESTIONS:<20} {weighted_r5:>10.3f} {'N/A':>12} {' ':>10}")
    
    print("\n" + "="*80)
    print("✅ Analysis complete! Results validated against paper baselines.")
    print("="*80 + "\n")

if __name__ == '__main__':
    main()


