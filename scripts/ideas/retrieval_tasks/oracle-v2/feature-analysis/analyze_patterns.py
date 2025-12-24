#!/usr/bin/env python3
"""
Analyze patterns between features and oracle selections.

This script combines oracle selections with task features to identify
predictive patterns for routing.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List
from collections import defaultdict, Counter
import pandas as pd
import numpy as np


def load_data(oracle_file: Path, features_file: Path) -> tuple:
    """Load oracle selections and features."""
    with open(oracle_file, 'r', encoding='utf-8') as f:
        oracle_data = json.load(f)
    
    with open(features_file, 'r', encoding='utf-8') as f:
        features_data = json.load(f)
    
    return oracle_data, features_data


def create_analysis_dataframe(oracle_data: Dict, features_data: Dict) -> pd.DataFrame:
    """Create a combined dataframe for analysis."""
    oracle_selections = oracle_data.get('oracle_selections', {})
    task_features = features_data.get('features', {})
    
    rows = []
    for task_id, oracle_info in oracle_selections.items():
        if task_id not in task_features:
            continue
        
        features = task_features[task_id]
        row = {
            'task_id': task_id,
            'oracle_combination': oracle_info['combination'],
            'oracle_retriever': oracle_info['retriever'],
            'oracle_strategy': oracle_info['strategy'],
            'oracle_score': oracle_info['score'],
            **features
        }
        rows.append(row)
    
    return pd.DataFrame(rows)


def convert_keys_to_strings(obj):
    """Recursively convert dictionary keys to strings for JSON serialization."""
    if isinstance(obj, dict):
        return {str(k): convert_keys_to_strings(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_keys_to_strings(item) for item in obj]
    elif isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj) if isinstance(obj, np.floating) else int(obj)
    else:
        return obj


def analyze_by_feature(df: pd.DataFrame, feature_col: str, output_dir: Path):
    """Analyze oracle selections by a specific feature."""
    if feature_col not in df.columns:
        print(f"  Warning: Feature '{feature_col}' not found")
        return
    
    results = {}
    
    # Overall distribution
    value_counts = df[feature_col].value_counts()
    results['value_counts'] = convert_keys_to_strings(value_counts.to_dict())
    
    # Oracle combination distribution by feature value
    combination_by_value = {}
    for value in df[feature_col].dropna().unique():
        subset = df[df[feature_col] == value]
        combo_dist = subset['oracle_combination'].value_counts().to_dict()
        combination_by_value[str(value)] = convert_keys_to_strings(combo_dist)
    
    results['combination_distribution'] = combination_by_value
    
    # Retriever distribution by feature value
    retriever_by_value = {}
    for value in df[feature_col].dropna().unique():
        subset = df[df[feature_col] == value]
        retriever_dist = subset['oracle_retriever'].value_counts().to_dict()
        retriever_by_value[str(value)] = convert_keys_to_strings(retriever_dist)
    
    results['retriever_distribution'] = retriever_by_value
    
    # Strategy distribution by feature value
    strategy_by_value = {}
    for value in df[feature_col].dropna().unique():
        subset = df[df[feature_col] == value]
        strategy_dist = subset['oracle_strategy'].value_counts().to_dict()
        strategy_by_value[str(value)] = convert_keys_to_strings(strategy_dist)
    
    results['strategy_distribution'] = strategy_by_value
    
    # Save results
    output_file = output_dir / f"analysis_{feature_col}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    
    return results


def print_summary_statistics(df: pd.DataFrame):
    """Print summary statistics."""
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    print(f"\nTotal tasks analyzed: {len(df)}")
    
    print("\nOracle Combination Distribution:")
    combo_counts = df['oracle_combination'].value_counts()
    for combo, count in combo_counts.items():
        pct = (count / len(df)) * 100
        retriever, strategy = combo.split('_', 1)
        print(f"  {retriever.upper():<8} {strategy.capitalize():<12} {count:>4} ({pct:>5.1f}%)")
    
    print("\nOracle Retriever Distribution:")
    retriever_counts = df['oracle_retriever'].value_counts()
    for retriever, count in retriever_counts.items():
        pct = (count / len(df)) * 100
        print(f"  {retriever.upper():<8} {count:>4} ({pct:>5.1f}%)")
    
    print("\nOracle Strategy Distribution:")
    strategy_counts = df['oracle_strategy'].value_counts()
    for strategy, count in strategy_counts.items():
        pct = (count / len(df)) * 100
        print(f"  {strategy.capitalize():<12} {count:>4} ({pct:>5.1f}%)")
    
    # Feature statistics
    print("\nFeature Statistics:")
    numeric_features = ['query_length_chars', 'query_length_words', 'turn_id']
    for feat in numeric_features:
        if feat in df.columns:
            print(f"\n  {feat}:")
            print(f"    Mean: {df[feat].mean():.2f}")
            print(f"    Median: {df[feat].median():.2f}")
            print(f"    Min: {df[feat].min()}")
            print(f"    Max: {df[feat].max()}")


def analyze_categorical_features(df: pd.DataFrame, output_dir: Path):
    """Analyze categorical features."""
    categorical_features = [
        'domain', 'answerability', 'question_type', 'multi_turn_type',
        'is_first_turn', 'has_question_mark', 'has_wh_word'
    ]
    
    print("\n" + "=" * 80)
    print("CATEGORICAL FEATURE ANALYSIS")
    print("=" * 80)
    
    for feature in categorical_features:
        if feature not in df.columns:
            continue
        
        print(f"\n{feature.upper()}:")
        results = analyze_by_feature(df, feature, output_dir)
        
        if results:
            # Print top patterns
            for value, combo_dist in list(results['combination_distribution'].items())[:5]:
                total = sum(combo_dist.values())
                print(f"  {value}:")
                for combo, count in sorted(combo_dist.items(), key=lambda x: x[1], reverse=True)[:3]:
                    pct = (count / total * 100) if total > 0 else 0.0
                    retriever, strategy = combo.split('_', 1)
                    print(f"    {retriever.upper():<8} {strategy.capitalize():<12} {count:>3} ({pct:>5.1f}%)")


def analyze_numeric_features(df: pd.DataFrame, output_dir: Path):
    """Analyze numeric features by binning."""
    numeric_features = ['query_length_chars', 'query_length_words', 'turn_id', 'conversation_length', 'num_previous_turns']
    
    print("\n" + "=" * 80)
    print("NUMERIC FEATURE ANALYSIS")
    print("=" * 80)
    
    for feature in numeric_features:
        if feature not in df.columns:
            continue
        
        print(f"\n{feature.upper()}:")
        
        # Create bins
        if feature == 'turn_id':
            bins = [0, 1, 2, 3, 5, 10, float('inf')]
            labels = ['1', '2', '3', '4-5', '6-10', '11+']
        elif feature in ['conversation_length', 'num_previous_turns']:
            bins = [0, 1, 2, 3, 5, 10, float('inf')]
            labels = ['0', '1', '2', '3-4', '5-9', '10+']
        elif feature == 'query_length_words':
            bins = [0, 5, 10, 15, 20, float('inf')]
            labels = ['1-5', '6-10', '11-15', '16-20', '21+']
        else:  # query_length_chars
            bins = [0, 20, 40, 60, 80, float('inf')]
            labels = ['1-20', '21-40', '41-60', '61-80', '81+']
        
        df[f'{feature}_binned'] = pd.cut(df[feature], bins=bins, labels=labels, right=False)
        
        results = analyze_by_feature(df, f'{feature}_binned', output_dir)
        
        if results:
            # Print patterns
            for value, combo_dist in results['combination_distribution'].items():
                total = sum(combo_dist.values())
                if total < 5:  # Skip bins with too few samples
                    continue
                print(f"  {value}:")
                top_combo = max(combo_dist.items(), key=lambda x: x[1])
                retriever, strategy = top_combo[0].split('_', 1)
                pct = (top_combo[1] / total * 100) if total > 0 else 0.0
                print(f"    Most common: {retriever.upper()} {strategy.capitalize()} ({pct:.1f}%)")


def analyze_bm25_vs_elser(df: pd.DataFrame, output_dir: Path):
    """Specifically analyze cases where BM25 outperforms ELSER."""
    print("\n" + "=" * 80)
    print("BM25 vs ELSER ANALYSIS")
    print("=" * 80)
    
    # Find tasks where BM25 is selected
    bm25_tasks = df[df['oracle_retriever'] == 'bm25'].copy()
    elser_tasks = df[df['oracle_retriever'] == 'elser'].copy()
    
    print(f"\nBM25 selected: {len(bm25_tasks)} tasks ({len(bm25_tasks)/len(df)*100:.1f}%)")
    print(f"ELSER selected: {len(elser_tasks)} tasks ({len(elser_tasks)/len(df)*100:.1f}%)")
    
    # Compare features
    comparison_features = [
        'domain', 'question_type', 'multi_turn_type', 'answerability',
        'query_length_words', 'turn_id', 'is_first_turn'
    ]
    
    results = {}
    for feature in comparison_features:
        if feature not in df.columns:
            continue
        
        if df[feature].dtype in ['object', 'category', 'bool']:
            # Categorical comparison
            bm25_dist = bm25_tasks[feature].value_counts(normalize=True).to_dict()
            elser_dist = elser_tasks[feature].value_counts(normalize=True).to_dict()
            results[feature] = {
                'bm25_distribution': convert_keys_to_strings(bm25_dist),
                'elser_distribution': convert_keys_to_strings(elser_dist)
            }
        else:
            # Numeric comparison
            results[feature] = {
                'bm25_mean': float(bm25_tasks[feature].mean()) if len(bm25_tasks) > 0 else None,
                'bm25_median': float(bm25_tasks[feature].median()) if len(bm25_tasks) > 0 else None,
                'elser_mean': float(elser_tasks[feature].mean()) if len(elser_tasks) > 0 else None,
                'elser_median': float(elser_tasks[feature].median()) if len(elser_tasks) > 0 else None,
            }
    
    # Save results
    output_file = output_dir / "bm25_vs_elser_analysis.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Print summary
    print("\nFeature Comparison:")
    for feature, data in list(results.items())[:5]:
        print(f"\n  {feature}:")
        if 'bm25_distribution' in data:
            print("    BM25 top values:")
            for val, pct in sorted(data['bm25_distribution'].items(), key=lambda x: x[1], reverse=True)[:3]:
                print(f"      {val}: {pct*100:.1f}%")
            print("    ELSER top values:")
            for val, pct in sorted(data['elser_distribution'].items(), key=lambda x: x[1], reverse=True)[:3]:
                print(f"      {val}: {pct*100:.1f}%")
        else:
            print(f"    BM25 mean: {data.get('bm25_mean', 'N/A')}")
            print(f"    ELSER mean: {data.get('elser_mean', 'N/A')}")


def analyze_lastturn_vs_rewrite(df: pd.DataFrame, output_dir: Path):
    """Specifically analyze when Lastturn vs Rewrite is optimal."""
    print("\n" + "=" * 80)
    print("LASTTURN vs REWRITE ANALYSIS")
    print("=" * 80)
    
    # Find tasks where Lastturn vs Rewrite is selected
    lastturn_tasks = df[df['oracle_strategy'] == 'lastturn'].copy()
    rewrite_tasks = df[df['oracle_strategy'] == 'rewrite'].copy()
    
    print(f"\nLastturn selected: {len(lastturn_tasks)} tasks ({len(lastturn_tasks)/len(df)*100:.1f}%)")
    print(f"Rewrite selected: {len(rewrite_tasks)} tasks ({len(rewrite_tasks)/len(df)*100:.1f}%)")
    
    # Compare features
    comparison_features = [
        'domain', 'question_type', 'multi_turn_type', 'answerability',
        'query_length_words', 'turn_id', 'is_first_turn', 'conversation_length', 'num_previous_turns'
    ]
    
    results = {}
    for feature in comparison_features:
        if feature not in df.columns:
            continue
        
        if df[feature].dtype in ['object', 'category', 'bool']:
            # Categorical comparison
            lastturn_dist = lastturn_tasks[feature].value_counts(normalize=True).to_dict()
            rewrite_dist = rewrite_tasks[feature].value_counts(normalize=True).to_dict()
            results[feature] = {
                'lastturn_distribution': convert_keys_to_strings(lastturn_dist),
                'rewrite_distribution': convert_keys_to_strings(rewrite_dist)
            }
        else:
            # Numeric comparison
            results[feature] = {
                'lastturn_mean': float(lastturn_tasks[feature].mean()) if len(lastturn_tasks) > 0 else None,
                'lastturn_median': float(lastturn_tasks[feature].median()) if len(lastturn_tasks) > 0 else None,
                'rewrite_mean': float(rewrite_tasks[feature].mean()) if len(rewrite_tasks) > 0 else None,
                'rewrite_median': float(rewrite_tasks[feature].median()) if len(rewrite_tasks) > 0 else None,
            }
    
    # Save results
    output_file = output_dir / "lastturn_vs_rewrite_analysis.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Print summary
    print("\nFeature Comparison:")
    for feature, data in list(results.items())[:5]:
        print(f"\n  {feature}:")
        if 'lastturn_distribution' in data:
            print("    Lastturn top values:")
            for val, pct in sorted(data['lastturn_distribution'].items(), key=lambda x: x[1], reverse=True)[:3]:
                print(f"      {val}: {pct*100:.1f}%")
            print("    Rewrite top values:")
            for val, pct in sorted(data['rewrite_distribution'].items(), key=lambda x: x[1], reverse=True)[:3]:
                print(f"      {val}: {pct*100:.1f}%")
        else:
            print(f"    Lastturn mean: {data.get('lastturn_mean', 'N/A')}")
            print(f"    Rewrite mean: {data.get('rewrite_mean', 'N/A')}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze patterns between features and oracle selections"
    )
    parser.add_argument(
        "--oracle-file",
        type=str,
        default="oracle_selections.json",
        help="Oracle selections JSON file"
    )
    parser.add_argument(
        "--features-file",
        type=str,
        default="task_features.json",
        help="Task features JSON file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Output directory for analysis results"
    )
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent
    oracle_file = script_dir / args.oracle_file
    features_file = script_dir / args.features_file
    output_dir = script_dir / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("PATTERN ANALYSIS")
    print("=" * 80)
    print(f"Oracle file: {oracle_file}")
    print(f"Features file: {features_file}")
    print(f"Output directory: {output_dir}")
    
    # Load data
    print("\nLoading data...")
    oracle_data, features_data = load_data(oracle_file, features_file)
    
    # Create dataframe
    print("Creating analysis dataframe...")
    df = create_analysis_dataframe(oracle_data, features_data)
    print(f"  Created dataframe with {len(df)} rows and {len(df.columns)} columns")
    
    # Save combined dataframe
    combined_file = output_dir / "combined_data.csv"
    df.to_csv(combined_file, index=False)
    print(f"  Saved combined data to {combined_file}")
    
    # Print summary statistics
    print_summary_statistics(df)
    
    # Analyze categorical features
    analyze_categorical_features(df, output_dir)
    
    # Analyze numeric features
    analyze_numeric_features(df, output_dir)
    
    # Specific analyses mentioned in ideas document
    analyze_bm25_vs_elser(df, output_dir)
    analyze_lastturn_vs_rewrite(df, output_dir)
    
    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()

