#!/usr/bin/env python3
"""
Evaluate strategy prediction model performance.

This script evaluates how well the strategy-only model performs compared to
oracle strategy selection, using a FIXED retriever (ELSER by default).

Key metrics:
- Oracle: Best strategy for the fixed retriever (not best combo across all retrievers)
- Baseline: Best static strategy for the fixed retriever (typically 'rewrite')
- Gap Captured: (predicted - baseline) / (oracle - baseline) - the true improvement
"""

import json
import argparse
from pathlib import Path
from typing import Dict
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


def load_model_and_encoders(model_dir: Path):
    """Load trained strategy model and label encoders."""
    model_file = model_dir / "strategy_model.pkl"
    encoders_file = model_dir / "strategy_label_encoders.json"
    
    model = joblib.load(model_file)
    
    with open(encoders_file, 'r') as f:
        encoders_dict = json.load(f)
    
    return model, encoders_dict


def prepare_features(df: pd.DataFrame, encoders_dict: Dict) -> pd.DataFrame:
    """Prepare features using saved encoders."""
    exclude_cols = ['task_id', 'conversation_id', 'oracle_combination', 
                    'oracle_retriever', 'oracle_strategy', 'oracle_score']
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    X = df[feature_cols].copy()
    
    # Apply label encoders
    for col in X.columns:
        if col in encoders_dict:
            le = LabelEncoder()
            le.classes_ = np.array(encoders_dict[col])
            X[col] = X[col].fillna('MISSING')
            # Map to encoded values
            def encode_value(x, encoder=le):
                return encoder.transform([x])[0] if x in encoder.classes_ else 0
            X[col] = X[col].apply(encode_value)
    
    # Fill numeric missing values
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
    
    return X


def get_test_set_indices(df: pd.DataFrame, model_dir: Path, script_dir: Path, test_size: float = 0.2) -> list:
    """Get test set indices, either from saved file or by recreating the split."""
    # First check main models directory for consistency with full routing model
    main_models_dir = script_dir / "models"
    main_test_indices_file = main_models_dir / "test_indices.json"
    
    # Then check strategy-only directory
    test_indices_file = model_dir / "test_indices.json"
    
    if main_test_indices_file.exists():
        print("Loading test set indices from main models directory (for consistency)...")
        with open(main_test_indices_file, 'r') as f:
            test_indices = json.load(f)
    elif test_indices_file.exists():
        print("Loading test set indices from strategy-only directory...")
        with open(test_indices_file, 'r') as f:
            test_indices = json.load(f)
    else:
        print("Test indices file not found. Recreating train/test split...")
        # Recreate the same split used in training
        y = df['oracle_combination'].str.split('_').str[1] if 'oracle_strategy' not in df.columns else df['oracle_strategy']
        _, test_indices = train_test_split(
            df.index,
            test_size=test_size,
            random_state=42,
            stratify=y
        )
        return test_indices.tolist()
    
    # Convert to int if they're stored as strings
    test_indices = [int(idx) if isinstance(idx, str) else idx for idx in test_indices]
    return test_indices


def load_retrieval_results(task_id: str, results_base_dir: Path) -> Dict[str, Dict[str, float]]:
    """Load retrieval results for a task to calculate performance."""
    results = {}
    retrievers = ['bm25', 'elser', 'bge']
    strategies = ['lastturn', 'rewrite', 'questions']
    
    for retriever in retrievers:
        results[retriever] = {}
        retriever_dir = results_base_dir / retriever / "results"
        
        for strategy in strategies:
            # Try to find the file (could be domain-specific or all)
            for pattern in [f"{retriever}_all_{strategy}_evaluated.jsonl",
                          f"{retriever}_*_{strategy}_evaluated.jsonl"]:
                files = list(retriever_dir.glob(pattern))
                if files:
                    filepath = files[0]
                    break
            else:
                continue
            
            # Load the specific task
            try:
                with open(filepath, 'r') as f:
                    for line in f:
                        data = json.loads(line.strip())
                        if data.get('task_id') == task_id:
                            results[retriever][strategy] = data.get('retriever_scores', {})
                            break
            except (json.JSONDecodeError, IOError):
                continue
    
    return results


def calculate_strategy_performance(
    df: pd.DataFrame,
    model,
    encoders_dict: Dict,
    results_base_dir: Path,
    test_indices: list,
    fixed_retriever: str = 'elser',
    baseline_strategy: str = 'rewrite'
) -> Dict:
    """
    Calculate performance of strategy prediction model on test set.
    
    Uses a FIXED retriever for fair evaluation:
    - Predicted score: fixed_retriever + predicted_strategy
    - Oracle score: fixed_retriever + best_strategy_for_that_retriever
    - Baseline score: fixed_retriever + baseline_strategy (best static)
    """
    # Split into train and test
    test_df = df.loc[test_indices].copy()
    train_df = df.drop(test_indices).copy()
    
    print(f"\nDataset split:")
    print(f"  Training set: {len(train_df)} samples")
    print(f"  Test set: {len(test_df)} samples")
    
    # Prepare features for test set only
    X_test = prepare_features(test_df, encoders_dict)
    
    # Get oracle strategy (for the fixed retriever, not the overall oracle)
    # We'll calculate this per-task based on which strategy gives best score for fixed_retriever
    
    # Predict strategy
    predicted_strategies = model.predict(X_test)
    test_df['predicted_strategy'] = predicted_strategies
    
    # Calculate retrieval performance with FIXED retriever
    predicted_scores = []
    oracle_scores = []
    baseline_scores = []
    oracle_strategies_for_fixed_retriever = []
    
    print(f"\nCalculating retrieval performance on {len(test_df)} test tasks...")
    print(f"(Using FIXED retriever: {fixed_retriever})")
    print(f"(Baseline strategy: {baseline_strategy})")
    
    strategies = ['lastturn', 'rewrite', 'questions']
    
    for idx, row in test_df.iterrows():
        task_id = row['task_id']
        predicted_strategy = row['predicted_strategy']
        
        # Load retrieval results
        retrieval_results = load_retrieval_results(task_id, results_base_dir)
        
        if not retrieval_results or fixed_retriever not in retrieval_results:
            continue
        
        try:
            retriever_results = retrieval_results.get(fixed_retriever, {})
            
            # Get predicted score: fixed_retriever + predicted_strategy
            predicted_score_dict = retriever_results.get(predicted_strategy, {})
            predicted_score = predicted_score_dict.get('ndcg_cut_5', 0.0) if isinstance(predicted_score_dict, dict) else 0.0
            
            # Get oracle score: best strategy for this fixed retriever
            best_oracle_score = 0.0
            best_oracle_strategy = baseline_strategy
            for strategy in strategies:
                score_dict = retriever_results.get(strategy, {})
                score = score_dict.get('ndcg_cut_5', 0.0) if isinstance(score_dict, dict) else 0.0
                if score > best_oracle_score:
                    best_oracle_score = score
                    best_oracle_strategy = strategy
            
            # Get baseline score: fixed_retriever + baseline_strategy (best static)
            baseline_score_dict = retriever_results.get(baseline_strategy, {})
            baseline_score = baseline_score_dict.get('ndcg_cut_5', 0.0) if isinstance(baseline_score_dict, dict) else 0.0
            
            predicted_scores.append(predicted_score)
            oracle_scores.append(best_oracle_score)
            baseline_scores.append(baseline_score)
            oracle_strategies_for_fixed_retriever.append(best_oracle_strategy)
            
        except (ValueError, AttributeError, KeyError):
            continue
    
    # Calculate accuracy against fixed-retriever oracle strategy
    # (not the overall oracle which may use a different retriever)
    valid_test_df = test_df.iloc[:len(oracle_strategies_for_fixed_retriever)].copy()
    valid_test_df['oracle_strategy_fixed'] = oracle_strategies_for_fixed_retriever
    
    accuracy = accuracy_score(
        valid_test_df['oracle_strategy_fixed'], 
        valid_test_df['predicted_strategy'].iloc[:len(oracle_strategies_for_fixed_retriever)]
    )
    
    # Classification report against fixed-retriever oracle
    class_report = classification_report(
        valid_test_df['oracle_strategy_fixed'],
        valid_test_df['predicted_strategy'].iloc[:len(oracle_strategies_for_fixed_retriever)],
        output_dict=True,
        zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(
        valid_test_df['oracle_strategy_fixed'],
        valid_test_df['predicted_strategy'].iloc[:len(oracle_strategies_for_fixed_retriever)]
    )
    unique_labels = sorted(set(valid_test_df['oracle_strategy_fixed'].unique()) | 
                          set(valid_test_df['predicted_strategy'].iloc[:len(oracle_strategies_for_fixed_retriever)].unique()))
    
    avg_predicted = np.mean(predicted_scores) if predicted_scores else 0.0
    avg_oracle = np.mean(oracle_scores) if oracle_scores else 0.0
    avg_baseline = np.mean(baseline_scores) if baseline_scores else 0.0
    
    # Calculate gap captured: (predicted - baseline) / (oracle - baseline)
    oracle_baseline_gap = avg_oracle - avg_baseline
    gap_captured = ((avg_predicted - avg_baseline) / oracle_baseline_gap * 100) if oracle_baseline_gap > 0 else 0.0
    
    return {
        'fixed_retriever': fixed_retriever,
        'baseline_strategy': baseline_strategy,
        'strategy_accuracy': accuracy,
        'avg_predicted_score': avg_predicted,
        'avg_oracle_score': avg_oracle,
        'avg_baseline_score': avg_baseline,
        'performance_gap': avg_oracle - avg_predicted,
        'performance_captured_ratio': (avg_predicted / avg_oracle * 100) if avg_oracle > 0 else 0.0,
        'gap_captured': gap_captured,
        'improvement_over_baseline': avg_predicted - avg_baseline,
        'improvement_over_baseline_pct': ((avg_predicted - avg_baseline) / avg_baseline * 100) if avg_baseline > 0 else 0.0,
        'test_size': len(test_df),
        'train_size': len(train_df),
        'valid_samples': len(predicted_scores),
        'classification_report': class_report,
        'confusion_matrix': cm.tolist(),
        'labels': unique_labels
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate strategy prediction model performance with FIXED retriever"
    )
    parser.add_argument(
        "--data-file",
        type=str,
        default="combined_data.csv",
        help="Combined data CSV file"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models-strategy-only",
        help="Directory containing trained model"
    )
    parser.add_argument(
        "--results-base-dir",
        type=str,
        default="scripts/baselines/retrieval_scripts",
        help="Base directory for retrieval results"
    )
    parser.add_argument(
        "--fixed-retriever",
        type=str,
        default="elser",
        choices=["bm25", "elser", "bge"],
        help="Fixed retriever to use for evaluation (default: elser)"
    )
    parser.add_argument(
        "--baseline-strategy",
        type=str,
        default="rewrite",
        choices=["lastturn", "rewrite", "questions"],
        help="Baseline strategy (best static strategy, default: rewrite)"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test set size (used if test_indices.json not found)"
    )
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent
    project_root = script_dir.parents[4]
    data_file = script_dir / args.data_file
    model_dir = script_dir / args.model_dir
    results_base_dir = (project_root / args.results_base_dir).resolve()
    
    print("=" * 80)
    print("EVALUATING STRATEGY PREDICTION MODEL (FIXED RETRIEVER)")
    print("=" * 80)
    print(f"Data file: {data_file}")
    print(f"Model directory: {model_dir}")
    print(f"Results base directory: {results_base_dir}")
    print(f"Fixed retriever: {args.fixed_retriever}")
    print(f"Baseline strategy: {args.baseline_strategy}")
    
    # Load data
    print("\nLoading data...")
    df = pd.read_csv(data_file)
    print(f"  Loaded {len(df)} samples")
    
    # Load model
    print("Loading model...")
    model, encoders_dict = load_model_and_encoders(model_dir)
    print("  Model loaded")
    
    # Get test set indices
    test_indices = get_test_set_indices(df, model_dir, script_dir, args.test_size)
    print(f"  Test set size: {len(test_indices)} samples")
    
    # Calculate performance on test set only
    results = calculate_strategy_performance(
        df, model, encoders_dict, results_base_dir, test_indices,
        fixed_retriever=args.fixed_retriever,
        baseline_strategy=args.baseline_strategy
    )
    
    # Print results
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS (TEST SET ONLY)")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Fixed Retriever: {results['fixed_retriever']}")
    print(f"  Baseline Strategy: {results['baseline_strategy']}")
    print(f"\nDataset Split:")
    print(f"  Training set: {results['train_size']} samples")
    print(f"  Test set: {results['test_size']} samples")
    print(f"  Valid samples (with retrieval results): {results['valid_samples']}")
    
    print(f"\nStrategy Prediction Accuracy (vs {results['fixed_retriever']} oracle):")
    print(f"  Accuracy: {results['strategy_accuracy']:.4f} ({results['strategy_accuracy']*100:.2f}%)")
    
    print(f"\nRetrieval Performance (nDCG@5) - Fixed Retriever: {results['fixed_retriever']}")
    print(f"  Average Predicted Score ({results['fixed_retriever']} + predicted_strategy): {results['avg_predicted_score']:.4f}")
    print(f"  Average Oracle Score ({results['fixed_retriever']} + best_strategy): {results['avg_oracle_score']:.4f}")
    print(f"  Average Baseline Score ({results['fixed_retriever']}_{results['baseline_strategy']}): {results['avg_baseline_score']:.4f}")
    print(f"\nPerformance Analysis:")
    print(f"  Performance Gap (oracle - predicted): {results['performance_gap']:.4f}")
    print(f"  Performance Ratio (predicted/oracle): {results['performance_captured_ratio']:.2f}%")
    print(f"  Gap Captured ((pred-base)/(oracle-base)): {results['gap_captured']:.2f}%")
    print(f"  Improvement over Baseline: {results['improvement_over_baseline']:.4f} ({results['improvement_over_baseline_pct']:.2f}%)")
    
    # Print per-class performance
    print(f"\nPer-Strategy Performance (vs {results['fixed_retriever']} oracle):")
    for label, metrics in results['classification_report'].items():
        if isinstance(metrics, dict) and 'precision' in metrics:
            print(f"  {label}:")
            print(f"    Precision: {metrics['precision']:.4f}")
            print(f"    Recall: {metrics['recall']:.4f}")
            print(f"    F1-Score: {metrics['f1-score']:.4f}")
            print(f"    Support: {metrics['support']}")
    
    # Save results
    output_file = model_dir / "strategy_evaluation_summary.json"
    summary_results = {
        'fixed_retriever': results['fixed_retriever'],
        'baseline_strategy': results['baseline_strategy'],
        'strategy_accuracy': results['strategy_accuracy'],
        'avg_predicted_score': results['avg_predicted_score'],
        'avg_oracle_score': results['avg_oracle_score'],
        'avg_baseline_score': results['avg_baseline_score'],
        'performance_gap': results['performance_gap'],
        'performance_captured_ratio': results['performance_captured_ratio'],
        'gap_captured': results['gap_captured'],
        'improvement_over_baseline': results['improvement_over_baseline'],
        'improvement_over_baseline_pct': results['improvement_over_baseline_pct'],
        'test_size': results['test_size'],
        'train_size': results['train_size'],
        'valid_samples': results['valid_samples']
    }
    with open(output_file, 'w') as f:
        json.dump(summary_results, f, indent=2)
    print(f"\nSummary results saved to: {output_file}")
    
    # Save detailed results
    detailed_file = model_dir / "strategy_evaluation_detailed.json"
    with open(detailed_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Detailed results (including confusion matrix) saved to: {detailed_file}")


if __name__ == "__main__":
    main()

