#!/usr/bin/env python3
"""
Evaluate routing model performance against oracle.

This script evaluates how well the routing model captures oracle performance gains.
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
    """Load trained model and label encoders."""
    model_file = model_dir / "routing_model.pkl"
    encoders_file = model_dir / "label_encoders.json"
    
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


def get_test_set_indices(df: pd.DataFrame, model_dir: Path, test_size: float = 0.2) -> list:
    """Get test set indices, either from saved file or by recreating the split."""
    test_indices_file = model_dir / "test_indices.json"
    
    if test_indices_file.exists():
        print("Loading test set indices from saved file...")
        with open(test_indices_file, 'r') as f:
            test_indices = json.load(f)
        # Convert to int if they're stored as strings
        test_indices = [int(idx) if isinstance(idx, str) else idx for idx in test_indices]
        return test_indices
    else:
        print("Test indices file not found. Recreating train/test split...")
        # Recreate the same split used in training
        y = df['oracle_combination']
        _, test_indices = train_test_split(
            df.index,
            test_size=test_size,
            random_state=42,
            stratify=y
        )
        return test_indices.tolist()


def calculate_routing_performance(
    df: pd.DataFrame,
    model,
    encoders_dict: Dict,
    results_base_dir: Path,
    test_indices: list
) -> Dict:
    """Calculate performance of routing model on test set only."""
    # Split into train and test
    test_df = df.loc[test_indices].copy()
    train_df = df.drop(test_indices).copy()
    
    print(f"\nDataset split:")
    print(f"  Training set: {len(train_df)} samples")
    print(f"  Test set: {len(test_df)} samples")
    
    # Prepare features for test set only
    X_test = prepare_features(test_df, encoders_dict)
    
    # Predict on test set
    predictions = model.predict(X_test)
    test_df['predicted_combination'] = predictions
    
    # Calculate accuracy on test set
    accuracy = accuracy_score(test_df['oracle_combination'], predictions)
    
    # Calculate detailed metrics
    print(f"\nCalculating detailed metrics on test set...")
    class_report = classification_report(
        test_df['oracle_combination'],
        predictions,
        output_dict=True
    )
    
    # Confusion matrix
    cm = confusion_matrix(test_df['oracle_combination'], predictions)
    unique_labels = sorted(test_df['oracle_combination'].unique())
    
    # Calculate performance metrics using actual retrieval scores
    # For each task in test set, get the score of the predicted combination
    predicted_scores = []
    oracle_scores = []
    baseline_scores = []  # Best single retriever+strategy baseline
    
    print(f"\nCalculating retrieval performance on {len(test_df)} test tasks...")
    for idx, row in test_df.iterrows():
        task_id = row['task_id']
        predicted_combo = row['predicted_combination']
        oracle_combo = row['oracle_combination']
        
        # Load retrieval results
        retrieval_results = load_retrieval_results(task_id, results_base_dir)
        
        if not retrieval_results:
            continue
        
        # Get scores
        try:
            pred_parts = predicted_combo.split('_', 1)
            oracle_parts = oracle_combo.split('_', 1)
            
            if len(pred_parts) != 2 or len(oracle_parts) != 2:
                continue
                
            pred_retriever, pred_strategy = pred_parts
            oracle_retriever, oracle_strategy = oracle_parts
            
            # Get scores - structure is results[retriever][strategy] = {'ndcg_cut_5': ...}
            pred_score_dict = retrieval_results.get(pred_retriever, {}).get(pred_strategy, {})
            pred_score = pred_score_dict.get('ndcg_cut_5', 0.0) if isinstance(pred_score_dict, dict) else 0.0
            
            oracle_score_dict = retrieval_results.get(oracle_retriever, {}).get(oracle_strategy, {})
            oracle_score = oracle_score_dict.get('ndcg_cut_5', 0.0) if isinstance(oracle_score_dict, dict) else 0.0
            
            # Calculate baseline (most common combination: elser_lastturn)
            baseline_score_dict = retrieval_results.get('elser', {}).get('lastturn', {})
            baseline_score = baseline_score_dict.get('ndcg_cut_5', 0.0) if isinstance(baseline_score_dict, dict) else 0.0
            
            predicted_scores.append(pred_score)
            oracle_scores.append(oracle_score)
            baseline_scores.append(baseline_score)
        except (ValueError, AttributeError, KeyError):
            # Skip this task if we can't parse the combination or get scores
            continue
    
    avg_predicted = np.mean(predicted_scores) if predicted_scores else 0.0
    avg_oracle = np.mean(oracle_scores) if oracle_scores else 0.0
    avg_baseline = np.mean(baseline_scores) if baseline_scores else 0.0
    
    # Calculate per-retriever and per-strategy accuracy
    test_df['predicted_retriever'] = test_df['predicted_combination'].str.split('_').str[0]
    test_df['predicted_strategy'] = test_df['predicted_combination'].str.split('_').str[1]
    # Extract oracle retriever and strategy from oracle_combination if columns don't exist
    if 'oracle_retriever' not in test_df.columns:
        test_df['oracle_retriever'] = test_df['oracle_combination'].str.split('_').str[0]
    if 'oracle_strategy' not in test_df.columns:
        test_df['oracle_strategy'] = test_df['oracle_combination'].str.split('_').str[1]
    
    retriever_accuracy = accuracy_score(test_df['oracle_retriever'], test_df['predicted_retriever'])
    strategy_accuracy = accuracy_score(test_df['oracle_strategy'], test_df['predicted_strategy'])
    
    return {
        'routing_accuracy': accuracy,
        'retriever_accuracy': retriever_accuracy,
        'strategy_accuracy': strategy_accuracy,
        'avg_predicted_score': avg_predicted,
        'avg_oracle_score': avg_oracle,
        'avg_baseline_score': avg_baseline,
        'performance_gap': avg_oracle - avg_predicted,
        'performance_captured': (avg_predicted / avg_oracle * 100) if avg_oracle > 0 else 0.0,
        'improvement_over_baseline': avg_predicted - avg_baseline,
        'improvement_over_baseline_pct': ((avg_predicted - avg_baseline) / avg_baseline * 100) if avg_baseline > 0 else 0.0,
        'test_size': len(test_df),
        'train_size': len(train_df),
        'classification_report': class_report,
        'confusion_matrix': cm.tolist(),
        'labels': unique_labels
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate routing model performance"
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
        default="models",
        help="Directory containing trained model"
    )
    parser.add_argument(
        "--results-base-dir",
        type=str,
        default="scripts/baselines/retrieval_scripts",
        help="Base directory for retrieval results"
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
    print("EVALUATING ROUTING MODEL")
    print("=" * 80)
    print(f"Data file: {data_file}")
    print(f"Model directory: {model_dir}")
    print(f"Results base directory: {results_base_dir}")
    
    # Load data
    print("\nLoading data...")
    df = pd.read_csv(data_file)
    print(f"  Loaded {len(df)} samples")
    
    # Load model
    print("Loading model...")
    model, encoders_dict = load_model_and_encoders(model_dir)
    print("  Model loaded")
    
    # Get test set indices
    test_indices = get_test_set_indices(df, model_dir, args.test_size)
    print(f"  Test set size: {len(test_indices)} samples")
    
    # Calculate performance on test set only
    results = calculate_routing_performance(df, model, encoders_dict, results_base_dir, test_indices)
    
    # Print results
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS (TEST SET ONLY)")
    print("=" * 80)
    print(f"\nDataset Split:")
    print(f"  Training set: {results['train_size']} samples")
    print(f"  Test set: {results['test_size']} samples")
    
    print(f"\nAccuracy Metrics:")
    print(f"  Routing Accuracy (exact match): {results['routing_accuracy']:.4f} ({results['routing_accuracy']*100:.2f}%)")
    print(f"  Retriever Accuracy: {results['retriever_accuracy']:.4f} ({results['retriever_accuracy']*100:.2f}%)")
    print(f"  Strategy Accuracy: {results['strategy_accuracy']:.4f} ({results['strategy_accuracy']*100:.2f}%)")
    
    print(f"\nRetrieval Performance (nDCG@5):")
    print(f"  Average Predicted Score: {results['avg_predicted_score']:.4f}")
    print(f"  Average Oracle Score: {results['avg_oracle_score']:.4f}")
    print(f"  Average Baseline Score (elser_lastturn): {results['avg_baseline_score']:.4f}")
    print(f"  Performance Gap: {results['performance_gap']:.4f}")
    print(f"  Performance Captured: {results['performance_captured']:.2f}%")
    print(f"  Improvement over Baseline: {results['improvement_over_baseline']:.4f} ({results['improvement_over_baseline_pct']:.2f}%)")
    
    # Print per-class performance
    print(f"\nPer-Class Performance:")
    for label, metrics in results['classification_report'].items():
        if isinstance(metrics, dict) and 'precision' in metrics:
            print(f"  {label}:")
            print(f"    Precision: {metrics['precision']:.4f}")
            print(f"    Recall: {metrics['recall']:.4f}")
            print(f"    F1-Score: {metrics['f1-score']:.4f}")
            print(f"    Support: {metrics['support']}")
    
    # Save results (excluding confusion matrix and classification report for readability)
    output_file = model_dir / "evaluation_summary.json"
    summary_results = {
        'routing_accuracy': results['routing_accuracy'],
        'retriever_accuracy': results['retriever_accuracy'],
        'strategy_accuracy': results['strategy_accuracy'],
        'avg_predicted_score': results['avg_predicted_score'],
        'avg_oracle_score': results['avg_oracle_score'],
        'avg_baseline_score': results['avg_baseline_score'],
        'performance_gap': results['performance_gap'],
        'performance_captured': results['performance_captured'],
        'improvement_over_baseline': results['improvement_over_baseline'],
        'improvement_over_baseline_pct': results['improvement_over_baseline_pct'],
        'test_size': results['test_size'],
        'train_size': results['train_size']
    }
    with open(output_file, 'w') as f:
        json.dump(summary_results, f, indent=2)
    print(f"\nSummary results saved to: {output_file}")
    
    # Save detailed results including confusion matrix
    detailed_file = model_dir / "evaluation_detailed.json"
    with open(detailed_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Detailed results (including confusion matrix) saved to: {detailed_file}")


if __name__ == "__main__":
    main()

