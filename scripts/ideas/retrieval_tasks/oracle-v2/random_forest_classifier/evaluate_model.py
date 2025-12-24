#!/usr/bin/env python3
"""
Evaluate Random Forest routing model performance against oracle.

This script evaluates how well the trained Random Forest model captures the oracle's
performance gains by comparing predicted combinations to oracle selections.
"""

import json
import argparse
from pathlib import Path
from typing import Dict
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


def load_model_and_encoders(model_dir: Path):
    """
    Load trained model and label encoders.
    
    Args:
        model_dir: Directory containing model files
        
    Returns:
        model: Trained Random Forest classifier
        encoders_dict: Dictionary mapping feature names to encoder classes
    """
    model_file = model_dir / "random_forest_model.pkl"
    encoders_file = model_dir / "label_encoders.json"
    
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}")
    if not encoders_file.exists():
        raise FileNotFoundError(f"Encoders file not found: {encoders_file}")
    
    model = joblib.load(model_file)
    
    with open(encoders_file, 'r') as f:
        encoders_dict = json.load(f)
    
    return model, encoders_dict


def prepare_features(df: pd.DataFrame, encoders_dict: Dict) -> pd.DataFrame:
    """
    Prepare features using saved label encoders.
    
    Args:
        df: DataFrame with raw features
        encoders_dict: Dictionary of label encoder classes
        
    Returns:
        X: Prepared feature matrix
    """
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
    
    # Fill numeric missing values with median
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
    
    return X


def load_retrieval_results(task_id: str, results_base_dir: Path) -> Dict[str, Dict[str, float]]:
    """
    Load retrieval results for a specific task.
    
    Args:
        task_id: Task identifier
        results_base_dir: Base directory containing retrieval results
        
    Returns:
        Dictionary mapping retriever -> strategy -> scores
    """
    results = {}
    retrievers = ['bm25', 'elser', 'bge']
    strategies = ['lastturn', 'rewrite', 'questions']
    
    for retriever in retrievers:
        results[retriever] = {}
        retriever_dir = results_base_dir / retriever / "results"
        
        if not retriever_dir.exists():
            continue
        
        for strategy in strategies:
            # Try to find the file (could be domain-specific or all)
            filepath = None
            for pattern in [f"{retriever}_all_{strategy}_evaluated.jsonl",
                          f"{retriever}_*_{strategy}_evaluated.jsonl"]:
                files = list(retriever_dir.glob(pattern))
                if files:
                    filepath = files[0]
                    break
            
            if not filepath:
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


def calculate_performance_metrics(
    df: pd.DataFrame,
    model,
    encoders_dict: Dict,
    results_base_dir: Path,
    test_size: float = 0.2,
    random_state: int = 42,
    sample_size: int = None
) -> Dict:
    """
    Calculate performance metrics comparing model predictions to oracle.
    
    Args:
        df: DataFrame with features and oracle selections
        model: Trained Random Forest model
        encoders_dict: Label encoders dictionary
        results_base_dir: Base directory for retrieval results
        test_size: Fraction of data to use for testing (must match training)
        random_state: Random seed (must match training)
        sample_size: Number of tasks to sample for performance calculation (None = all test set)
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Split data the same way as training
    print(f"\nSplitting data (train: {1-test_size:.1%}, test: {test_size:.1%})...")
    df_train, df_test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df['oracle_combination']  # Maintain class distribution
    )
    print(f"  Training samples: {len(df_train)}")
    print(f"  Test samples: {len(df_test)}")
    
    # Prepare features for test set only
    X_test = prepare_features(df_test, encoders_dict)
    
    # Predict on test set only
    predictions = model.predict(X_test)
    df_test = df_test.copy()
    df_test['predicted_combination'] = predictions
    
    # Calculate routing accuracy on test set only
    routing_accuracy = accuracy_score(df_test['oracle_combination'], predictions)
    
    # Calculate actual performance scores on test set
    # Sample if requested (for faster evaluation)
    eval_df = df_test if sample_size is None else df_test.sample(n=min(sample_size, len(df_test)), random_state=42)
    
    print(f"\nCalculating performance scores on {len(eval_df)} tasks...")
    predicted_scores = []
    oracle_scores = []
    
    for idx, row in eval_df.iterrows():
        task_id = row['task_id']
        predicted_combo = row['predicted_combination']
        oracle_combo = row['oracle_combination']
        
        # Load retrieval results
        retrieval_results = load_retrieval_results(task_id, results_base_dir)
        
        # Get scores for predicted and oracle combinations
        pred_parts = predicted_combo.split('_', 1)
        oracle_parts = oracle_combo.split('_', 1)
        
        if len(pred_parts) == 2:
            pred_retriever, pred_strategy = pred_parts
            pred_score = retrieval_results.get(pred_retriever, {}).get(pred_strategy, {}).get('ndcg_cut_5', 0.0)
        else:
            pred_score = 0.0
        
        if len(oracle_parts) == 2:
            oracle_retriever, oracle_strategy = oracle_parts
            oracle_score = retrieval_results.get(oracle_retriever, {}).get(oracle_strategy, {}).get('ndcg_cut_5', 0.0)
        else:
            oracle_score = 0.0
        
        predicted_scores.append(pred_score)
        oracle_scores.append(oracle_score)
    
    avg_predicted = np.mean(predicted_scores) if predicted_scores else 0.0
    avg_oracle = np.mean(oracle_scores) if oracle_scores else 0.0
    performance_gap = avg_oracle - avg_predicted
    performance_captured = (avg_predicted / avg_oracle * 100) if avg_oracle > 0 else 0.0
    
    return {
        'routing_accuracy': float(routing_accuracy),
        'avg_predicted_score': float(avg_predicted),
        'avg_oracle_score': float(avg_oracle),
        'performance_gap': float(performance_gap),
        'performance_captured': float(performance_captured),
        'n_evaluated_tasks': len(eval_df),
        'n_test_tasks': len(df_test),
        'n_total_tasks': len(df)
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Random Forest routing model"
    )
    parser.add_argument(
        "--test-data-file",
        type=str,
        default=None,
        help="Path to test data CSV file (if None, uses --data-file)"
    )
    parser.add_argument(
        "--data-file",
        type=str,
        default="../feature-analysis/combined_data.csv",
        help="Path to combined data CSV file (used if --test-data-file not specified)"
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
        "--sample-size",
        type=int,
        default=None,
        help="Number of tasks to sample for performance calculation (default: all test set)"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test set size (must match training, default: 0.2)"
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed (must match training, default: 42)"
    )
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent
    project_root = script_dir.parents[4]
    
    # Resolve paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parents[4]
    
    # Determine test data file
    if args.test_data_file:
        if Path(args.test_data_file).is_absolute():
            test_data_file = Path(args.test_data_file)
        else:
            test_data_file = (script_dir / args.test_data_file).resolve()
    else:
        if Path(args.data_file).is_absolute():
            test_data_file = Path(args.data_file)
        else:
            test_data_file = (script_dir / args.data_file).resolve()
    
    model_dir = script_dir / args.model_dir
    results_base_dir = (project_root / args.results_base_dir).resolve()
    
    print("=" * 80)
    print("EVALUATING RANDOM FOREST ROUTING MODEL")
    print("=" * 80)
    print(f"Test data file: {test_data_file}")
    print(f"Model directory: {model_dir}")
    print(f"Results base directory: {results_base_dir}")
    if args.sample_size:
        print(f"Sample size: {args.sample_size} tasks")
    
    # Load test data
    print("\nLoading test data...")
    df = pd.read_csv(test_data_file)
    print(f"  Loaded {len(df)} test samples")
    
    # Load model
    print("\nLoading model...")
    model, encoders_dict = load_model_and_encoders(model_dir)
    print("  Model loaded successfully")
    print(f"  Number of classes: {len(model.classes_)}")
    
    # Calculate performance
    print("\n" + "-" * 80)
    results = calculate_performance_metrics(
        df, model, encoders_dict, results_base_dir,
        test_size=args.test_size,
        random_state=args.random_state,
        sample_size=args.sample_size
    )
    
    # Print results
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(f"\nRouting Accuracy: {results['routing_accuracy']:.4f} ({results['routing_accuracy']*100:.2f}%)")
    print("  → Percentage of queries where model predicts the oracle's choice")
    
    print("\nPerformance Scores (nDCG@5):")
    print(f"  Average Predicted Score: {results['avg_predicted_score']:.4f}")
    print(f"  Average Oracle Score:    {results['avg_oracle_score']:.4f}")
    print(f"  Performance Gap:         {results['performance_gap']:.4f}")
    print(f"  Performance Captured:    {results['performance_captured']:.2f}%")
    print(f"  → Model captures {results['performance_captured']:.1f}% of oracle's performance gain")
    
    print("\nEvaluation Details:")
    print(f"  Total tasks: {results['n_total_tasks']}")
    print(f"  Test set tasks: {results['n_test_tasks']}")
    print(f"  Evaluated tasks (for performance scores): {results['n_evaluated_tasks']}")
    
    # Detailed classification report
    X = prepare_features(df, encoders_dict)
    predictions = model.predict(X)
    
    print("\n" + "-" * 80)
    print("Detailed Classification Report:")
    print("-" * 80)
    print(classification_report(df['oracle_combination'], predictions))
    
    # Save results
    print("\n" + "-" * 80)
    print("Saving evaluation results...")
    output_file = model_dir / "evaluation_summary.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved: {output_file}")
    
    print("\n" + "=" * 80)
    print("Evaluation complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()

