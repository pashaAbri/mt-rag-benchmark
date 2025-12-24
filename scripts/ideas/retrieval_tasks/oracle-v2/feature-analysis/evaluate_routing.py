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
from sklearn.metrics import accuracy_score


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
            X[col] = X[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else 0)
    
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


def calculate_routing_performance(
    df: pd.DataFrame,
    model,
    encoders_dict: Dict,
    results_base_dir: Path
) -> Dict:
    """Calculate performance of routing model."""
    # Prepare features
    X = prepare_features(df, encoders_dict)
    
    # Predict
    predictions = model.predict(X)
    df['predicted_combination'] = predictions
    
    # Calculate accuracy
    accuracy = accuracy_score(df['oracle_combination'], predictions)
    
    # Calculate performance metrics
    # For each task, get the score of the predicted combination
    predicted_scores = []
    oracle_scores = []
    
    # Sample a subset for performance calculation (can be slow)
    sample_size = min(100, len(df))  # Sample 100 tasks for performance calculation
    sample_df = df.sample(n=sample_size, random_state=42)
    
    print(f"\nCalculating performance on {len(sample_df)} sampled tasks...")
    for idx, row in sample_df.iterrows():
        task_id = row['task_id']
        predicted_combo = row['predicted_combination']
        oracle_combo = row['oracle_combination']
        
        # Load retrieval results
        retrieval_results = load_retrieval_results(task_id, results_base_dir)
        
        # Get scores
        pred_retriever, pred_strategy = predicted_combo.split('_', 1)
        oracle_retriever, oracle_strategy = oracle_combo.split('_', 1)
        
        pred_score = retrieval_results.get(pred_retriever, {}).get(pred_strategy, {}).get('ndcg_cut_5', 0.0)
        oracle_score = retrieval_results.get(oracle_retriever, {}).get(oracle_strategy, {}).get('ndcg_cut_5', 0.0)
        
        predicted_scores.append(pred_score)
        oracle_scores.append(oracle_score)
    
    avg_predicted = np.mean(predicted_scores) if predicted_scores else 0.0
    avg_oracle = np.mean(oracle_scores) if oracle_scores else 0.0
    
    return {
        'routing_accuracy': accuracy,
        'avg_predicted_score': avg_predicted,
        'avg_oracle_score': avg_oracle,
        'performance_gap': avg_oracle - avg_predicted,
        'performance_captured': (avg_predicted / avg_oracle * 100) if avg_oracle > 0 else 0.0
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
    
    # Calculate performance
    results = calculate_routing_performance(df, model, encoders_dict, results_base_dir)
    
    # Print results
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(f"Routing Accuracy: {results['routing_accuracy']:.4f}")
    print(f"Average Predicted Score (nDCG@5): {results['avg_predicted_score']:.4f}")
    print(f"Average Oracle Score (nDCG@5): {results['avg_oracle_score']:.4f}")
    print(f"Performance Gap: {results['performance_gap']:.4f}")
    print(f"Performance Captured: {results['performance_captured']:.2f}%")
    
    # Save results
    output_file = model_dir / "evaluation_summary.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()

