#!/usr/bin/env python3
"""
Train routing models to predict optimal retriever+strategy combinations.

This script trains classifiers using extracted features to predict oracle selections.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Prepare features and labels for training."""
    # Select feature columns (exclude metadata and target)
    exclude_cols = ['task_id', 'conversation_id', 'oracle_combination', 
                    'oracle_retriever', 'oracle_strategy', 'oracle_score']
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Handle missing values
    X = df[feature_cols].copy()
    
    # Encode categorical variables
    label_encoders = {}
    for col in X.columns:
        if X[col].dtype == 'object':
            le = LabelEncoder()
            X[col] = X[col].fillna('MISSING')
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le
    
    # Fill numeric missing values with median
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
    
    # Target: oracle_combination
    y = df['oracle_combination'].copy()
    
    return X, y, label_encoders, feature_cols


def train_model(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> Dict:
    """Train a routing model."""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # Train Random Forest classifier
    print("Training Random Forest classifier...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nModel Performance:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop 10 Most Important Features:")
    for idx, row in feature_importance.head(10).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")
    
    return {
        'model': model,
        'accuracy': accuracy,
        'feature_importance': feature_importance.to_dict('records'),
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': y_pred
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train routing model to predict oracle selections"
    )
    parser.add_argument(
        "--data-file",
        type=str,
        default="combined_data.csv",
        help="Combined data CSV file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Output directory for models"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test set size"
    )
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent
    data_file = script_dir / args.data_file
    output_dir = script_dir / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("TRAINING ROUTING MODEL")
    print("=" * 80)
    print(f"Data file: {data_file}")
    print(f"Output directory: {output_dir}")
    
    # Load data
    print("\nLoading data...")
    df = pd.read_csv(data_file)
    print(f"  Loaded {len(df)} samples")
    
    # Prepare features
    print("Preparing features...")
    X, y, label_encoders, feature_cols = prepare_features(df)
    print(f"  Features: {len(feature_cols)}")
    print(f"  Target classes: {y.nunique()}")
    
    # Train model
    results = train_model(X, y, test_size=args.test_size)
    
    # Save model
    model_file = output_dir / "routing_model.pkl"
    joblib.dump(results['model'], model_file)
    print(f"\nModel saved to: {model_file}")
    
    # Save label encoders
    encoders_file = output_dir / "label_encoders.json"
    encoders_dict = {col: list(le.classes_) for col, le in label_encoders.items()}
    with open(encoders_file, 'w') as f:
        json.dump(encoders_dict, f, indent=2)
    print(f"Label encoders saved to: {encoders_file}")
    
    # Save feature importance
    importance_file = output_dir / "feature_importance.json"
    with open(importance_file, 'w') as f:
        json.dump(results['feature_importance'], f, indent=2)
    print(f"Feature importance saved to: {importance_file}")
    
    # Save evaluation results
    eval_file = output_dir / "evaluation_results.json"
    eval_data = {
        'accuracy': results['accuracy'],
        'feature_importance': results['feature_importance']
    }
    with open(eval_file, 'w') as f:
        json.dump(eval_data, f, indent=2)
    print(f"Evaluation results saved to: {eval_file}")


if __name__ == "__main__":
    main()

