#!/usr/bin/env python3
"""
Train a Random Forest classifier to predict optimal retriever+strategy combinations.

This script trains a Random Forest model using query features to predict which of the
9 possible combinations (3 retrievers × 3 strategies) will perform best for each query.
"""

import json
import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, Dict, list]:
    """
    Prepare features and labels for training.
    
    Args:
        df: DataFrame with features and oracle selections
        
    Returns:
        X: Feature matrix
        y: Target labels (oracle combinations)
        label_encoders: Dictionary of label encoders for categorical features
        feature_cols: List of feature column names
    """
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
    
    # Target: oracle_combination (e.g., "bm25_lastturn", "elser_rewrite")
    y = df['oracle_combination'].copy()
    
    return X, y, label_encoders, feature_cols


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_estimators: int = 50,
    max_depth: int = 10,
    min_samples_split: int = 10,
    min_samples_leaf: int = 5,
    random_state: int = 42
) -> Tuple[RandomForestClassifier, Dict]:
    """
    Train a Random Forest classifier and evaluate on test set.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        n_estimators: Number of trees in the forest
        max_depth: Maximum depth of trees
        min_samples_split: Minimum samples required to split a node
        min_samples_leaf: Minimum samples required at a leaf node
        random_state: Random seed for reproducibility
        
    Returns:
        model: Trained Random Forest classifier
        results: Dictionary with evaluation metrics
    """
    print("Training Random Forest classifier...")
    print("  Parameters:")
    print(f"    - Number of trees: {n_estimators}")
    print(f"    - Max depth: {max_depth}")
    print(f"    - Min samples split: {min_samples_split}")
    print(f"    - Min samples leaf: {min_samples_leaf}")
    
    # Initialize and train Random Forest
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=-1,  # Use all available CPU cores
        verbose=0
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate on test set
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Get classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    results = {
        'accuracy': accuracy,
        'classification_report': report,
        'feature_importance': feature_importance.to_dict('records'),
        'n_classes': len(model.classes_),
        'classes': list(model.classes_)
    }
    
    return model, results


def main():
    parser = argparse.ArgumentParser(
        description="Train Random Forest classifier for routing"
    )
    parser.add_argument(
        "--train-data-file",
        type=str,
        default=None,
        help="Path to training data CSV file (if None, will split --data-file)"
    )
    parser.add_argument(
        "--data-file",
        type=str,
        default="../feature-analysis/combined_data.csv",
        help="Path to combined data CSV file (used if --train-data-file not specified)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Output directory for trained model"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data to use for testing (default: 0.2)"
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=50,
        help="Number of trees in the forest (default: 50)"
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=10,
        help="Maximum depth of trees (default: 10)"
    )
    parser.add_argument(
        "--min-samples-split",
        type=int,
        default=10,
        help="Minimum samples required to split a node (default: 10)"
    )
    parser.add_argument(
        "--min-samples-leaf",
        type=int,
        default=5,
        help="Minimum samples required at a leaf node (default: 5)"
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent
    
    if args.train_data_file:
        # Use separate training file
        if Path(args.train_data_file).is_absolute():
            train_data_file = Path(args.train_data_file)
        else:
            train_data_file = (script_dir / args.train_data_file).resolve()
        
        # Check if file exists
        if not train_data_file.exists():
            print(f"\nERROR: Training data file not found: {train_data_file}")
            print("\nTo create synthetic training data, you need to:")
            print("1. Extract oracle selections from synthetic retrieval results")
            print("2. Extract features from synthetic conversations")
            print("3. Combine them using analyze_patterns.py")
            print("\nSee README_SYNTHETIC.md for detailed instructions.")
            sys.exit(1)
        
        print("=" * 80)
        print("TRAINING RANDOM FOREST CLASSIFIER")
        print("=" * 80)
        print(f"Training data file: {train_data_file}")
        print(f"Output directory: {script_dir / args.output_dir}")
        print(f"Random state: {args.random_state}")
        
        # Load training data
        print("\nLoading training data...")
        df_train = pd.read_csv(train_data_file)
        print(f"  Loaded {len(df_train)} training samples")
        print(f"  Features: {len(df_train.columns)} columns")
        
        # Prepare features
        print("\nPreparing features...")
        X_train, y_train, label_encoders, feature_cols = prepare_features(df_train)
        print(f"  Feature columns: {len(feature_cols)}")
        print(f"  Target classes: {y_train.nunique()}")
        print(f"  Classes: {sorted(y_train.unique())}")
        
        # For training-only mode, use a small validation split
        X_train_final, X_test, y_train_final, y_test = train_test_split(
            X_train, y_train,
            test_size=0.1,  # Small validation set
            random_state=args.random_state,
            stratify=y_train
        )
        print(f"\nUsing {len(X_train_final)} samples for training, {len(X_test)} for validation")
        X_train = X_train_final
        y_train = y_train_final
        
    else:
        # Use single file with train/test split
        if Path(args.data_file).is_absolute():
            data_file = Path(args.data_file)
        else:
            data_file = (script_dir / args.data_file).resolve()
        
        output_dir = script_dir / args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("=" * 80)
        print("TRAINING RANDOM FOREST CLASSIFIER")
        print("=" * 80)
        print(f"Data file: {data_file}")
        print(f"Output directory: {output_dir}")
        print(f"Test size: {args.test_size}")
        print(f"Random state: {args.random_state}")
        
        # Load data
        print("\nLoading data...")
        df = pd.read_csv(data_file)
        print(f"  Loaded {len(df)} samples")
        print(f"  Features: {len(df.columns)} columns")
        
        # Prepare features
        print("\nPreparing features...")
        X, y, label_encoders, feature_cols = prepare_features(df)
        print(f"  Feature columns: {len(feature_cols)}")
        print(f"  Target classes: {y.nunique()}")
        print(f"  Classes: {sorted(y.unique())}")
        
        # Split data
        print(f"\nSplitting data (train: {1-args.test_size:.1%}, test: {args.test_size:.1%})...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=args.test_size, 
            random_state=args.random_state, 
            stratify=y  # Maintain class distribution
        )
        print(f"  Training samples: {len(X_train)}")
        print(f"  Test samples: {len(X_test)}")
    
    output_dir = script_dir / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Train model
    print("\n" + "-" * 80)
    model, results = train_random_forest(
        X_train, y_train, X_test, y_test,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.random_state
    )
    
    # Print results
    print("\n" + "=" * 80)
    print("TRAINING RESULTS")
    print("=" * 80)
    print(f"\nTest Set Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"Number of classes: {results['n_classes']}")
    
    print("\nClassification Report:")
    y_test_pred = model.predict(X_test)
    print(classification_report(y_test, y_test_pred))
    
    print("\nTop 10 Most Important Features:")
    for i, feat in enumerate(results['feature_importance'][:10], 1):
        print(f"  {i:2d}. {feat['feature']:<25} {feat['importance']:.4f}")
    
    # Save model
    print("\n" + "-" * 80)
    print("Saving model and artifacts...")
    
    model_file = output_dir / "random_forest_model.pkl"
    joblib.dump(model, model_file)
    print(f"  Model saved: {model_file}")
    
    # Save label encoders
    encoders_file = output_dir / "label_encoders.json"
    encoders_dict = {col: list(le.classes_) for col, le in label_encoders.items()}
    with open(encoders_file, 'w') as f:
        json.dump(encoders_dict, f, indent=2)
    print(f"  Label encoders saved: {encoders_file}")
    
    # Save feature importance
    importance_file = output_dir / "feature_importance.json"
    with open(importance_file, 'w') as f:
        json.dump(results['feature_importance'], f, indent=2)
    print(f"  Feature importance saved: {importance_file}")
    
    # Save training results
    training_results = {
        'accuracy': results['accuracy'],
        'n_classes': results['n_classes'],
        'classes': results['classes'],
        'n_train_samples': len(X_train),
        'n_test_samples': len(X_test),
        'feature_importance': results['feature_importance'],
        'model_parameters': {
            'n_estimators': args.n_estimators,
            'max_depth': args.max_depth,
            'min_samples_split': args.min_samples_split,
            'min_samples_leaf': args.min_samples_leaf,
            'random_state': args.random_state
        }
    }
    
    results_file = output_dir / "training_results.json"
    with open(results_file, 'w') as f:
        json.dump(training_results, f, indent=2)
    print(f"  Training results saved: {results_file}")
    
    print("\n" + "=" * 80)
    print("Training complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()

