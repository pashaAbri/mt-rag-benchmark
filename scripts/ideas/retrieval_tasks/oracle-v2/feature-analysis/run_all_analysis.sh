#!/bin/bash
# Run all feature analysis scripts in sequence

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "Running Feature Analysis Pipeline"
echo "=========================================="
echo ""

# Step 1: Extract oracle selections
echo "Step 1/5: Extracting oracle selections..."
python3 extract_oracle_selections.py --domain all
echo ""

# Step 2: Extract features
echo "Step 2/5: Extracting task features..."
python3 extract_features.py --domain all
echo ""

# Step 3: Analyze patterns
echo "Step 3/5: Analyzing patterns..."
python3 analyze_patterns.py
echo ""

# Step 4: Train routing model
echo "Step 4/5: Training routing model..."
python3 train_routing_model.py --data-file combined_data.csv
echo ""

# Step 5: Evaluate routing model
echo "Step 5/5: Evaluating routing model..."
python3 evaluate_routing.py --data-file combined_data.csv --model-dir models
echo ""

echo "=========================================="
echo "All analysis complete!"
echo "=========================================="

