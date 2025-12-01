#!/bin/bash

# Create checkpoint directory
mkdir -p checkpoint

# Download Llama-3-8B-Instruct checkpoint
# Note: This is a large file (~7-8GB). Ensure you have space.
CHECKPOINT_DIR="checkpoint/msmarco-v1-passage-dev-small.original-bm25-1000.original-Meta-Llama-3-8B-Instruct-neg2-top1000"
if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "Downloading Llama-3-8B-Instruct checkpoint..."
    # Using wget with the Google Drive link might require confirmation or specific tools like gdown.
    # For now, I'll output instructions if manual download is easier, or try a direct wget if possible.
    # The link provided in the repo is a Google Drive view link. Direct download usually requires gdown.
    
    echo "Please download the checkpoint manually from: https://drive.google.com/file/d/15VvHS9jV-1J8RwfGJCmaJTNEgaP-5_qC/view?usp=share_link"
    echo "Unzip it and place the 'checkpoint-2675' folder inside '$CHECKPOINT_DIR'"
    
    # Alternatively, if gdown is installed:
    # pip install gdown
    # gdown 15VvHS9jV-1J8RwfGJCmaJTNEgaP-5_qC -O checkpoint/llama3_8b_instruct.zip
    # unzip checkpoint/llama3_8b_instruct.zip -d checkpoint/
else
    echo "Checkpoint directory already exists."
fi

# Create output directory
mkdir -p output

echo "Setup complete. Please ensure requirements are installed: pip install -r requirements.txt"
echo "Note: QPP-GenRE scripts are located in scripts/ideas/retrieval_tasks/qpp_genre_experiment/QPP-GenRE-original-scripts"
