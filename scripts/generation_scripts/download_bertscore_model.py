"""
Downloads microsoft/deberta-xlarge-mnli for BertScore algorithmic metrics.
"""
import os
from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from dotenv import load_dotenv

load_dotenv()

script_dir = Path(__file__).parent
cache_dir = script_dir / ".cache" / "huggingface"
cache_dir.mkdir(parents=True, exist_ok=True)

os.environ['HF_HOME'] = str(cache_dir)
os.environ['TRANSFORMERS_CACHE'] = str(cache_dir / "transformers")

print(f"Downloading to: {cache_dir}\n")

# BertScore model for algorithmic metrics
print("microsoft/deberta-xlarge-mnli (~1.5GB)")
model_name = "microsoft/deberta-xlarge-mnli"
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    cache_dir=cache_dir,
    token=os.environ.get('HF_TOKEN')
)
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    cache_dir=cache_dir,
    token=os.environ.get('HF_TOKEN')
)
print("Downloaded\n")

print(f"Complete! Cache: {cache_dir}")
os.system(f"du -sh {cache_dir}")
