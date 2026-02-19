"""
Download the Epstein Files dataset from HuggingFace.

This pulls ~20K raw text records from the 'teyler/epstein-files-20k' dataset
and saves them locally as JSON. You only need to run this once.

Output: data/raw.json
"""

import os
import json
from datasets import load_dataset

# We import paths from config so everything stays consistent
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, RAW_DATA_PATH, HF_DATASET_NAME

def main():
    # Create the data folder if it doesn't exist yet
    os.makedirs(DATA_DIR, exist_ok=True)

    # Pull the dataset from HuggingFace (this downloads it to a cache first)
    print(f"📥 Downloading dataset: {HF_DATASET_NAME}")
    dataset = load_dataset(HF_DATASET_NAME, split="train")
    print(f"✅ Downloaded {len(dataset)} records")

    # Convert to a plain list of dicts — easier to work with than HF's Dataset object
    rows = [row for row in dataset]

    # Save as JSON locally so we don't need to re-download every time
    with open(RAW_DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f"💾 Saved to {RAW_DATA_PATH}")
    print(f"📊 Sample record keys: {list(rows[0].keys())}")
    
if __name__ == "__main__":
    main()