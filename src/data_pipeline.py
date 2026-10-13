import os
import random
import torch
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

# ==========================================
# Rule 6: Reproducibility & Environment Setup
# ==========================================
SEED = 42

def seed_everything(seed=42):
    """
    Sets deterministic seeds across python, numpy, and PyTorch.
    Comment: Ensures identical train/val splits and model initializations.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

seed_everything(SEED)

# ==========================================
# Task 3.1 Data Preparation (TREC-6)
# ==========================================
# Mapping IDs 0-5 to official class names and string labels for T5
# ID order: 0: ABBR, 1: ENTY, 2: DESC, 3: HUM, 4: LOC, 5: NUM
LABEL_MAPPING = {
    0: {"coarse": "ABBR", "t5_target": "abbreviation"},
    1: {"coarse": "ENTY", "t5_target": "entity"},
    2: {"coarse": "DESC", "t5_target": "description"},
    3: {"coarse": "HUM",  "t5_target": "human"},
    4: {"coarse": "LOC",  "t5_target": "location"},
    5: {"coarse": "NUM",  "t5_target": "number"}
}

def load_and_split_trec6():
    """
    Loads TREC-6 dataset and performs 90/10 stratified split on training data.
    Comment: Protocol requires 5,452 train rows (90% train / 10% val with seed 42)
    and 500 test rows. Preserves original row indices.
    """
    try:
        # Load official dataset from Hugging Face
        dataset = load_dataset("cogcomp/trec")
        train_raw = dataset["train"]
        test_raw = dataset["test"]
    except Exception as e:
        print(f"Standard load failed ({e}), using fallback handling.")
        # Fallback handling for CogComp/trec bytes issue if needed
        dataset = load_dataset("trec")
        train_raw = dataset["train"]
        test_raw = dataset["test"]

    # Convert to DataFrames
    df_train = pd.DataFrame({
        'text': train_raw['text'],
        'label': train_raw['coarse_label']
    })
    
    df_test = pd.DataFrame({
        'text': test_raw['text'],
        'label': test_raw['coarse_label']
    })

    # Rule 5: Rule out uninitialized state; explicitly verify dataset sizes
    assert len(df_train) == 5452, f"Expected 5452 training examples, got {len(df_train)}"
    assert len(df_test) == 500, f"Expected 500 test examples, got {len(df_test)}"

    # Stratified 90/10 split with seed 42
    indices = np.arange(len(df_train))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=0.10,
        random_state=SEED,
        stratify=df_train['label']
    )

    train_df = df_train.iloc[train_idx].reset_index(drop=True)
    val_df = df_train.iloc[val_idx].reset_index(drop=True)
    test_df = df_test.copy().reset_index(drop=True)

    print(f"Data successfully split:")
    print(f"  - Training samples:   {len(train_df)}")
    print(f"  - Validation samples: {len(val_df)}")
    print(f"  - Test samples:       {len(test_df)}")

    return train_df, val_df, test_df, train_idx, val_idx

if __name__ == "__main__":
    train_df, val_df, test_df, train_idx, val_idx = load_and_split_trec6()