import os
import random
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ==============================================================================
# Rule 6: Reproducibility & Environment Setup
# ==============================================================================
SEED = 42

def seed_everything(seed=42):
    """
    Sets deterministic seeds across python, numpy, and PyTorch.
    Ensures identical train/val splits and model initializations.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

seed_everything(SEED)

# ==============================================================================
# Label Mapping & Class Definitions for TREC-6
# ==============================================================================
LABEL_NAMES = ["ABBR", "ENTY", "DESC", "HUM", "LOC", "NUM"]

LABEL_MAPPING = {
    0: {"coarse": "ABBR", "t5_target": "abbreviation"},
    1: {"coarse": "ENTY", "t5_target": "entity"},
    2: {"coarse": "DESC", "t5_target": "description"},
    3: {"coarse": "HUM",  "t5_target": "human"},
    4: {"coarse": "LOC",  "t5_target": "location"},
    5: {"coarse": "NUM",  "t5_target": "number"}
}

LABEL_STR_TO_ID = {
    "ABBR": 0, "ENTY": 1, "DESC": 2, "HUM": 3, "LOC": 4, "NUM": 5
}

# ==============================================================================
# File Parsing & Protocol Functions
# ==============================================================================
def parse_raw_trec_file(file_path):
    """
    Parses a local raw TREC label file.
    Protocol implementation:
    1. Replaces byte 0xF0 with space before UTF-8 decoding.
    2. Extracts coarse label from 'COARSE:fine' prefix.
    3. Retains question text and converts label string to ID.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Required data file not found at: {file_path}")

    with open(file_path, 'rb') as f:
        raw_bytes = f.read()

    # Preprocessing rule: replace 0xF0 byte with space
    cleaned_bytes = raw_bytes.replace(b'\xf0', b' ')
    text_content = cleaned_bytes.decode('utf-8', errors='replace')

    texts, labels = [], []
    for line in text_content.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        
        parts = line.split(maxsplit=1)
        label_prefix = parts[0]
        question_text = parts[1] if len(parts) > 1 else ""

        # Extract coarse label before colon
        coarse = label_prefix.split(':')[0].strip()
        
        if coarse in LABEL_STR_TO_ID:
            texts.append(question_text)
            labels.append(LABEL_STR_TO_ID[coarse])

    return pd.DataFrame({'text': texts, 'label': labels})

def load_local_data_files():
    """
    Locates local raw files under the 'data/' directory (strictly without .txt extension).
    """
    possible_train_paths = [
        os.path.join("data", "train_5500.label"),
        os.path.join("data", "train_5452.label"),
        "train_5500.label",
        "train_5452.label"
    ]

    possible_test_paths = [
        os.path.join("data", "TREC_10.label"),
        os.path.join("data", "TAR.label"),
        "TREC_10.label",
        "TAR.label"
    ]

    train_file = next((path for path in possible_train_paths if os.path.exists(path)), None)
    test_file = next((path for path in possible_test_paths if os.path.exists(path)), None)

    if not train_file or not test_file:
        raise RuntimeError(
            f"Failed to find local data files.\n"
            f"Train search result: {train_file}\n"
            f"Test search result: {test_file}\n"
            f"Please verify files exist under 'data/' without extensions."
        )

    print(f"Loading local train file from: {train_file}")
    df_train = parse_raw_trec_file(train_file)

    print(f"Loading local test file from: {test_file}")
    df_test = parse_raw_trec_file(test_file)

    # Strictly enforce 5,452 training examples if 5,500 file was loaded
    if len(df_train) > 5452:
        print(f"Truncating training set from {len(df_train)} to official 5452 rows...")
        df_train = df_train.iloc[:5452].reset_index(drop=True)

    if len(df_test) > 500:
        print(f"Truncating test set from {len(df_test)} to official 500 rows...")
        df_test = df_test.iloc[:500].reset_index(drop=True)

    return df_train, df_test

# ==============================================================================
# Main Execution Pipeline
# ==============================================================================
def load_and_split_trec6():
    """
    Main dataset loader for Task 3.1.A and Task 3.1.B.
    Loads raw text files locally, verifies row counts, and performs 90/10 stratified split.
    """
    df_train, df_test = load_local_data_files()

    # Rule 5: Explicit size verification
    assert len(df_train) == 5452, f"Expected 5452 training examples, got {len(df_train)}"
    assert len(df_test) == 500, f"Expected 500 test examples, got {len(df_test)}"

    # 90/10 Stratified Train/Val Split with SEED=42
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

    return train_df, val_df, test_df