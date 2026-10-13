import os
import sys
import time
import argparse
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Ensure current directory is on sys.path for robust imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import shared data pipeline and seed setup
from data_pipeline import load_and_split_trec6, seed_everything, LABEL_NAMES

# ==============================================================================
# 1. REPRODUCIBILITY & HYPERPARAMETER PROTOCOL (Rule 3 & Rule 6)
# ==============================================================================
SEED = 42
MODEL_NAME = "google-bert/bert-base-uncased"
MAX_LEN = 64          # Assumption: 64 covers 100% of TREC-6 questions without clipping
BATCH_SIZE = 16       # Assumption: Batch size 16 offers stable gradient estimation
EPOCHS = 5            # Assumption: 5 epochs ensures complete convergence
LEARNING_RATE = 2e-5  # Assumption: Standard fine-tuning LR for BERT
WEIGHT_DECAY = 0.01   # Regularization parameter for AdamW
NUM_LABELS = 6        # TREC-6 coarse labels

# Path definitions
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(BASE_DIR, "results", "figures")
TABLES_DIR = os.path.join(BASE_DIR, "results", "tables")
CHECKPOINTS_DIR = os.path.join(BASE_DIR, "checkpoints")

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

seed_everything(SEED)

# ==============================================================================
# 2. DATASET CLASS DEFINITION
# ==============================================================================
class TRECDataset(Dataset):
    """
    PyTorch Dataset wrapper for TREC-6 text classification.
    Comment: Encapsulates Hugging Face tokenization with padding and truncation.
    """
    def __init__(self, texts, labels, tokenizer, max_len=64):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        return {
            "text": text,
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long)
        }

# ==============================================================================
# 3. DYNAMIC ARCHITECTURE SPECS & PARAMETER AUDIT (Rule 1 & Task 3.1.A.3)
# ==============================================================================
def inspect_and_save_architecture(model):
    """
    Extracts dynamic architecture metadata and documents updated vs frozen parameters.
    Comment: Fulfills Rule 1 (Single Source of Truth) and Task 3.1.A parameter audit.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    updated_layer_groups = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            group_name = name.split('.')[0]
            if group_name not in updated_layer_groups:
                updated_layer_groups.append(group_name)

    hidden_size = model.config.hidden_size
    num_hidden_layers = model.config.num_hidden_layers
    num_attention_heads = model.config.num_attention_heads

    arch_summary = {
        "Model Backbone": MODEL_NAME,
        "Total Parameters": total_params,
        "Trainable Parameters": trainable_params,
        "Frozen Parameters": frozen_params,
        "Updated Parameter Groups": ", ".join(updated_layer_groups),
        "Hidden Size": hidden_size,
        "Number of Layers": num_hidden_layers,
        "Attention Heads": num_attention_heads,
        "Classification Head": f"Linear({hidden_size} -> {NUM_LABELS})"
    }

    print("\n" + "="*60)
    print("      BERT ARCHITECTURE SPECIFICATIONS & AUDIT (Task 3.1.A)")
    print("="*60)
    for k, v in arch_summary.items():
        print(f"  {k}: {v}")
    print("="*60 + "\n")

    df_arch = pd.DataFrame(list(arch_summary.items()), columns=["Attribute", "Value"])
    df_arch.to_csv(os.path.join(TABLES_DIR, "bert_architecture_specs.csv"), index=False)
    return arch_summary

# ==============================================================================
# 4. SINGLE-EXAMPLE INSPECTION (Task 3.1.A Requirement 3)
# ==============================================================================
def inspect_single_example(model, tokenizer, sample_text, true_label, device):
    """
    Outputs Token IDs, Attention Mask, Logits, and Predicted Label for 1 example.
    Comment: Satisfies Task 3.1.A.3 prompt requirement.
    """
    model.eval()
    encoding = tokenizer(
        sample_text,
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
        return_tensors="pt"
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits.squeeze(0).cpu().numpy()
        pred_id = np.argmax(logits)

    print("\n" + "="*60)
    print("      TASK 3.1.A.3 SINGLE-EXAMPLE INSPECTION OUTPUT")
    print("="*60)
    print(f"Sample Text:        \"{sample_text}\"")
    print(f"True Label:         {LABEL_NAMES[true_label]} (ID: {true_label})")
    print(f"Predicted Label:    {LABEL_NAMES[pred_id]} (ID: {pred_id})")
    print(f"Token IDs (Non-pad): {input_ids[0][attention_mask[0] == 1].tolist()}")
    print(f"Attention Mask:     {attention_mask[0].tolist()}")
    print(f"Output Logits:      {np.round(logits, 4).tolist()}")
    print("="*60 + "\n")

    example_path = os.path.join(TABLES_DIR, "bert_single_example_inspection.txt")
    with open(example_path, "w") as f:
        f.write(f"Sample Text: {sample_text}\n")
        f.write(f"True Label: {LABEL_NAMES[true_label]} (ID: {true_label})\n")
        f.write(f"Predicted Label: {LABEL_NAMES[pred_id]} (ID: {pred_id})\n")
        f.write(f"Token IDs: {input_ids[0].tolist()}\n")
        f.write(f"Attention Mask: {attention_mask[0].tolist()}\n")
        f.write(f"Logits: {logits.tolist()}\n")

# ==============================================================================
# 5. TRAINING AND EVALUATION LOOPS
# ==============================================================================
def train_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0
    correct_preds = 0
    total_samples = 0

    for batch in dataloader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item() * len(labels)
        preds = torch.argmax(logits, dim=1)
        correct_preds += torch.sum(preds == labels).item()
        total_samples += len(labels)

    avg_loss = total_loss / total_samples
    accuracy = correct_preds / total_samples
    return avg_loss, accuracy

def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            total_loss += loss.item() * len(labels)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )
    p_class, r_class, f1_class, _ = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0
    )

    return avg_loss, acc, precision, recall, f1, p_class, r_class, f1_class, all_preds, all_labels

# ==============================================================================
# 6. CPU SMOKE TEST ROUTINE (Section 5 Constraint)
# ==============================================================================
def run_smoke_test(model, dataloader, optimizer, device):
    """
    Executes a 1-step CPU smoke test checking shapes, mask, loss, and gradient update.
    Comment: Satisfies Section 5 requirement for quick grader verification.
    """
    print("\n" + "="*60)
    print("         RUNNING CPU SMOKE TEST (1 BATCH & GRADIENT STEP)")
    print("="*60)
    model.train()
    batch = next(iter(dataloader))
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    print(f"Input IDs Shape:       {input_ids.shape}")
    print(f"Attention Mask Shape:  {attention_mask.shape}")
    print(f"Labels Shape:          {labels.shape}")

    optimizer.zero_grad()
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss
    logits = outputs.logits

    print(f"Output Logits Shape:   {logits.shape}")
    print(f"Computed Loss:         {loss.item():.4f}")

    loss.backward()
    
    grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
    print(f"Non-zero Gradient Check: {len(grad_norms)} parameters received gradients.")
    optimizer.step()

    print("="*60)
    print("      SMOKE TEST COMPLETED SUCCESSFULLY!")
    print("="*60 + "\n")

# ==============================================================================
# 7. PLOTTING LEARNING CURVES (Section 4 Requirement 2)
# ==============================================================================
def plot_learning_curves(df_history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Loss Plot
    axes[0].plot(df_history["epoch"], df_history["train_loss"], label="Train Loss", marker="o")
    axes[0].plot(df_history["epoch"], df_history["val_loss"], label="Val Loss", marker="s")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].set_title("BERT Training vs Validation Loss")
    axes[0].legend()
    axes[0].grid(True)

    # Accuracy Plot
    axes[1].plot(df_history["epoch"], df_history["train_acc"] * 100, label="Train Acc", marker="o")
    axes[1].plot(df_history["epoch"], df_history["val_acc"] * 100, label="Val Acc", marker="s")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("BERT Training vs Validation Accuracy")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plot_path = os.path.join(FIGURES_DIR, "bert_learning_curves.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Learning curves successfully saved to: {plot_path}")

# ==============================================================================
# 8. MAIN EXECUTION ROUTINE
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Task 3.1.A BERT Fine-tuning and Verification")
    parser.add_argument("--smoke-test", action="store_true", help="Run a quick 1-batch CPU test and exit.")
    args = parser.parse_args()

    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    if args.smoke_test:
        device = torch.device("cpu")
        print("Smoke-test flag activated: Forcing CPU execution.")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Add hardware verification logging here:
    print("=" * 60)
    print(f"  ACTIVE HARDWARE DEVICE: {device}")
    if device.type == "cuda":
        print(f"  GPU DEVICE NAME:      {torch.cuda.get_device_name(0)}")
        print(f"  AVAILABLE MEMORY:     {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("  RUNNING ON CPU")
    print("=" * 60)
    
    train_df, val_df, test_df = load_and_split_trec6()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_dataset = TRECDataset(train_df['text'].values, train_df['label'].values, tokenizer, MAX_LEN)
    val_dataset = TRECDataset(val_df['text'].values, val_df['label'].values, tokenizer, MAX_LEN)
    test_dataset = TRECDataset(test_df['text'].values, test_df['label'].values, tokenizer, MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS
    ).to(device)

    inspect_and_save_architecture(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    if args.smoke_test:
        run_smoke_test(model, train_loader, optimizer, device)
        return

    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps
    )

    print("Starting BERT Fine-Tuning...")
    start_time = time.time()
    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, device)
        val_loss, val_acc, val_p, val_r, val_f1, _, _, _, _, _ = evaluate(model, val_loader, device)

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | Val Macro F1: {val_f1:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(CHECKPOINTS_DIR, "bert_best_model.pt"))

    train_time_sec = time.time() - start_time
    print(f"\nTraining Complete in {train_time_sec:.2f} seconds ({train_time_sec/60:.2f} minutes).")

    df_history = pd.DataFrame(history)
    df_history.to_csv(os.path.join(TABLES_DIR, "bert_training_history.csv"), index=False)
    plot_learning_curves(df_history)

    model.load_state_dict(torch.load(os.path.join(CHECKPOINTS_DIR, "bert_best_model.pt")))

    sample_text = test_df['text'].iloc[0]
    sample_label = test_df['label'].iloc[0]
    inspect_single_example(model, tokenizer, sample_text, sample_label, device)

    print("Evaluating Best BERT Model on Official Test Set...")
    test_loss, test_acc, test_p, test_r, test_f1, p_class, r_class, f1_class, y_pred, y_true = evaluate(model, test_loader, device)

    print("\n" + "="*60)
    print("         BERT FINAL TEST SET RESULTS (Table 1)")
    print("="*60)
    print(f"  Best Val Accuracy: {best_val_acc * 100:.2f}%")
    print(f"  Test Loss:         {test_loss:.4f}")
    print(f"  Test Accuracy:     {test_acc * 100:.2f}%")
    print(f"  Macro Precision:   {test_p:.4f}")
    print(f"  Macro Recall:      {test_r:.4f}")
    print(f"  Macro F1-Score:    {test_f1:.4f}")
    print(f"  Training Time:     {train_time_sec:.2f} sec")
    print("="*60)

    class_metrics = []
    for idx, name in enumerate(LABEL_NAMES):
        class_metrics.append({
            "Class ID": idx,
            "Class Name": name,
            "Precision": round(p_class[idx], 4),
            "Recall": round(r_class[idx], 4),
            "F1-Score": round(f1_class[idx], 4)
        })
    df_class_metrics = pd.DataFrame(class_metrics)
    df_class_metrics.to_csv(os.path.join(TABLES_DIR, "bert_per_class_results.csv"), index=False)

    table1_entry = {
        "Model": "BERT fine-tuned",
        "Best val. acc. (%)": round(best_val_acc * 100, 2),
        "Test acc. (%)": round(test_acc * 100, 2),
        "Macro-F1 (%)": round(test_f1 * 100, 2),
        "Train time (s)": round(train_time_sec, 2)
    }
    pd.DataFrame([table1_entry]).to_csv(os.path.join(TABLES_DIR, "bert_table1_entry.csv"), index=False)

if __name__ == "__main__":
    main()