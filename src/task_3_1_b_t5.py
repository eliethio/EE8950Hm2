import os
import time
import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, get_linear_schedule_with_warmup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

from data_pipeline import load_and_split_trec6, seed_everything, LABEL_MAPPING, LABEL_NAMES

# ==============================================================================
# Target Mapping Configuration
# ==============================================================================
TARGET_STR_TO_ID = {v["t5_target"]: k for k, v in LABEL_MAPPING.items()}
PREFIX = "classify question: "

# ==============================================================================
# Argument Parsing
# ==============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Task 3.1.B T5-small Text-to-Text Classification on TREC-6")
    parser.add_argument("--smoke-test", action="store_true", help="Run 1-batch CPU smoke test")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate for T5 fine-tuning")
    parser.add_argument("--max-input-len", type=int, default=64, help="Max input sequence length")
    parser.add_argument("--max-target-len", type=int, default=5, help="Max output generation length")
    return parser.parse_args()

# ==============================================================================
# PyTorch Dataset for T5
# ==============================================================================
class T5TRECDataset(Dataset):
    def __init__(self, df, tokenizer, max_input_len=64, max_target_len=5):
        self.texts = [PREFIX + str(t) for t in df['text'].tolist()]
        self.raw_labels = df['label'].tolist()
        self.target_strings = [LABEL_MAPPING[lbl]["t5_target"] for lbl in self.raw_labels]
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        target_str = self.target_strings[idx]
        label_id = self.raw_labels[idx]

        input_enc = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_input_len,
            return_tensors='pt'
        )

        target_enc = self.tokenizer(
            target_str,
            truncation=True,
            padding='max_length',
            max_length=self.max_target_len,
            return_tensors='pt'
        )

        labels = target_enc['input_ids'].flatten()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            'input_ids': input_enc['input_ids'].flatten(),
            'attention_mask': input_enc['attention_mask'].flatten(),
            'labels': labels,
            'target_str': target_str,
            'label_id': torch.tensor(label_id, dtype=torch.long)
        }

# ==============================================================================
# Evaluation via Generation (Greedy Decoding)
# ==============================================================================
def evaluate_t5(model, dataloader, tokenizer, device, max_new_tokens=5):
    model.eval()
    total_loss = 0.0
    all_preds_id = []
    all_labels_id = []
    generated_texts = []
    invalid_count = 0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            label_ids = batch['label_id'].numpy()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()

            gen_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False
            )

            decoded_strs = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

            for gen_str, true_id in zip(decoded_strs, label_ids):
                clean_str = gen_str.lower().strip()
                generated_texts.append(clean_str)
                total_samples += 1

                if clean_str in TARGET_STR_TO_ID:
                    pred_id = TARGET_STR_TO_ID[clean_str]
                else:
                    pred_id = -1
                    invalid_count += 1

                all_preds_id.append(pred_id)
                all_labels_id.append(true_id)

    avg_loss = total_loss / len(dataloader)
    invalid_rate = (invalid_count / total_samples) * 100.0 if total_samples > 0 else 0.0

    acc = accuracy_score(all_labels_id, all_preds_id)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels_id, all_preds_id, labels=list(range(6)), average='macro', zero_division=0
    )

    return avg_loss, acc, precision, recall, f1, invalid_rate, generated_texts, all_preds_id, all_labels_id

# ==============================================================================
# Main Pipeline
# ==============================================================================
def main():
    args = parse_args()
    seed_everything(42)

    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    if args.smoke_test:
        device = torch.device("cpu")
        print("Smoke-test flag activated: Forcing CPU execution.", flush=True)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60, flush=True)
    print(f"  ACTIVE HARDWARE DEVICE: {device}", flush=True)
    if device.type == "cuda":
        print(f"  GPU DEVICE NAME:      {torch.cuda.get_device_name(0)}", flush=True)
        print(f"  AVAILABLE MEMORY:     {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB", flush=True)
    else:
        print("  RUNNING ON CPU", flush=True)
    print("=" * 60, flush=True)

    train_df, val_df, test_df = load_and_split_trec6()

    model_name = "google-t5/t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\n" + "=" * 60, flush=True)
    print("      T5 ARCHITECTURE SPECIFICATIONS & AUDIT (Task 3.1.B)", flush=True)
    print("=" * 60, flush=True)
    print(f"  Model Backbone:      {model_name}", flush=True)
    print(f"  Total Parameters:    {total_params}", flush=True)
    print(f"  Trainable Params:    {trainable_params}", flush=True)
    print(f"  Frozen Parameters:   {total_params - trainable_params}", flush=True)
    print(f"  Input Task Prefix:   '{PREFIX}'", flush=True)
    print(f"  Target Classes:      {list(TARGET_STR_TO_ID.keys())}", flush=True)
    print("=" * 60, flush=True)

    train_dataset = T5TRECDataset(train_df, tokenizer, max_input_len=args.max_input_len, max_target_len=args.max_target_len)
    val_dataset = T5TRECDataset(val_df, tokenizer, max_input_len=args.max_input_len, max_target_len=args.max_target_len)
    test_dataset = T5TRECDataset(test_df, tokenizer, max_input_len=args.max_input_len, max_target_len=args.max_target_len)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    if args.smoke_test:
        print("\n" + "=" * 60, flush=True)
        print("         RUNNING CPU SMOKE TEST (1 BATCH & GRADIENT STEP)", flush=True)
        print("=" * 60, flush=True)
        batch = next(iter(train_loader))
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        model.train()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()

        grad_count = sum(1 for p in model.parameters() if p.grad is not None and p.grad.sum() != 0)

        print(f"Input IDs Shape:       {input_ids.shape}", flush=True)
        print(f"Attention Mask Shape:  {attention_mask.shape}", flush=True)
        print(f"Target Labels Shape:   {labels.shape}", flush=True)
        print(f"Computed Loss:         {loss.item():.4f}", flush=True)
        print(f"Non-zero Gradient Check: {grad_count} parameters received gradients.", flush=True)
        print("=" * 60, flush=True)
        print("      SMOKE TEST COMPLETED SUCCESSFULLY!", flush=True)
        print("=" * 60, flush=True)
        return

    optimizer = AdamW(model.parameters(), lr=args.lr, eps=1e-8)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    best_val_acc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_f1': [], 'val_invalid_rate': []}

    print("\nStarting T5 Text-to-Text Fine-Tuning...", flush=True)
    start_time = time.time()

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        val_loss, val_acc, val_prec, val_rec, val_f1, val_invalid, _, _, _ = evaluate_t5(
            model, val_loader, tokenizer, device, max_new_tokens=args.max_target_len
        )

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        history['val_invalid_rate'].append(val_invalid)

        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {avg_train_loss:.4g} | Val Loss: {val_loss:.4g} | "
              f"Val Acc: {val_acc*100:.4g}% | Val Macro F1: {val_f1:.4g} | Val Invalid Rate: {val_invalid:.4g}%", flush=True)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "checkpoints/t5_best_model.pt")

    total_time = time.time() - start_time
    print(f"\nTraining Complete in {total_time:.4g} seconds ({total_time/60:.4g} minutes).", flush=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    epochs_range = range(1, args.epochs + 1)

    ax1.plot(epochs_range, history['train_loss'], 'b-o', label='Train Loss')
    ax1.plot(epochs_range, history['val_loss'], 'r-s', label='Val Loss')
    ax1.set_title('T5 Training & Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs_range, [acc * 100 for acc in history['val_acc']], 'r-s', label='Val Acc (%)')
    ax2.set_title('T5 Validation Generation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig("results/figures/t5_learning_curves.png", dpi=300)
    plt.close()

    model.load_state_dict(torch.load("checkpoints/t5_best_model.pt"))
    test_loss, test_acc, test_prec, test_rec, test_f1, test_invalid, gen_texts, test_preds, test_labels = evaluate_t5(
        model, test_loader, tokenizer, device, max_new_tokens=args.max_target_len
    )

    print("\n" + "=" * 60, flush=True)
    print("         T5 FINAL TEST SET RESULTS", flush=True)
    print("=" * 60, flush=True)
    print(f"  Best Val Accuracy: {best_val_acc*100:.4g}%", flush=True)
    print(f"  Test Loss:         {test_loss:.4g}", flush=True)
    print(f"  Test Accuracy:     {test_acc*100:.4g}%", flush=True)
    print(f"  Macro Precision:   {test_prec:.4g}", flush=True)
    print(f"  Macro Recall:      {test_rec:.4g}", flush=True)
    print(f"  Macro F1-Score:    {test_f1:.4g}", flush=True)
    print(f"  Invalid Output Rate:{test_invalid:.4g}%", flush=True)
    print(f"  Training Time:     {total_time:.4g} sec", flush=True)
    print("=" * 60, flush=True)

    examples_output = ["============================================================",
                       "      TASK 3.1.B 5-EXAMPLE GENERATION INSPECTION",
                       "============================================================"]
    for i in range(min(5, len(test_df))):
        raw_text = test_df.iloc[i]['text']
        gt_id = test_df.iloc[i]['label']
        gt_str = LABEL_MAPPING[gt_id]["t5_target"]
        gen_str = gen_texts[i]
        
        examples_output.append(
            f"Example {i+1}:\n"
            f"  Input Question:   \"{raw_text}\"\n"
            f"  Generated Text:   \"{gen_str}\"\n"
            f"  Ground-Truth:     \"{gt_str}\" ({LABEL_NAMES[gt_id]})\n"
            f"  Exact Match:      {'YES' if gen_str == gt_str else 'NO'}\n"
        )
    examples_text = "\n".join(examples_output)
    print("\n" + examples_text, flush=True)

    with open("results/tables/t5_examples_inspection.txt", "w") as f:
        f.write(examples_text)

    t5_table_df = pd.DataFrame([{
        "Model": "T5-small",
        "Test Loss": test_loss,
        "Test Accuracy (%)": test_acc * 100,
        "Macro Precision": test_prec,
        "Macro Recall": test_rec,
        "Macro F1": test_f1,
        "Invalid Rate (%)": test_invalid,
        "Training Time (s)": total_time
    }])
    t5_table_df.to_csv("results/tables/t5_table_entry.csv", index=False)

if __name__ == "__main__":
    main()