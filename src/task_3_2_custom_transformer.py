import os
import math
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from data_pipeline import load_and_split_trec6, seed_everything, LABEL_NAMES

# ==============================================================================
# Argument Parsing
# ==============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Task 3.2 Custom Transformer Classifier on TREC-6")
    parser.add_argument("--smoke-test", action="store_true", help="Run 1-batch CPU smoke test")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--d-model", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--num-heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--d-ff", type=int, default=512, help="FFN hidden dimension")
    parser.add_argument("--depth", type=int, default=2, help="Number of encoder blocks")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout probability")
    parser.add_argument("--max-len", type=int, default=64, help="Max sequence length")
    return parser.parse_args()

# ==============================================================================
# Custom Modules
# ==============================================================================
class CustomMultiHeadAttention(nn.Module):
    """
    Explicit Multi-Head Self-Attention with Bidirectional Masking:
    Attention(Q, K, V) = softmax( (Q * K^T) / sqrt(d_h) + M ) * V
    """
    def __init__(self, d_model=128, num_heads=4, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_h = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, L, _ = x.size()

        Q = self.q_proj(x).view(B, L, self.num_heads, self.d_h).transpose(1, 2)
        K = self.k_proj(x).view(B, L, self.num_heads, self.d_h).transpose(1, 2)
        V = self.v_proj(x).view(B, L, self.num_heads, self.d_h).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_h)
        if mask is not None:
            scores = scores + mask

        attn_weights = self.dropout(F.softmax(scores, dim=-1))
        attn_out = torch.matmul(attn_weights, V).transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.out_proj(attn_out)

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model=128, num_heads=4, d_ff=512, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CustomMultiHeadAttention(d_model, num_heads, dropout)
        self.dropout1 = nn.Dropout(dropout)

        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        x = x + self.dropout1(self.attn(self.ln1(x), mask=mask))
        x = x + self.ffn(self.ln2(x))
        return x

class CustomTransformerClassifier(nn.Module):
    def __init__(self, vocab_size, d_model=128, num_heads=4, d_ff=512, depth=2, num_classes=6, max_len=512, dropout=0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_emb = SinusoidalPositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(d_model, num_heads, d_ff, dropout) for _ in range(depth)
        ])
        
        self.ln_final = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, input_ids, attention_mask=None):
        B, L = input_ids.size()
        x = self.dropout(self.pos_emb(self.token_emb(input_ids)))

        mask = (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9 if attention_mask is not None else None

        for block in self.blocks:
            x = block(x, mask=mask)

        x = self.ln_final(x)

        # Masked Mean Pooling
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).float()
            sum_embeddings = torch.sum(x * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            pooled = sum_embeddings / sum_mask
        else:
            pooled = x.mean(dim=1)

        return self.classifier(pooled)

# ==============================================================================
# Dataset Definition
# ==============================================================================
class TRECDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=64):
        self.texts = df['text'].tolist()
        self.labels = df['label'].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': enc['input_ids'].flatten(),
            'attention_mask': enc['attention_mask'].flatten(),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# ==============================================================================
# Main Pipeline
# ==============================================================================
def main():
    args = parse_args()
    seed_everything(42)

    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # 1. Device Hardware Details Printing
    if args.smoke_test:
        device = torch.device("cpu")
        device_name = "CPU (Smoke-Test Forced)"
        avail_mem = "N/A"
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            device_name = torch.cuda.get_device_name(0)
            avail_mem = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        else:
            device_name = "CPU"
            avail_mem = "N/A"

    print("=" * 60, flush=True)
    print(f"  ACTIVE HARDWARE DEVICE: {device.type.upper()}", flush=True)
    print(f"  DEVICE NAME:            {device_name}", flush=True)
    print(f"  AVAILABLE MEMORY:       {avail_mem}", flush=True)
    print("=" * 60, flush=True)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = CustomTransformerClassifier(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        depth=args.depth,
        dropout=args.dropout
    ).to(device)

    # 2. Print Explicit Architecture Dimensions Table
    d_h = args.d_model // args.num_heads
    print("\n" + "=" * 70, flush=True)
    print("      CUSTOM TRANSFORMER ENCODER ARCHITECTURE TABLE (Task 3.2.A)")
    print("=" * 70, flush=True)
    print(f"  Stage / Component               | Formula / Tensor Expression      | Tensor Shape (B={args.batch_size}, L={args.max_len})", flush=True)
    print("-" * 70, flush=True)
    print(f"  1. Input Token IDs              | Sequence Indices                 | [{args.batch_size}, {args.max_len}]", flush=True)
    print(f"  2. Token + Positional Embedding | Emb(x) + SinusoidalPE            | [{args.batch_size}, {args.max_len}, {args.d_model}]", flush=True)
    print(f"  3. Multi-Head Projections (Q/K/V)| Linear(d_model, d_model) -> View | [{args.batch_size}, {args.num_heads}, {args.max_len}, {d_h}]", flush=True)
    print(f"  4. Scaled Attention Scores      | Softmax( (Q*K^T)/sqrt(d_h) + M ) | [{args.batch_size}, {args.num_heads}, {args.max_len}, {args.max_len}]", flush=True)
    print(f"  5. Attention Output (Merged)    | (Scores * V) -> Out Linear       | [{args.batch_size}, {args.max_len}, {args.d_model}]", flush=True)
    print(f"  6. Feed-Forward Output (FFN)    | Linear(d_model, d_ff) -> GELU    | [{args.batch_size}, {args.max_len}, {args.d_model}]", flush=True)
    print(f"  7. Masked Mean Pooling          | sum(x * mask) / sum(mask)        | [{args.batch_size}, {args.d_model}]", flush=True)
    print(f"  8. Classification Logits        | Linear(d_model, num_classes)     | [{args.batch_size}, 6]", flush=True)
    print("=" * 70 + "\n", flush=True)

    # --------------------------------------------------------------------------
    # SMOKE TEST BLOCK
    # --------------------------------------------------------------------------
    if args.smoke_test:
        print("=" * 60, flush=True)
        print("         RUNNING CPU SMOKE TEST (1 BATCH & GRADIENT STEP)", flush=True)
        print("=" * 60, flush=True)
        dummy_in = torch.randint(1, 1000, (2, 16)).to(device)
        dummy_mask = torch.ones(2, 16).to(device)
        dummy_label = torch.tensor([0, 1]).to(device)

        model.train()
        logits = model(dummy_in, dummy_mask)
        loss = F.cross_entropy(logits, dummy_label)
        loss.backward()

        grad_count = sum(1 for p in model.parameters() if p.grad is not None and p.grad.sum() != 0)

        print(f"Input Shape:          {dummy_in.shape}", flush=True)
        print(f"Logits Output Shape:  {logits.shape} (Expected: [2, 6])", flush=True)
        print(f"Computed Loss:        {loss.item():.4f}", flush=True)
        print(f"Non-zero Gradients:   {grad_count} parameter tensors updated.", flush=True)
        print("=" * 60, flush=True)
        print("      SMOKE TEST COMPLETED SUCCESSFULLY!", flush=True)
        print("=" * 60, flush=True)
        return

    # --------------------------------------------------------------------------
    # FULL TRAINING PIPELINE & LEARNING CURVES
    # --------------------------------------------------------------------------
    train_df, val_df, test_df = load_and_split_trec6()

    train_loader = DataLoader(TRECDataset(train_df, tokenizer, args.max_len), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TRECDataset(val_df, tokenizer, args.max_len), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(TRECDataset(test_df, tokenizer, args.max_len), batch_size=args.batch_size, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss()

    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    best_val_acc = 0.0
    start_time = time.time()

    print("Starting Custom Transformer Encoder Training on TREC-6...", flush=True)
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        train_preds, train_labels = [], []

        for batch in train_loader:
            optimizer.zero_grad()
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            logits = model(ids, mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

        epoch_train_loss = running_loss / len(train_loader)
        epoch_train_acc = accuracy_score(train_labels, train_preds)

        # Validation Loop
        model.eval()
        val_running_loss = 0.0
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                ids = batch['input_ids'].to(device)
                mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                logits = model(ids, mask)
                v_loss = criterion(logits, labels)
                val_running_loss += v_loss.item()

                val_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        epoch_val_loss = val_running_loss / len(val_loader)
        epoch_val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = precision_recall_fscore_support(val_labels, val_preds, average='macro', zero_division=0)[2]

        # Track history for learning curves
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        train_accs.append(epoch_train_acc * 100)
        val_accs.append(epoch_val_acc * 100)

        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc*100:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc*100:.2f}% | Val Macro-F1: {val_f1*100:.2f}%", flush=True)

        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), "checkpoints/custom_transformer_best.pt")

    total_time = time.time() - start_time

    # 3. Plot and Save Learning Curves
    epochs_range = range(1, args.epochs + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(epochs_range, train_losses, 'o-', label='Train Loss')
    ax1.plot(epochs_range, val_losses, 's-', label='Validation Loss')
    ax1.set_title('Custom Transformer: Training & Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Cross-Entropy Loss')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs_range, train_accs, 'o-', label='Train Accuracy')
    ax2.plot(epochs_range, val_accs, 's-', label='Validation Accuracy')
    ax2.set_title('Custom Transformer: Training & Validation Accuracy (%)')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig("results/figures/custom_transformer_learning_curves.png", dpi=300)
    plt.close()
    print("Saved learning curve plots to: results/figures/custom_transformer_learning_curves.png", flush=True)

    # Final Test Evaluation
    model.load_state_dict(torch.load("checkpoints/custom_transformer_best.pt"))
    model.eval()
    test_preds, test_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            logits = model(ids, mask)
            test_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            test_labels.extend(batch['label'].numpy())

    test_acc = accuracy_score(test_labels, test_preds)
    test_f1 = precision_recall_fscore_support(test_labels, test_preds, average='macro', zero_division=0)[2]

    out_df = pd.DataFrame([{
        "Model": "Custom Transformer",
        "Best val. acc. (%)": round(best_val_acc * 100, 2),
        "Test acc. (%)": round(test_acc * 100, 2),
        "Macro-F1 (%)": round(test_f1 * 100, 2),
        "Train time (s)": round(total_time, 2)
    }])
    out_df.to_csv("results/tables/custom_transformer_table1_entry.csv", index=False)
    print(f"Saved Task 3.2 output to: results/tables/custom_transformer_table1_entry.csv", flush=True)

if __name__ == "__main__":
    main()