import os
import time
import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision.datasets import CIFAR10, CIFAR100
import torchvision.transforms as T
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def parse_args():
    parser = argparse.ArgumentParser(description="Task 3.4 Custom Vision Transformer from Scratch")
    parser.add_argument("--epochs", type=int, default=15, help="Epochs for scratch training")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for AdamW")
    parser.add_argument("--data-dir", type=str, default="./data", help="CIFAR root directory")
    parser.add_argument("--smoke-test", action="store_true", help="Run quick 5-second sanity audit")
    return parser.parse_args()

class ManualMultiHeadAttention(nn.Module):
    """
    Custom Multi-Head Self-Attention built without torch.nn.MultiheadAttention.
    """
    def __init__(self, embed_dim=128, num_heads=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, L, C = x.shape
        Q = self.q_proj(x).reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled Dot-Product Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attn_out = torch.matmul(attn_weights, V)

        attn_out = attn_out.transpose(1, 2).reshape(B, L, C)
        return self.out_proj(attn_out)

class CustomEncoderBlock(nn.Module):
    """
    Custom Transformer Encoder Block (Pre-LN Architecture)
    """
    def __init__(self, embed_dim=128, num_heads=4, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = ManualMultiHeadAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_chans=3, embed_dim=128):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2  # (32/4)^2 = 64
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        x = self.proj(x).flatten(2).transpose(1, 2)  # [B, 64, 128]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # [B, 65, 128]
        x = x + self.pos_embed
        return x

class CustomViT(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_chans=3, num_classes=10, 
                 embed_dim=128, num_heads=4, num_layers=1, mlp_ratio=4.0):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_chans, embed_dim)
        self.blocks = nn.ModuleList([
            CustomEncoderBlock(embed_dim, num_heads, mlp_ratio) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        cls_rep = x[:, 0]  # Extract [CLS] token representation
        logits = self.head(cls_rep)
        return logits

def run_smoke_test(device):
    print("=" * 70)
    print("             TASK 3.4 CUSTOM ViT SMOKE TEST AUDIT")
    print("=" * 70)
    print(f"  Active Device: {device}")

    model_1b = CustomViT(num_layers=1, num_classes=10).to(device)
    model_2b = CustomViT(num_layers=2, num_classes=100).to(device)

    p1 = sum(p.numel() for p in model_1b.parameters())
    p2 = sum(p.numel() for p in model_2b.parameters())

    print(f"\n[CHECK 1] Parameter Counts:")
    print(f"  ✓ Custom ViT (1 Block, 10 Classes):  {p1:,} Parameters")
    print(f"  ✓ Custom ViT (2 Blocks, 100 Classes): {p2:,} Parameters")

    mock_input = torch.randn(4, 3, 32, 32).to(device)
    out1 = model_1b(mock_input)
    out2 = model_2b(mock_input)

    print(f"\n[CHECK 2] Output Shapes:")
    print(f"  1-Block Output Shape: {list(out1.shape)} (Expected: [4, 10])")
    print(f"  2-Block Output Shape: {list(out2.shape)} (Expected: [4, 100])")
    assert out1.shape == (4, 10) and out2.shape == (4, 100), "Shape Mismatch!"

    optimizer = torch.optim.AdamW(model_1b.parameters(), lr=1e-3)
    loss = nn.CrossEntropyLoss()(out1, torch.tensor([0, 1, 2, 3], device=device))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    all_grads = all(p.grad is not None and torch.isfinite(p.grad).all() for p in model_1b.parameters())
    print(f"\n[CHECK 3] Gradient Flow Audit: {'✓ PASSED' if all_grads else 'FAILED'}")
    assert all_grads, "Gradient flow error!"

    print("\n" + "=" * 70)
    print("      ✓ ALL TASK 3.4 SMOKE TESTS PASSED SUCCESSFULLY!")
    print("=" * 70 + "\n")

def train_and_eval_custom_vit(model_name, dataset_name, num_blocks, num_classes, 
                               train_loader, val_loader, test_loader, args, device):
    print("\n" + "=" * 70)
    print(f"  TRAINING {model_name} ON {dataset_name} ({num_classes} CLASSES)")
    print("=" * 70)

    model = CustomViT(img_size=32, patch_size=4, num_classes=num_classes, 
                      embed_dim=128, num_heads=4, num_layers=num_blocks).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total Parameters:     {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss()

    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    best_val_acc = 0.0
    ckpt_path = f"checkpoints/custom_vit_{num_blocks}b_{dataset_name.lower().replace('-', '')}_best.pt"

    start_train_time = time.time()

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        preds_all, labels_all = [], []

        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            logits = model(x_b)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds_all.extend(torch.argmax(logits, dim=1).cpu().numpy())
            labels_all.extend(y_b.cpu().numpy())

        epoch_train_loss = running_loss / len(train_loader)
        epoch_train_acc = accuracy_score(labels_all, preds_all)

        model.eval()
        val_running_loss = 0.0
        val_preds_all, val_labels_all = [], []
        with torch.no_grad():
            for x_b, y_b in val_loader:
                x_b, y_b = x_b.to(device), y_b.to(device)
                logits = model(x_b)
                v_loss = criterion(logits, y_b)
                val_running_loss += v_loss.item()
                val_preds_all.extend(torch.argmax(logits, dim=1).cpu().numpy())
                val_labels_all.extend(y_b.cpu().numpy())

        epoch_val_loss = val_running_loss / len(val_loader)
        epoch_val_acc = accuracy_score(val_labels_all, val_preds_all)
        val_f1 = precision_recall_fscore_support(val_labels_all, val_preds_all, average='macro', zero_division=0)[2]

        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        train_accs.append(epoch_train_acc * 100)
        val_accs.append(epoch_val_acc * 100)

        print(f"Epoch {epoch+1:02d}/{args.epochs:02d} | Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc*100:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc*100:.2f}% | Val F1: {val_f1*100:.2f}%")

        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), ckpt_path)

    total_train_time = time.time() - start_train_time

    model.load_state_dict(torch.load(ckpt_path))
    model.eval()
    test_preds_all, test_labels_all = [], []
    with torch.no_grad():
        for x_b, y_b in test_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            logits = model(x_b)
            test_preds_all.extend(torch.argmax(logits, dim=1).cpu().numpy())
            test_labels_all.extend(y_b.cpu().numpy())

    test_acc = accuracy_score(test_labels_all, test_preds_all)
    test_f1 = precision_recall_fscore_support(test_labels_all, test_preds_all, average='macro', zero_division=0)[2]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    epochs_range = range(1, args.epochs + 1)

    ax1.plot(epochs_range, train_losses, 'o-', label='Train Loss')
    ax1.plot(epochs_range, val_losses, 's-', label='Val Loss')
    ax1.set_title(f'{model_name} ({dataset_name}): Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Cross-Entropy Loss')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs_range, train_accs, 'o-', label='Train Acc')
    ax2.plot(epochs_range, val_accs, 's-', label='Val Acc')
    ax2.set_title(f'{model_name} ({dataset_name}): Accuracy (%)')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(f"results/figures/custom_vit_{num_blocks}b_{dataset_name.lower().replace('-', '')}_curves.png", dpi=300)
    plt.close()

    return {
        "Dataset": dataset_name,
        "Model": model_name,
        "Extracted Feat Time (s)": "N/A",
        "Head Train Time (s)": "N/A",
        "Total Pipeline Time (s)": round(total_train_time, 2),
        "Best Val Acc (%)": round(best_val_acc * 100, 2),
        "Test Acc (%)": round(test_acc * 100, 2),
        "Macro-F1 (%)": round(test_f1 * 100, 2),
        "Total Params": total_params,
        "Trainable Params": trainable_params
    }

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    if args.smoke_test:
        run_smoke_test(device)
        return

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    results_summary = []
    datasets = [("CIFAR-10", CIFAR10, 10), ("CIFAR-100", CIFAR100, 100)]
    models_config = [("Custom ViT: 1 block", 1), ("Custom ViT: 2 blocks", 2)]

    for dataset_name, dataset_cls, num_classes in datasets:
        full_train = dataset_cls(root=args.data_dir, train=True, download=True, transform=transform)
        test_ds = dataset_cls(root=args.data_dir, train=False, download=True, transform=transform)

        generator = torch.Generator().manual_seed(42)
        train_ds, val_ds = torch.utils.data.random_split(full_train, [45000, 5000], generator=generator)

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

        for model_name, num_blocks in models_config:
            res = train_and_eval_custom_vit(
                model_name, dataset_name, num_blocks, num_classes,
                train_loader, val_loader, test_loader, args, device
            )
            results_summary.append(res)

    summary_df = pd.DataFrame(results_summary)
    summary_df.to_csv("results/tables/custom_vit_summary.csv", index=False)
    print("\n" + "=" * 70)
    print("                TASK 3.4 CUSTOM ViT SUMMARY")
    print("=" * 70)
    print(summary_df.to_string(index=False))
    print("=" * 70)

if __name__ == "__main__":
    main()