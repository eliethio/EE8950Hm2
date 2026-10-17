import os
import time
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torchvision
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.models import vit_b_16, ViT_B_16_Weights
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def parse_args():
    parser = argparse.ArgumentParser(description="Task 3.3 ViT-B/16 Transfer Learning on CIFAR-10 and CIFAR-100")
    parser.add_argument("--epochs", type=int, default=10, help="Epochs for linear head training")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for head training")
    parser.add_argument("--extract-batch-size", type=int, default=64, help="Batch size for feature extraction")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for AdamW")
    parser.add_argument("--data-dir", type=str, default="./data", help="CIFAR data root directory")
    parser.add_argument("--smoke-test", action="store_true", help="Run quick 5-second sanity audit on mock data")
    return parser.parse_args()

def run_smoke_test(device):
    """
    Embedded Smoke Test Audit: Verifies model loading, frozen parameter status,
    feature extraction dimensions, and head gradient flow in under 5 seconds.
    """
    print("=" * 70)
    print("           TASK 3.3 PRETRAINED ViT SMOKE TEST AUDIT")
    print("=" * 70)
    print(f"  Torchvision Version: {torchvision.__version__}")
    print(f"  Active Device:       {device}")

    # 1. Model Loading & Weights
    print("\n[CHECK 1] Model Loading & Weights Verification...")
    weights = ViT_B_16_Weights.IMAGENET1K_V1
    transform = weights.transforms()
    model = vit_b_16(weights=weights).to(device)
    print("  ✓ ViT-B/16 loaded cleanly with IMAGENET1K_V1 weights.")

    # 2. Freeze Check
    print("\n[CHECK 2] Backbone Parameter Freezing Audit...")
    for p in model.parameters():
        p.requires_grad = False
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  ✓ Backbone Trainable Parameters: {trainable_params} (Expected: 0)")
    assert trainable_params == 0, "Backbone parameters are not frozen!"

    # 3. Feature Shape Verification
    print("\n[CHECK 3] Mock Feature Extraction Shape Audit...")
    model.eval()
    mock_imgs = torch.randn(4, 3, 224, 224).to(device)
    with torch.no_grad():
        x = model._process_input(mock_imgs)
        n = x.shape[0]
        batch_class_token = model.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = model.encoder(x)
        cls_features = x[:, 0]

    print(f"  Mock Input:       {list(mock_imgs.shape)}")
    print(f"  Extracted Output: {list(cls_features.shape)} (Expected: [4, 768])")
    assert cls_features.shape == (4, 768), "Feature shape mismatch!"

    # 4. Head Training Pass
    print("\n[CHECK 4] Linear Head Gradient Flow Test...")
    in_features = model.heads.head.in_features
    head_cifar10 = nn.Linear(in_features, 10).to(device)
    optimizer = torch.optim.AdamW(head_cifar10.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    mock_feats = torch.randn(8, 768).to(device)
    mock_labels = torch.randint(0, 10, (8,)).to(device)

    optimizer.zero_grad()
    logits = head_cifar10(mock_feats)
    loss = criterion(logits, mock_labels)
    loss.backward()
    optimizer.step()

    head_grads = all(p.grad is not None and torch.isfinite(p.grad).all() for p in head_cifar10.parameters())
    print(f"  Head Gradient Flow Check: {'✓ PASSED' if head_grads else 'FAILED'}")
    assert head_grads, "Gradients failed to flow through linear head!"

    print("\n" + "=" * 70)
    print("      ✓ ALL TASK 3.3 SMOKE TESTS PASSED SUCCESSFULLY!")
    print("=" * 70 + "\n")

def extract_vit_features(model, dataloader, device):
    """
    Passes images through frozen ViT backbone to extract the [CLS] token representation.
    """
    model.eval()
    all_features = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs = imgs.to(device)
            x = model._process_input(imgs)
            n = x.shape[0]
            batch_class_token = model.class_token.expand(n, -1, -1)
            x = torch.cat([batch_class_token, x], dim=1)
            x = model.encoder(x)
            cls_features = x[:, 0]
            
            all_features.append(cls_features.cpu())
            all_labels.append(labels)

    features = torch.cat(all_features, dim=0)
    labels = torch.cat(all_labels, dim=0)
    return features, labels

def train_and_eval_head(dataset_name, num_classes, in_features, train_feats, train_labels, 
                        val_feats, val_labels, test_feats, test_labels, args, device):
    """
    Trains a randomly initialized linear head on extracted frozen features.
    """
    print(f"\n" + "=" * 70)
    print(f"      TRAINING LINEAR HEAD FOR {dataset_name} ({num_classes} CLASSES)")
    print("=" * 70)

    train_ds = TensorDataset(train_feats, train_labels)
    val_ds = TensorDataset(val_feats, val_labels)
    test_ds = TensorDataset(test_feats, test_labels)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    head = nn.Linear(in_features, num_classes).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss()

    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    best_val_acc = 0.0
    
    start_head_time = time.time()

    for epoch in range(args.epochs):
        head.train()
        running_loss = 0.0
        preds_all, labels_all = [], []

        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            logits = head(x_b)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds_all.extend(torch.argmax(logits, dim=1).cpu().numpy())
            labels_all.extend(y_b.cpu().numpy())

        epoch_train_loss = running_loss / len(train_loader)
        epoch_train_acc = accuracy_score(labels_all, preds_all)

        # Validation
        head.eval()
        val_running_loss = 0.0
        val_preds_all, val_labels_all = [], []
        with torch.no_grad():
            for x_b, y_b in val_loader:
                x_b, y_b = x_b.to(device), y_b.to(device)
                logits = head(x_b)
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
            torch.save(head.state_dict(), f"checkpoints/vit_head_{dataset_name.lower().replace('-', '')}_best.pt")

    head_train_time = time.time() - start_head_time

    # Test Evaluation
    head.load_state_dict(torch.load(f"checkpoints/vit_head_{dataset_name.lower().replace('-', '')}_best.pt"))
    head.eval()
    test_preds_all, test_labels_all = [], []
    with torch.no_grad():
        for x_b, y_b in test_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            logits = head(x_b)
            test_preds_all.extend(torch.argmax(logits, dim=1).cpu().numpy())
            test_labels_all.extend(y_b.cpu().numpy())

    test_acc = accuracy_score(test_labels_all, test_preds_all)
    test_f1 = precision_recall_fscore_support(test_labels_all, test_preds_all, average='macro', zero_division=0)[2]

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accs": train_accs,
        "val_accs": val_accs,
        "best_val_acc": best_val_acc * 100,
        "test_acc": test_acc * 100,
        "test_f1": test_f1 * 100,
        "head_train_time": head_train_time
    }

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # Trigger smoke test if requested
    if args.smoke_test:
        run_smoke_test(device)
        return

    print("=" * 70)
    print(f"  Torchvision Version:        {torchvision.__version__}")
    print(f"  Selected Weight Enum:       ViT_B_16_Weights.IMAGENET1K_V1")
    print(f"  Active Device:              {device}")
    if torch.cuda.is_available():
        print(f"  Device Name:                {torch.cuda.get_device_name(0)}")
    print("=" * 70)

    # 1. Load Pretrained Model & Transforms
    weights = ViT_B_16_Weights.IMAGENET1K_V1
    transform = weights.transforms()
    print(f"\nOfficial ViT Transforms:\n{transform}")

    model = vit_b_16(weights=weights).to(device)

    # 2. Freeze Backbone & Replace Head
    in_features = model.heads.head.in_features
    for param in model.parameters():
        param.requires_grad = False

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params_before = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal Model Parameters:       {total_params:,}")
    print(f"Trainable Parameters (Frozen): {trainable_params_before}")

    results_summary = []
    datasets = [("CIFAR-10", CIFAR10, 10), ("CIFAR-100", CIFAR100, 100)]

    for dataset_name, dataset_cls, num_classes in datasets:
        print("\n" + "#" * 70)
        print(f"              PROCESSING DATASET: {dataset_name}")
        print("#" * 70)

        full_train = dataset_cls(root=args.data_dir, train=True, download=True, transform=transform)
        test_ds = dataset_cls(root=args.data_dir, train=False, download=True, transform=transform)

        generator = torch.Generator().manual_seed(42)
        train_ds, val_ds = torch.utils.data.random_split(full_train, [45000, 5000], generator=generator)

        train_loader = DataLoader(train_ds, batch_size=args.extract_batch_size, shuffle=False, num_workers=2)
        val_loader = DataLoader(val_ds, batch_size=args.extract_batch_size, shuffle=False, num_workers=2)
        test_loader = DataLoader(test_ds, batch_size=args.extract_batch_size, shuffle=False, num_workers=2)

        print(f"\nExtracting and caching frozen [CLS] features for {dataset_name}...")
        start_extract_time = time.time()
        
        train_feats, train_labels = extract_vit_features(model, train_loader, device)
        val_feats, val_labels = extract_vit_features(model, val_loader, device)
        
        train_val_extract_time = time.time() - start_extract_time
        test_feats, test_labels = extract_vit_features(model, test_loader, device)

        print(f"Feature Extraction Completed in {train_val_extract_time:.2f}s (Train + Val)")
        print(f"Features Shape: Train {train_feats.shape}, Val {val_feats.shape}, Test {test_feats.shape}")

        res = train_and_eval_head(
            dataset_name, num_classes, in_features,
            train_feats, train_labels, val_feats, val_labels,
            test_feats, test_labels, args, device
        )

        total_pipeline_time = train_val_extract_time + res["head_train_time"]

        print(f"\nSummary for {dataset_name}:")
        print(f"  Feature Extraction Time (Train+Val): {train_val_extract_time:.2f} s")
        print(f"  Head Training Time:                  {res['head_train_time']:.2f} s")
        print(f"  Total Training Pipeline Time:        {total_pipeline_time:.2f} s")
        print(f"  Best Val Accuracy:                   {res['best_val_acc']:.2f}%")
        print(f"  Test Accuracy:                       {res['test_acc']:.2f}%")
        print(f"  Test Macro-F1:                       {res['test_f1']:.2f}%")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        epochs_range = range(1, args.epochs + 1)

        ax1.plot(epochs_range, res["train_losses"], 'o-', label='Train Loss')
        ax1.plot(epochs_range, res["val_losses"], 's-', label='Val Loss')
        ax1.set_title(f'ViT-B/16 Head ({dataset_name}): Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Cross-Entropy Loss')
        ax1.legend()
        ax1.grid(True)

        ax2.plot(epochs_range, res["train_accs"], 'o-', label='Train Acc')
        ax2.plot(epochs_range, res["val_accs"], 's-', label='Val Acc')
        ax2.set_title(f'ViT-B/16 Head ({dataset_name}): Accuracy (%)')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        curve_path = f"results/figures/vit_transfer_{dataset_name.lower().replace('-', '')}_learning_curves.png"
        plt.savefig(curve_path, dpi=300)
        plt.close()

        results_summary.append({
            "Dataset": dataset_name,
            "Model": "ViT-B/16 (Frozen Backbone)",
            "Extracted Feat Time (s)": round(train_val_extract_time, 2),
            "Head Train Time (s)": round(res["head_train_time"], 2),
            "Total Pipeline Time (s)": round(total_pipeline_time, 2),
            "Best Val Acc (%)": round(res["best_val_acc"], 2),
            "Test Acc (%)": round(res["test_acc"], 2),
            "Macro-F1 (%)": round(res["test_f1"], 2)
        })

    summary_df = pd.DataFrame(results_summary)
    summary_df.to_csv("results/tables/vit_transfer_summary.csv", index=False)
    print("\n" + "=" * 70)
    print("           TASK 3.3 PRETRAINED VIT TRANSFER LEARNING SUMMARY")
    print("=" * 70)
    print(summary_df.to_string(index=False))
    print("=" * 70)

if __name__ == "__main__":
    main()