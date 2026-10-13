import torch
import torch.nn as nn
from task_3_2_custom_transformer import CustomTransformerClassifier

def run_verifications():
    print("=" * 70)
    print("      TASK 3.2.B: CORRECTNESS EVIDENCE & VERIFICATION AUDIT")
    print("=" * 70)

    vocab_size = 30522  # BERT tokenizer vocabulary size
    model = CustomTransformerClassifier(
        vocab_size=vocab_size,
        d_model=128,
        num_heads=4,
        d_ff=512,
        depth=2,
        num_classes=6,
        dropout=0.1
    )

    # --------------------------------------------------------------------------
    # CHECK 1: Forward-Pass Shapes & Divisibility Verification
    # --------------------------------------------------------------------------
    print("\n[CHECK 1] Shape & Divisibility Verification:")
    assert model.blocks[0].attn.d_model % model.blocks[0].attn.num_heads == 0, (
        "d_model must be divisible by H!"
    )
    d_h = model.blocks[0].attn.d_h
    print(f"  ✓ d_model (128) is perfectly divisible by H (4) -> d_h = {d_h}")

    # Combination 1: Batch 8, Length 32
    combo1_in = torch.randint(1, 1000, (8, 32))
    combo1_mask = torch.ones(8, 32)
    out1 = model(combo1_in, combo1_mask)
    print(f"  Combo 1 Input: [8, 32]   --> Output Logits Shape: {list(out1.shape)} (Expected: [8, 6])")
    assert out1.shape == (8, 6), "Combo 1 output shape mismatch!"

    # Combination 2: Batch 16, Length 64
    combo2_in = torch.randint(1, 1000, (16, 64))
    combo2_mask = torch.ones(16, 64)
    out2 = model(combo2_in, combo2_mask)
    print(f"  Combo 2 Input: [16, 64]  --> Output Logits Shape: {list(out2.shape)} (Expected: [16, 6])")
    assert out2.shape == (16, 6), "Combo 2 output shape mismatch!"

    # --------------------------------------------------------------------------
    # CHECK 2: Padding Invariance Audit (Dropout Disabled)
    # --------------------------------------------------------------------------
    print("\n[CHECK 2] Padding Invariance Audit (Dropout Disabled):")
    model.eval()  # Disable dropout for deterministic evaluation

    # Original sequence with 5 real tokens and 5 padded tokens (pad_id = 0)
    ids_padded = torch.tensor([[101, 2054, 2003, 1037, 102, 0, 0, 0, 0, 0]])
    mask_padded = torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]])

    # Change token IDs ONLY at masked padding positions (indices 5..9)
    ids_modified = torch.tensor([[101, 2054, 2003, 1037, 102, 999, 888, 777, 666, 555]])
    mask_modified = torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]])

    with torch.no_grad():
        logits_orig = model(ids_padded, mask_padded)
        logits_mod = model(ids_modified, mask_modified)

    diff = torch.max(torch.abs(logits_orig - logits_mod)).item()
    print(f"  Logits Max Absolute Difference: {diff:.8e}")
    assert diff < 1e-5, f"Padding leak detected! Logits changed by {diff:.8e}"
    print("  ✓ PASSED: Logits are strictly unchanged when altering padded token IDs (< 1e-5 tolerance).")

    # --------------------------------------------------------------------------
    # CHECK 3: Finite Loss, Gradient Flow & 32-Sample Overfitting Check
    # --------------------------------------------------------------------------
    print("\n[CHECK 3] Gradient Flow & Overfitting Tiny Batch:")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    tiny_ids = torch.randint(1, 1000, (32, 32))
    tiny_mask = torch.ones(32, 32)
    tiny_targets = torch.randint(0, 6, (32,))

    initial_loss = None
    for step in range(50):
        optimizer.zero_grad()
        logits = model(tiny_ids, tiny_mask)
        loss = criterion(logits, tiny_targets)
        
        if step == 0:
            initial_loss = loss.item()

        loss.backward()
        optimizer.step()

    # Check finite gradients across all active parameter tensors
    has_grads = all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in model.parameters()
        if p.requires_grad
    )

    print(f"  Initial Batch Loss (Step 0):   {initial_loss:.4f}")
    print(f"  Overfit Batch Loss (Step 50):  {loss.item():.4f}")
    print(f"  Finite Gradient Check:        {'✓ PASSED' if has_grads and torch.isfinite(loss) else 'FAILED'}")

    assert loss.item() < 0.1, f"Model failed to overfit tiny 32-sample batch! Loss = {loss.item():.4f}"
    print("  ✓ PASSED: Model overfits 32-sample batch cleanly (loss < 0.10).")
    print("=" * 70)


if __name__ == "__main__":
    run_verifications()