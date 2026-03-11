"""
Quantitative trading autoresearch training script. Single-GPU, single-file.
Uses MultiModalTransformer for LOB + Trade data classification.
Training runs for a fixed time budget of 5 minutes (wall clock).

Usage: uv run train.py
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import gc
import math
import time
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F

from prepare import (
    SEQ_LEN, TIME_BUDGET, NUM_CLASSES, MODEL_CONFIG_PATH,
    make_dataloader, evaluate_loss,
)
from Model.multi_modal_transformer import MultiModalTransformer

# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly, no CLI flags needed)
# ---------------------------------------------------------------------------

# Training
DEVICE_BATCH_SIZE = 64       # per-device batch size (reduce if OOM)
LEARNING_RATE = 3e-4         # peak learning rate for AdamW
WEIGHT_DECAY = 1e-4          # AdamW weight decay
ADAM_BETAS = (0.9, 0.999)    # Adam beta1, beta2
NUM_WORKERS = 4              # dataloader workers

# Schedule
WARMUP_RATIO = 0.0           # fraction of time budget for LR warmup
WARMDOWN_RATIO = 0.5         # fraction of time budget for LR warmdown
FINAL_LR_FRAC = 0.0          # final LR as fraction of initial

# Model (loaded from YAML config, but can override here)
USE_CONFIG_FILE = True       # if True, load from MODEL_CONFIG_PATH

# Inline model config (used when USE_CONFIG_FILE = False)
INLINE_CONFIG = dict(
    lob_config=dict(
        in_channels=4,
        base_channels=16,
        num_layers=3,
        time_strides=[5, 2, 2],
        level_strides=[2, 2, 2],
        kernel_sizes={'time': 3, 'level': 2},
        dropout=0.1,
    ),
    trade_config=dict(
        in_features=23,
        hidden_channels=[16, 16, 16],
        kernel_size=[5, 2, 2],
        time_stride=[5, 2, 2],
        dropout=0.1,
    ),
    fusion_config=dict(
        strategy='gated',
        d_model=32,
        use_layer_norm=False,
    ),
    transformer_config=dict(
        d_model=32,
        nhead=4,
        num_layers=1,
        dim_feedforward=128,
        dropout=0.3,
        positional_encoding='sinusoidal',
        max_seq_len=500,
    ),
    output_config=dict(
        num_classes=NUM_CLASSES,
        return_regression=False,
        pooling='last',
    ),
)

# ---------------------------------------------------------------------------
# LR Schedule (based on progress = training_time / TIME_BUDGET)
# ---------------------------------------------------------------------------

def get_lr_multiplier(progress):
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    elif progress < 1.0 - WARMDOWN_RATIO:
        return 1.0
    else:
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC

# ---------------------------------------------------------------------------
# Setup: model, optimizer, dataloader
# ---------------------------------------------------------------------------

t_start = time.time()
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.set_float32_matmul_precision("high")
device = torch.device("cuda")
autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

# Build model
if USE_CONFIG_FILE:
    print(f"Loading model config from: {MODEL_CONFIG_PATH}")
    model = MultiModalTransformer.from_config(MODEL_CONFIG_PATH)
else:
    print("Using inline model config")
    model = MultiModalTransformer(**INLINE_CONFIG)

model = model.to(device)

num_params = sum(p.numel() for p in model.parameters())
num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model parameters: {num_params:,} total, {num_trainable:,} trainable")
print(f"Model parameters (M): {num_params / 1e6:.2f}")

# Loss function
criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    betas=ADAM_BETAS,
)

# Scaler for mixed precision
scaler = torch.amp.GradScaler("cuda")

# Infinite training dataloader (recreated when exhausted)
train_loader_iter = None
train_epoch = 0

def next_batch():
    """Get next batch from dataloader, recreating when exhausted."""
    global train_loader_iter, train_epoch
    while True:
        if train_loader_iter is None:
            train_epoch += 1
            loader = make_dataloader(
                'train', DEVICE_BATCH_SIZE, seq_len=SEQ_LEN,
                samples_per_epoch=200_000, num_workers=NUM_WORKERS,
            )
            train_loader_iter = iter(loader)
        try:
            return next(train_loader_iter), train_epoch
        except StopIteration:
            train_loader_iter = None

print(f"Optimizer: AdamW (lr={LEARNING_RATE}, wd={WEIGHT_DECAY})")
print(f"Sequence length: {SEQ_LEN}")
print(f"Batch size: {DEVICE_BATCH_SIZE}")
print(f"Time budget: {TIME_BUDGET}s")

# Prefetch first batch
(lob, trade, labels), epoch = next_batch()

# ---------------------------------------------------------------------------
# Training loop — runs for fixed TIME_BUDGET
# ---------------------------------------------------------------------------

t_start_training = time.time()
step = 0
smooth_train_loss = 0
total_training_time = 0
total_samples_seen = 0

model.train()

while True:
    torch.cuda.synchronize()
    t0 = time.time()

    lob = lob.to(device, non_blocking=True)
    trade = trade.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)

    # Forward
    optimizer.zero_grad(set_to_none=True)
    with autocast_ctx:
        inputs = {'lob': lob, 'trade': trade}
        logits = model(inputs)
        loss = criterion(logits, labels)

    # Backward with gradient scaling
    train_loss_f = loss.detach().item()
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # LR schedule based on time progress
    progress = min(total_training_time / TIME_BUDGET, 1.0)
    lrm = get_lr_multiplier(progress)
    for pg in optimizer.param_groups:
        pg['lr'] = LEARNING_RATE * lrm

    scaler.step(optimizer)
    scaler.update()

    total_samples_seen += logits.size(0)

    # Prefetch next batch
    (lob, trade, labels), epoch = next_batch()

    # EMA smoothed loss
    ema_beta = 0.95
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
    debiased = smooth_train_loss / (1 - ema_beta ** (step + 1))

    torch.cuda.synchronize()
    t1 = time.time()
    dt = t1 - t0

    if step > 5:
        total_training_time += dt

    # Logging
    pct_done = 100 * progress
    remaining = max(0, TIME_BUDGET - total_training_time)

    print(
        f"\rstep {step:05d} ({pct_done:.1f}%) | loss: {debiased:.4f} | "
        f"lrm: {lrm:.2f} | dt: {dt*1000:.0f}ms | "
        f"epoch: {epoch} | remaining: {remaining:.0f}s    ",
        end="", flush=True,
    )

    # Fast fail
    if train_loss_f > 100:
        print("\nFAIL: loss exploded")
        exit(1)

    # GC management (Python's GC causes ~500ms stalls)
    if step == 0:
        gc.collect()
        gc.freeze()
        gc.disable()
    elif (step + 1) % 5000 == 0:
        gc.collect()

    step += 1

    # Time's up — but only stop after warmup steps so we don't count compilation
    if step > 10 and total_training_time >= TIME_BUDGET:
        break

print()  # newline after \r training log

# ---------------------------------------------------------------------------
# Final evaluation
# ---------------------------------------------------------------------------

print("Evaluating on validation set...")
model.eval()
with autocast_ctx:
    val_loss = evaluate_loss(model, device=device, batch_size=DEVICE_BATCH_SIZE,
                             seq_len=SEQ_LEN)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

t_end = time.time()
startup_time = t_start_training - t_start
peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

print("---")
print(f"val_loss:         {val_loss:.6f}")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
print(f"total_samples_M:  {total_samples_seen / 1e6:.1f}")
print(f"num_steps:        {step}")
print(f"num_params_M:     {num_params / 1e6:.2f}")
