"""
Quantitative trading autoresearch training script. Single-GPU, single-file.
Uses MultiModalTransformer for LOB + Trade data classification.

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
    SEQ_LEN, NUM_EPOCHS, EPOCH_TIME_BUDGET, TOTAL_TIME_BUDGET,
    NUM_CLASSES, MODEL_CONFIG_PATH,
    make_dataloader, evaluate_loss,
)
from Model.multi_modal_transformer import MultiModalTransformer

# ---------------------------------------------------------------------------
# MuonAdamW Optimizer (from original autoresearch, single GPU only)
# ---------------------------------------------------------------------------

polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]

@torch.compile(dynamic=False, fullgraph=True)
def adamw_step_fused(p, grad, exp_avg, exp_avg_sq, step_t, lr_t, beta1_t, beta2_t, eps_t, wd_t):
    p.mul_(1 - lr_t * wd_t)
    exp_avg.lerp_(grad, 1 - beta1_t)
    exp_avg_sq.lerp_(grad.square(), 1 - beta2_t)
    bias1 = 1 - beta1_t ** step_t
    bias2 = 1 - beta2_t ** step_t
    denom = (exp_avg_sq / bias2).sqrt() + eps_t
    step_size = lr_t / bias1
    p.add_(exp_avg / denom, alpha=-step_size)

@torch.compile(dynamic=False, fullgraph=True)
def muon_step_fused(stacked_grads, stacked_params, momentum_buffer, second_momentum_buffer,
                    momentum_t, lr_t, wd_t, beta2_t, ns_steps, red_dim):
    momentum = momentum_t.to(stacked_grads.dtype)
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)
    g = stacked_grads.lerp_(momentum_buffer, momentum)
    X = g.bfloat16()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)
    if g.size(-2) > g.size(-1):
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X.mT @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else:
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X
    g = X
    beta2 = beta2_t.to(g.dtype)
    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = g.size(red_dim)
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size
    v_norm = v_norm_sq.sqrt()
    second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2)
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
    g = g * final_scale.to(g.dtype)
    lr = lr_t.to(g.dtype)
    wd = wd_t.to(g.dtype)
    mask = (g * stacked_params) >= 0
    stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)


class MuonAdamW(torch.optim.Optimizer):
    """Combined optimizer: Muon for 2D matrix params, AdamW for others."""

    def __init__(self, param_groups):
        super().__init__(param_groups, defaults={})
        self._adamw_step_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta1_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_eps_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_momentum_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")

    def _step_adamw(self, group):
        for p in group['params']:
            if p.grad is None:
                continue
            grad = p.grad
            state = self.state[p]
            if not state:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)
            state['step'] += 1
            self._adamw_step_t.fill_(state['step'])
            self._adamw_lr_t.fill_(group['lr'])
            self._adamw_beta1_t.fill_(group['betas'][0])
            self._adamw_beta2_t.fill_(group['betas'][1])
            self._adamw_eps_t.fill_(group['eps'])
            self._adamw_wd_t.fill_(group['weight_decay'])
            adamw_step_fused(p, grad, state['exp_avg'], state['exp_avg_sq'],
                            self._adamw_step_t, self._adamw_lr_t, self._adamw_beta1_t,
                            self._adamw_beta2_t, self._adamw_eps_t, self._adamw_wd_t)

    def _step_muon(self, group):
        params = group['params']
        if not params:
            return
        p = params[0]
        state = self.state[p]
        num_params = len(params)
        shape, device, dtype = p.shape, p.device, p.dtype
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(num_params, *shape, dtype=dtype, device=device)
        if "second_momentum_buffer" not in state:
            state_shape = (num_params, shape[-2], 1) if shape[-2] >= shape[-1] else (num_params, 1, shape[-1])
            state["second_momentum_buffer"] = torch.zeros(state_shape, dtype=dtype, device=device)
        red_dim = -1 if shape[-2] >= shape[-1] else -2
        stacked_grads = torch.stack([p.grad for p in params])
        stacked_params = torch.stack(params)
        self._muon_momentum_t.fill_(group["momentum"])
        self._muon_beta2_t.fill_(group["beta2"] if group["beta2"] is not None else 0.0)
        self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1])**0.5)
        self._muon_wd_t.fill_(group["weight_decay"])
        muon_step_fused(stacked_grads, stacked_params,
                        state["momentum_buffer"], state["second_momentum_buffer"],
                        self._muon_momentum_t, self._muon_lr_t, self._muon_wd_t,
                        self._muon_beta2_t, group["ns_steps"], red_dim)
        torch._foreach_copy_(params, list(stacked_params.unbind(0)))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            if group['kind'] == 'adamw':
                self._step_adamw(group)
            elif group['kind'] == 'muon':
                self._step_muon(group)

# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly, no CLI flags needed)
# ---------------------------------------------------------------------------

# Training
DEVICE_BATCH_SIZE = 64       # per-device batch size (reduce if OOM)
SAMPLES_PER_EPOCH = 200_000  # random samples per epoch
NUM_WORKERS = 4              # dataloader workers

# Optimizer — Muon (for 2D weight matrices)
MATRIX_LR = 0.02             # Muon learning rate
MUON_NS_STEPS = 5            # polar express Newton-Schulz steps
MUON_WEIGHT_DECAY = 0.1      # cautious weight decay for Muon

# Optimizer — AdamW (for embeddings, biases, norms, conv, 1D params)
ADAMW_LR = 3e-4              # AdamW learning rate
ADAM_BETAS = (0.8, 0.95)     # Adam beta1, beta2

# Schedule
WARMUP_RATIO = 0.05          # fraction of total budget for LR warmup
WARMDOWN_RATIO = 0.3         # fraction of total budget for LR warmdown
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
# Schedules (all based on progress = training_time / TOTAL_TIME_BUDGET)
# ---------------------------------------------------------------------------

def get_lr_multiplier(progress):
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    elif progress < 1.0 - WARMDOWN_RATIO:
        return 1.0
    else:
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC

def get_muon_momentum(step):
    frac = min(step / 300, 1)
    return (1 - frac) * 0.85 + frac * 0.95

def get_weight_decay(progress):
    return MUON_WEIGHT_DECAY * (1 - progress)

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

# ---------------------------------------------------------------------------
# Optimizer setup: Muon for 2D matrices, AdamW for everything else
# ---------------------------------------------------------------------------

matrix_params = []
adamw_params = []

for name, p in model.named_parameters():
    if not p.requires_grad:
        continue
    if p.ndim == 2 and min(p.shape) > 1:
        matrix_params.append(p)
    else:
        adamw_params.append(p)

print(f"Muon params: {sum(p.numel() for p in matrix_params):,} "
      f"({len(matrix_params)} tensors)")
print(f"AdamW params: {sum(p.numel() for p in adamw_params):,} "
      f"({len(adamw_params)} tensors)")

param_groups = []

if adamw_params:
    param_groups.append(dict(
        kind='adamw', params=adamw_params, lr=ADAMW_LR,
        betas=ADAM_BETAS, eps=1e-8, weight_decay=0.0,
    ))

for shape in sorted({p.shape for p in matrix_params}):
    group_params = [p for p in matrix_params if p.shape == shape]
    param_groups.append(dict(
        kind='muon', params=group_params, lr=MATRIX_LR,
        momentum=0.95, ns_steps=MUON_NS_STEPS, beta2=0.95,
        weight_decay=MUON_WEIGHT_DECAY,
    ))

optimizer = MuonAdamW(param_groups)
for group in optimizer.param_groups:
    group["initial_lr"] = group["lr"]

print(f"Optimizer: MuonAdamW ({len(optimizer.param_groups)} param groups)")
print(f"Sequence length: {SEQ_LEN}")
print(f"Batch size: {DEVICE_BATCH_SIZE}")
print(f"Samples per epoch: {SAMPLES_PER_EPOCH:,}")
print(f"Epochs: {NUM_EPOCHS}")
print(f"Epoch time budget: {EPOCH_TIME_BUDGET}s")
print(f"Total time budget: {TOTAL_TIME_BUDGET}s")

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

t_start_training = time.time()
global_step = 0
smooth_train_loss = 0
total_training_time = 0
total_samples_seen = 0

for epoch in range(NUM_EPOCHS):
    print(f"\n{'='*60}")
    print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
    print(f"{'='*60}")

    train_loader = make_dataloader(
        'train', DEVICE_BATCH_SIZE, seq_len=SEQ_LEN,
        samples_per_epoch=SAMPLES_PER_EPOCH, num_workers=NUM_WORKERS,
    )

    t_epoch_start = time.time()
    epoch_loss_sum = 0
    epoch_correct = 0
    epoch_samples = 0
    epoch_steps = 0

    model.train()

    for step, (lob, trade, labels) in enumerate(train_loader):
        torch.cuda.synchronize()
        t0 = time.time()

        lob = lob.to(device, non_blocking=True)
        trade = trade.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Forward + backward
        with autocast_ctx:
            inputs = {'lob': lob, 'trade': trade}
            logits = model(inputs)
            loss = criterion(logits, labels)
        train_loss_f = loss.detach().item()
        loss.backward()

        # Schedules based on time progress
        progress = min(total_training_time / TOTAL_TIME_BUDGET, 1.0)
        lrm = get_lr_multiplier(progress)
        muon_momentum = get_muon_momentum(global_step)
        muon_wd = get_weight_decay(progress)
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * lrm
            if group['kind'] == 'muon':
                group["momentum"] = muon_momentum
                group["weight_decay"] = muon_wd

        optimizer.step()
        model.zero_grad(set_to_none=True)

        # Metrics
        batch_size = labels.size(0)
        preds = logits.argmax(dim=-1)
        correct = (preds == labels).sum().item()

        epoch_loss_sum += train_loss_f * batch_size
        epoch_correct += correct
        epoch_samples += batch_size
        total_samples_seen += batch_size

        # EMA smoothed loss
        ema_beta = 0.95
        smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
        debiased = smooth_train_loss / (1 - ema_beta ** (global_step + 1))

        torch.cuda.synchronize()
        t1 = time.time()
        dt = t1 - t0

        if global_step > 5:
            total_training_time += dt

        # Logging
        epoch_elapsed = time.time() - t_epoch_start
        acc = epoch_correct / epoch_samples * 100 if epoch_samples > 0 else 0
        remaining_epoch = max(0, EPOCH_TIME_BUDGET - epoch_elapsed)
        remaining_total = max(0, TOTAL_TIME_BUDGET - total_training_time)

        print(
            f"\r  step {epoch_steps:04d} | loss: {debiased:.4f} | "
            f"acc: {acc:.1f}% | lrm: {lrm:.2f} | "
            f"dt: {dt*1000:.0f}ms | "
            f"epoch_rem: {remaining_epoch:.0f}s | total_rem: {remaining_total:.0f}s    ",
            end="", flush=True,
        )

        # Fast fail
        if train_loss_f > 100:
            print("\nFAIL: loss exploded")
            exit(1)

        # GC management (Python's GC causes ~500ms stalls)
        if global_step == 0:
            gc.collect()
            gc.freeze()
            gc.disable()
        elif (global_step + 1) % 5000 == 0:
            gc.collect()

        global_step += 1
        epoch_steps += 1

        # Time budget check
        epoch_elapsed = time.time() - t_epoch_start
        if epoch_elapsed >= EPOCH_TIME_BUDGET:
            print(f"\n  [Epoch time budget reached: {epoch_elapsed:.1f}s]")
            break
        if total_training_time >= TOTAL_TIME_BUDGET:
            print(f"\n  [Total time budget reached: {total_training_time:.1f}s]")
            break

    # Epoch summary
    epoch_time = time.time() - t_epoch_start
    epoch_avg_loss = epoch_loss_sum / epoch_samples if epoch_samples > 0 else float('inf')
    epoch_acc = epoch_correct / epoch_samples * 100 if epoch_samples > 0 else 0
    print(f"\n  Epoch {epoch + 1} done: {epoch_steps} steps, "
          f"avg_loss={epoch_avg_loss:.4f}, acc={epoch_acc:.1f}%, "
          f"time={epoch_time:.1f}s")

    # Total time budget check
    if total_training_time >= TOTAL_TIME_BUDGET:
        print("[Total time budget reached, stopping]")
        break

print()

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
print(f"num_steps:        {global_step}")
print(f"num_params_M:     {num_params / 1e6:.2f}")
print(f"epochs_completed: {min(epoch + 1, NUM_EPOCHS)}")
