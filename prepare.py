"""
Data preparation and runtime utilities for quantitative trading autoresearch.

Loads LOB (Limit Order Book) and trade data from local numpy files.
Provides dataloader and evaluation utilities used by train.py.

Data format:
    - lob_data:    (T, 4, 10) — [ask_price, bid_price, ask_notional, bid_notional] x 10 levels
    - trade_data:  (T, 23)    — 23 aggregated trade features per 100ms window
    - labels_ret:  (T,)       — future 180s return for classification

Usage:
    python prepare.py          # verify data exists and print stats
"""

import os
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

SEQ_LEN = 3000               # sliding window size (100ms intervals = 200 seconds)
TIME_BUDGET = 300             # training time budget in seconds (5 minutes)
NUM_CLASSES = 3               # [Down(0), Stationary(1), Up(2)]
EVAL_SAMPLES = 10000          # number of samples for validation evaluation

# ---------------------------------------------------------------------------
# Data directories
# ---------------------------------------------------------------------------

TRAIN_DATA_DIR = '/root/autodl-tmp/train_data'
TEST_DATA_DIR = '/root/autodl-tmp/test_data'
MODEL_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Model', 'model_config.yaml')

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

_data_cache = {}
_threshold_cache = None


def load_data(data_dir):
    """
    Load LOB, trade, and label data from a directory.
    Uses memory-mapped files for efficiency with large datasets.
    Returns cached results on repeated calls.
    """
    if data_dir in _data_cache:
        return _data_cache[data_dir]

    lob_path = os.path.join(data_dir, 'lob_data.npy')
    trade_path = os.path.join(data_dir, 'trade_data.npy')
    labels_path = os.path.join(data_dir, 'trade_labels_ret.npy')

    for p in [lob_path, trade_path, labels_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Data file not found: {p}")

    lob = np.load(lob_path, mmap_mode='r')
    trade = np.load(trade_path, mmap_mode='r')
    labels_ret = np.load(labels_path, mmap_mode='r')

    _data_cache[data_dir] = (lob, trade, labels_ret)
    return lob, trade, labels_ret


def compute_label_thresholds(returns):
    """
    Compute classification thresholds from continuous returns using terciles.
    Bottom 1/3 → Down(0), Middle 1/3 → Stationary(1), Top 1/3 → Up(2).
    Thresholds are cached after first computation.
    """
    global _threshold_cache
    if _threshold_cache is not None:
        return _threshold_cache

    if hasattr(returns, 'shape') and len(returns) > 1_000_000:
        # Subsample for faster percentile computation on large arrays
        rng = np.random.default_rng(42)
        indices = rng.choice(len(returns), size=1_000_000, replace=False)
        sample = np.array(returns[indices])
    else:
        sample = np.array(returns)

    _threshold_cache = np.percentile(sample, [33.33, 66.67])
    return _threshold_cache


def returns_to_classes(returns, thresholds):
    """Convert continuous returns to 3-class labels."""
    labels = np.ones(len(returns), dtype=np.int64)  # default: Stationary(1)
    labels[returns < thresholds[0]] = 0  # Down
    labels[returns > thresholds[1]] = 2  # Up
    return labels


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class QuantDataset(Dataset):
    """
    Sliding window dataset for LOB + Trade multimodal data.

    Each sample is a window of SEQ_LEN consecutive timesteps.
    The label corresponds to the last timestep in the window.

    Output shapes per sample:
        lob:   (4, SEQ_LEN, 10)  — [C, T, L] for Conv2d
        trade: (23, SEQ_LEN)     — [F, T] for Conv1d
        label: scalar (int64)    — class index {0, 1, 2}
    """

    def __init__(self, lob_data, trade_data, labels, seq_len):
        self.lob = lob_data
        self.trade = trade_data
        self.labels = labels
        self.seq_len = seq_len
        self.n_samples = len(lob_data) - seq_len

    def __len__(self):
        return max(0, self.n_samples)

    def __getitem__(self, idx):
        end = idx + self.seq_len
        # LOB: (T, C, L) → (C, T, L)
        lob = torch.from_numpy(self.lob[idx:end].copy()).permute(1, 0, 2).float()
        # Trade: (T, F) → (F, T)
        trade = torch.from_numpy(self.trade[idx:end].copy()).t().contiguous().float()
        # Label at window end
        label = torch.tensor(self.labels[end - 1], dtype=torch.long)
        return lob, trade, label


# ---------------------------------------------------------------------------
# Dataloader factory
# ---------------------------------------------------------------------------

def make_dataloader(split, batch_size, seq_len=SEQ_LEN,
                    samples_per_epoch=None, num_workers=4):
    """
    Create a DataLoader for training or validation.

    Args:
        split: 'train' or 'val'
        batch_size: batch size
        seq_len: sliding window length
        samples_per_epoch: number of random samples per epoch (None = auto)
        num_workers: DataLoader workers

    Returns:
        DataLoader yielding (lob, trade, label) batches
    """
    assert split in ['train', 'val']
    data_dir = TRAIN_DATA_DIR if split == 'train' else TEST_DATA_DIR

    lob, trade, labels_ret = load_data(data_dir)

    # Always use training set thresholds for label generation
    train_lob, train_trade, train_ret = load_data(TRAIN_DATA_DIR)
    thresholds = compute_label_thresholds(train_ret)
    labels = returns_to_classes(np.array(labels_ret), thresholds)

    dataset = QuantDataset(lob, trade, labels, seq_len)

    if split == 'train':
        if samples_per_epoch is None:
            samples_per_epoch = min(len(dataset), 200_000)
        sampler = RandomSampler(dataset, replacement=True,
                                num_samples=samples_per_epoch)
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                          num_workers=num_workers, pin_memory=True,
                          drop_last=True, persistent_workers=num_workers > 0)
    else:
        eval_samples = min(EVAL_SAMPLES, len(dataset))
        sampler = RandomSampler(dataset, replacement=False,
                                num_samples=eval_samples)
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                          num_workers=num_workers, pin_memory=True,
                          drop_last=False, persistent_workers=num_workers > 0)


# ---------------------------------------------------------------------------
# Evaluation (DO NOT CHANGE — this is the fixed metric)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_loss(model, device='cuda', batch_size=64, seq_len=SEQ_LEN):
    """
    Evaluate model on validation set. Returns average cross-entropy loss.
    Lower is better.

    Uses a fixed number of random validation samples (EVAL_SAMPLES)
    for consistent and comparable evaluation across experiments.
    """
    model.eval()
    val_loader = make_dataloader('val', batch_size, seq_len=seq_len, num_workers=2)
    total_loss = 0.0
    total_samples = 0

    for lob, trade, labels in val_loader:
        lob = lob.to(device, non_blocking=True)
        trade = trade.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        inputs = {'lob': lob, 'trade': trade}
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(inputs)
        loss = F.cross_entropy(logits, labels)

        total_loss += loss.item() * labels.size(0)
        total_samples += labels.size(0)

    model.train()
    return total_loss / total_samples if total_samples > 0 else float('inf')


# ---------------------------------------------------------------------------
# Main — data verification
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Quantitative Trading Autoresearch — Data Verification")
    print("=" * 60)

    for name, data_dir in [("Train", TRAIN_DATA_DIR), ("Test", TEST_DATA_DIR)]:
        print(f"\n--- {name} data: {data_dir} ---")
        try:
            lob, trade, labels_ret = load_data(data_dir)
            print(f"  lob_data shape:    {lob.shape}  (T, C=4, L=10)")
            print(f"  trade_data shape:  {trade.shape}  (T, F=23)")
            print(f"  labels_ret shape:  {labels_ret.shape}  (T,)")
            print(f"  timesteps:         {len(lob):,}")

            ret_sample = np.array(labels_ret[:min(100000, len(labels_ret))])
            print(f"  returns range:     [{ret_sample.min():.6f}, {ret_sample.max():.6f}]")
            print(f"  returns mean:      {ret_sample.mean():.6f}")
            print(f"  returns std:       {ret_sample.std():.6f}")

            n_windows = len(lob) - SEQ_LEN
            print(f"  possible windows:  {n_windows:,} (seq_len={SEQ_LEN})")
        except FileNotFoundError as e:
            print(f"  NOT FOUND: {e}")

    # Label thresholds
    try:
        _, _, train_ret = load_data(TRAIN_DATA_DIR)
        thresholds = compute_label_thresholds(train_ret)
        print(f"\nLabel thresholds (33/67 percentile): {thresholds}")

        labels = returns_to_classes(np.array(train_ret[:min(100000, len(train_ret))]),
                                    thresholds)
        for c in range(NUM_CLASSES):
            pct = (labels == c).sum() / len(labels) * 100
            class_name = ['Down', 'Stationary', 'Up'][c]
            print(f"  Class {c} ({class_name}): {pct:.1f}%")
    except Exception as e:
        print(f"\nCould not compute thresholds: {e}")

    print(f"\nModel config: {MODEL_CONFIG_PATH}")
    print(f"Config exists: {os.path.exists(MODEL_CONFIG_PATH)}")
    print("\nDone!")
