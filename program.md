# autoresearch

This is an experiment to have the LLM do its own research on quantitative trading models.

## Domain

The task is **short-term crypto price direction prediction** (3-class classification: Down / Stationary / Up) using high-frequency limit order book (LOB) and trade data at 100ms resolution. The prediction horizon is 180 seconds.

Data modalities:
- **LOB data**: `(T, 4, 10)` — ask price, bid price, ask notional, bid notional across 10 levels.
- **Trade data**: `(T, 23)` — 23 aggregated trade features per 100ms window (volume, VWAP, order flow imbalance, etc.).
- **Labels**: `(T,)` — continuous future returns, discretized into 3 classes using training set terciles.

The baseline model is a **MultiModalTransformer** with:
1. LOB encoder (causal 2D CNN with level + time downsampling)
2. Trade encoder (causal 1D CNN)
3. Gated feature fusion
4. Transformer backbone
5. Classification head

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar10`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `program.md` — this file, experiment guide.
   - `prepare.py` — fixed constants, data loading, dataloader, evaluation. Do not modify.
   - `train.py` — the file you modify. Model configuration, optimizer, hyperparameters, training loop.
   - `Model/` — the model architecture directory. You may also modify files here.
   - `Model/model_config.yaml` — model configuration (architecture, dimensions, etc.).
4. **Verify data exists**: Check that `/root/autodl-tmp/train_data/` and `/root/autodl-tmp/test_data/` contain the numpy data files (`lob_data.npy`, `trade_data.npy`, `trade_labels_ret.npy`). If not, tell the human to prepare the data.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU. The training script runs for **3 epochs with a time budget of 10 minutes per epoch** (max 30 minutes total). You launch it simply as: `uv run train.py`.

**What you CAN do:**
- Modify `train.py` — this is the primary file you edit. Model configuration, optimizer, hyperparameters, training loop, batch size, etc.
- Modify `Model/` — architecture changes (encoders, fusion, backbone, pooling, etc.). There are additional modules available that aren't used in the baseline (event-driven encoders, hierarchical encoders, structured LOB encoders, sparse-aware trade encoders, various fusion strategies, advanced pooling methods). Explore and experiment with them.
- Modify `Model/model_config.yaml` — model dimensions, layer counts, dropout, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation, data loading, dataloader, and training constants.
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Modify the evaluation harness. The `evaluate_loss` function in `prepare.py` is the ground truth metric.

**The goal is simple: get the lowest val_loss.** Since each epoch has a time budget, you have room to experiment with model architecture and hyperparameters. Everything is fair game: change the architecture, the optimizer, the hyperparameters, the batch size, the model size. The only constraint is that the code runs without crashing and finishes within the time budget.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful val_loss gains, but it should not blow up dramatically.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Output format

Once the script finishes it prints a summary like this:

```
---
val_loss:         1.098612
training_seconds: 300.1
total_seconds:    325.9
peak_vram_mb:     4500.2
total_samples_M:  0.6
num_steps:        9375
num_params_M:     0.12
epochs_completed: 3
```

You can extract the key metric from the log file:

```
grep "^val_loss:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 5 columns:

```
commit	val_loss	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. val_loss achieved (e.g. 1.234567) — use 0.000000 for crashes
3. peak memory in GB, round to .1f (e.g. 4.5 — divide peak_vram_mb by 1024) — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	val_loss	memory_gb	status	description
a1b2c3d	1.098612	4.4	keep	baseline
b2c3d4e	1.082300	4.5	keep	increase d_model to 64
c3d4e5f	1.105000	4.4	discard	switch to cross_attention fusion
d4e5f6g	0.000000	0.0	crash	double transformer layers (OOM)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar10`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `train.py` and/or `Model/` files with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `uv run train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "^val_loss:\|^peak_vram_mb:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv
8. If val_loss improved (lower), you "advance" the branch, keeping the git commit
9. If val_loss is equal or worse, you git reset back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment should take ~30 minutes total (3 epochs × 10 min + a few minutes for startup and eval overhead). If a run exceeds 45 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**Research directions to explore**:
- Model architecture: try different encoders (EventDrivenTradeEncoder, HierarchicalTemporalEncoder, StructuredLOBEncoder, SparseAwareTradeEncoder), different fusion strategies (cross_attention, late_concat, gated), different pooling methods.
- Hyperparameters: d_model, nhead, num_layers, dim_feedforward, dropout, learning rate, batch size, weight decay.
- Training tricks: gradient accumulation, label smoothing, class weighting, data augmentation (time shift, noise injection), warm restarts.
- Sequence length: shorter windows for faster training, longer for more context.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — re-read the model code for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~30 minutes then you can run approx 12/hour, for a total of about 25 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!