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

Each experiment runs on a single GPU. The training script runs for a **fixed time budget of 5 minutes** (wall clock training time, excluding startup/compilation). You launch it simply as: `uv run train.py`.

**What you CAN do:**
- Modify `train.py` — this is the primary file you edit. Model configuration, optimizer, hyperparameters, training loop, batch size, etc.
- Modify `Model/` — architecture changes (encoders, fusion, backbone, pooling, etc.). There are additional modules available that aren't used in the baseline (event-driven encoders, hierarchical encoders, structured LOB encoders, sparse-aware trade encoders, various fusion strategies, advanced pooling methods). Explore and experiment with them.
- Modify `Model/model_config.yaml` — model dimensions, layer counts, dropout, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation, data loading, dataloader, and training constants (time budget, sequence length, etc).
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Modify the evaluation harness. The `evaluate_loss` function in `prepare.py` is the ground truth metric.

**The goal is simple: get the lowest val_loss.** Since the time budget is fixed, you don't need to worry about training time — it's always 5 minutes. The only constraint is that the code runs without crashing and finishes within the time budget.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful val_loss gains, but it should not blow up dramatically.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Research philosophy — READ THIS CAREFULLY

You are not a hyperparameter tuner. You are a **quantitative researcher**. The biggest gains in this task will come from understanding the data and designing better model architectures, NOT from tweaking learning rates, dropout, or d_model by small amounts.

**DO NOT waste runs on pure hyperparameter sweeps.** Changing `lr` from 3e-4 to 1e-4, or `dropout` from 0.1 to 0.2, or `d_model` from 32 to 64 — these are the lowest-value experiments you can run. They rarely produce meaningful improvements and burn precious experiment slots. If you find yourself about to run a "change one number" experiment, stop and think harder.

**Priority hierarchy** (spend your experiments in roughly this order):

1. **Data understanding and feature engineering (HIGHEST VALUE)**
   Think deeply about what the raw data actually represents. LOB data has spatial structure across levels — are we exploiting that? Trade data has 23 features — what are they, how do they relate to each other, and are some more predictive than others? Can you engineer better input representations? Consider:
   - Is raw price the right input, or should we use returns, log-returns, or spreads?
   - Does the model see order flow imbalance directly, or could we construct it from raw features?
   - Are there cross-level or cross-feature interactions the encoder should be designed to capture?
   - Can we give the model explicit microstructure signals (mid-price changes, volume-weighted spreads, trade arrival rates)?

2. **Architecture design (HIGH VALUE)**
   The baseline uses a simple CNN encoder + Transformer backbone. There are many more powerful ways to model this data:
   - Does trade data even need a CNN encoder? Maybe a simple linear projection or MLP per-timestep is better. Maybe no encoder at all — just project features directly.
   - LOB has a natural 2D structure (level × channel). Is a 2D CNN the best way to capture level interactions, or would attention across levels work better?
   - The fusion of LOB and trade modalities is critical. Gated fusion is just one strategy. Think about cross-attention between modalities, or late fusion, or hierarchical approaches.
   - The Transformer backbone is generic. Would a different temporal model (e.g. Mamba, pure convolutions, or a hybrid) be better for this data's characteristics?
   - The classification head is trivial (last-step pooling + linear). Is there a better aggregation strategy?
   - Think about what inductive biases are appropriate for financial time series — locality, causality, scale invariance.

3. **Training methodology (MEDIUM VALUE)**
   - Loss function design: label smoothing, focal loss for class imbalance, auxiliary losses.
   - Data augmentation specific to financial data (time shifts, noise injection to prices).
   - Curriculum learning, sample weighting.

4. **Hyperparameter tuning (LOW VALUE — do this last, sparingly)**
   Only tune hyperparameters after you've exhausted architectural ideas. And when you do, make large moves (2x or 0.5x), not small ones.

**Think like a quant researcher, not an ML engineer.** The person who designed these features made choices. Understand those choices. Challenge them. The model architecture should reflect your understanding of market microstructure — how limit order books work, how trades happen, what signals predict price movement. If your experiment idea doesn't come from a hypothesis about the data or the task, it's probably not a good experiment.

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
```

Note that the script is configured to always stop after 5 minutes, so depending on the computing platform of this computer the numbers might look different. You can extract the key metric from the log file:

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
2. **Think first, code second.** Before touching code, formulate a clear hypothesis: "I believe X will improve val_loss because Y." The hypothesis should be grounded in data understanding or architectural reasoning, not "maybe a bigger number is better." Then implement the change in `train.py` and/or `Model/` files.
3. git commit
4. Run the experiment: `uv run train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "^val_loss:\|^peak_vram_mb:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv
8. If val_loss improved (lower), you "advance" the branch, keeping the git commit
9. If val_loss is equal or worse, you git reset back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment should take ~5 minutes total (+ a few seconds for startup and eval overhead). If a run exceeds 10 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**Concrete ideas to get you started** (but don't limit yourself to these — the best ideas will come from your own analysis of the data and the model):

- **Rethink the trade encoder**: The baseline applies a 1D CNN to 23 features. But are all 23 features equally useful? Maybe some are redundant or noisy. Try: (a) no encoder — just a linear layer per timestep, (b) feature-grouped encoding — different groups of trade features processed differently, (c) attention across features within each timestep.
- **Rethink the LOB encoder**: LOB data has rich spatial structure. Bid/ask asymmetry, depth profiles, spread dynamics. The 2D CNN treats all this uniformly. Try: (a) separate bid/ask pathways that interact later, (b) explicit spread/imbalance computation before encoding, (c) attention across levels.
- **Rethink fusion**: LOB and trade data capture different aspects of market activity. Maybe the model should learn when to trust each modality. Try: (a) cross-attention where trade queries attend to LOB keys, (b) modality-specific predictions that are ensembled, (c) conditional gating based on market regime.
- **Rethink the backbone**: Is a Transformer the right temporal model here? Financial time series have specific properties: non-stationarity, regime changes, varying volatility. Try: (a) replacing Transformer with convolutions or Mamba, (b) multi-scale temporal modeling, (c) removing the backbone entirely and relying on the encoders.
- **Rethink the input representation**: Instead of feeding raw features, engineer more meaningful inputs. Try: (a) log-returns instead of raw prices, (b) z-score normalization per feature, (c) relative features (this level vs best level), (d) handcrafted microstructure features as additional inputs.
- **Rethink the prediction head**: The baseline takes the last timestep and classifies. But the prediction-relevant signal might be distributed across time. Try: (a) attention pooling, (b) multi-scale pooling, (c) auxiliary predictions at intermediate timesteps.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — re-read the model code and the data format for new angles, think about what market microstructure signals the model is missing, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~5 minutes then you can run approx 12/hour, for a total of about 100 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!
