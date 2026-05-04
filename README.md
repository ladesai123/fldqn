# FLDQN — Federated Deep Q‑Network for Fog Task Scheduling

## Overview
This repository contains a research-focused implementation of **FLDQN** (Federated Deep Q‑Network) for task scheduling in a fog computing environment. It simulates multiple fog nodes, each with its own DQN scheduler, and periodically aggregates model weights via federated learning. The script also includes baseline comparisons and an enhanced variant (**E‑FLDQN**) with additional innovations.

## What it does
- Loads GoCJ task datasets and converts MI values into scheduling features.
- Uses K‑Means to assign task priorities.
- Simulates a fog environment with multiple nodes and VMs.
- Trains FLDQN agents with federated weight aggregation.
- Compares against baselines (Standalone DQN, GGCN, LSTM).
- Implements E‑FLDQN (dynamic K, prioritized replay, SLA‑weighted aggregation, priority‑scaled rewards).
- Produces plots and printed metric tables for evaluation.

## Key components
- **Dataset loading**: GoCJ MI values → task features.
- **Priority clustering**: K‑Means on execution time & deadline.
- **Environment**: VM capacities, queueing, SLA checks, rewards.
- **FLDQN**: per‑node DQN agents + federated aggregation.
- **Baselines**: DQN (no FL), GGCN heuristic, LSTM scheduler.
- **E‑FLDQN**: enhanced federated DQN with four improvements.

## Repository structure
```
.
├── fldqn_main_review_2.py   # Main implementation and experiments
├── explanation.md           # Detailed algorithm walkthrough
├── GoCJ_Dataset_100.txt     # GoCJ dataset (100 tasks)
├── GoCJ_Dataset_500.txt     # GoCJ dataset (500 tasks)
├── GoCJ_Dataset_1000 (2).txt # GoCJ dataset (1000 tasks)
└── README.md
```

## Requirements
This project is a single Python script. Typical dependencies include:
- Python 3.8+
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- tensorflow

Install dependencies (example):
```
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow
```

## Usage
1. **Update dataset paths** in `fldqn_main_review_2.py` (the script currently points to Colab paths).  
   Replace the following with local paths, for example:
   - `./GoCJ_Dataset_100.txt`
   - `./GoCJ_Dataset_500.txt`
   - `./GoCJ_Dataset_1000 (2).txt`

2. **Run the script**:
```
python fldqn_main_review_2.py
```

3. **Adjust runtime** (optional):  
   Change `NUM_EPISODES` in the script to control training length.

## Outputs
The script prints comparison tables and generates plots such as:
- `comparison_<dataset>.png`
- `5alg_<dataset>.png`

These visualize makespan, energy, SLA violations, rejection rate, and throughput across algorithms.

## Notes & limitations
- The script is a direct conversion from a Colab notebook and runs end‑to‑end experiments by default.
- Training can be compute‑intensive; reduce `NUM_EPISODES` for faster runs.
- Dataset paths must be updated for local execution.
- The 1000‑task dataset keeps its original filename: `GoCJ_Dataset_1000 (2).txt`.

## Documentation
For a detailed, step‑by‑step explanation of the algorithms and formulas, see `explanation.md`.
