# headsup-cfr

A deep learning poker bot for heads-up no-limit Texas Hold'em, trained using Deep Counterfactual Regret Minimization (Deep CFR).

## How it works

Two neural networks learn to play poker by playing millions of hands against themselves. The advantage network predicts regrets for each action, and the strategy network learns the average policy. An exploitative wrapper detects opponent tendencies in real time and adjusts accordingly.

## Results

Trained at 30k iterations and evaluated over 50,000 hands per opponent:

| Opponent | Win Rate (bb/100) |
|----------|------------------|
| Random | +549 |
| Calling Station | +188 |
| Tight-Aggressive | +19 |
| Loose-Aggressive | +431 |

## Usage

```bash
pip install torch

# Play against the bot
python play.py

# Run benchmarks
python benchmark.py

# Retrain from scratch
python play.py --retrain --iters 30000
```

## Files

- `bot.py` — Deep CFR training and inference
- `hand_eval.py` — Hand evaluation and equity estimation
- `play.py` — Interactive terminal game
- `benchmark.py` — Benchmark suite against baseline opponents
