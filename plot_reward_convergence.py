#!/usr/bin/env python3
"""
Reward Convergence Plot

Parses training.log for:
  - Per-episode reward:  "Episode reward: <value> at step <step>"
  - Per-update avg reward (from update= line, field avg_reward=...)

Generates a publication-ready convergence figure with:
  - Raw episode reward (transparent scatter)
  - Smoothed trend (moving average)
  - Optional: multiple runs overlay / PPO vs SPF comparison

Usage:
    python plot_reward_convergence.py                        # default: training.log
    python plot_reward_convergence.py --logs training.log spf_training.log --labels PPO SPF
    python plot_reward_convergence.py --window 20            # wider smoothing
"""

import re
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

# ── Parsing ──────────────────────────────────────────────────────────

def parse_episode_rewards(log_path: str):
    """
    Parse lines of the form:
        Episode reward: -1234.56 at step 12345
    Returns (steps, rewards) arrays.
    """
    steps, rewards = [], []
    if not os.path.exists(log_path):
        print(f"Warning: {log_path} not found")
        return np.array(steps), np.array(rewards)

    # Pattern 1:  "Episode reward: <val> at step <step>"
    pat1 = re.compile(r"Episode reward:\s*([-\d.eE+]+)\s+at step\s*(\d+)")
    # Pattern 2 (fallback):  update line with avg_reward field
    pat2 = re.compile(r"update=\s*\d+\s+steps=\s*(\d+).*?avg_reward=([-\d.eE+]+)")

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = pat1.search(line)
            if m:
                rewards.append(float(m.group(1)))
                steps.append(int(m.group(2)))
                continue
            m = pat2.search(line)
            if m:
                steps.append(int(m.group(1)))
                rewards.append(float(m.group(2)))

    return np.array(steps), np.array(rewards)

def smooth(values: np.ndarray, window: int) -> np.ndarray:
    """Exponential Moving Average (EMA) for smoother learning curves."""
    if len(values) < 2:
        return values.copy()
    
    alpha = 2.0 / (window + 1.0)
    smoothed = np.empty_like(values)
    smoothed[0] = values[0]
    
    for i in range(1, len(values)):
        smoothed[i] = alpha * values[i] + (1 - alpha) * smoothed[i - 1]
        
    return smoothed

# ── Plotting ─────────────────────────────────────────────────────────

def plot_reward_convergence(
    log_paths: list[str],
    labels: list[str],
    window: int = 10,
    output: str = "plots/reward_convergence.png",
):
    fig, ax = plt.subplots(figsize=(12, 5))

    colors = ["#2196F3", "#F44336", "#4CAF50", "#FF9800"]

    any_data = False
    for idx, (path, label) in enumerate(zip(log_paths, labels)):
        steps, rewards = parse_episode_rewards(path)
        if len(steps) == 0:
            print(f"  \u26a0  No episode reward data in {path}")
            continue
        any_data = True
        c = colors[idx % len(colors)]

        # Raw (scatter, semi-transparent)
        ax.scatter(steps, rewards, s=8, alpha=0.15, color=c, edgecolors="none")

        # Smoothed trend
        smoothed = smooth(rewards, window)
        ax.plot(steps, smoothed, color=c, linewidth=2, label=f"{label} (avg window={window})")

        # Print summary
        print(f"  {label}: {len(rewards)} episodes | "
              f"first={rewards[0]:.1f}  last={rewards[-1]:.1f}  "
              f"min={rewards.min():.1f}  max={rewards.max():.1f}  "
              f"last-{window}-avg={rewards[-window:].mean():.1f}")

    if not any_data:
        print("\n\u2718 No reward data found. Make sure the log contains 'Episode reward: <val> at step <step>'")
        return

    ax.set_xlabel("Training Steps", fontsize=12)
    ax.set_ylabel("Episode Reward", fontsize=12)
    ax.set_title("Reward Convergence", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    print(f"\n\u2714 Saved to {output}")
    plt.close(fig)

# ── CLI ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Plot reward convergence")
    parser.add_argument(
        "--logs", nargs="+", default=["training.log"],
        help="Log file(s) to parse (default: training.log)"
    )
    parser.add_argument(
        "--labels", nargs="+", default=None,
        help="Legend labels for each log file"
    )
    parser.add_argument(
        "--window", type=int, default=10,
        help="Moving average window size (default: 10)"
    )
    parser.add_argument(
        "--output", type=str, default="plots/reward_convergence.png",
        help="Output image path"
    )
    args = parser.parse_args()

    labels = args.labels or [os.path.splitext(os.path.basename(p))[0] for p in args.logs]
    if len(labels) < len(args.logs):
        labels.extend([f"run_{i}" for i in range(len(labels), len(args.logs))])

    plot_reward_convergence(args.logs, labels, args.window, args.output)

if __name__ == "__main__":
    main()
