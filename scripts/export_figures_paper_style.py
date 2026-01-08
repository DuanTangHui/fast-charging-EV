"""Export summary figures from metrics logs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_dir", default="runs")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    metrics_path = runs_dir / "cycle0" / "metrics.jsonl"
    if not metrics_path.exists():
        print(f"Metrics not found: {metrics_path}")
        return

    rewards = []
    epochs = []
    with metrics_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            epochs.append(record.get("epoch", 0))
            rewards.append(record.get("reward", 0.0))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, rewards, marker="o")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Reward")
    ax.set_title("Cycle0 Reward Trend")
    fig.tight_layout()
    out_path = runs_dir / "cycle0" / "reward_trend.png"
    fig.savefig(out_path)
    plt.close(fig)


if __name__ == "__main__":
    main()
