from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def run_cmd(args: list[str]) -> None:
    print("[RUN]", " ".join(args))
    subprocess.run(args, check=True, cwd=ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-platform launcher for other_method full experiment.")
    parser.add_argument("config", nargs="?", default="configs/pack_3p6s_spme.yaml")
    parser.add_argument("outroot", nargs="?", default="runs/other_method")
    parser.add_argument("seed", nargs="?", default="7")
    args = parser.parse_args()

    py = sys.executable
    run_cmd(
        [
            py,
            "scripts/other_method/train_other_methods.py",
            "--config",
            args.config,
            "--output",
            args.outroot,
            "--seed",
            str(args.seed),
        ]
    )

    run_cmd(
        [
            py,
            "scripts/other_method/evaluate_on_real_env.py",
            "--config",
            args.config,
            "--manifest",
            f"{args.outroot}/models/manifest.json",
            "--output",
            f"{args.outroot}/eval",
            "--seed",
            str(args.seed),
        ]
    )


if __name__ == "__main__":
    main()