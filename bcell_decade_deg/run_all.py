"""Run the corrected extreme-age sensitivity pipeline in dependency order."""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path


STAGES = [
    "01_prep.py",
    "02_deg.py",
    "03_gsea.py",
    "04_decade_boxplots.py",
    "05_variance_age.py",
]


def main() -> None:
    scripts = Path(__file__).resolve().parent / "scripts"
    started = time.perf_counter()
    for stage in STAGES:
        stage_started = time.perf_counter()
        print(f"\n=== {stage} ===", flush=True)
        subprocess.run([sys.executable, str(scripts / stage)], check=True)
        print(f"=== {stage} finished in {time.perf_counter() - stage_started:.1f}s ===",
              flush=True)
    print(f"\nAll stages finished in {(time.perf_counter() - started) / 60:.2f} min.")


if __name__ == "__main__":
    main()
