"""Save the per-game outputs of an evaluation run into results/ (committed to git).

Kept: each game's JSON summary (spec, user turns, Moshi text tokens with times, events, per-frame
Moshi level), its .score.json (per-turn classes, latencies, rewards), specs.json and
timing_worker*.json. Not kept: audio (.wav/.mp3), user token arrays (.npy), worker shard files.
If the kept files total more than MAX_MB, they are written as one <dest>.tar.gz instead of a folder.

  python -m fruitbench.archive runs/base_test_s1 results/base_games/test_s1
"""

from __future__ import annotations

import shutil
import sys
import tarfile
from pathlib import Path

MAX_MB = 20.0


def kept_files(run_dir: Path) -> list[Path]:
    out = []
    for f in sorted(run_dir.glob("*.json")):
        if f.name.startswith("shard") or f.stat().st_size == 0:
            continue
        out.append(f)
    return out


def archive_run(run_dir: Path, dest: Path, max_mb: float = MAX_MB) -> Path:
    run_dir, dest = Path(run_dir), Path(dest)
    files = kept_files(run_dir)
    total = sum(f.stat().st_size for f in files) / 1e6
    tgz = dest.with_name(dest.name + ".tar.gz")
    if dest.exists():
        shutil.rmtree(dest)
    if tgz.exists():
        tgz.unlink()
    dest.parent.mkdir(parents=True, exist_ok=True)
    if total > max_mb:
        with tarfile.open(tgz, "w:gz") as tar:
            for f in files:
                tar.add(f, arcname=f"{dest.name}/{f.name}")
        out = tgz
    else:
        dest.mkdir(parents=True)
        for f in files:
            shutil.copy2(f, dest / f.name)
        out = dest
    print(f"archived {len(files)} files ({total:.1f} MB) from {run_dir} -> {out}")
    return out


if __name__ == "__main__":
    archive_run(Path(sys.argv[1]), Path(sys.argv[2]))
