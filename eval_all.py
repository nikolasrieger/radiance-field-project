#!/usr/bin/env python3
"""Run evaluate_splat.py over all *_30 runs and print a loss table."""
from __future__ import annotations

import subprocess
import sys
import re
import tempfile
from pathlib import Path

RUNS_DIR = Path(__file__).resolve().parent / "runs"
N_EVAL   = 3
SEED     = 0

runs = sorted(p for p in RUNS_DIR.iterdir() if p.is_dir() and p.name.endswith("_30"))
if not runs:
    sys.exit(f"No *_30 run dirs found in {RUNS_DIR}")

results: list[dict] = []

for run in runs:
    ckpt = run / "ckpt_final.pt"
    if not ckpt.exists():
        candidates = sorted(run.glob("ckpt_*.pt"))
        ckpt = candidates[-1] if candidates else None
    if ckpt is None:
        print(f"[SKIP] {run.name}  — no checkpoint found")
        results.append({"run": run.name, "error": "no checkpoint"})
        continue

    print(f"[EVAL] {run.name} ...", flush=True)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_out = tmp.name
    cmd = [
        sys.executable, str(Path(__file__).parent / "evaluate_splat.py"),
        "--run",    str(run),
        "--n-eval", str(N_EVAL),
        "--seed",   str(SEED),
        "--out",    tmp_out,
        "--all-losses",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    Path(tmp_out).unlink(missing_ok=True)
    out  = proc.stdout + proc.stderr

    if proc.returncode != 0:
        # grab last non-empty line as the error hint
        err_lines = [l.strip() for l in out.splitlines() if l.strip()]
        hint = err_lines[-1] if err_lines else "unknown error"
        print(f"       FAILED: {hint}")
        results.append({"run": run.name, "error": hint[:60]})
        continue

    row: dict = {"run": run.name}
    for line in out.splitlines():
        m = re.search(r"Photo.*?:\s+([\d.]+)", line)
        if m:
            row["photo"] = float(m.group(1))
        m = re.search(r"Depth.*?:\s+([\d.]+)", line)
        if m:
            row["depth"] = float(m.group(1))
        m = re.search(r"Normal.*?:\s+([\d.]+)", line)
        if m:
            row["normal"] = float(m.group(1))
        m = re.search(r"Full.*?:\s+([\d.]+)", line)
        if m:
            row["full"] = float(m.group(1))
    results.append(row)
    p = row.get('photo', float('nan'))
    d = row.get('depth', float('nan'))
    n = row.get('normal', float('nan'))
    f = row.get('full', float('nan'))
    print(f"       photo={p:.6f} | depth={d:.6f} | normal={n:.6f} | full={f:.6f}")

# ── table ─────────────────────────────────────────────────────────────────────
col_run = max(len(r["run"]) for r in results)
col_w   = 12

def fmt(v): return f"{v:.6f}".center(col_w) if isinstance(v, float) else "—".center(col_w)

header = "  ".join([
    f"{'Run':<{col_run}}",
    f"{'Photo':^{col_w}}", f"{'Depth':^{col_w}}", f"{'Normal':^{col_w}}",
    f"{'Full (weighted)':^{col_w+2}}",
])
sep = "─" * len(header)

print()
print(sep)
print(f"  Eval table  ({N_EVAL} views per run, seed={SEED})")
print(sep)
print(header)
print(sep)

for r in results:
    if "error" in r:
        print(f"  {r['run']:<{col_run}}  ERROR: {r['error']}")
        continue
    row = "  ".join([
        f"{r['run']:<{col_run}}",
        fmt(r.get("photo")),
        fmt(r.get("depth")),
        fmt(r.get("normal")),
        fmt(r.get("full")).center(col_w+2),
    ])
    print(row)

print(sep)
