#!/usr/bin/env python3
"""Op-level eager/autograd benchmark regression gate.

Complements tools/bench_gate.py (the paired model-level gate vs llama.cpp):
this gate runs the op-level bench suite (`bench*` build steps) against a
locally recorded per-machine baseline, so dispatch-latency and
training-throughput regressions surface without an external reference.

Gating rules:

- timings (ns/op or ms) gate with a tolerance band over the median of N
  repeats; rows whose cross-repeat coefficient of variation is high are
  reported as NOISY instead of failing, mirroring bench_gate.py;
- allocs_per_op gates EXACTLY: allocation counts are deterministic, so any
  increase fails and any decrease asks for a re-record;
- checksums gate EXACTLY: a checksum change is numerical drift, not noise;
- bytes_per_op / live_bytes are recorded for reference, not gated.

Usage:
  python3 tools/opbench_gate.py record
  python3 tools/opbench_gate.py check
  python3 tools/opbench_gate.py check --suites facade,mlp --repeats 5
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import platform
import re
import statistics
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE_DIR = ROOT / "bench" / "baselines"

NOISY_COV = 0.12


@dataclass(frozen=True)
class Suite:
    name: str
    step: str
    kind: str  # "csv" | "ce" | "scatter" | "optim"
    # Multiplier on --tol. The optim suite runs ~1 minute of sustained
    # bandwidth-saturating work, so its late rows execute on a heat-soaked
    # SoC whose throttle state varies run to run: same-binary A/B showed
    # 10-18% row-level drift with rotating offenders. A wider band keeps the
    # gate honest there without loosening the short suites.
    tol_mul: float = 1.0


SUITES: dict[str, Suite] = {
    "packed-gemm": Suite("packed-gemm", "bench-packed-gemm", "csv"),
    "facade": Suite("facade", "bench-facade", "csv"),
    "mlp": Suite("mlp", "bench", "csv"),
    "backward-diamond": Suite("backward-diamond", "bench-backward-diamond", "csv"),
    "attention-backward": Suite("attention-backward", "bench-attention-backward", "csv"),
    "einsum": Suite("einsum", "bench-einsum", "csv"),
    "conv": Suite("conv", "bench-conv", "csv"),
    "ce": Suite("ce", "bench-ce", "ce"),
    "scatter": Suite("scatter", "bench-scatter", "scatter"),
    "optim": Suite("optim", "bench-optim", "optim", tol_mul=2.5),
}

# CSV columns that are measurements (or derived from measurements); everything
# else identifies the row and becomes part of its key.
CSV_METRIC_COLUMNS = {
    "ns_per_op",
    "allocs_per_op",
    "bytes_per_op",
    "live_bytes",
    "checksum",
    "speedup_vs_serial_vjp_x",
    "approx_gflops",
}

CE_ROW = re.compile(
    r"^(.+?)\s{2,}(\d+) x (\d+)\s+([\d.]+) ms\s+([\d.]+) Gelem/s\s+\((\d+) iters\)$"
)
SCATTER_ROW = re.compile(
    r"^scatter-add (.+?)\s{2,}(\d+) idx -> (\d+) x (\d+)\s+([\d.]+) ms\s+([\d.]+) GB/s\s+\((\d+) iters\)$"
)
OPTIM_ROW = re.compile(
    r"^(\S+)\s+([\d.]+) M\s+([\d.]+) ms\s+(?:[\d.]+ GB/s|-)\s+([\d.]+) ms$"
)
BACKEND_NOTE = re.compile(r"backend=(\w+)")


@dataclass
class Row:
    key: str
    time_value: float  # ns/op for csv suites, ms for table suites
    time_unit: str
    allocs: int | None
    bytes_per_op: int | None
    checksum: str | None


def run_suite(suite: Suite) -> str:
    cmd = ["zig", "build", suite.step, "-Doptimize=ReleaseFast", "--", "--prod-allocator"]
    proc = subprocess.run(
        cmd, cwd=ROOT, capture_output=True, text=True, timeout=1200
    )
    if proc.returncode != 0:
        tail = proc.stderr.strip().splitlines()[-15:]
        raise RuntimeError(
            f"{suite.name}: `{' '.join(cmd)}` exited {proc.returncode}\n" + "\n".join(tail)
        )
    return proc.stdout


def parse_backend(text: str) -> str | None:
    match = BACKEND_NOTE.search(text)
    return match.group(1) if match else None


def parse_csv_suite(text: str) -> tuple[list[Row], str | None]:
    lines = text.splitlines()
    header_index = next(
        (i for i, line in enumerate(lines) if "ns_per_op" in line and "," in line), None
    )
    if header_index is None:
        raise ValueError("no CSV header with ns_per_op found")
    reader = csv.DictReader(io.StringIO("\n".join(lines[header_index:])))
    rows: list[Row] = []
    backend = None
    for record in reader:
        if record.get("ns_per_op") is None:
            continue
        backend = record.get("backend", backend)
        key_parts = [
            f"{name}={value}"
            for name, value in record.items()
            if name not in CSV_METRIC_COLUMNS and name != "runtime" and value is not None
        ]
        rows.append(
            Row(
                key="/".join(key_parts),
                time_value=float(record["ns_per_op"]),
                time_unit="ns/op",
                allocs=int(record["allocs_per_op"]) if "allocs_per_op" in record else None,
                bytes_per_op=int(record["bytes_per_op"]) if "bytes_per_op" in record else None,
                checksum=record.get("checksum"),
            )
        )
    return rows, backend


def parse_table_suite(text: str, kind: str) -> tuple[list[Row], str | None]:
    rows: list[Row] = []
    section = None
    for line in text.splitlines():
        line = line.rstrip()
        if kind == "ce":
            match = CE_ROW.match(line)
            if match:
                name, height, width, ms, _gelems, iters = match.groups()
                rows.append(
                    Row(
                        key=f"{name.strip()}/{height}x{width}/iters={iters}",
                        time_value=float(ms),
                        time_unit="ms",
                        allocs=None,
                        bytes_per_op=None,
                        checksum=None,
                    )
                )
        elif kind == "scatter":
            match = SCATTER_ROW.match(line)
            if match:
                name, count, vocab, dim, ms, _gbs, iters = match.groups()
                rows.append(
                    Row(
                        key=f"{name.strip()}/{count}idx/{vocab}x{dim}/iters={iters}",
                        time_value=float(ms),
                        time_unit="ms",
                        allocs=None,
                        bytes_per_op=None,
                        checksum=None,
                    )
                )
        elif kind == "optim":
            if line.startswith("embedding ["):
                section = "embedding"
                continue
            match = OPTIM_ROW.match(line)
            if match and match.group(1) != "optimizer":
                name, params, ms, _scaled = match.groups()
                rows.append(
                    Row(
                        key=f"{section or 'block'}/{name}/params={params}M",
                        time_value=float(ms),
                        time_unit="ms",
                        allocs=None,
                        bytes_per_op=None,
                        checksum=None,
                    )
                )
    return rows, parse_backend(text)


def parse_suite(suite: Suite, text: str) -> tuple[list[Row], str | None]:
    if suite.kind == "csv":
        return parse_csv_suite(text)
    return parse_table_suite(text, suite.kind)


def collect(suite: Suite, repeats: int) -> tuple[dict[str, dict], str | None]:
    samples: dict[str, list[Row]] = {}
    backend = None
    for _ in range(repeats):
        rows, found_backend = parse_suite(suite, run_suite(suite))
        backend = backend or found_backend
        if not rows:
            raise RuntimeError(f"{suite.name}: parser produced no rows")
        for row in rows:
            samples.setdefault(row.key, []).append(row)
    result: dict[str, dict] = {}
    for key, row_samples in samples.items():
        times = [row.time_value for row in row_samples]
        median = statistics.median(times)
        cov = statistics.pstdev(times) / statistics.fmean(times) if len(times) > 1 else 0.0
        allocs_values = {row.allocs for row in row_samples}
        checksum_values = {row.checksum for row in row_samples}
        stable = len(allocs_values) == 1 and len(checksum_values) == 1
        first = row_samples[0]
        result[key] = {
            "time": median,
            "unit": first.time_unit,
            "samples": times,
            "cov": round(cov, 4),
            "allocs": first.allocs,
            "bytes_per_op": first.bytes_per_op,
            "checksum": first.checksum,
            "stable": stable,
        }
    return result, backend


def short_hostname() -> str:
    # The DNS suffix changes with the network; keep baselines keyed to the box.
    return socket.gethostname().split(".")[0]


def environment() -> dict:
    zig_version = subprocess.run(
        ["zig", "version"], capture_output=True, text=True
    ).stdout.strip()
    return {
        "hostname": short_hostname(),
        "machine": platform.machine(),
        "system": platform.system(),
        "zig": zig_version,
        "optimize": "ReleaseFast",
        "allocator": "prod",
    }


def baseline_path(args: argparse.Namespace) -> Path:
    if args.baseline:
        return Path(args.baseline)
    return DEFAULT_BASELINE_DIR / f"opbench-{short_hostname()}.json"


def selected_suites(args: argparse.Namespace) -> list[Suite]:
    if args.suites == "all":
        return list(SUITES.values())
    names = [name.strip() for name in args.suites.split(",") if name.strip()]
    unknown = [name for name in names if name not in SUITES]
    if unknown:
        sys.exit(f"unknown suites: {', '.join(unknown)} (known: {', '.join(SUITES)})")
    return [SUITES[name] for name in names]


def cmd_record(args: argparse.Namespace) -> int:
    path = baseline_path(args)
    baseline: dict = {
        "version": 1,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "repeats": args.repeats,
        "env": environment(),
        "suites": {},
    }
    if path.exists() and args.suites != "all":
        baseline_existing = json.loads(path.read_text())
        baseline["suites"] = baseline_existing.get("suites", {})
    backend = None
    for suite in selected_suites(args):
        print(f"[record] {suite.name} ({suite.step}) x{args.repeats} ...", flush=True)
        rows, found_backend = collect(suite, args.repeats)
        backend = backend or found_backend
        baseline["suites"][suite.name] = rows
        unstable = [key for key, row in rows.items() if not row["stable"]]
        for key in unstable:
            print(f"  UNSTABLE across repeats (allocs/checksum): {key}")
    baseline["env"]["backend"] = backend
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(baseline, indent=2, sort_keys=True) + "\n")
    total = sum(len(rows) for rows in baseline["suites"].values())
    print(f"[record] wrote {total} rows to {path}")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    path = baseline_path(args)
    if not path.exists():
        sys.exit(f"no baseline at {path}; run `record` first")
    baseline = json.loads(path.read_text())
    env = environment()
    recorded_env = baseline.get("env", {})
    for field in ("hostname", "machine", "zig"):
        if recorded_env.get(field) != env.get(field) and not args.force:
            sys.exit(
                f"baseline {field}={recorded_env.get(field)!r} != current {env.get(field)!r};"
                " re-record or pass --force"
            )

    failures: list[str] = []
    noisy: list[str] = []
    improved: list[str] = []
    missing: list[str] = []
    transient: list[str] = []
    checked = 0

    for suite in selected_suites(args):
        recorded_rows = baseline["suites"].get(suite.name)
        if recorded_rows is None:
            missing.append(f"{suite.name}: not in baseline")
            continue
        print(f"[check] {suite.name} ({suite.step}) x{args.repeats} ...", flush=True)
        current_rows, _ = collect(suite, args.repeats)
        suite_tol = args.tol * suite.tol_mul
        # Timing exceedances are confirmed on a second run of the suite after
        # a cooldown: transient interference (background load, a heat-soaked
        # SoC) must not fail the gate, and a real kernel regression
        # reproduces. Deterministic metrics (allocs, checksums) never retry.
        time_suspects: list[tuple[str, dict]] = []
        for key, recorded in recorded_rows.items():
            current = current_rows.get(key)
            label = f"{suite.name}:{key}"
            if current is None:
                missing.append(f"{label}: row disappeared")
                continue
            checked += 1

            ratio = current["time"] / recorded["time"] if recorded["time"] else 1.0
            time_note = (
                f"{recorded['time']:.3f} -> {current['time']:.3f} {recorded['unit']}"
                f" ({ratio:.3f}x)"
            )
            is_noisy = current["cov"] > NOISY_COV or recorded["cov"] > NOISY_COV
            if ratio > 1.0 + suite_tol:
                if is_noisy:
                    noisy.append(f"{label}: NOISY slower {time_note} cov={current['cov']}")
                else:
                    time_suspects.append((key, recorded))
            elif ratio < 1.0 - suite_tol:
                improved.append(f"{label}: faster {time_note}")

            if recorded["stable"] and current["stable"]:
                if recorded["allocs"] is not None and current["allocs"] != recorded["allocs"]:
                    delta = f"allocs/op {recorded['allocs']} -> {current['allocs']}"
                    if current["allocs"] > recorded["allocs"]:
                        failures.append(f"{label}: {delta}")
                    else:
                        improved.append(f"{label}: {delta} (re-record to lock in)")
                if (
                    recorded["checksum"] is not None
                    and current["checksum"] != recorded["checksum"]
                ):
                    failures.append(
                        f"{label}: checksum {recorded['checksum']} -> {current['checksum']}"
                    )
        for key in current_rows:
            if key not in recorded_rows:
                missing.append(f"{suite.name}:{key}: new row (re-record to cover)")

        if time_suspects:
            print(
                f"[check] {suite.name}: {len(time_suspects)} timing exceedance(s),"
                f" confirming after {args.retry_cooldown_s}s cooldown ...",
                flush=True,
            )
            time.sleep(args.retry_cooldown_s)
            retry_rows, _ = collect(suite, args.repeats)
            for key, recorded in time_suspects:
                label = f"{suite.name}:{key}"
                retry = retry_rows.get(key)
                if retry is None:
                    missing.append(f"{label}: row disappeared on retry")
                    continue
                ratio = retry["time"] / recorded["time"] if recorded["time"] else 1.0
                time_note = (
                    f"{recorded['time']:.3f} -> {retry['time']:.3f} {recorded['unit']}"
                    f" ({ratio:.3f}x on retry)"
                )
                if ratio > 1.0 + suite_tol:
                    failures.append(f"{label}: slower {time_note}")
                else:
                    transient.append(f"{label}: {time_note}")

    for title, entries in (
        ("IMPROVED", improved),
        ("NOISY", noisy),
        ("TRANSIENT (passed on retry)", transient),
        ("MISSING/NEW", missing),
        ("FAIL", failures),
    ):
        if entries:
            print(f"\n{title} ({len(entries)}):")
            for entry in entries:
                print(f"  {entry}")
    print(
        f"\n[check] {checked} rows checked: {len(failures)} fail, {len(noisy)} noisy,"
        f" {len(transient)} transient, {len(improved)} improved, {len(missing)} missing/new"
    )
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("record", "check"))
    parser.add_argument("--suites", default="all", help="comma list or 'all'")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--tol", type=float, default=0.10, help="timing tolerance band")
    parser.add_argument("--baseline", default=None, help="baseline JSON path")
    parser.add_argument("--force", action="store_true", help="ignore env mismatch")
    parser.add_argument(
        "--retry-cooldown-s",
        type=int,
        default=30,
        help="cooldown before re-running a suite to confirm timing exceedances",
    )
    args = parser.parse_args()
    if args.command == "record":
        return cmd_record(args)
    return cmd_check(args)


if __name__ == "__main__":
    sys.exit(main())
