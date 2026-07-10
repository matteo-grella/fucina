#!/usr/bin/env python3
"""Paired Fucina-vs-llama.cpp benchmark gate.

The gate is deliberately conservative:

- every row runs as Fucina->llama and llama->Fucina by default, so process order
  and thermal drift are visible in the samples;
- rows with high cross-sample coefficient of variation fail as NOISY instead of
  becoming benchmark claims;
- raw stdout/stderr and exact command lines are saved for every subprocess;
- Fucina prefill/decode still includes the final logits/sampler work that
  llama-bench skips in its pp/tg loops, so passing rows are conservative for
  Fucina. Failing rows should be followed by a no-logits A/B before being
  treated as true kernel regressions.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import re
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LLAMA_BENCH = ROOT / "refs/llama.cpp/build-cpu/bin/llama-bench"
DEFAULT_QWEN = ROOT / "zig-out/bin/fucina-zig-qwen3"
DEFAULT_GEMMA = ROOT / "zig-out/bin/fucina-zig-gemma4"

FULL_LENGTHS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64, 127, 128, 129, 256)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    kind: str
    path: Path
    prompt_seed: tuple[int, ...]
    short_seed: tuple[int, ...]
    short_len: int


CATALOG: dict[str, ModelSpec] = {
    "qwen3-0.6b-f16": ModelSpec(
        "qwen3-0.6b-f16", "qwen3", ROOT / "models/Qwen3-0.6B-f16.gguf", (151644, 872, 198, 9707), (151644, 872, 198, 9707), 8
    ),
    "qwen3-0.6b-q8_0": ModelSpec(
        "qwen3-0.6b-q8_0", "qwen3", ROOT / "models/Qwen3-0.6B-Q8_0.gguf", (151644, 872, 198, 9707), (151644, 872, 198, 9707), 8
    ),
    "qwen3-0.6b-q6_k": ModelSpec(
        "qwen3-0.6b-q6_k", "qwen3", ROOT / "models/Qwen3-0.6B-Q6_K.gguf", (151644, 872, 198, 9707), (151644, 872, 198, 9707), 8
    ),
    "qwen3-0.6b-q5_k_m": ModelSpec(
        "qwen3-0.6b-q5_k_m", "qwen3", ROOT / "models/Qwen3-0.6B-Q5_K_M.gguf", (151644, 872, 198, 9707), (151644, 872, 198, 9707), 8
    ),
    "qwen3-0.6b-q5_k_s": ModelSpec(
        "qwen3-0.6b-q5_k_s", "qwen3", ROOT / "models/Qwen3-0.6B-Q5_K_S.gguf", (151644, 872, 198, 9707), (151644, 872, 198, 9707), 8
    ),
    "qwen3-0.6b-q4_k_m": ModelSpec(
        "qwen3-0.6b-q4_k_m", "qwen3", ROOT / "models/Qwen3-0.6B-Q4_K_M.gguf", (151644, 872, 198, 9707), (151644, 872, 198, 9707), 8
    ),
    "qwen3-0.6b-q4_k_s": ModelSpec(
        "qwen3-0.6b-q4_k_s", "qwen3", ROOT / "models/Qwen3-0.6B-Q4_K_S.gguf", (151644, 872, 198, 9707), (151644, 872, 198, 9707), 8
    ),
    "qwen3-1.7b-q4_k_m": ModelSpec(
        "qwen3-1.7b-q4_k_m", "qwen3", ROOT / "models/Qwen3-1.7B-Q4_K_M.gguf", (151644, 872, 198, 9707), (151644, 872, 198, 9707), 8
    ),
    "qwen3moe-30b-q5_k_m": ModelSpec(
        "qwen3moe-30b-q5_k_m",
        "qwen3",
        ROOT / "models/Qwen3-30B-A3B-Instruct-2507-Q5_K_M.gguf",
        (785, 6722, 198, 9707),
        (785, 6722, 198, 9707),
        8,
    ),
    "qwen3moe-30b-q6_k": ModelSpec(
        "qwen3moe-30b-q6_k",
        "qwen3",
        ROOT / "models/Qwen3-30B-A3B-Instruct-2507-Q6_K.gguf",
        (785, 6722, 198, 9707),
        (785, 6722, 198, 9707),
        8,
    ),
    "gemma4-26b-q6_k": ModelSpec(
        "gemma4-26b-q6_k",
        "gemma4",
        ROOT / "models/gemma-4-26B-A4B-it-UD-Q6_K.gguf",
        (2, 818, 235, 108),
        (2, 818, 235, 108),
        8,
    ),
}


FUCINA_PREFILL_RE = re.compile(
    r"prefill:\s*([0-9.]+)\s*±\s*([0-9.]+)\s*tok/s\s*\(min\s*([0-9.]+),\s*max\s*([0-9.]+)\)"
)
FUCINA_DECODE_RE = re.compile(
    r"decode\s*:\s*([0-9.]+)\s*±\s*([0-9.]+)\s*tok/s\s*\(min\s*([0-9.]+),\s*max\s*([0-9.]+)\)"
)


def ids_repeated(seed: tuple[int, ...], n: int) -> str:
    out: list[int] = []
    while len(out) < n:
        out.extend(seed)
    return ",".join(str(x) for x in out[:n])


def parse_csv_ints(raw: str) -> list[int]:
    try:
        values = [int(part) for part in raw.split(",") if part]
    except ValueError as exc:
        raise SystemExit(f"invalid integer list: {raw}") from exc
    if not values or any(v < 0 for v in values):
        raise SystemExit(f"invalid integer list: {raw}")
    return values


def parse_tasks(raw: str) -> list[str]:
    tasks: list[str] = []
    for part in raw.split(","):
        name = part.strip().lower()
        if name in ("pp", "prefill"):
            tasks.append("prefill")
        elif name in ("tg", "decode"):
            tasks.append("decode")
        elif name:
            raise SystemExit(f"unknown task {part!r}; use prefill,decode")
    if not tasks:
        raise SystemExit("no benchmark tasks selected")
    return tasks


def git_text(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def render_cmd(cmd: list[str]) -> str:
    shown: list[str] = []
    for arg in cmd:
        if "," in arg and len(arg) > 80:
            shown.append(f"<{arg.count(',') + 1} token ids>")
        else:
            shown.append(arg)
    return " ".join(shown)


def prewarm_model(path: Path, block_mb: int, max_gb: float) -> None:
    size = path.stat().st_size
    if max_gb > 0 and size > max_gb * 1024**3:
        print(f"  prewarm skipped: {path.name} is {size / 1024**3:.1f} GiB > cap {max_gb:.1f} GiB", flush=True)
        return
    block = max(1, block_mb) * 1024 * 1024
    total = 0
    with path.open("rb", buffering=0) as f:
        while True:
            data = f.read(block)
            if not data:
                break
            total += len(data)
    print(f"  prewarmed {path.name}: {total / 1024**3:.2f} GiB read", flush=True)


def run_logged(cmd: list[str], out_dir: Path, stem: str, timeout_s: int | None) -> dict[str, Any]:
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    print(f"$ {render_cmd(cmd)}", flush=True)
    start = time.monotonic()
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout_s,
    )
    elapsed = time.monotonic() - start
    stdout_path = raw_dir / f"{stem}.stdout"
    stderr_path = raw_dir / f"{stem}.stderr"
    cmd_path = raw_dir / f"{stem}.cmd"
    stdout_path.write_text(proc.stdout)
    stderr_path.write_text(proc.stderr)
    cmd_path.write_text(render_cmd(cmd) + "\n")
    if proc.returncode != 0:
        print(proc.stdout, end="")
        print(proc.stderr, end="", file=sys.stderr)
        raise RuntimeError(f"command failed with exit {proc.returncode}: {render_cmd(cmd)}")
    return {
        "cmd": cmd,
        "cmd_rendered": render_cmd(cmd),
        "stdout": stdout_path.relative_to(out_dir).as_posix(),
        "stderr": stderr_path.relative_to(out_dir).as_posix(),
        "elapsed_s": elapsed,
    }


def parse_llama(stdout: str, task: str, n_prompt: int, n_gen: int) -> float:
    rows = json.loads(stdout)
    for row in rows:
        row_prompt = int(row.get("n_prompt") or 0)
        row_gen = int(row.get("n_gen") or 0)
        if task == "prefill" and row_prompt == n_prompt and row_gen == 0:
            return float(row["avg_ts"])
        # llama-bench emits a separate prompt-processing row and a generation
        # row for `-p P -n N`; recent builds record the tg row as
        # n_prompt=0,n_gen=N rather than n_prompt=P,n_gen=N.
        if task == "decode" and row_gen == n_gen and row_prompt in (0, n_prompt):
            return float(row["avg_ts"])
    raise ValueError(f"could not find llama {task} row for p={n_prompt} n={n_gen}:\n{stdout}")


def parse_fucina(stdout: str, task: str) -> float:
    regex = FUCINA_PREFILL_RE if task == "prefill" else FUCINA_DECODE_RE
    match = regex.search(stdout)
    if not match:
        raise ValueError(f"could not parse Fucina {task} throughput:\n{stdout}")
    return float(match.group(1))


def fucina_runner(spec: ModelSpec, args: argparse.Namespace) -> Path:
    return args.fucina_gemma if spec.kind == "gemma4" else args.fucina_qwen


def fucina_cmd(spec: ModelSpec, args: argparse.Namespace, task: str, n: int) -> list[str]:
    runner = fucina_runner(spec, args)
    if task == "prefill":
        token_ids = ids_repeated(spec.prompt_seed, n)
        if spec.kind == "gemma4":
            return [str(runner), str(spec.path), token_ids, "--gen", "0", "--bench", str(args.fucina_reps)]
        return [str(runner), str(spec.path), token_ids, "--gen", "1", "--bench", str(args.fucina_reps)]
    token_ids = ids_repeated(spec.short_seed, spec.short_len)
    # Qwen's --gen counts the free post-prefill sample plus decode steps.
    qwen_gen = args.decode_tokens + 1 if spec.kind == "qwen3" else args.decode_tokens
    return [str(runner), str(spec.path), token_ids, "--gen", str(qwen_gen), "--bench", str(args.fucina_reps)]


def llama_cmd(spec: ModelSpec, args: argparse.Namespace, task: str, n: int) -> list[str]:
    if task == "prefill":
        return [
            str(args.llama_bench),
            "-m",
            str(spec.path),
            "-ngl",
            "0",
            "-t",
            str(args.threads),
            "-p",
            str(n),
            "-n",
            "0",
            "-r",
            str(args.llama_reps),
            "-o",
            "json",
        ]
    return [
        str(args.llama_bench),
        "-m",
        str(spec.path),
        "-ngl",
        "0",
        "-t",
        str(args.threads),
        "-p",
        str(spec.short_len),
        "-n",
        str(args.decode_tokens),
        "-r",
        str(args.llama_reps),
        "-o",
        "json",
    ]


def sample_cv(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = statistics.mean(values)
    return 0.0 if mu == 0 else statistics.stdev(values) / mu


def summarize_row(row: dict[str, Any], min_ratio: float, max_cv: float, allow_noisy: bool) -> None:
    f_vals = [s["tok_s"] for s in row["samples"] if s["engine"] == "fucina"]
    l_vals = [s["tok_s"] for s in row["samples"] if s["engine"] == "llama"]
    if not f_vals or not l_vals:
        row["verdict"] = "MISSING"
        row["pass"] = False
        return
    f_med = statistics.median(f_vals)
    l_med = statistics.median(l_vals)
    ratio = f_med / l_med if l_med else math.inf
    f_cv = sample_cv(f_vals)
    l_cv = sample_cv(l_vals)
    noisy = max(f_cv, l_cv) > max_cv
    row.update(
        {
            "fucina_median_tok_s": f_med,
            "llama_median_tok_s": l_med,
            "ratio": ratio,
            "fucina_cv": f_cv,
            "llama_cv": l_cv,
            "noisy": noisy,
        }
    )
    if noisy and not allow_noisy:
        row["verdict"] = "NOISY"
        row["pass"] = False
    elif ratio < min_ratio:
        row["verdict"] = "FAIL"
        row["pass"] = False
    else:
        row["verdict"] = "PASS"
        row["pass"] = True


def write_reports(out_dir: Path, meta: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    payload = {"meta": meta, "rows": rows}
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "# Fucina vs llama.cpp Benchmark Gate",
        "",
        f"- created: `{meta['created_utc']}`",
        f"- git: `{meta['git_head']}`",
        f"- min ratio: `{meta['min_ratio']}`",
        f"- max CV: `{meta['max_cv']}`",
        "",
        "Passing rows are conservative for Fucina: Fucina still measures final logits/sampler work that llama-bench skips in pp/tg loops.",
        "",
        "| verdict | model | task | n | Fucina tok/s | llama tok/s | ratio | Fucina CV | llama CV |",
        "|:--|:--|:--|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        n = row["n_prompt"] if row["task"] == "prefill" else row["n_gen"]
        lines.append(
            "| {verdict} | {model} | {task} | {n} | {f:.2f} | {l:.2f} | {r:.3f} | {fcv:.1%} | {lcv:.1%} |".format(
                verdict=row.get("verdict", "?"),
                model=row["model"],
                task=row["task"],
                n=n,
                f=row.get("fucina_median_tok_s", 0.0),
                l=row.get("llama_median_tok_s", 0.0),
                r=row.get("ratio", 0.0),
                fcv=row.get("fucina_cv", 0.0),
                lcv=row.get("llama_cv", 0.0),
            )
        )
    (out_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n")

    fields = [
        "verdict",
        "model",
        "task",
        "n_prompt",
        "n_gen",
        "fucina_median_tok_s",
        "llama_median_tok_s",
        "ratio",
        "fucina_cv",
        "llama_cv",
    ]
    tsv = ["\t".join(fields)]
    for row in rows:
        tsv.append("\t".join(str(row.get(field, "")) for field in fields))
    (out_dir / "results.tsv").write_text("\n".join(tsv) + "\n")


def validate_inputs(args: argparse.Namespace, specs: list[ModelSpec]) -> None:
    needed = [args.llama_bench]
    if any(spec.kind == "qwen3" for spec in specs):
        needed.append(args.fucina_qwen)
    if any(spec.kind == "gemma4" for spec in specs):
        needed.append(args.fucina_gemma)
    for exe in needed:
        if not exe.exists():
            raise SystemExit(f"missing executable: {exe} (build ReleaseFast first, or pass an explicit path)")
    missing_models = [spec.path for spec in specs if not spec.path.exists()]
    if missing_models and not args.skip_missing:
        raise SystemExit("missing model(s):\n" + "\n".join(f"  {p}" for p in missing_models))


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", default="qwen3-0.6b-q6_k", help="comma-separated model keys; use --list-models to inspect")
    p.add_argument("--list-models", action="store_true", help="print model catalog and exit")
    p.add_argument("--tasks", default="prefill", help="comma-separated: prefill,decode")
    p.add_argument("--lengths", default=",".join(str(x) for x in FULL_LENGTHS), help="prefill prompt lengths")
    p.add_argument("--rounds", type=int, default=1, help="paired-order rounds")
    p.add_argument("--fucina-reps", type=int, default=3, help="per-process Fucina warm bench reps")
    p.add_argument("--llama-reps", type=int, default=3, help="per-process llama-bench reps")
    p.add_argument("--decode-tokens", type=int, default=32, help="decode steps for tg comparison")
    p.add_argument("--threads", type=int, default=8, help="llama.cpp thread count; Fucina uses FUCINA_MAX_THREADS/build default")
    p.add_argument("--min-ratio", type=float, default=1.0, help="minimum Fucina/llama median ratio required to pass")
    p.add_argument("--max-cv", type=float, default=0.08, help="fail rows whose Fucina or llama cross-sample CV exceeds this")
    p.add_argument("--allow-noisy", action="store_true", help="do not fail noisy rows")
    p.add_argument("--cooldown-s", type=float, default=0.0, help="sleep after every subprocess")
    p.add_argument("--single-order", action="store_true", help="run only one engine order instead of both order pairs")
    p.add_argument("--first", choices=("fucina", "llama"), default="fucina", help="engine that runs first when --single-order is used")
    p.add_argument("--prewarm-model", action="store_true", help="read the GGUF before each subprocess to stabilize page cache")
    p.add_argument("--prewarm-block-mb", type=int, default=64)
    p.add_argument("--prewarm-max-gb", type=float, default=0.0, help="0 means no size cap")
    p.add_argument("--timeout-s", type=int, default=0, help="per-subprocess timeout; 0 disables")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--skip-missing", action="store_true", help="skip missing models instead of failing")
    p.add_argument("--build", action="store_true", help="run `zig build -Doptimize=ReleaseFast` before benchmarking")
    p.add_argument("--llama-bench", type=Path, default=DEFAULT_LLAMA_BENCH)
    p.add_argument("--fucina-qwen", type=Path, default=DEFAULT_QWEN)
    p.add_argument("--fucina-gemma", type=Path, default=DEFAULT_GEMMA)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.list_models:
        for spec in CATALOG.values():
            status = "present" if spec.path.exists() else "missing"
            print(f"{spec.name:<22} {spec.kind:<6} {status:<7} {spec.path.relative_to(ROOT)}")
        return 0

    if args.build:
        subprocess.check_call(["zig", "build", "-Doptimize=ReleaseFast"], cwd=ROOT)

    model_keys = [m for m in (part.strip() for part in args.models.split(",")) if m]
    unknown = [m for m in model_keys if m not in CATALOG]
    if unknown:
        raise SystemExit(f"unknown model key(s): {', '.join(unknown)}")
    specs = [CATALOG[m] for m in model_keys]
    if args.skip_missing:
        specs = [spec for spec in specs if spec.path.exists()]
    if not specs:
        raise SystemExit("no models selected")
    validate_inputs(args, specs)

    tasks = parse_tasks(args.tasks)
    lengths = parse_csv_ints(args.lengths)
    if args.rounds < 1:
        raise SystemExit("--rounds must be >= 1")
    if args.fucina_reps < 2:
        raise SystemExit("--fucina-reps must be >= 2; Qwen emits generation output, not warm-bench stats, for --bench 1")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.out_dir or (ROOT / "compare" / f"bench-gate-{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    meta: dict[str, Any] = {
        "created_utc": stamp,
        "root": str(ROOT),
        "git_head": git_text(["rev-parse", "--short", "HEAD"]),
        "git_status_short": git_text(["status", "--short"]),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "models": [spec.name for spec in specs],
        "tasks": tasks,
        "lengths": lengths,
        "rounds": args.rounds,
        "fucina_reps": args.fucina_reps,
        "llama_reps": args.llama_reps,
        "decode_tokens": args.decode_tokens,
        "threads": args.threads,
        "min_ratio": args.min_ratio,
        "max_cv": args.max_cv,
        "cooldown_s": args.cooldown_s,
        "order_pairs": not args.single_order,
        "fairness_note": "Fucina includes final logits/sampler work; llama-bench pp/tg does not. Passing rows are conservative for Fucina.",
    }

    print(f"writing benchmark gate output to {out_dir}", flush=True)
    rows: list[dict[str, Any]] = []
    timeout_s = args.timeout_s or None
    orders = [[args.first, "llama" if args.first == "fucina" else "fucina"]]
    if not args.single_order:
        orders = [["fucina", "llama"], ["llama", "fucina"]]

    try:
        for round_i in range(1, args.rounds + 1):
            for spec in specs:
                row_tasks: list[tuple[str, int]] = []
                if "prefill" in tasks:
                    row_tasks.extend(("prefill", n) for n in lengths)
                if "decode" in tasks:
                    row_tasks.append(("decode", spec.short_len))
                for task, n in row_tasks:
                    n_prompt = n if task == "prefill" else spec.short_len
                    n_gen = 0 if task == "prefill" else args.decode_tokens
                    row = {
                        "round": round_i,
                        "model": spec.name,
                        "task": task,
                        "n_prompt": n_prompt,
                        "n_gen": n_gen,
                        "samples": [],
                    }
                    print(f"\nround {round_i}/{args.rounds} {spec.name} {task} p={n_prompt} n={n_gen}", flush=True)
                    for order_i, order in enumerate(orders, 1):
                        for engine_i, engine in enumerate(order, 1):
                            if args.prewarm_model:
                                prewarm_model(spec.path, args.prewarm_block_mb, args.prewarm_max_gb)
                            stem = f"r{round_i}_{spec.name}_{task}_p{n_prompt}_n{n_gen}_o{order_i}_{engine}"
                            if engine == "fucina":
                                cmd = fucina_cmd(spec, args, task, n)
                                log = run_logged(cmd, out_dir, stem, timeout_s)
                                stdout = (out_dir / log["stdout"]).read_text()
                                tok_s = parse_fucina(stdout, task)
                            else:
                                cmd = llama_cmd(spec, args, task, n)
                                log = run_logged(cmd, out_dir, stem, timeout_s)
                                stdout = (out_dir / log["stdout"]).read_text()
                                tok_s = parse_llama(stdout, task, n_prompt, n_gen)
                            row["samples"].append(
                                {
                                    "engine": engine,
                                    "order": order_i,
                                    "position": engine_i,
                                    "tok_s": tok_s,
                                    **log,
                                }
                            )
                            if args.cooldown_s > 0:
                                print(f"  cooldown {args.cooldown_s:.1f}s", flush=True)
                                time.sleep(args.cooldown_s)
                    summarize_row(row, args.min_ratio, args.max_cv, args.allow_noisy)
                    rows.append(row)
                    write_reports(out_dir, meta, rows)
                    print(
                        "{verdict}: Fucina {f:.2f} tok/s, llama {l:.2f} tok/s, ratio {r:.3f}, CV f/l {fcv:.1%}/{lcv:.1%}".format(
                            verdict=row["verdict"],
                            f=row.get("fucina_median_tok_s", 0.0),
                            l=row.get("llama_median_tok_s", 0.0),
                            r=row.get("ratio", 0.0),
                            fcv=row.get("fucina_cv", 0.0),
                            lcv=row.get("llama_cv", 0.0),
                        ),
                        flush=True,
                    )
    except Exception as exc:
        write_reports(out_dir, meta, rows)
        print(f"bench gate aborted: {exc}", file=sys.stderr)
        print(f"partial results: {out_dir}", file=sys.stderr)
        return 2

    failed = [row for row in rows if not row.get("pass")]
    write_reports(out_dir, meta, rows)
    print(f"\nsummary: {len(rows) - len(failed)} pass, {len(failed)} fail/noisy")
    print(f"raw results: {out_dir}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
