#!/usr/bin/env python3
"""Run the randomized B200 cotenant/SM-target cross-product.

Every matrix point is a separate mblas-bench process. Failed points are kept
visible, retried after the randomized pass, and cause a nonzero final status if
all attempts fail.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as dt
import json
import os
import pathlib
import random
import socket
import subprocess
import sys
from typing import Iterable


DEVICE_SMS = 148
ACTUAL_AVAILABLE_SMS = (148, 136, 128, 120, 112)
SM_COUNT_TARGETS = (0, 148, 136, 128, 120, 112)
DEFAULT_SIZES = (128, 256, 512, 1024, 2048, 4096, 8192)
DATATYPES = ("bf16", "xf32")
RUN_FIELDS = (
    "repeat",
    "datatype",
    "size",
    "actual_available_sms",
    "started_utc",
    "finished_utc",
    "attempt",
)


@dataclasses.dataclass(frozen=True)
class Point:
    repeat: int
    datatype: str
    size: int
    actual_available_sms: int
    cotenant_workgroups: int
    sm_count_target: int

    @property
    def key(self) -> tuple[int, str, int, int, int]:
        return (
            self.repeat,
            self.datatype,
            self.size,
            self.actual_available_sms,
            self.sm_count_target,
        )


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def generate_matrix(
    repeats: int = 5,
    sizes: Iterable[int] = DEFAULT_SIZES,
    seed: int = 20261009,
) -> list[Point]:
    """Generate deterministic config shuffles within each repeat/type/shape block."""
    points: list[Point] = []
    for repeat in range(1, repeats + 1):
        for datatype in DATATYPES:
            for size in sizes:
                configs = [
                    (available, DEVICE_SMS - available, target)
                    for available in ACTUAL_AVAILABLE_SMS
                    for target in SM_COUNT_TARGETS
                ]
                block_seed = f"{seed}:{repeat}:{datatype}:{size}"
                random.Random(block_seed).shuffle(configs)
                points.extend(
                    Point(repeat, datatype, size, available, workgroups, target)
                    for available, workgroups, target in configs
                )
    return points


def benchmark_command(binary: pathlib.Path, device: str, point: Point) -> list[str]:
    data_type = "bf16_r" if point.datatype == "bf16" else "f32_r"
    compute_type = "f32_r" if point.datatype == "bf16" else "xf32_r"
    size = str(point.size)
    return [
        str(binary),
        "-m", size,
        "-n", size,
        "-k", size,
        "--a_type", data_type,
        "--b_type", data_type,
        "--c_type", data_type,
        "--d_type", data_type,
        "--compute_type", compute_type,
        "--function", "matmul",
        "--driver", "cublaslt",
        "--device", device,
        "--iters", "20",
        "--cold_iters", "5",
        "--requested_solution_num", "1",
        "--cotenant_workgroups", str(point.cotenant_workgroups),
        "--cotenant_max_occupancy", "1",
        "--cublaslt_sm_count_target", str(point.sm_count_target),
    ]


def parse_benchmark_csv(stdout: str) -> dict[str, str]:
    lines = stdout.splitlines()
    for index, line in enumerate(lines):
        if not line.startswith("transA_option,"):
            continue
        header = next(csv.reader([line]))
        if header and header[-1] == "":
            header.pop()
        for value_line in lines[index + 1 :]:
            values = next(csv.reader([value_line]))
            if values and values[-1] == "":
                values.pop()
            if len(values) == len(header):
                return dict(zip(header, values))
        break
    raise ValueError("benchmark output did not contain one complete CSV row")


def validate_result(point: Point, row: dict[str, str]) -> None:
    expected = {
        "cotenant_workgroups": point.cotenant_workgroups,
        "cotenant_max_occupancy": 1,
        "cublaslt_sm_count_target": point.sm_count_target,
        "device_sm_count": DEVICE_SMS,
        "actual_cotenant_occupancy": 1 if point.cotenant_workgroups else 0,
    }
    for field, value in expected.items():
        if field not in row:
            raise ValueError(f"benchmark output is missing {field}")
        if int(row[field]) != value:
            raise ValueError(f"{field}={row[field]}, expected {value}")
    if point.cotenant_workgroups and int(row["cotenant_shared_bytes_per_block"]) <= 0:
        raise ValueError("contended result reports no shared-memory reservation")
    for field in (
        "heuristic_results_returned",
        "cublaslt_version",
        "heuristic_state",
        "heuristic_workspace_bytes",
        "heuristic_waves_count",
        "algo_id",
        "algo_tile_id",
        "algo_stages_id",
        "algo_splitk_num",
        "algo_reduction_scheme",
        "algo_cta_swizzle",
        "algo_custom_option",
        "algo_inner_shape_id",
        "algo_cluster_shape_id",
    ):
        if field not in row or row[field] == "":
            raise ValueError(f"benchmark output is missing {field}")
    if int(row["heuristic_results_returned"]) != 1:
        raise ValueError("primary experiment must return exactly one heuristic result")
    if int(row["heuristic_state"]) != 0:
        raise ValueError(f"heuristic result failed with state {row['heuristic_state']}")
    if int(row["cublaslt_version"]) <= 0:
        raise ValueError("benchmark reported an invalid cuBLASLt version")
    if int(row["heuristic_workspace_bytes"]) < 0:
        raise ValueError("benchmark reported a negative heuristic workspace size")


def command_record(command: list[str]) -> dict[str, object]:
    try:
        result = subprocess.run(command, text=True, capture_output=True, check=False)
        return {
            "command": command,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except OSError as error:
        return {"command": command, "returncode": None, "stdout": "", "stderr": str(error)}


def metadata(binary: pathlib.Path, args: argparse.Namespace) -> dict[str, object]:
    query = (
        "index,uuid,name,driver_version,temperature.gpu,power.draw,"
        "clocks.current.graphics,clocks.current.memory"
    )
    return {
        "started_utc": utc_now(),
        "hostname": socket.gethostname(),
        "environment": {
            name: os.environ.get(name, "")
            for name in (
                "CUDA_VISIBLE_DEVICES",
                "SLURM_JOB_ID",
                "SLURM_JOB_GPUS",
                "SLURM_GPUS_ON_NODE",
            )
        },
        "matrix": {
            "device_sms": DEVICE_SMS,
            "actual_available_sms": ACTUAL_AVAILABLE_SMS,
            "sm_count_targets": SM_COUNT_TARGETS,
            "datatypes": DATATYPES,
            "sizes": args.sizes,
            "repeats": args.repeats,
            "cold_iters": 5,
            "timed_iters": 20,
            "requested_solution_num": 1,
            "cotenant_max_occupancy": 1,
            "seed": args.seed,
            "randomization": "configuration order shuffled within each repeat/datatype/shape block",
        },
        "binary": str(binary.resolve()),
        "git_commit": command_record(["git", "rev-parse", "HEAD"]),
        "git_status": command_record(["git", "status", "--short"]),
        "nvidia_smi_L": command_record(["nvidia-smi", "-L"]),
        "nvidia_smi_query": command_record(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"]
        ),
        "nvcc_version": command_record(["nvcc", "--version"]),
    }


def run_point(
    binary: pathlib.Path,
    device: str,
    point: Point,
    attempt: int,
    log,
) -> tuple[dict[str, str] | None, dict[str, object] | None]:
    command = benchmark_command(binary, device, point)
    started = utc_now()
    environment = os.environ.copy()
    environment["CUBLAS_BENCH_FREQ"] = "1"
    result = subprocess.run(command, text=True, capture_output=True, env=environment, check=False)
    finished = utc_now()
    print(f"[{finished}] attempt={attempt} rc={result.returncode} {' '.join(command)}", file=log)
    if result.stdout:
        print(result.stdout, file=log, end="" if result.stdout.endswith("\n") else "\n")
    if result.stderr:
        print(result.stderr, file=log, end="" if result.stderr.endswith("\n") else "\n")
    log.flush()

    try:
        if result.returncode != 0:
            raise RuntimeError(f"benchmark exited with status {result.returncode}")
        row = parse_benchmark_csv(result.stdout)
        validate_result(point, row)
    except (RuntimeError, ValueError, KeyError) as error:
        return None, {
            "repeat": point.repeat,
            "datatype": point.datatype,
            "size": point.size,
            "actual_available_sms": point.actual_available_sms,
            "cotenant_workgroups": point.cotenant_workgroups,
            "cublaslt_sm_count_target": point.sm_count_target,
            "attempt": attempt,
            "started_utc": started,
            "finished_utc": finished,
            "returncode": result.returncode,
            "error": str(error),
            "stderr": result.stderr.strip(),
        }

    prefix = {
        "repeat": str(point.repeat),
        "datatype": point.datatype,
        "size": str(point.size),
        "actual_available_sms": str(point.actual_available_sms),
        "started_utc": started,
        "finished_utc": finished,
        "attempt": str(attempt),
    }
    return prefix | row, None


def write_csv(path: pathlib.Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=pathlib.Path, default=pathlib.Path("build/mblas-bench"))
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--sizes", type=int, nargs="+", default=list(DEFAULT_SIZES))
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.repeats < 1 or args.max_attempts < 1 or any(size < 1 for size in args.sizes):
        parser.error("repeats, max-attempts, and all sizes must be positive")
    if not args.device.isdigit():
        parser.error("the primary experiment requires exactly one numeric CUDA device")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    points = generate_matrix(args.repeats, args.sizes, args.seed)
    if args.dry_run:
        for point in points:
            print(json.dumps(dataclasses.asdict(point), sort_keys=True))
        print(f"points={len(points)}", file=sys.stderr)
        return 0
    if not args.binary.is_file():
        raise SystemExit(f"benchmark binary does not exist: {args.binary}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    meta = metadata(args.binary, args)
    meta_path = args.output_dir / "metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")

    successful: dict[tuple[int, str, int, int, int], dict[str, str]] = {}
    failures: list[dict[str, object]] = []
    pending = points
    with (args.output_dir / "full.log").open("w") as log, (
        args.output_dir / "progress.jsonl"
    ).open("w") as progress:
        for attempt in range(1, args.max_attempts + 1):
            next_pending: list[Point] = []
            for point in pending:
                row, failure = run_point(args.binary, args.device, point, attempt, log)
                if row is not None:
                    successful[point.key] = row
                    print(json.dumps({"status": "success", **row}), file=progress)
                else:
                    failures.append(failure or {})
                    next_pending.append(point)
                    print(json.dumps({"status": "failure", **(failure or {})}), file=progress)
                progress.flush()
            pending = next_pending
            if not pending:
                break

    ordered_rows = [successful[point.key] for point in points if point.key in successful]
    write_csv(args.output_dir / "raw.csv", ordered_rows)
    write_csv(args.output_dir / "failures.csv", failures)
    meta.update(
        {
            "finished_utc": utc_now(),
            "expected_rows": len(points),
            "completed_rows": len(successful),
            "failed_points": len(pending),
            "failure_attempts": len(failures),
        }
    )
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    if pending:
        print(f"{len(pending)} points still failed after {args.max_attempts} attempts", file=sys.stderr)
        return 1
    (args.output_dir / "COMPLETE").write_text(f"completed {len(points)} points at {utc_now()}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
