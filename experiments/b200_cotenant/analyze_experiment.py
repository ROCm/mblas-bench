#!/usr/bin/env python3
"""Validate and summarize B200 cotenant/SM-target experiment results."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import statistics
from collections import defaultdict


ALGO_FIELDS = (
    "algo_id",
    "algo_tile_id",
    "algo_stages_id",
    "algo_splitk_num",
    "algo_reduction_scheme",
    "algo_cta_swizzle",
    "algo_custom_option",
    "algo_inner_shape_id",
    "algo_cluster_shape_id",
)
T_CRITICAL_975 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


def read_csv(path: pathlib.Path) -> list[dict[str, str]]:
    with path.open(newline="") as source:
        return list(csv.DictReader(source))


def write_csv(path: pathlib.Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean_ci95(values: list[float]) -> tuple[float, float, float]:
    mean = statistics.mean(values)
    if len(values) < 2:
        return mean, math.nan, math.nan
    standard_error = statistics.stdev(values) / math.sqrt(len(values))
    critical = T_CRITICAL_975.get(len(values) - 1, 1.96)
    margin = critical * standard_error
    return mean, mean - margin, mean + margin


def signature(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(row[field] for field in ALGO_FIELDS)


def expected_keys(metadata: dict[str, object]) -> set[tuple[int, str, int, int, int]]:
    matrix = metadata["matrix"]
    assert isinstance(matrix, dict)
    return {
        (repeat, datatype, int(size), int(available), int(target))
        for repeat in range(1, int(matrix["repeats"]) + 1)
        for datatype in matrix["datatypes"]
        for size in matrix["sizes"]
        for available in matrix["actual_available_sms"]
        for target in matrix["sm_count_targets"]
    }


def row_key(row: dict[str, str]) -> tuple[int, str, int, int, int]:
    return (
        int(row["repeat"]),
        row["datatype"],
        int(row["size"]),
        int(row["actual_available_sms"]),
        int(row["cublaslt_sm_count_target"]),
    )


def validate(rows: list[dict[str, str]], metadata: dict[str, object]) -> None:
    keys = [row_key(row) for row in rows]
    duplicates = len(keys) - len(set(keys))
    expected = expected_keys(metadata)
    missing = expected - set(keys)
    unexpected = set(keys) - expected
    if duplicates or missing or unexpected:
        raise ValueError(
            f"incomplete matrix: duplicates={duplicates}, missing={len(missing)}, "
            f"unexpected={len(unexpected)}"
        )
    uuids = {row["device_uuid"] for row in rows if row.get("device_uuid")}
    if any(not row.get("device_uuid") for row in rows) or len(uuids) != 1:
        raise ValueError(f"results contain invalid or multiple GPU UUIDs: {sorted(uuids)}")
    versions = {row["cublaslt_version"] for row in rows}
    if "" in versions or len(versions) != 1:
        raise ValueError(f"results contain invalid or mixed cuBLASLt versions: {sorted(versions)}")
    matrix = metadata["matrix"]
    assert isinstance(matrix, dict)
    device_sms = int(matrix["device_sms"])
    for row in rows:
        workgroups = int(row["cotenant_workgroups"])
        occupancy = int(row["actual_cotenant_occupancy"])
        available = int(row["actual_available_sms"])
        if int(row["device_sm_count"]) != device_sms:
            raise ValueError(f"unexpected device SM count for {row_key(row)}")
        if int(row["cotenant_max_occupancy"]) != 1:
            raise ValueError(f"primary matrix must request occupancy one: {row_key(row)}")
        if workgroups != device_sms - available:
            raise ValueError(f"workgroup/availability mismatch for {row_key(row)}")
        if occupancy != (1 if workgroups else 0):
            raise ValueError(
                f"invalid cotenant occupancy for {row_key(row)}: {occupancy}"
            )
        if workgroups and int(row["cotenant_shared_bytes_per_block"]) <= 0:
            raise ValueError(f"missing cotenant shared-memory reservation for {row_key(row)}")
        if int(row["shared_bytes_per_sm"]) <= 0:
            raise ValueError(f"missing shared-memory-per-SM metadata for {row_key(row)}")
        if int(row["solution_index"]) != 0:
            raise ValueError(f"primary matrix must use the first heuristic solution: {row_key(row)}")


def summarize(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, int, int, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = (
            row["datatype"],
            int(row["size"]),
            int(row["actual_available_sms"]),
            int(row["cublaslt_sm_count_target"]),
        )
        groups[key].append(row)

    result: list[dict[str, object]] = []
    for key, group in sorted(groups.items()):
        throughput = [float(row["cuBLAS-Gflops"]) for row in group]
        times = [float(row["cuBLAS-us"]) for row in group]
        throughput_mean = statistics.mean(throughput)
        result.append(
            {
                "datatype": key[0],
                "size": key[1],
                "actual_available_sms": key[2],
                "cublaslt_sm_count_target": key[3],
                "samples": len(group),
                "throughput_mean_gflops": throughput_mean,
                "throughput_median_gflops": statistics.median(throughput),
                "throughput_stdev_gflops": statistics.stdev(throughput) if len(group) > 1 else 0.0,
                "throughput_cv_percent": (
                    100.0 * statistics.stdev(throughput) / throughput_mean
                    if len(group) > 1 and throughput_mean
                    else 0.0
                ),
                "time_mean_us": statistics.mean(times),
                "time_median_us": statistics.median(times),
                "algorithm_variants": len({signature(row) for row in group}),
                "cublaslt_version": group[0]["cublaslt_version"],
                "workspace_bytes_values": "|".join(
                    sorted({row["heuristic_workspace_bytes"] for row in group}, key=int)
                ),
                "waves_count_values": "|".join(sorted({row["heuristic_waves_count"] for row in group})),
                "min_avg_sysclk_mhz": min(float(row.get("avg_sysclk_mhz", 0) or 0) for row in group),
                "max_avg_sysclk_mhz": max(float(row.get("avg_sysclk_mhz", 0) or 0) for row in group),
                "min_avg_memclk_mhz": min(float(row.get("avg_memclk_mhz", 0) or 0) for row in group),
                "max_avg_memclk_mhz": max(float(row.get("avg_memclk_mhz", 0) or 0) for row in group),
                "min_temperature_c": min(float(row.get("min_temperature_c", 0) or 0) for row in group),
                "max_temperature_c": max(float(row.get("max_temperature_c", 0) or 0) for row in group),
                "min_power_w": min(float(row.get("min_power_w", 0) or 0) for row in group),
                "max_power_w": max(float(row.get("max_power_w", 0) or 0) for row in group),
            }
        )
    return result


def paired_ratios(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    indexed = {row_key(row): row for row in rows}
    groups: dict[tuple[str, int, int, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        available = int(row["actual_available_sms"])
        target = int(row["cublaslt_sm_count_target"])
        if target == 0:
            continue
        repeat = int(row["repeat"])
        base_key = (repeat, row["datatype"], int(row["size"]), available)
        no_target = indexed[base_key + (0,)]
        correct_target = indexed[base_key + (available,)]
        groups[(row["datatype"], int(row["size"]), available, target)].append(
            {
                "repeat": repeat,
                "ratio_vs_no_target": (
                    float(row["cuBLAS-Gflops"]) / float(no_target["cuBLAS-Gflops"])
                ),
                "ratio_vs_correct_target": (
                    float(row["cuBLAS-Gflops"]) / float(correct_target["cuBLAS-Gflops"])
                ),
                "algorithm_changed_vs_no_target": signature(row) != signature(no_target),
                "algorithm_changed_vs_correct_target": signature(row) != signature(correct_target),
            }
        )

    result: list[dict[str, object]] = []
    for key, pairs in sorted(groups.items()):
        ratios_vs_no = [float(pair["ratio_vs_no_target"]) for pair in pairs]
        ratios_vs_correct = [float(pair["ratio_vs_correct_target"]) for pair in pairs]
        mean_no, low_no, high_no = mean_ci95(ratios_vs_no)
        mean_correct, low_correct, high_correct = mean_ci95(ratios_vs_correct)
        relation = "exact" if key[3] == key[2] else ("optimistic" if key[3] > key[2] else "conservative")
        result.append(
            {
                "datatype": key[0],
                "size": key[1],
                "actual_available_sms": key[2],
                "cublaslt_sm_count_target": key[3],
                "target_relation": relation,
                "samples": len(pairs),
                "target_over_no_target_mean": mean_no,
                "target_over_no_target_ci95_low": low_no,
                "target_over_no_target_ci95_high": high_no,
                "target_over_correct_target_mean": mean_correct,
                "target_over_correct_target_ci95_low": low_correct,
                "target_over_correct_target_ci95_high": high_correct,
                "algorithm_changed_vs_no_target_repeats": sum(
                    bool(pair["algorithm_changed_vs_no_target"]) for pair in pairs
                ),
                "algorithm_changed_vs_correct_target_repeats": sum(
                    bool(pair["algorithm_changed_vs_correct_target"]) for pair in pairs
                ),
            }
        )
    return result


def report_markdown(
    rows: list[dict[str, str]],
    summary: list[dict[str, object]],
    ratios: list[dict[str, object]],
) -> str:
    changed = sum(int(row["algorithm_changed_vs_no_target_repeats"]) for row in ratios)
    return "\n".join(
        [
            "# B200 cotenant SM-target analysis",
            "",
            f"- Complete invocation rows: {len(rows)}",
            f"- Summary groups: {len(summary)}",
            f"- Paired target/no-target/correct-target groups: {len(ratios)}",
            f"- Target/no-target repeat pairs with a changed algorithm: {changed}",
            "",
            "`summary.csv` contains throughput/time dispersion, algorithm variants, and telemetry ranges.",
            "`paired_ratios.csv` contains repeat-paired exact, optimistic, and conservative target ratios against both no-target and correct-target controls, with 95% Student-t intervals.",
            "Selection effects are identified by changed algorithm configurations; ratios without a configuration change isolate execution effects more closely.",
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=pathlib.Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metadata = json.loads((args.results_dir / "metadata.json").read_text())
    rows = read_csv(args.results_dir / "raw.csv")
    validate(rows, metadata)
    summary = summarize(rows)
    ratios = paired_ratios(rows)
    write_csv(args.results_dir / "summary.csv", summary)
    write_csv(args.results_dir / "paired_ratios.csv", ratios)
    (args.results_dir / "analysis.md").write_text(
        report_markdown(rows, summary, ratios)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
