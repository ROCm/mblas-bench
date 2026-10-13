from __future__ import annotations

import unittest
import pathlib
import tempfile

from experiments.b200_cotenant.analyze_experiment import mean_ci95, paired_ratios
from experiments.b200_cotenant.run_experiment import (
    ACTUAL_AVAILABLE_SMS,
    DATATYPES,
    DEFAULT_SIZES,
    SM_COUNT_TARGETS,
    Point,
    generate_matrix,
    parse_benchmark_csv,
    validate_result,
    write_csv,
)


class MatrixTest(unittest.TestCase):
    def test_full_matrix_is_complete_unique_and_deterministic(self) -> None:
        matrix = generate_matrix()
        expected = 5 * len(DATATYPES) * len(DEFAULT_SIZES) * len(ACTUAL_AVAILABLE_SMS) * len(SM_COUNT_TARGETS)
        self.assertEqual(expected, 2100)
        self.assertEqual(len(matrix), expected)
        self.assertEqual(len({point.key for point in matrix}), expected)
        self.assertEqual(matrix, generate_matrix())

    def test_actual_pressure_and_reported_target_are_independent(self) -> None:
        block = generate_matrix(repeats=1, sizes=[1024])
        pairs = {
            (point.actual_available_sms, point.cotenant_workgroups, point.sm_count_target)
            for point in block
            if point.datatype == "bf16"
        }
        expected = {
            (available, 148 - available, target)
            for available in ACTUAL_AVAILABLE_SMS
            for target in SM_COUNT_TARGETS
        }
        self.assertEqual(pairs, expected)


class ParsingTest(unittest.TestCase):
    def test_empty_csv_replaces_stale_results(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "raw.csv"
            path.write_text("stale,data\n1,2\n")
            write_csv(path, [])
            self.assertEqual(path.read_text(), "")

    def test_parses_trailing_comma_csv_amid_logs(self) -> None:
        output = "note\ntransA_option,cublaslt_sm_count_target,cuBLAS-us,\nN,148,10.5,\n"
        self.assertEqual(
            parse_benchmark_csv(output),
            {"transA_option": "N", "cublaslt_sm_count_target": "148", "cuBLAS-us": "10.5"},
        )

    def test_student_t_interval_contains_mean(self) -> None:
        mean, low, high = mean_ci95([1.0, 1.1, 0.9, 1.05, 0.95])
        self.assertLess(low, mean)
        self.assertLess(mean, high)

    def test_validates_complete_solution_metadata(self) -> None:
        point = Point(1, "bf16", 1024, 128, 20, 128)
        row = {
            "cotenant_workgroups": "20",
            "cotenant_max_occupancy": "1",
            "cublaslt_sm_count_target": "128",
            "device_sm_count": "148",
            "device_uuid": "GPU-00000000-0000-0000-0000-000000000001",
            "actual_cotenant_occupancy": "1",
            "cotenant_shared_bytes_per_block": "232448",
            "heuristic_results_returned": "1",
            "cublaslt_version": "130000",
            "heuristic_state": "0",
            "heuristic_workspace_bytes": "4194304",
            "heuristic_waves_count": "1.0",
        }
        row.update({field: "0" for field in (
            "algo_id", "algo_tile_id", "algo_stages_id", "algo_splitk_num",
            "algo_reduction_scheme", "algo_cta_swizzle", "algo_custom_option",
            "algo_inner_shape_id", "algo_cluster_shape_id",
        )})
        validate_result(point, row)

    def test_pairs_exact_optimistic_and_conservative_targets(self) -> None:
        rows = []
        for repeat in range(1, 3):
            for target, throughput, algo in (
                (0, 100.0, "1"),
                (148, 102.0, "2"),
                (128, 110.0, "3"),
                (120, 99.0, "1"),
            ):
                row = {
                    "repeat": str(repeat),
                    "datatype": "bf16",
                    "size": "1024",
                    "actual_available_sms": "128",
                    "cublaslt_sm_count_target": str(target),
                    "cuBLAS-Gflops": str(throughput),
                }
                row.update({field: algo for field in (
                    "algo_id", "algo_tile_id", "algo_stages_id",
                    "algo_splitk_num", "algo_reduction_scheme",
                    "algo_cta_swizzle", "algo_custom_option",
                    "algo_inner_shape_id", "algo_cluster_shape_id",
                )})
                rows.append(row)
        comparisons = paired_ratios(rows)
        self.assertEqual(
            {row["target_relation"] for row in comparisons},
            {"exact", "optimistic", "conservative"},
        )
        exact = next(row for row in comparisons if row["target_relation"] == "exact")
        self.assertAlmostEqual(exact["target_over_no_target_mean"], 1.1)


if __name__ == "__main__":
    unittest.main()
