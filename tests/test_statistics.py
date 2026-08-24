from __future__ import annotations

import unittest

from src.benchmark.statistics import summarize_latencies


class LatencyStatisticsTests(unittest.TestCase):
    def test_summary_contains_tail_percentiles_and_throughput(self) -> None:
        result = summarize_latencies([1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertEqual(result["count"], 5)
        self.assertEqual(result["median_ms"], 3.0)
        self.assertEqual(result["minimum_ms"], 1.0)
        self.assertEqual(result["maximum_ms"], 5.0)
        self.assertGreaterEqual(result["p99_ms"], result["p95_ms"])
        self.assertAlmostEqual(
            result["sequential_throughput_per_second_from_median"], 1000 / 3
        )

    def test_non_positive_values_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            summarize_latencies([1.0, 0.0])


if __name__ == "__main__":
    unittest.main()

