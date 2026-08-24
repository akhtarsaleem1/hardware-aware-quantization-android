from __future__ import annotations

import unittest

from src.selection.pareto import pareto_efficient_indices


class ParetoTests(unittest.TestCase):
    def test_dominated_configuration_is_removed(self) -> None:
        rows = [
            {"accuracy": 0.90, "latency": 10.0, "size": 5.0},
            {"accuracy": 0.90, "latency": 12.0, "size": 6.0},
            {"accuracy": 0.92, "latency": 15.0, "size": 4.0},
        ]
        result = pareto_efficient_indices(
            rows, maximize=["accuracy"], minimize=["latency", "size"]
        )
        self.assertEqual(result, [0, 2])

    def test_missing_objective_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            pareto_efficient_indices(
                [{"accuracy": 0.9}], maximize=["accuracy"], minimize=["latency"]
            )


if __name__ == "__main__":
    unittest.main()

