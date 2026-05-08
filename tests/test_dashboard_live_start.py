from __future__ import annotations

from edge_iiot_dashboard import sum_numeric_values


def test_sum_numeric_values_ignores_non_numeric_values() -> None:
    assert sum_numeric_values([1, "2", "error", None, 3.0]) == 6
