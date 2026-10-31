import pandas as pd
import pytest

from modelpilot.cost import add_cost, query_cost


def test_query_cost_uses_per_million_prices():
    # 129 input tokens at $0.27/1M + 367 output tokens at $0.85/1M
    assert query_cost(129, 367, 0.27, 0.85) == pytest.approx(0.00034678)


def test_query_cost_zero_output_tokens_costs_input_only():
    assert query_cost(1_000_000, 0, 0.2, 5.0) == pytest.approx(0.2)


def test_add_cost_joins_prices_per_model(candidates):
    results = pd.DataFrame(
        {"model_name": ["cheap", "pricey"], "input_tokens": [1_000_000, 1_000_000], "output_tokens": [0, 1_000_000]}
    )
    with_cost = add_cost(results, candidates)
    assert with_cost["cost"].tolist() == pytest.approx([0.1, 6.0])


def test_add_cost_rejects_unknown_model(candidates):
    results = pd.DataFrame({"model_name": ["mystery"], "input_tokens": [10], "output_tokens": [10]})
    with pytest.raises(ValueError):
        add_cost(results, candidates)
