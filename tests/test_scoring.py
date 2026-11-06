"""model_summary, evaluate and the baseline strategies."""

import pandas as pd
import pytest

from modelpilot.baselines import always, best_model, cheapest_model, oracle
from modelpilot.evaluation import evaluate
from modelpilot.report import pareto_frontier
from modelpilot.summary import model_summary


def test_model_summary_averages_and_sorts_best_first(results):
    summary = model_summary(results)
    assert summary.index.tolist() == ["pricey", "cheap", "mid"]  # cheap beats mid on cost tie-break
    assert summary.loc["pricey", "avg_score"] == pytest.approx(2 / 3)
    assert summary.loc["cheap", "avg_cost"] == pytest.approx(0.001)
    assert summary.loc["cheap", "queries"] == 3


def test_cheapest_and_best_model(results):
    summary = model_summary(results)
    assert cheapest_model(summary) == "cheap"
    assert best_model(summary) == "pricey"


def test_always_routes_every_query_to_one_model(results):
    choices = always(results, "mid")
    assert sorted(choices.index) == [0, 1, 2]
    assert set(choices) == {"mid"}


def test_oracle_prefers_highest_score_then_lowest_cost(results):
    choices = oracle(results)
    assert choices[0] == "cheap"   # everyone right -> cheapest
    assert choices[1] == "pricey"  # only pricey right
    assert choices[2] == "cheap"   # nobody right -> all tie at 0 -> cheapest


def test_oracle_uses_partial_scores():
    results = pd.DataFrame(
        {"embedding_id": [0, 0], "model_name": ["a", "b"], "performance": [0.4, 0.6], "cost": [0.1, 0.2]}
    )
    assert oracle(results)[0] == "b"


def test_evaluate_reports_avg_score_and_total_cost(results):
    metrics = evaluate(results, oracle(results))
    assert metrics["avg_score"] == pytest.approx(2 / 3)
    assert metrics["total_cost"] == pytest.approx(0.001 + 0.1 + 0.001)
    assert metrics["queries"] == 3


def test_evaluate_rejects_choice_missing_from_results(results):
    with pytest.raises(ValueError):
        evaluate(results, pd.Series({0: "unknown-model"}))


def test_pareto_frontier_keeps_only_undominated_rows():
    table = pd.DataFrame(
        {"router": ["a", "b", "c", "d"], "total_cost": [1.0, 2.0, 3.0, 2.5], "avg_score": [0.5, 0.7, 0.9, 0.6]}
    )
    assert pareto_frontier(table)["router"].tolist() == ["a", "b", "c"]  # d costs more than b and scores less
