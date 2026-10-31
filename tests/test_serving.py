"""Which model /chat actually calls, given what NVIDIA serves."""

import pytest

from modelpilot.serving import ServedModel, choose_served_model
from modelpilot.summary import model_summary


def test_routed_model_used_when_available(candidates, results):
    served = choose_served_model("mid", candidates, model_summary(results), {"v/mid", "v/cheap"})
    assert served == ServedModel("mid", "v/mid", fallback=False)


def test_falls_back_to_best_available_by_train_score(candidates, results):
    # pricey is routed but unavailable; of the available ones, cheap beats mid on the cost tie-break.
    served = choose_served_model("pricey", candidates, model_summary(results), {"v/cheap", "v/mid"})
    assert served == ServedModel("cheap", "v/cheap", fallback=True)


def test_fallback_ignores_non_candidate_ids(candidates, results):
    served = choose_served_model("cheap", candidates, model_summary(results), {"v/pricey", "other/model"})
    assert served == ServedModel("pricey", "v/pricey", fallback=True)


def test_raises_when_no_candidate_is_available(candidates, results):
    with pytest.raises(RuntimeError):
        choose_served_model("cheap", candidates, model_summary(results), {"other/model"})
