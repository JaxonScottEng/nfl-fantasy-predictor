"""
Slow tier: the real-data gates. Excluded by default and from CI.

    pytest -m slow

Two properties, both about drift:

1. The evaluation harness still reproduces train.py exactly. If it stops doing so,
   every number the harness reports describes something other than what training
   actually does.
2. The locked-in figures in CLAUDE.md, the regression hook and this test still
   agree. The expectations are IMPORTED from the hook rather than copied, so the
   three cannot silently diverge -- copying them here would create a fourth place
   to forget to update.
"""
import importlib.util
import os

import pytest

from conftest import PROJECT_ROOT

pytestmark = [pytest.mark.slow, pytest.mark.needs_data]


def _load_hook():
    hook_path = os.path.join(PROJECT_ROOT, ".claude", "hooks", "check_wr_regression.py")
    spec = importlib.util.spec_from_file_location("check_wr_regression", hook_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_harness_reproduces_training_exactly(skip_if_no_cache):
    from evaluate import check_fidelity

    failures = check_fidelity(verbose=False)
    assert failures == [], f"harness diverged from train.py: {failures}"


def test_locked_in_top_40_numbers_still_hold(skip_if_no_cache):
    import config
    from evaluate import run_evaluation, summarize

    expected = _load_hook().EXPECTED_MAE

    rows = run_evaluation(
        seasons=[config.VALIDATION_SEASON],
        pool="top_n_by_projection",
        verbose=False,
    )
    summary = summarize(rows, metric_ids=["mae"])

    for position, comparators in expected.items():
        for comparator, target in comparators.items():
            match = summary[(summary["position"] == position)
                            & (summary["comparator"] == comparator)]
            assert len(match) == 1, f"no result for {position}/{comparator}"
            actual = round(float(match["value"].iloc[0]), 3)
            assert actual == pytest.approx(target, abs=0.0005), (
                f"{position} {comparator}: expected {target}, got {actual}"
            )
