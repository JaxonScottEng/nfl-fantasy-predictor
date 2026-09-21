"""
The registry seam, plus a guard that the ids other modules depend on still resolve.

CLAUDE.md claims a "registry + recipe" extension convention. These tests make that
claim enforceable: adding a metric, pool or comparator is a decorated function, and
a rename that would break the GUI at runtime fails here instead.
"""
import pytest

from registry import Registry


def test_register_and_retrieve():
    reg = Registry("widget")

    @reg.register("alpha", label="Alpha", caveat="approximate", size=3)
    def alpha():
        return "a"

    assert reg.get("alpha")() == "a"
    assert reg.available() == ["alpha"]
    assert reg.label("alpha") == "Alpha"
    assert reg.caveat("alpha") == "approximate"
    assert reg.meta("alpha", "size") == 3
    assert "alpha" in reg
    assert len(reg) == 1


def test_duplicate_id_is_rejected():
    reg = Registry("widget")
    reg.register("alpha")(lambda: None)

    with pytest.raises(ValueError, match="already registered"):
        reg.register("alpha")(lambda: None)


def test_unknown_id_lists_what_is_available():
    reg = Registry("widget")
    reg.register("alpha")(lambda: None)

    with pytest.raises(KeyError) as excinfo:
        reg.get("missing")
    assert "alpha" in str(excinfo.value)


def test_label_defaults_to_the_id():
    reg = Registry("widget")
    reg.register("alpha")(lambda: None)
    assert reg.label("alpha") == "alpha"


# --- integration guards -------------------------------------------------------

def test_shipped_comparator_resolves():
    from comparators import COMPARATORS, SHIPPED_COMPARATOR_ID
    assert SHIPPED_COMPARATOR_ID in COMPARATORS


def test_default_comparator_ids_all_resolve():
    from comparators import COMPARATORS, DEFAULT_COMPARATOR_IDS, ALL_POINT_COMPARATOR_IDS
    for comparator_id in set(DEFAULT_COMPARATOR_IDS) | set(ALL_POINT_COMPARATOR_IDS):
        assert COMPARATORS.get(comparator_id) is not None


def test_default_metric_ids_all_resolve():
    from metrics import METRICS, DEFAULT_METRIC_IDS
    for metric_id in DEFAULT_METRIC_IDS:
        assert METRICS.get(metric_id) is not None


def test_gui_range_methods_resolve():
    """The GUI builds its selector from available_methods(); a rename breaks it."""
    from gui import prediction_ranges

    methods = prediction_ranges.available_methods()
    assert methods, "at least one range method must be registered"
    for method in methods:
        assert prediction_ranges.METHOD_LABELS.get(method)
        assert prediction_ranges.METHOD_CAVEATS.get(method)


def test_hook_expectations_cover_every_active_position():
    """
    The regression hook hardcodes the locked-in numbers. If a position is added to
    config without updating the hook, drift stops being detected for it.
    """
    import importlib.util
    import os

    import config
    from conftest import PROJECT_ROOT

    hook_path = os.path.join(PROJECT_ROOT, ".claude", "hooks", "check_wr_regression.py")
    spec = importlib.util.spec_from_file_location("check_wr_regression", hook_path)
    hook = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hook)

    for position in config.ACTIVE_POSITIONS:
        assert position in hook.EXPECTED_MAE, f"hook has no expectations for {position}"
