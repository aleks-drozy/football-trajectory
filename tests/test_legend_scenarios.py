"""Tests for the legend-arc scenarios.

The two guards on `improving_like` are the whole point of this module: without
them the page publishes a youngster scoring 2.8 goals/90 and reaching 1,800
career goals, which is not a scenario but an arithmetic artefact. So the tests
pin the guards, not the arithmetic.
"""

from __future__ import annotations

import pytest

from scripts.legend_scenarios import SCALE_BAND, build_scenarios

# A legend who was already elite young: rate 0.60 at 18, peaks at 1.20 (2x).
ELITE_YOUNG = {18: (2000, 13), 19: (2000, 20), 20: (2000, 27)}
# A legend who was a nobody young: rate 0.09 at 18, peaks at 1.35 (15x!).
LATE_BLOOMER = {18: (2000, 2), 19: (2000, 15), 20: (2000, 30)}

CEILING = 1.96


def _kid(rate: float, age: int = 18, so_far: int = 30) -> dict:
    return {"age": age, "league": "ESP-La Liga", "rate": rate, "goals_so_far": so_far}


def test_at_rate_gives_the_legends_own_goals():
    """'Performing at his rate' means inheriting his actual goals, so the total
    is simply what he already has plus the legend's remaining output."""
    out = build_scenarios(_kid(0.60), ELITE_YOUNG, CEILING)
    sc = out["at_rate"]
    assert sc["career_total"] == 30 + 20 + 27
    assert [p["age"] for p in sc["path"]] == [19, 20]


def test_at_rate_is_independent_of_the_youngsters_own_rate():
    """It is the legend's rate by construction — a weaker kid at the same age
    and the same goals-so-far lands in the same place."""
    a = build_scenarios(_kid(0.60), ELITE_YOUNG, CEILING)["at_rate"]
    b = build_scenarios(_kid(0.10), ELITE_YOUNG, CEILING)["at_rate"]
    assert a["career_total"] == b["career_total"]


def test_improving_like_survives_a_comparable_anchor():
    """Kid at 0.60, legend at 0.65 => scale 0.92, multiplier peaks at 2x =>
    implied peak 1.2/90. Sane on both counts, so the mode is offered."""
    out = build_scenarios(_kid(0.60), ELITE_YOUNG, CEILING)
    sc = out["improving_like"]
    assert sc is not None
    assert SCALE_BAND[0] <= sc["anchor_scale"] <= SCALE_BAND[1]
    assert max(p["rate_per90"] for p in sc["path"]) <= CEILING


def test_improving_like_suppressed_when_not_comparable():
    """An elite kid against a late-bloomer's teenage baseline: scale 6.7x. The
    multiplier would be measuring the legend's weakness, not his growth."""
    out = build_scenarios(_kid(0.60), LATE_BLOOMER, CEILING)
    assert out["improving_like"] is None
    assert "not comparable" in out["improving_like_reason"]


def test_a_late_bloomers_own_growth_is_transplantable_to_a_similar_kid():
    """The ceiling must not be trigger-happy: a kid who starts exactly where the
    late bloomer started, and grows exactly as he did, lands on the late
    bloomer's own peak (1.35/90) — steep, but a real career, so it is allowed."""
    out = build_scenarios(_kid(0.09), LATE_BLOOMER, CEILING)
    assert out["improving_like"] is not None
    assert max(p["rate_per90"] for p in out["improving_like"]["path"]) == pytest.approx(1.35)


def test_improving_like_suppressed_when_the_growth_factor_is_untransplantable():
    """The ratio guard alone is insufficient — this is the Endrick/Ronaldo case.

    The kid sits *inside* the scale band (0.18 vs the legend's 0.09 is 2.0x, the
    edge), so guard 1 waves it through. But the legend's own peak is 15x his
    teenage rate, so the scenario implies 0.18 x 15 = 2.7 goals/90 — nearly
    double the best season ever played. Only the ceiling catches this.
    """
    out = build_scenarios(_kid(0.18), LATE_BLOOMER, CEILING)
    assert out["improving_like"] is None
    assert "plausibility ceiling" in out["improving_like_reason"]


def test_a_suppressed_mode_still_leaves_at_rate_usable():
    """Suppression is per-mode: the robust scenario must survive."""
    out = build_scenarios(_kid(0.60), LATE_BLOOMER, CEILING)
    assert out["improving_like"] is None
    assert out["at_rate"] is not None


def test_unavailable_when_the_legend_has_no_season_at_that_age():
    out = build_scenarios(_kid(0.60, age=25), ELITE_YOUNG, CEILING)
    assert out["at_rate"] is None and out["improving_like"] is None
    assert "no season at age 25" in out["unavailable"]


def test_unavailable_when_the_arc_ends_at_the_youngsters_age():
    out = build_scenarios(_kid(0.60, age=20), ELITE_YOUNG, CEILING)
    assert out["at_rate"] is None
    assert "ends at 20" in out["unavailable"]


def test_cumulative_path_is_monotonic_and_starts_above_goals_so_far():
    sc = build_scenarios(_kid(0.60, so_far=42), ELITE_YOUNG, CEILING)["at_rate"]
    cums = [p["cumulative"] for p in sc["path"]]
    assert cums[0] >= 42
    assert cums == sorted(cums)


def test_record_flag_reports_whether_the_scenario_beats_it():
    """La Liga record is 474; this toy arc cannot get near it."""
    sc = build_scenarios(_kid(0.60), ELITE_YOUNG, CEILING)["at_rate"]
    assert sc["record"]["goals"] == 474
    assert sc["record"]["beats"] is False
    assert sc["record"]["broken_at_age"] is None


def test_zero_baseline_legend_does_not_divide_by_zero():
    arc = {18: (2000, 0), 19: (2000, 20)}
    out = build_scenarios(_kid(0.60), arc, CEILING)
    assert out["improving_like"] is None
    assert "did not score" in out["improving_like_reason"]
    assert out["at_rate"] is not None
