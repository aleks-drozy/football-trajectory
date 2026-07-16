"""Charts for the writeup. Run after run_validate.py and run_project.py.

Usage:
    python scripts/make_charts.py     # -> charts/*.png
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import schema as S
from model.aging import fit_aging_curve

ROOT = Path(__file__).resolve().parents[1]
CHARTS = ROOT / "charts"
RESULTS = ROOT / "results"

plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True, "grid.alpha": 0.3})


def chart_aging(panel: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    for ax, pos in zip(axes, ("FW", "MF", "DF")):
        for variant, style in (("survivors", "-"), ("dropouts", "--")):
            curve = fit_aging_curve(panel, S.NPG90, variant=variant)
            by_age = curve.curve.get(pos, {})
            ages = [a for a in sorted(by_age) if 17 <= a <= 36]
            if not ages:
                continue
            ax.plot(ages, [by_age[a] for a in ages], style, label=variant, linewidth=1.6)
        ax.axhline(0, color="k", linewidth=0.6)
        ax.axvline(21, color="grey", linewidth=0.6, linestyle=":")
        ax.set_title(f"{pos}: non-penalty goals/90 vs age")
        ax.set_xlabel("age")
    axes[0].set_ylabel("offset vs age 21")
    axes[0].legend(fontsize=8)
    fig.suptitle(
        "Aging curves — survivor bias makes decline look kinder than it is", fontsize=10
    )
    fig.tight_layout()
    fig.savefig(CHARTS / "aging_curves.png", bbox_inches="tight")
    plt.close(fig)


def chart_calibration(gate: dict) -> None:
    fig, ax = plt.subplots(figsize=(6, 3.6))
    metrics = list(gate["gates"])
    width = 0.35
    for i, metric in enumerate(metrics):
        hs = [g["horizon"] for g in gate["gates"][metric]]
        cov = [g["calibration"]["coverage"] for g in gate["gates"][metric]]
        x = np.arange(len(hs)) + (i - 0.5) * width
        bars = ax.bar(x, cov, width, label=metric)
        for b, c in zip(bars, cov):
            ax.text(b.get_x() + b.get_width() / 2, c + 0.01, f"{c:.0%}",
                    ha="center", fontsize=7)
    ax.axhspan(0.72, 0.88, color="green", alpha=0.12, label="pass band (72-88%)")
    ax.axhline(0.80, color="green", linestyle="--", linewidth=1, label="nominal 80%")
    ax.set_xticks(np.arange(len(gate["gates"][metrics[0]])))
    ax.set_xticklabels([f"+{g['horizon']}" for g in gate["gates"][metrics[0]]])
    ax.set_xlabel("horizon (seasons ahead)")
    ax.set_ylabel("actuals inside the 80% interval")
    ax.set_ylim(0, 1.05)
    ax.set_title("Calibration against the pre-registered band")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(CHARTS / "calibration.png", bbox_inches="tight")
    plt.close(fig)


def chart_skill(gate: dict) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    labels, diffs, los, his = [], [], [], []
    for metric in gate["gates"]:
        for g in gate["gates"][metric]:
            labels.append(f"{metric[:3]} +{g['horizon']}")
            diffs.append(g["skill"]["diff"])
            los.append(g["skill"]["ci_lo"])
            his.append(g["skill"]["ci_hi"])
    y = np.arange(len(labels))
    err = [np.array(diffs) - np.array(los), np.array(his) - np.array(diffs)]
    colors = ["tab:green" if h < 0 else "tab:red" for h in his]
    ax.errorbar(diffs, y, xerr=err, fmt="o", ecolor="grey", capsize=3, linestyle="none")
    ax.scatter(diffs, y, c=colors, zorder=3)
    ax.axvline(0, color="k", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("MAE difference vs best baseline (negative = model better)")
    ax.set_title("Skill: the whole CI must sit left of zero to pass")
    fig.tight_layout()
    fig.savefig(CHARTS / "skill.png", bbox_inches="tight")
    plt.close(fig)


def chart_k_grid(gate: dict) -> None:
    fig, ax = plt.subplots(figsize=(6, 3.4))
    for metric, scores in gate["K_scores"].items():
        ks = sorted(float(k) for k in scores)
        vals = [scores[str(k)] if str(k) in scores else scores[f"{k}"] for k in ks]
        ax.plot(ks, vals, "o-", label=f"{metric} (best K={gate['fitted_K'][metric]})")
    ax.set_xscale("log")
    ax.set_xlabel("shrinkage constant K (in 90s)")
    ax.set_ylabel("weighted MAE (training)")
    ax.set_title("K is fitted, not chosen")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(CHARTS / "k_grid.png", bbox_inches="tight")
    plt.close(fig)


def chart_projection(proj: dict) -> None:
    players = proj["players"]
    trusted = proj["verdict"]["any_horizon_proven"]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    names = [p["player"] for p in players]
    med = [p["future_non_penalty_goals"]["median"] for p in players]
    lo = [p["future_non_penalty_goals"]["p10"] for p in players]
    hi = [p["future_non_penalty_goals"]["p90"] for p in players]
    order = np.argsort(med)
    y = np.arange(len(names))
    med, lo, hi = np.array(med)[order], np.array(lo)[order], np.array(hi)[order]
    names = [names[i] for i in order]
    ax.errorbar(med, y, xerr=[med - lo, hi - med], fmt="o", capsize=3,
                color="tab:blue" if trusted else "tab:red", ecolor="grey")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("projected remaining Big-5 league non-penalty goals (median, 10-90%)")
    title = "Career projections"
    if not trusted:
        title += " — UNCALIBRATED, shown to demonstrate the failure"
    ax.set_title(title, fontsize=10, color="black" if trusted else "darkred")
    fig.tight_layout()
    fig.savefig(CHARTS / "projections.png", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    CHARTS.mkdir(exist_ok=True)
    panel = pd.read_parquet(ROOT / "data" / "panel.parquet")
    chart_aging(panel)
    print("-> charts/aging_curves.png")

    gate_path = RESULTS / "gate.json"
    if gate_path.exists():
        gate = json.loads(gate_path.read_text())
        chart_calibration(gate)
        chart_skill(gate)
        chart_k_grid(gate)
        print("-> charts/calibration.png, skill.png, k_grid.png")

    proj_path = RESULTS / "projections.json"
    if proj_path.exists():
        chart_projection(json.loads(proj_path.read_text()))
        print("-> charts/projections.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
