# How to read this repo

Start here, then follow whichever thread you care about. Nothing below needs the
network or a re-scrape — the panel is committed.

## The 60-second version

```bash
cd C:\Users\Alex\Projects\football-trajectory
.venv\Scripts\python -m pytest        # 83 tests, offline, ~8s
```

Then read **[README.md](README.md)** — the verdict and headline numbers.

## The five-minute version, in order

1. **[README.md](README.md)** — what was asked, what came back. The verdict table
   and the two headline numbers (K = 110; NOT PROVEN at all six gates).
2. **[DESIGN.md](DESIGN.md)** — the **pre-registration**. This is the point of
   the project: it was committed at `c0a6260` *before any modelling code existed*,
   so the gate could not be moved once results arrived. §7 is the gate. §A is the
   amendment log — every change made after contact with the data, with its cost.
3. **[WRITEUP.md](WRITEUP.md)** — the full story. §2 verdict, §3 *why* calibration
   failed, §4 the Yamal answer, §5 data findings, §6 method, §8 what the tests
   caught.

To see the pre-registration really does predate the results:

```bash
git log --oneline            # c0a6260 is the first commit; results land 3 commits later
git show c0a6260 --stat
```

## The results themselves

```bash
type results\gate.json                       # the verdict, per metric per horizon
type results\gate_fresh.json                 # v2 on 2025-26 (the one clean, unseen season)
type results\projections.json                # the 10 young players, incl. Yamal + records
type results\calibration_diagnostic.json     # why calibration failed, broken out
```

**Read `gate_fresh.json` before `gate.json`.** Every gate file carries a
`test_status` field. `gate.json` says `SECOND LOOK` — v2 was built after seeing
the v1 result, so that split is spent. `gate_fresh.json` says `CLEAN`: season
2025-26 was never used to evaluate anything, and the finding replicates there on
its own. That is the number with standing.

Prefer pictures:

```
charts\calibration.png    every bar sits ABOVE the pass band -> under-confident
charts\skill.png          every CI sits LEFT of zero -> real skill, growing with horizon
charts\aging_curves.png   the dropouts curve sits BELOW survivors -> survivor bias, visible
charts\k_grid.png         K is fitted, not chosen
charts\projections.png    the fan, self-labelled UNCALIBRATED in red
```

Quick look at Yamal specifically:

```bash
.venv\Scripts\python -c "import json;d=json.load(open('results/projections.json',encoding='utf-8'));y=[p for p in d['players'] if p['player']=='Lamine Yamal'][0];print(json.dumps(y,indent=2))"
```

## The code, in dependency order

| Read | For |
|---|---|
| `data/loader.py` | raw → panel. The `(name, born)` identity key, `audit_identity`, and the Bundesliga label repair |
| `model/talent.py` | the shrinkage estimator. `posterior()` is the crux: the Marcel formula *is* a Gamma-Poisson posterior mean, so fitted K also gives uncertainty |
| `model/aging.py` | delta-method curves; the `survivors` vs `dropouts` survivor-bias question |
| `model/minutes.py` | availability. **v2: now conditioned on talent** — read the module docstring for why the three design choices are what they are |
| `mc/simulate.py` | the four uncertainty sources, kept separate |
| `validation/gate.py` | the verdict machinery. Tested against known-answer constructions |
| `run_validate.py` | the three-way split and why the variant choice happens on an inner split |

## The interactive explorer

`viz/trajectory_explorer.html` — open in any browser, no server needed. 16
young attackers, fan charts for goals/assists/minutes, a cumulative career
view, and a **"Compare with"** selector that grafts a legend's real career arc
onto the selected youngster: Messi, Cristiano Ronaldo, Mbappé, Haaland,
Benzema, Agüero or Rooney (career view only; scenarios, not forecasts, and
labelled as such — active players' arcs are marked as still being written).
Built by `scripts/build_viz.py` from `viz/template.html` + the results JSONs —
edit the template, never the built file.

## Re-running anything

```bash
.venv\Scripts\python run_validate.py            # -> results/gate.json      (~10 min)
.venv\Scripts\python run_validate.py --fresh    # -> results/gate_fresh.json
.venv\Scripts\python run_project.py             # -> results/projections.json (needs a gate first)
.venv\Scripts\python scripts\diagnose_calibration.py
.venv\Scripts\python scripts\make_charts.py     # -> charts/*.png
.venv\Scripts\python scripts\export_trajectories.py   # -> results/trajectories.json
.venv\Scripts\python scripts\legend_scenarios.py      # -> results/legend_scenarios.json
.venv\Scripts\python scripts\build_viz.py             # -> viz/trajectory_explorer.html
```

Only `scripts/fetch_data.py` touches the network (headless Chrome, slow, FBref
rate-limits). You should not need it — `data/panel.parquet` is committed.

## If you only read four things

- `DESIGN.md` §7 — the gate, fixed in advance.
- `WRITEUP.md` §3 — the calibration failure, diagnosed rather than excused.
- `WRITEUP.md` §4 — the Yamal answer and why the model is probably too pessimistic.
- `model/minutes.py` docstring — the v2 talent conditioning, the fix for exactly
  that pessimism.
