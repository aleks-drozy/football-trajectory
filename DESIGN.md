# Pre-registration — Football Trajectory Projection

**Written before any model was fitted or any result was seen.** Committed to git ahead of the
implementation so the gate below cannot be moved after the fact. Everything in this file is a
commitment, not a description of what happened. The honest verdict lives in `WRITEUP.md`.

---

## 1. The question

Can cohort-based aging curves plus Monte Carlo simulation produce **calibrated** multi-season
projections of a young footballer's output (non-penalty goals, assists, minutes) in the Big-5
European leagues — and do those projections **beat naive baselines out-of-sample**?

The interesting deliverable is *not* "here is Lamine Yamal's projected goal total." Any Monte Carlo
loop can emit a fan chart, and it will look authoritative regardless of whether it is meaningful.
The deliverable is a **defensible answer to whether such a fan chart can be trusted at all**, and a
projection only gets published with the trust label the validation actually earns.

## 2. Why this is not the obvious version

The naive project resamples a player's own two or three seasons thousands of times and reports a
tight interval. That interval is close to meaningless:

- ~2 seasons of a teenager is a tiny sample; bootstrap precision is not accuracy.
- It bakes in "this player's current rate is his true rate," which is exactly the thing that
  regresses hardest.
- It ignores aging entirely — an 18-year-old and a 30-year-old are not the same process.
- It has no way to be wrong, so it cannot be checked.

This project instead estimates true talent with shrinkage against a cohort, applies an aging curve
estimated from thousands of player-seasons, models availability (minutes) non-parametrically, and
then **tests the whole apparatus out-of-sample** against the baselines it must beat to be worth
anything.

## 3. Scope boundary (explicit)

Data is FBref Big-5 domestic-league player-seasons only. Therefore:

- **We project Big-5 domestic-league output, not global career output.** A player moving to the
  Saudi Pro League, MLS, or Eredivisie leaves our data and reads as zero minutes. That is a
  modelling boundary, not a prediction of retirement.
- Cup and continental goals (Champions League, Copa del Rey, FA Cup) are **not** counted.
- Any record comparison must therefore be a **domestic-league** record, or it is apples-to-oranges.
  Record thresholds live in a config file with sources and are labelled league-only.

## 4. Data

- Source: FBref via `soccerdata`, "Big 5 European Leagues Combined".
- Seasons: 2017-2018 .. 2025-2026 (xG/xAG are available from 2017-18; earlier seasons would lose
  the finishing-stable signal the talent model depends on).
- Grain: one row per player-season.
- Identity key: `(player_name, born_year)`. FBref's season tables carry no stable player ID.
  Name-collision and spelling-drift risk is real and will be **measured and reported**, not assumed
  away.
- Age convention: `age = season_start_year - born_year` (derived from `born`, which is stable,
  rather than FBref's `age` string).

## 5. Method

### 5.1 Target metrics
- **npG/90** — non-penalty goals per 90. Penalties are a role/team artefact and are separated.
- **Ast/90** — assists per 90.
- **Minutes per season** — availability, modelled separately and explicitly.

Career totals accumulate as `Σ_seasons (minutes/90) × rate`.

### 5.2 True talent — empirical-Bayes shrinkage (Marcel-style)
Observed per-90 rates are noisy, especially for low-minute teenagers:

```
talent_hat = (Σ_s w_s · events_s + K · prior_rate) / (Σ_s w_s · n90_s + K)
```

- `w_s`: recency weights over the player's recent seasons.
- `prior_rate`: position × age-bucket cohort mean.
- `K`: the shrinkage constant. **`K` is fitted from data** by minimising out-of-sample error on a
  training split. Hand-picking `K` is where naive implementations quietly cheat, so it is fitted and
  the fitted value is reported.
- npxG/90 is used as a co-signal alongside npG/90 (finishing regresses harder than shot volume);
  the blend weight is fitted, not assumed.

### 5.3 Aging curve — delta method
For consecutive seasons of the same player at age `a → a+1`, weight each delta by the harmonic mean
of the two seasons' 90s, average per age, and chain into a curve.

**Known bias, stated up front:** the delta method suffers *survivor bias* — players who decline lose
minutes or drop out and so never contribute their next delta, which biases the curve optimistically.
Mitigation: build **two** variants — (a) raw survivors-only, (b) dropouts included at replacement
level — and let the out-of-sample validation in §6 choose between them. The choice is made by the
gate, not by which curve looks nicer.

### 5.4 Minutes / availability — non-parametric
Minutes are strongly bimodal (regular starter vs squad player) and shaped by injury, rotation and
transfer. Rather than impose a parametric shape, sample next-season minutes from the **empirical
conditional distribution** given (current minutes bucket, age). Zero minutes is a real, retained
outcome — it captures injury years, benching, and departure from the Big-5.

### 5.5 Monte Carlo
Per simulation path: draw talent from its posterior, then for each future season — age-adjust via
the aging curve (including curve uncertainty), sample minutes, sample events
(`goals ~ Poisson(rate_adj × n90)`, likewise assists), and accumulate. Poisson is the starting event
model; if validation shows intervals are too narrow, over-dispersion (negative binomial) is the
documented next step rather than a silent patch.

### 5.6 Player selection
The projected cohort is chosen **from the data**, not from memory: the youngest high-minute players
in the most recent season (age ≤ 21, ranked by minutes), plus Lamine Yamal by name as the motivating
case. This avoids cherry-picking names that happen to flatter the model.

## 6. Validation — the actual point of the project

Fit on seasons ≤ T, project forward, compare against what really happened.

- **Split:** aging curves, `K`, and all fitted quantities are estimated on training seasons only.
  Held-out seasons are never touched during fitting.
- **Horizons:** +1, +2, +3 seasons.
- **Baselines the model must beat:**
  1. **Persistence** — "next season looks like last season."
  2. **Cohort mean** — "becomes an average player of his age and position."

## 7. Pre-registered gate

Decided now, before results exist. Evaluated per horizon (+1, +2, +3):

1. **Calibration.** The nominal 80% central prediction interval must contain the actual outcome
   between **72% and 88%** of the time on held-out player-seasons.
2. **Skill.** Projection MAE must beat the better of the two baselines in §6, with a bootstrap 95%
   CI on the MAE difference that **excludes zero**.
3. **No concentration.** The skill result must survive leave-one-group-out across leagues and
   positions — if dropping any single league or position flips the verdict, the edge is
   concentrated and does not count.

**A horizon passes only if all three hold.** Any failure ⇒ **NOT PROVEN** at that horizon, reported
as such.

## 8. The honesty commitment

If the model is not calibrated, **the Yamal projection is published with an explicit
"not trustworthy" label**, and the fan chart is presented as a demonstration of the failure rather
than as a forecast. A well-drawn but uncalibrated interval is a lie with error bars; the failure
mode of this genre of project is publishing exactly that. Reporting an honest null here is a
better result than an unfalsifiable fan chart, and is the intended outcome if that is what the data
says.

## 9. Known limitations (acknowledged in advance)

- Big-5 domestic league only (§3).
- Survivor bias in the delta method (§5.3) — mitigated, not eliminated.
- `(name, born)` identity is imperfect (§4).
- Team context (a striker at Barcelona vs. at a relegation side) is not modelled in v1; it is
  absorbed into noise and listed as future work.
- Teenage samples are small by nature; the whole point of §7's gate is to find out whether that
  makes the exercise hopeless. It might. That is a publishable answer.
