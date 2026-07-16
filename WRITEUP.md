# Can you project a teenager's career? — method, results, honest verdict

## 1. The question

If Lamine Yamal keeps performing the way he has, how many goals and assists will
he finish with, and will he break records?

The tempting way to answer it is to resample his own two or three seasons ten
thousand times and publish the resulting fan chart. That chart would look
authoritative and mean almost nothing. It rests on a tiny sample; it assumes his
current rate *is* his true rate, which is the single thing that regresses
hardest; it ignores aging entirely; and — fatally — it has no way of being
wrong, so nobody can ever check it.

So this project asks the question behind the question:

> Can cohort-based aging curves plus Monte Carlo produce **calibrated**
> multi-season projections of a young footballer's output, and do they **beat
> naive baselines out-of-sample**?

The deliverable is not Yamal's number. It is a defensible answer to whether
anyone's number can be trusted.

The gate was written and committed (`c0a6260`) **before any modelling code
existed**, so it could not be moved once results arrived. It is in
[DESIGN.md](DESIGN.md), along with a log of every amendment made after contact
with the data.

## 2. Verdict — NOT PROVEN at every horizon, and it fails the interesting way

All six gates (goals and assists × +1/+2/+3 seasons) fail. But *how* they fail
is the finding:

| | goals +1 | goals +2 | goals +3 | assists +1 | assists +2 | assists +3 |
|---|---|---|---|---|---|---|
| **Calibration** — 80% interval must cover 72-88% | 91.5% ✗ | 91.2% ✗ | 90.4% ✗ | 92.2% ✗ | 91.8% ✗ | 89.4% ✗ |
| **Skill** — must beat best baseline, CI excluding 0 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| **Concentration** — must survive leave-one-group-out | ✓ 0/9 | ✓ 0/9 | ✓ 0/9 | ✓ 0/9 | ✓ 0/9 | ✓ 0/9 |
| **Verdict** | NOT PROVEN | NOT PROVEN | NOT PROVEN | NOT PROVEN | NOT PROVEN | NOT PROVEN |

**The skill is real, and it is broad.**

| horizon | model MAE | best baseline | baseline MAE | diff [95% CI] |
|---|---|---|---|---|
| goals +1 | 1.279 | persistence | 1.642 | −0.363 [−0.441, −0.290] |
| goals +2 | 1.200 | cohort mean | 1.758 | −0.558 [−0.633, −0.481] |
| goals +3 | 1.093 | cohort mean | 1.805 | −0.712 [−0.792, −0.626] |
| assists +1 | 0.985 | persistence | 1.342 | −0.357 [−0.414, −0.297] |
| assists +2 | 0.962 | cohort mean | 1.428 | −0.466 [−0.518, −0.415] |
| assists +3 | 0.900 | cohort mean | 1.451 | −0.551 [−0.609, −0.494] |

Every CI sits well clear of zero, and dropping **any** single league or position
never flips the result (0 of 9 leave-one-group-out checks fail at every
horizon). The advantage also *grows* with horizon — which is exactly what a real
aging model should do. "He'll repeat last season" decays as the horizon extends;
a model that knows how players age does not.

**The uncertainty is not honest — and it fails the unusual way.** The nominal
80% interval catches ~90-92% of outcomes. The model is **under-confident**: its
intervals are too *wide*. Almost every projection system in this genre fails by
being too narrow — confidently wrong. This one is cautiously right.

That is still a failure. An interval labelled 80% that behaves like 91% is
mislabelled, and the pre-registration says so. The honest summary:

> **Useful point projections. Untrustworthy intervals.**

Per DESIGN.md's §8 honesty commitment, the Yamal projection therefore ships
with an explicit NOT TRUSTWORTHY label (§4), as a demonstration of that failure
rather than a forecast.

## 3. Where the calibration failure comes from

"Too wide" needed a cause, not an excuse. Two hypotheses make opposite
predictions:

1. **Discreteness + zero-inflation.** Most of a 2,010-player Big-5 cohort are
   defenders and midfielders whose predictive distribution is mostly zero. If
   the 10th percentile is 0 and the 90th is 1, an actual of 0 is "covered"
   automatically — coverage overshoots for reasons unrelated to the model's
   caution. Under this hypothesis the overshoot concentrates in low-expectation
   players and shrinks among high scorers.
2. **Genuinely inflated spread.** The simulation stacks four uncertainty
   sources; if one is double-counted, intervals are too wide for *everyone*.
   Under this hypothesis the overshoot is roughly flat across the board.

`scripts/diagnose_calibration.py` splits coverage by predicted level and by
position to separate them. **The answer is hypothesis (1), decisively.**

At +1 season:

| model expects | n | coverage | mean interval width | mean actual |
|---|---|---|---|---|
| median 0 goals | 1,045 | **93.0%** | 1.6 | 0.50 |
| median 1-2 | 663 | 91.3% | 4.4 | 1.73 |
| median 3-5 | 222 | **86.5%** | 8.8 | 4.62 |
| median 6+ | 76 | **86.8%** | 14.0 | 8.13 |

| position | n | coverage | width |
|---|---|---|---|
| GK | 144 | **100.0%** | **0.0** |
| DF | 672 | 94.2% | 1.8 |
| MF | 854 | 88.6% | 3.8 |
| FW | 340 | 89.7% | 9.1 |

**98-100% of the cohort has a 10th percentile of exactly 0**, so "covering zero"
is close to automatic for most of it. The goalkeepers are the reductio: 144 of
them, interval width **0.0**, coverage **100%** — the model predicts [0, 0], they
score 0, and every one counts as "covered". That is not the model being cautious;
it is the test being unable to say anything about an atom at zero.

And coverage falls monotonically as the model expects more, landing at
**86.5-86.8% for players it expects to actually score — inside the 72-88% pass
band.** The aggregate 91.5% is carried by the ~1,200 defenders and goalkeepers
for whom the goals question is close to meaningless.

**This does not overturn the verdict.** The gate is pre-registered and it says
NOT PROVEN; rescuing it after the fact with a sub-population that happens to
pass is exactly the sin pre-registration exists to prevent, and the headline
stands. What the diagnostic changes is the *diagnosis*: the spread is not
obviously inflated, and the failure is concentrated where the test instrument
cannot work.

It also exposes a real flaw in my own pre-registration: **interval coverage is
the wrong calibration test for a discrete, zero-inflated target.** The right
instrument is a randomised PIT, which handles the atom at zero properly. I
pre-registered the cruder one, so I report it and live with it — swapping to a
nicer test now, having seen the result, would be indefensible. It is the first
item of future work, to be re-registered before it is run, and scored on a
cohort where the question means something (regular attackers) rather than on
every goalkeeper in Europe.

## 4. So what about Yamal?

Labelled **NOT TRUSTWORTHY**, as §2 requires. With that said, here is what the
model produces (Big-5 domestic league only, from age 18 to a notional 38). Both
model versions are shown, because the difference between them *is* the lesson:

| | v1 | **v2 (talent-conditioned)** |
|---|---|---|
| future non-penalty goals | 31 [7 - 72] | **45** [15 - 87] |
| future assists | 34 [7 - 80] | **51** [17 - 96] |
| **career total league goals** (incl. penalties, 30 already scored) | 69 [38 - 120] | **88** [50 - 142] |
| P(career ≥ 100) | 22% | **38%** |
| P(career ≥ 150) | 2.7% | **7.2%** |
| P(career ≥ 200) | 0.2% | 0.4% |
| **P(beating Messi's 474 La Liga)** | 0.0000 | **0.0000** |

Fixing one modelling flaw moved his median career total by **28%** (69 → 88) and
nearly doubled P(≥100). That is worth sitting with: the headline number was never
really a fact about Yamal, it was a fact about an assumption buried in the
availability model. It is why the paragraph below matters more than the table
above, and why the label is on both columns.

The interval still spans a factor of roughly six. That is the under-confidence of
§2, and it is why the label stays.

**What did not move: the record.** P(Messi's 474) is 0.0000 in both versions —
not close, and not sensitive to the fix. His v2 90th percentile is 142 career
Big-5 league goals; the record is 474. Even the most optimistic correction to
availability leaves the two an order of magnitude apart.

**Why v1 was this pessimistic — this is the diagnosis the fix came from.** The
median is driven almost entirely by *availability*, not by talent. Simulating the
v1 minutes chain alone from Yamal's actual state (2,262 minutes at 18) gives a
median of **9,794 career Big-5 minutes** (~109 × 90s) and:

| still in the Big-5 at age | 20 | 23 | 26 | 30 | 34 |
|---|---|---|---|---|---|
| probability | 84% | **54%** | 39% | 25% | 8% |

At his shrunk talent of ~0.32 npg/90, 109 × 90s ≈ 35 goals — which is the
median the full simulation reports. So the projection is a statement about
career length, not finishing.

Is that base rate wrong? Not as an empirical fact: that really is what happens to
the median teenager with 2,000+ Big-5 minutes. (I checked the obvious suspect —
that a thin cell was backing off to a pool contaminated with 34-year-olds — and
it isn't: the (2000-2500 minutes, under-20) cell has 48 observations of its own,
and its observed P(zero next season) is **0.0**. The drop-off is real and
happens later in the chain.)

The flaw was narrower and more interesting: **v1's availability model conditioned
on minutes and age, and never on talent.** A generational 18-year-old was handed
the dropout hazard of the *average* 18-year-old with similar minutes. Elite
players keep their place far longer than average ones, and v1 had no way to know
Yamal is elite beyond his rate — which it then shrinks by K = 110.

**§4b fixes exactly this**, and the diagnosis held: conditioning on talent raises
his P(still in the Big-5 at 30) from 0.25 to 0.43 and his median career total
from 69 to 88. The prediction that it would also *narrow* the interval was only
half right — the 10th percentile rose sharply (7 → 15 future goals), but the
spread stayed wide, because §3 had already shown the width is mostly the
discreteness of the target rather than the minutes model.

The honest answer to the original question: **the model says Yamal has
essentially no chance of Messi's record, and that conclusion survived fixing the
one flaw most likely to have caused it.** P(474) is 0.0000 in both versions; his
v2 90th percentile is 142 against a record of 474. The gap is not a modelling
artefact — it is what happens when you count only Big-5 domestic-league goals and
ask a teenager to sustain Messi's career for fifteen years.

## 4b. v2 — conditioning availability on talent

§4 identified the biggest flaw: availability conditioned on (minutes, age) and
**never on talent**, so a generational teenager inherited the average teenager's
dropout hazard. v2 fixes exactly that, with a talent tercile — attacking output
per 90, **absolute** cut points within position, **carried forward** from the last
qualifying season and never read from the future. (Absolute rather than
age-relative because keeping your place depends on how good you are, not how good
you are *for your age*: age-relative cuts would say a teenager stops being good
the moment he turns 24.)

**It works, on the measure that is clean.** Inner-split MAE — the split used to
choose the variant, never the test — improves **1.3574 → 1.3407**. On a
Yamal-like state (2,262 minutes at 18, which lands in the strong tercile, his
0.955/90 being nearly double the FW cut of 0.516):

| | v1 | v2 (strong) |
|---|---|---|
| median career Big-5 minutes | 9,794 | **14,939** |
| P(still in the Big-5 at 23) | 0.54 | **0.74** |
| P(still in the Big-5 at 30) | 0.25 | **0.43** |

89% of the 168 (minutes, age, talent) cells clear the support threshold on the
full panel (80% on the shorter training split), so the new dimension is genuinely
used rather than quietly backing off.

**On the gate, it moves skill and leaves calibration alone:**

| metric | h | calibration v1 → v2 | MAE v1 → v2 |
|---|---|---|---|
| goals | +1 | 91.5% → 91.3% | 1.279 → 1.271 |
| goals | +2 | 91.2% → 91.2% | 1.200 → 1.175 |
| goals | +3 | 90.4% → 90.5% | 1.093 → **1.063** |
| assists | +1 | 92.2% → 91.9% | 0.985 → 0.989 |
| assists | +2 | 91.8% → 91.7% | 0.962 → 0.945 |
| assists | +3 | 89.4% → 89.8% | 0.900 → **0.864** |

Skill improves at five of six horizons and the gain **grows with horizon** —
which is what it should do, since availability compounds the further out you
project. Calibration is untouched, also as expected: §3 showed that failure is
the test instrument meeting a zero-inflated target, not the width of the minutes
distribution. **Every horizon is still NOT PROVEN.**

**What this costs in evidential standing.** v2 was built *after* seeing the v1
gate, so re-running the same 2022-24 split is a **second look at a spent test
set**. The table above is reported for information and is labelled
`test_status: "SECOND LOOK"` in `results/gate.json`. It is not a pre-registered
result and must not be quoted as one. See DESIGN.md §A5.

### The clean test: 2025-26

Season 2025-26 was never used to evaluate anything — the inner split scored
2020-21, the gate scored 2022-24. `run_validate.py --fresh` trains through 2024
and tests v2 on 2025 alone: one horizon, but genuinely untouched data
(`results/gate_fresh.json`, `test_status: "CLEAN"`, 1,959 players).

| | calibration (need 72-88%) | skill | concentration | verdict |
|---|---|---|---|---|
| goals +1 | **92.2%** ✗ | MAE **1.248** vs persistence 1.633 — diff −0.386 [−0.459, −0.315] ✓ | 0/9 ✓ | **NOT PROVEN** |
| assists +1 | **92.2%** ✗ | MAE **1.040** vs cohort mean 1.376 — diff −0.336 [−0.384, −0.287] ✓ | 0/9 ✓ | **NOT PROVEN** |

**The v1 finding replicates on unseen data.** Different base season, different
test season, same signature: skill real and broad, intervals too wide, NOT
PROVEN. Compare the fresh goals +1 (cal 92.2%, MAE 1.248 vs 1.633, diff −0.386)
against v1's 2022 goals +1 (cal 91.5%, MAE 1.279 vs 1.642, diff −0.363) — the
numbers are near-identical across a completely separate split.

That matters more than any single gate. **"Useful point projections,
untrustworthy intervals" is not an artefact of one split** — it survived a clean
replication, which is the strongest claim this project makes.

## 5. Data

FBref Big-5 domestic leagues via `soccerdata`, 2017-18 .. 2025-26:
**24,057 player-seasons, 7,814 players.** FBref 403s plain HTTP requests, so the
fetch drives a real headless Chrome and caches every season to parquet.

**Scope boundary (explicit).** This projects **Big-5 domestic-league** output
only. A move to the Saudi Pro League, MLS or the Eredivisie leaves the data and
reads as zero minutes — a modelling boundary, not a prediction of retirement.
Champions League and domestic cups are not counted.

**Identity.** FBref's season tables carry no stable player ID, so identity is
`(player_name, born_year)`. That correctly separates the mononym collisions
(51 names are reused across birth years — Marcelo, Rafinha, Rafael). It fails
when two players share a name *and* a birth year, and the audit found a real
case: two Portuguese players named **Vitinha, both born 2000** (PSG's Vítor
Ferreira and Genoa's Vítor Oliveira) merge into a single 2025-26 row of **4,345
minutes** — more than any human can play in a season, which is exactly the
signature `audit_identity` looks for. Measured rate: 1 in 24,057 (0.004%). That
figure is a **lower bound**: only merges extreme enough to break the
minutes ceiling are detectable.

**A repaired data bug.** soccerdata returns a *null* league for every Bundesliga
row in the Big-5 Combined view — 4,330 rows, 18% of the panel. The rows are
fine; only the label is missing. `repair_missing_league` infers it and verifies
its own premise: the fetch requests exactly five leagues, so if precisely one
label is absent the unlabelled rows can only belong to it. If that ever stops
holding it raises rather than mislabelling a fifth of the data. Left unfixed,
the Bundesliga would simply have been invisible to the concentration check.

## 6. Method

**Talent — Gamma-Poisson shrinkage.** A teenager's observed rate is mostly
noise: 4 goals in 700 minutes is 0.51/90 and almost no evidence. Shrink toward a
position × age-bucket cohort prior in proportion to workload:

```
talent = (Σ_s w_s·events_s + K·prior) / (Σ_s w_s·n90_s + K)
```

**K is fitted, not chosen** — grid-searched against out-of-sample error on
training seasons. Hand-picking K is the quiet cheat here, because it is the one
knob that decides whether a hot teenage sample gets believed.

> **Fitted result: K = 110.** In units of 90s that is ≈9,900 minutes ≈ **3.5
> full seasons** of play before a player's own record outweighs the cohort
> prior. That single number is the real answer to "how much should we believe
> Yamal's 0.52 npg/90 at 18?" — and it was measured, not asserted. (160 was on
> the grid and scored worse, so this is not a boundary artefact.)

A useful accident: this shrinkage formula is exactly the posterior **mean** of a
Gamma-Poisson conjugate pair with prior `Gamma(K·prior, K)`. So the same fitted
K also delivers principled *uncertainty* — the Monte Carlo draws talent from
`Gamma(K·prior + Σw·events, K + Σw·n90)` rather than bolting an invented spread
onto a point estimate, and that posterior correctly narrows as a career
accumulates.

Side-finding: total goals fits **K = 75** against non-penalty goals' 110.
Penalties are more persistent than open play — penalty takers keep taking
penalties — so less shrinkage is optimal.

**Aging — delta method, per position, two variants.** Pair a player's
consecutive seasons, weight each delta by the harmonic mean of the two seasons'
workloads, average per age, chain into a curve. The method's known flaw is
**survivor bias**: a player who declines loses his place and never supplies his
next delta, so the surviving pairs make aging look kinder than it is. Two
variants were built — `survivors` (conditional on still playing) and `dropouts`
(vanished players re-enter at replacement level) — and the choice deferred to
validation. `survivors` won on the inner split (MAE 1.3574 vs 1.3699 — close).

**Minutes — non-parametric.** Minutes are bimodal, truncated, and shaped by
injury, rotation and transfer; a parametric shape would impose structure the
data lacks. Next-season minutes are sampled straight from the empirical
conditional distribution given (minutes bucket, age bucket). Zero is a real,
retained outcome, so injury, benching and departure need no separate model. The
fitted model knows, for example, that **84% of players who leave the Big-5 stay
gone**.

**Monte Carlo.** Four uncertainty sources, kept separate rather than merged into
one fudge factor: talent (Gamma posterior), availability (empirical minutes),
event noise (Poisson), and aging-curve uncertainty (a bootstrap ensemble of 30
curves, one assigned per path).

## 7. Validation design

Nine seasons, split three ways so nothing used to *choose* the model is used to
*judge* it:

```
2017-2019  inner-train    fit both aging variants
2020-2021  inner-select   pick the variant   <- selection happens here
2017-2021  final-train    refit with the chosen variant
2022-2024  TEST           the gate           <- touched exactly once
```

The pre-registration said "let out-of-sample validation choose the variant",
which taken literally leaks: scoring both variants on the test set gives the
model two attempts at the gate and reports the better. The choice moved to an
inner split carved from training data — stricter than the original wording
(amendment A2).

**Absence is scored as zero, not dropped.** A player who left the Big-5 really
did score zero Big-5 goals. Dropping those rows would grade the model only on
players who stuck around — the same survivor bias the aging curves exist to
resist.

## 8. What the tests caught

The suite (72 tests, no network) exists because this kind of code fails
silently — every bug below produced plausible-looking output:

- **`fit_K` read a column that did not exist.** `DataFrame.itertuples` silently
  renames columns that aren't valid Python identifiers, so `_from_season`
  became positional.
- **The aging chain broke whenever the reference age wasn't adjacent to data**,
  leaving neighbouring ages silently unset.
- **The Monte Carlo indexed the assist curve ensemble with the goal ensemble's
  index** — which crashes only when bootstrap resamples drop out at different
  rates.
- **`bootstrap_curves` cross-joined players with themselves.** Resampling
  duplicated players by `(name, born)`, and `build_pairs` then self-merges on
  that key — so a player drawn *k* times produced *k²* season pairs instead of
  *k*, weighting duplicated players quadratically. It also made the fit appear
  to hang.
- **Replacement level was not reliably weak.** The first version used the mean
  rate of the bottom minutes quartile, which includes hot young substitutes with
  inflated per-90 rates — so it could impute an *above-average* rate to a player
  who vanished, asserting that disappearing improved him. Now a rate quantile.

The gate itself is tested against constructions with known answers — a perfectly
calibrated predictor, an overconfident one, an underconfident one, a no-skill
model, and an edge that lives in a single league — to confirm it fails when it
should.

## 9. Limitations

- **No xG/xAG at this grain** (amendment A1). FBref's Big-5 Combined view serves
  no `passing` table and no Expected block. `shots/90` substitutes as the goals
  co-signal; **assists have no co-signal at all** and are the weaker projection.
- **Interval coverage is the wrong calibration test for this target** (§3) —
  pre-registered, so reported as-is; randomised PIT is the correct instrument.
- **Survivor bias is mitigated, not eliminated.**
- **Team context is not modelled.** A striker at Barcelona and the same striker
  at a relegation side get the same projection.
- **Identity collisions are detectable only at the extreme** (§4).
- **Record comparisons use total goals** (records count penalties) while the
  model's honest unit is non-penalty goals; total goals runs through the same
  machinery but was **not separately gated**.
- **The showcase selects on outcome** — it ranks young attackers by output, so
  it is not a random sample of young players. It has no bearing on the gate,
  which runs on the full unbiased cohort.

## 10. Future work, in order

1. **Re-register and run a randomised-PIT calibration test** — the current
   instrument cannot fairly assess a zero-inflated discrete target.
2. **Fetch per-league to recover xG/xAG**, and give assists a real co-signal.
3. **Model team context** — squad strength is the obvious omitted variable.
4. **Investigate the width directly.** If §3 finds the spread is genuinely
   inflated rather than an artefact of discreteness, the aging-curve ensemble
   and the minutes model are the first suspects for double-counting.
