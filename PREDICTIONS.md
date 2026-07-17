# Predictions ledger — 2026-27 season

**Frozen 2026-07-17**, before the 2026-27 season. The git history of this
file is the proof. Scored publicly when the season ends, by the rules below,
written down now so they cannot drift after the fact.

The model behind these numbers failed its own calibration gate (its 80%
intervals are honestly too wide — see [WRITEUP.md](WRITEUP.md)), so the
intervals here are the test as much as the medians: nominal 80% coverage is
itself a prediction being scored.

## The headline claim

**Lamine Yamal scores fewer non-penalty league goals in 2026-27 than the 13
he scored in 2025-26.** Median call: **5** [1-11].
Not because he isn't brilliant, but because every hot teenage season in the
data regressed, and the model refuses to make him the exception in advance.

## All sixteen calls (non-penalty goals, Big-5 league only)

| Player | Age | Club when frozen | npG median | 80% interval | Assists | Minutes |
|---|---|---|---|---|---|---|
| Lamine Yamal | 19 | Barcelona | **5** | 1-11 | 5 [1-11] | 1,696 |
| Yan Diomandé | 20 | RB Leipzig | **7** | 1-13 | 4 [0-9] | 1,900 |
| Nicolás Paz | 22 | Como | **6** | 1-10 | 4 [1-8] | 2,282 |
| Said El Mala | 20 | Köln | **7** | 1-14 | 3 [0-6] | 1,779 |
| Kenan Yıldız | 21 | Juventus | **5** | 1-10 | 3 [0-7] | 2,263 |
| Bazoumana Touré | 20 | Hoffenheim | **3** | 0-6 | 5 [1-10] | 1,902 |
| Arda Güler | 21 | Real Madrid | **4** | 0-9 | 5 [1-10] | 1,902 |
| Álvaro Rodríguez | 22 | Elche | **5** | 0-11 | 3 [0-6] | 1,944 |
| Afonso Moreira | 21 | Lyon | **3** | 0-6 | 4 [0-9] | 1,781 |
| Carlos Espí | 21 | Levante | **6** | 0-14 | 1 [0-3] | 1,270 |
| Désiré Doué | 21 | PSG | **4** | 0-10 | 3 [0-7] | 1,295 |
| Eli Junior Kroupi | 20 | Bournemouth | **6** | 1-12 | 2 [0-4] | 1,779 |
| Endrick | 20 | Lyon | **3** | 0-9 | 3 [0-8] | 1,289 |
| Jesus Rodríguez | 21 | Como | **2** | 0-5 | 4 [0-9] | 1,779 |
| Matias Fernandez-Pardo | 21 | Lille | **5** | 0-10 | 3 [0-7] | 1,803 |
| Diego Moreira | 22 | Strasbourg | **2** | 0-5 | 4 [0-8] | 1,928 |

## Scoring rules (fixed now)

1. **Point skill** — MAE of the medians vs what happens, against the
   persistence baseline (each player repeats his 2025-26 npG). The model
   must beat it.
2. **Interval honesty** — how many of the 16 land inside their 80% band.
   Nominal is 80%; the v1/v2 backtests say the true figure will run high.
3. **The headline claim** — right or wrong, no partial credit.
4. **No exclusions.** A transfer out of the Big 5 scores 0 (that is the
   model's stated scope); an injury counts as what happened. Every row is
   scored.

Run `python scripts/score_predictions.py` after the 2026-27 fetch to grade.
