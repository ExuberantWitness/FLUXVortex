# MODEL SCORECARD — candidate LESP-LEV models vs RoboEagle Fig17/18 measured

grid {'nc': 4, 'ns': 8, 'n_cycle': 2, 'steps_per_cycle': 60, 'wake_rows': 60} | a0_crit=0.23 | phase +90 | both wings | 34 keys / 170 pts

Ranked by USER LADDER: **trend > sign > >50%err > <20%err > MAE**. (M2==M1, M5==M4 in this framework — omitted.)

| rank | model | trend↑ | sign↑ | >50%err↓ | <20%err↑ | MAE(N)↓ | RMSE | description |
|---|---|---|---|---|---|---|---|---|
| 1 | **M1** | 88% | 91% | 62 | 15 | 2.33 | 2.78 | Hirato kelvin LEV |
| 2 | **M3** | 85% | 91% | 65 | 11 | 2.34 | 2.80 | varA0 + hold |
| 3 | **M4** | 85% | 89% | 61 | 16 | 2.34 | 2.81 | varA0 + hold_detach (rec.) |
| 4 | **ML** | 76% | 96% | 83 | 24 | 4.58 | 6.25 | legacy fp/sectional anchor |
| 5 | **M0** | 53% | 96% | 70 | 22 | 3.25 | 4.08 | attached UVLM floor |

**Winner: M1** (Hirato kelvin LEV) by the priority ladder.

Honest note: absolute under/over-prediction of held-LEV lift is the inviscid-LEV limit (Li JFM); trends/signs are the primary acceptance criteria. M0/ML bound the comparison (floor / legacy anchor).