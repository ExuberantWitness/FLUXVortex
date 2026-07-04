# GAP structure (model − experiment) — step-1 of gap→research→implement


## model fix  (322 matched pts)


### T: global gap regression (R²=0.93, resid RMSE 0.49N vs gap std 1.90N)
| term | coef | rank |
|---|---|---|
| 1 | +4.861 | 7 |
| f² | +0.623 | 2 |
| tw² | +0.737 | 6 |
| f²·tw² | -0.216 | 5 |
| aoa | +5.864 | 3 |
| U²/64 | -5.747 | 1 |
| f²·U²/64 | +0.131 | 4 |

### T: per-axis laws (gap fits along swept axes)
| axis | fixed | law | R² | n |
|---|---|---|---|---|
| f@tw0 | U6/aoa5 | gap = +2.55 +0.577·f² | 0.93 | 8 |
| f@tw0 | U8/aoa0 | gap = -0.83 +0.671·f² | 0.85 | 8 |
| f@tw0 | U8/aoa5 | gap = -0.18 +0.729·f² | 0.91 | 21 |
| f@tw0 | U8/aoa10 | gap = +0.04 +0.748·f² | 0.94 | 8 |
| f@tw0 | U8/aoa15 | gap = +0.25 +0.847·f² | 0.93 | 8 |
| f@tw0 | U10/aoa5 | gap = -3.28 +0.786·f² | 0.94 | 8 |
| tw@2.6Hz | U6/aoa5 | gap = +6.12 -0.10·tw²[rad] | 0.00 | 6 |
| tw@2.6Hz | U8/aoa0 | gap = +4.31 +0.00·tw²[rad] | 0.00 | 13 |
| tw@2.6Hz | U8/aoa5 | gap = +4.81 -0.51·tw²[rad] | 0.02 | 24 |
| tw@2.6Hz | U8/aoa10 | gap = +5.32 +0.19·tw²[rad] | 0.01 | 13 |
| tw@2.6Hz | U8/aoa15 | gap = +6.27 -3.39·tw²[rad] | 0.69 | 13 |
| tw@2.6Hz | U10/aoa5 | gap = +1.89 -2.43·tw²[rad] | 0.72 | 6 |
| aoa@tw0 | U8/f1.4 | gap = +0.67 +4.92·aoa[rad] | 0.93 | 6 |
| aoa@tw0 | U8/f1.5 | gap = +0.84 +5.28·aoa[rad] | 0.96 | 4 |
| aoa@tw0 | U8/f1.7 | gap = +1.38 +5.44·aoa[rad] | 0.93 | 6 |
| aoa@tw0 | U8/f2 | gap = +2.17 +6.31·aoa[rad] | 0.78 | 7 |
| aoa@tw0 | U8/f2.3 | gap = +2.83 +6.66·aoa[rad] | 0.70 | 7 |
| aoa@tw0 | U8/f2.5 | gap = +2.92 +8.78·aoa[rad] | 0.99 | 4 |
| aoa@tw0 | U8/f2.6 | gap = +4.10 +7.77·aoa[rad] | 0.44 | 11 |

### L: global gap regression (R²=0.52, resid RMSE 0.88N vs gap std 1.27N)
| term | coef | rank |
|---|---|---|
| 1 | -3.789 | 7 |
| f² | +0.596 | 2 |
| tw² | -1.962 | 5 |
| f²·tw² | +0.398 | 4 |
| aoa | +11.096 | 3 |
| U²/64 | +1.561 | 6 |
| f²·U²/64 | -0.487 | 1 |

### L: per-axis laws (gap fits along swept axes)
| axis | fixed | law | R² | n |
|---|---|---|---|---|
| f@tw0 | U6/aoa5 | gap = -0.83 +0.181·f² | 0.76 | 8 |
| f@tw0 | U8/aoa0 | gap = -1.55 +0.110·f² | 0.46 | 8 |
| f@tw0 | U8/aoa5 | gap = -0.86 -0.050·f² | 0.04 | 21 |
| f@tw0 | U8/aoa10 | gap = -0.98 +0.093·f² | 0.05 | 8 |
| f@tw0 | U8/aoa15 | gap = +0.30 -0.033·f² | 0.02 | 8 |
| f@tw0 | U10/aoa5 | gap = +1.06 -0.427·f² | 0.74 | 8 |
| tw@2.6Hz | U6/aoa5 | gap = +0.54 -0.15·tw²[rad] | 0.00 | 6 |
| tw@2.6Hz | U8/aoa0 | gap = -0.22 -4.00·tw²[rad] | 0.75 | 13 |
| tw@2.6Hz | U8/aoa5 | gap = -0.77 -0.53·tw²[rad] | 0.04 | 24 |
| tw@2.6Hz | U8/aoa10 | gap = +0.55 +0.87·tw²[rad] | 0.05 | 13 |
| tw@2.6Hz | U8/aoa15 | gap = +1.54 +4.73·tw²[rad] | 0.35 | 13 |
| tw@2.6Hz | U10/aoa5 | gap = -1.83 +3.36·tw²[rad] | 0.87 | 6 |
| aoa@tw0 | U8/f1.4 | gap = -1.27 +6.11·aoa[rad] | 0.58 | 6 |
| aoa@tw0 | U8/f1.5 | gap = -1.19 +5.50·aoa[rad] | 0.80 | 4 |
| aoa@tw0 | U8/f1.7 | gap = -1.54 +5.57·aoa[rad] | 0.67 | 6 |
| aoa@tw0 | U8/f2 | gap = -1.61 +3.87·aoa[rad] | 0.41 | 7 |
| aoa@tw0 | U8/f2.3 | gap = -1.39 +3.84·aoa[rad] | 0.39 | 7 |
| aoa@tw0 | U8/f2.5 | gap = -1.45 +4.32·aoa[rad] | 0.62 | 4 |
| aoa@tw0 | U8/f2.6 | gap = -1.10 +5.16·aoa[rad] | 0.25 | 11 |

## model H4  (322 matched pts)


### T: global gap regression (R²=0.93, resid RMSE 1.08N vs gap std 4.09N)
| term | coef | rank |
|---|---|---|
| 1 | +6.613 | 7 |
| f² | +0.304 | 4 |
| tw² | -6.581 | 3 |
| f²·tw² | -2.127 | 1 |
| aoa | -2.955 | 5 |
| U²/64 | -5.436 | 2 |
| f²·U²/64 | -0.043 | 6 |

### T: per-axis laws (gap fits along swept axes)
| axis | fixed | law | R² | n |
|---|---|---|---|---|
| f@tw0 | U6/aoa5 | gap = +3.17 +0.252·f² | 0.72 | 8 |
| f@tw0 | U8/aoa0 | gap = -0.40 +0.510·f² | 0.78 | 8 |
| f@tw0 | U8/aoa5 | gap = +0.32 +0.521·f² | 0.82 | 21 |
| f@tw0 | U8/aoa10 | gap = +0.50 +0.461·f² | 0.85 | 8 |
| f@tw0 | U8/aoa15 | gap = +0.33 +0.382·f² | 0.72 | 8 |
| f@tw0 | U10/aoa5 | gap = -3.21 +0.731·f² | 0.92 | 8 |
| tw@2.6Hz | U6/aoa5 | gap = +3.79 -13.35·tw²[rad] | 0.88 | 6 |
| tw@2.6Hz | U8/aoa0 | gap = +2.31 -18.91·tw²[rad] | 0.93 | 13 |
| tw@2.6Hz | U8/aoa5 | gap = +2.75 -20.57·tw²[rad] | 0.93 | 24 |
| tw@2.6Hz | U8/aoa10 | gap = +2.56 -20.11·tw²[rad] | 0.93 | 13 |
| tw@2.6Hz | U8/aoa15 | gap = +2.15 -24.30·tw²[rad] | 0.96 | 13 |
| tw@2.6Hz | U10/aoa5 | gap = +0.76 -27.55·tw²[rad] | 0.97 | 6 |
| aoa@tw0 | U8/f1.4 | gap = +0.79 +2.15·aoa[rad] | 0.48 | 6 |
| aoa@tw0 | U8/f1.5 | gap = +0.96 +1.99·aoa[rad] | 0.43 | 4 |
| aoa@tw0 | U8/f1.7 | gap = +1.68 +0.38·aoa[rad] | 0.01 | 6 |
| aoa@tw0 | U8/f2 | gap = +2.52 -1.03·aoa[rad] | 0.03 | 7 |
| aoa@tw0 | U8/f2.3 | gap = +2.97 -1.61·aoa[rad] | 0.05 | 7 |
| aoa@tw0 | U8/f2.5 | gap = +2.67 -0.01·aoa[rad] | 0.00 | 4 |
| aoa@tw0 | U8/f2.6 | gap = +3.67 -0.65·aoa[rad] | 0.01 | 11 |

### L: global gap regression (R²=0.73, resid RMSE 1.04N vs gap std 1.98N)
| term | coef | rank |
|---|---|---|
| 1 | -3.800 | 7 |
| f² | +0.387 | 3 |
| tw² | +3.621 | 2 |
| f²·tw² | +0.160 | 6 |
| aoa | +19.532 | 1 |
| U²/64 | +0.783 | 5 |
| f²·U²/64 | -0.254 | 4 |

### L: per-axis laws (gap fits along swept axes)
| axis | fixed | law | R² | n |
|---|---|---|---|---|
| f@tw0 | U6/aoa5 | gap = -0.98 +0.263·f² | 0.87 | 8 |
| f@tw0 | U8/aoa0 | gap = -1.40 -0.006·f² | 0.00 | 8 |
| f@tw0 | U8/aoa5 | gap = -0.87 -0.018·f² | 0.01 | 21 |
| f@tw0 | U8/aoa10 | gap = -1.15 +0.271·f² | 0.30 | 8 |
| f@tw0 | U8/aoa15 | gap = +1.46 +0.309·f² | 0.59 | 8 |
| f@tw0 | U10/aoa5 | gap = +0.82 -0.351·f² | 0.63 | 8 |
| tw@2.6Hz | U6/aoa5 | gap = +0.48 +2.61·tw²[rad] | 0.33 | 6 |
| tw@2.6Hz | U8/aoa0 | gap = -1.48 +7.24·tw²[rad] | 0.90 | 13 |
| tw@2.6Hz | U8/aoa5 | gap = -0.85 +3.21·tw²[rad] | 0.55 | 24 |
| tw@2.6Hz | U8/aoa10 | gap = +1.38 +1.66·tw²[rad] | 0.07 | 13 |
| tw@2.6Hz | U8/aoa15 | gap = +3.90 +7.03·tw²[rad] | 0.62 | 13 |
| tw@2.6Hz | U10/aoa5 | gap = -1.28 +7.34·tw²[rad] | 0.83 | 6 |
| aoa@tw0 | U8/f1.4 | gap = -1.71 +12.63·aoa[rad] | 0.77 | 6 |
| aoa@tw0 | U8/f1.5 | gap = -1.58 +12.46·aoa[rad] | 0.82 | 4 |
| aoa@tw0 | U8/f1.7 | gap = -2.11 +14.10·aoa[rad] | 0.78 | 6 |
| aoa@tw0 | U8/f2 | gap = -2.38 +15.45·aoa[rad] | 0.79 | 7 |
| aoa@tw0 | U8/f2.3 | gap = -2.34 +17.22·aoa[rad] | 0.80 | 7 |
| aoa@tw0 | U8/f2.5 | gap = -2.17 +17.31·aoa[rad] | 0.87 | 4 |
| aoa@tw0 | U8/f2.6 | gap = -2.15 +20.51·aoa[rad] | 0.80 | 11 |

### phase-dimension gap (Fig16 8/5/2Hz, filtered, harmonics of model−exp)
| kind | tw | mean(N) | 1/rev amp | 2/rev amp | 3/rev amp | max|gap| | @t/T |
|---|---|---|---|---|---|---|---|
| T | 0 | +2.76 | 0.30∠-107° | 4.34∠-138° | 0.26 | 7.39 | 0.81 |
| L | 0 | +1.74 | 14.07∠94° | 5.35∠-152° | 3.39 | 20.62 | 0.29 |
| T | 22.5 | +1.14 | 3.74∠-90° | 1.79∠-32° | 0.93 | 7.75 | 0.27 |
| L | 22.5 | +0.39 | 30.84∠92° | 4.06∠-145° | 5.05 | 39.06 | 0.28 |
| T | 45 | +6.71 | 4.66∠-111° | 10.36∠0° | 1.45 | 23.43 | 0.26 |
| L | 45 | +1.80 | 49.25∠92° | 4.77∠-131° | 7.25 | 62.21 | 0.28 |

## model H13  (52 matched pts)


### T: global gap regression (R²=0.96, resid RMSE 0.81N vs gap std 4.21N)
| term | coef | rank |
|---|---|---|
| 1 | +3.002 | 7 |
| f² | +0.507 | 4 |
| tw² | -8.686 | 1 |
| f²·tw² | -0.325 | 5 |
| aoa | +1.449 | 6 |
| U²/64 | -2.583 | 3 |
| f²·U²/64 | -0.696 | 2 |

### T: per-axis laws (gap fits along swept axes)
| axis | fixed | law | R² | n |
|---|---|---|---|---|
| f@tw0 | U6/aoa5 | gap = +2.73 -0.183·f² | 0.43 | 4 |
| f@tw0 | U10/aoa5 | gap = -2.92 +0.125·f² | 0.21 | 4 |
| tw@2.6Hz | U8/aoa15 | gap = +0.41 -14.43·tw²[rad] | 0.96 | 5 |

### L: global gap regression (R²=0.59, resid RMSE 1.11N vs gap std 1.74N)
| term | coef | rank |
|---|---|---|
| 1 | -3.164 | 7 |
| f² | +0.544 | 2 |
| tw² | +1.748 | 5 |
| f²·tw² | +0.360 | 3 |
| aoa | -8.833 | 4 |
| U²/64 | +0.665 | 6 |
| f²·U²/64 | -0.532 | 1 |

### L: per-axis laws (gap fits along swept axes)
| axis | fixed | law | R² | n |
|---|---|---|---|---|
| f@tw0 | U6/aoa5 | gap = -2.03 +0.097·f² | 0.41 | 4 |
| f@tw0 | U10/aoa5 | gap = -0.57 -0.988·f² | 0.80 | 4 |
| tw@2.6Hz | U8/aoa15 | gap = -5.06 +6.52·tw²[rad] | 0.76 | 5 |

### phase-dimension gap (Fig16 8/5/2Hz, filtered, harmonics of model−exp)
| kind | tw | mean(N) | 1/rev amp | 2/rev amp | 3/rev amp | max|gap| | @t/T |
|---|---|---|---|---|---|---|---|
| T | 0 | +0.23 | 0.15∠-72° | 1.80∠-150° | 0.20 | 2.20 | 0.80 |
| L | 0 | +4.35 | 2.48∠68° | 3.35∠-142° | 2.41 | 11.70 | 0.56 |
| T | 22.5 | +2.40 | 2.48∠-84° | 1.02∠40° | 0.58 | 6.92 | 0.29 |
| L | 22.5 | +3.35 | 11.12∠85° | 2.49∠-128° | 2.52 | 13.18 | 0.79 |
| T | 45 | +4.90 | 2.28∠-125° | 5.93∠15° | 0.97 | 13.87 | 0.28 |
| L | 45 | +1.32 | 24.14∠91° | 3.31∠-120° | 4.42 | 29.54 | 0.30 |