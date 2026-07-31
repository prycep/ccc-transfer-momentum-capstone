# 8–10 minute presentation outline

## 0:00–0:45 — Hook and research question

“Four-year transfer outcomes arrive too late to help us understand a cohort's early trajectory. I asked whether a small set of first-year academic momentum indicators could forecast those later outcomes at the college level—and whether machine learning adds anything beyond a linear model.”

State clearly that the unit is college × cohort, not individual students.

## 0:45–1:45 — Data

- Public DataVista General Admit Cohort exports.
- College-level cohorts from 2015-16 through 2019-20 for the mature outcome.
- Outcome: metric 650 at four years.
- Overall, all-program rows only.
- Briefly show the project schematic.

## 1:45–2:45 — Predictors

Name the five first-year indicators: course success, successful units, persistence, 30-unit completion, and transfer-level math and English completion. Add log cohort size as a control.

Explain the idea in plain language: these measures capture whether a cohort is accumulating successful academic progress early.

## 2:45–4:00 — Methods and honest validation

- Linear regression is the transparent baseline.
- K-nearest neighbors is the nonlinear comparison.
- Earlier cohorts train the models.
- 2018-19 selects the KNN setting.
- 2019-20 remains held out until the end.

Emphasize why a random row split would leak future cohort information and exaggerate confidence.

## 4:00–5:45 — Main results

Show `figures/main_results.svg`.

Report:

- Held-out sample: 111 colleges.
- Linear model: 4.02 percentage-point MAE, 4.65-point RMSE, and R² of 0.41.
- KNN (`k = 20`): 3.58 percentage-point MAE, 4.28-point RMSE, and R² of 0.50.
- KNN reduced MAE by 0.45 percentage points, or 11.1%; the paired-bootstrap 95% confidence interval for the improvement was 0.11 to 0.79 points.
- One-sentence answer: KNN improved prediction modestly and consistently, but not dramatically.

Translate MAE into percentage points for accessibility.

## 5:45–7:15 — Where the models fail

- Small colleges did not fail more: linear MAE was 4.01 points for the smallest quartile versus 4.03 for larger colleges; KNN was 3.27 versus 3.68.
- Explain positive residual = underprediction; negative = overprediction.
- Both models systematically overpredicted the pandemic-affected 2019-20 cohort: about 3.6 points for linear regression and 2.9 for KNN.
- Mention one or two large residuals only as examples, not rankings or causal judgments.

## 7:15–8:15 — Limitations

- College-level, not student-level.
- Only five mature cohort years.
- Missingness/suppression may be nonrandom.
- Many institutional and contextual factors are omitted.
- Prediction is not causation.

## 8:15–9:00 — Conclusion

Answer all three questions directly:

1. Yes, the early indicators predicted meaningful cross-college variation.
2. Held-out R² was 0.41 for linear regression and 0.50 for KNN, with errors of roughly 3.6–4.0 percentage points.
3. KNN was modestly better: an 11.1% MAE reduction, not a transformative gain.

Close with: “The practical value is not a definitive ranking of colleges. It is an auditable early-warning benchmark—and the residuals tell us where that benchmark stops being trustworthy.”
