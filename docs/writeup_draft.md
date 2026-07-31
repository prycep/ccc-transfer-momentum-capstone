# Can first-year academic momentum predict four-year transfer outcomes?

## Abstract

California community colleges serve students with diverse goals and institutional contexts, yet four-year transfer outcomes are observed only after a substantial delay. This project asks whether a small set of college-level first-year academic momentum indicators can predict the percentage of an entering General Admit cohort that transfers to a four-year institution within four years. Public DataVista General Admit Cohort exports were assembled into a college-by-cohort-year panel. Ordinary least squares was compared with a nonlinear K-nearest-neighbors regressor using chronological validation rather than a random split. The complete-case analysis contained 550 college-cohort observations from 111 colleges. On the held-out 2019-20 cohort, linear regression achieved a mean absolute error (MAE) of 4.02 percentage points and R² of 0.41; KNN achieved an MAE of 3.58 percentage points and R² of 0.50. KNN reduced MAE by 0.45 percentage points, or 11.1%, with a paired-bootstrap 95% confidence interval of 0.11 to 0.79 percentage points. Both models systematically overpredicted transfer for the pandemic-affected cohort. The findings should be interpreted as institutional forecasting evidence, not student-level prediction or causal inference.

## Research question and motivation

The primary question is: **Can a small set of first-year academic momentum indicators predict four-year transfer outcomes among first-time California Community College cohorts, to what extent, and does a supervised machine-learning method meaningfully improve on linear regression?**

Four-year transfer is a delayed outcome. If early, readily observed institutional indicators carry useful predictive information, colleges and researchers may be able to identify cohorts whose longer-run outcomes differ from historical expectations while there is still time to investigate. The goal is prediction and error characterization—not to claim that changing any one indicator will necessarily cause transfer rates to rise.

## Data and unit of analysis

The data come from the California Community Colleges Chancellor's Office DataVista General Admit Cohort bulk-download system. The unit of analysis is one college × entering cohort year. The export contains many demographic disaggregations; this study uses only overall, all-program rows so that each metric contributes at most one value per college and cohort.

The preferred outcome is DataVista metric 650 at the four-year timeframe: the percentage of the overall entering General Admit cohort that transferred to a four-year postsecondary institution within four years. This denominator aligns more directly with the research question than metric 620, which conditions on a narrower population. Mature four-year target values are available for the 2015-16 through 2019-20 entering cohorts. The 2020-21 cohort can describe first-year momentum but is excluded from four-year model evaluation when its outcome is not yet present.

## Predictors

The pre-specified predictor set contains five first-year momentum measures and one institutional-size control:

1. Course success rate (408, one year).
2. Average successfully completed semester units (428, one year).
3. Persistence to the subsequent primary term (453, one year).
4. Percentage completing at least 30 degree-applicable units (458, one year).
5. Percentage completing transfer-level mathematics and English (501, one year).
6. The natural logarithm of one plus cohort size (122, one year).

Rates are represented as proportions between zero and one. Average units and log cohort size retain their natural scales before training-only standardization.

## Data processing

Each college export is read in compressed form. Rows are filtered to the seven required metric/timeframe combinations, `All Programs`, and `Overall / All` for both disaggregation dimensions. DataVista's `\N` marker is treated as missing. The retained rows are pivoted into a college × cohort-year panel. Non-college administrative or continuing-education locales appearing in the college selector are flagged and excluded from the primary modeling sample. A complete-case model row requires all six predictors and the four-year target; raw and incomplete rows remain in the processed panel for auditing.

## Modeling and validation

The linear baseline is ordinary least squares. The comparison model is K-nearest-neighbors regression, a supervised nonlinear method that predicts from outcomes of nearby observations in standardized feature space. KNN was selected because the number of mature cohort years is small and does not support honest tuning of a highly flexible ensemble with many hyperparameters.

The split follows time:

- 2015-16 through 2017-18: development training cohorts.
- 2018-19: validation cohort used to select the KNN neighbor count.
- 2015-16 through 2018-19: final training data.
- 2019-20: untouched final test cohort.

All feature means and standard deviations are estimated using the relevant training sample only. Models are evaluated using mean absolute error (MAE), root mean squared error (RMSE), and R² across colleges in the held-out cohort. A paired college-level bootstrap estimates a 95% confidence interval for the difference in MAE between KNN and linear regression. An ML improvement is considered meaningful only if it is nontrivial in magnitude and the uncertainty interval does not make the direction ambiguous.

## Results

The final complete-case sample contained 550 college-cohort observations from 111 colleges. Annual usable sample sizes ranged from 109 to 111 colleges. The held-out 2019-20 test cohort contained 111 colleges.

On the 2018-19 validation cohort, linear regression achieved an MAE of 2.50 percentage points, RMSE of 3.13 percentage points, and R² of 0.69. Validation selected `k = 20` for KNN; this model achieved an MAE of 2.22 percentage points, RMSE of 2.80 percentage points, and R² of 0.75.

Performance deteriorated for both methods on the held-out 2019-20 cohort. Linear regression had an MAE of 4.02 percentage points, RMSE of 4.65 percentage points, and R² of 0.41. KNN had an MAE of 3.58 percentage points, RMSE of 4.28 percentage points, and R² of 0.50. KNN therefore reduced MAE by 0.45 percentage points, an 11.1% relative reduction. In paired bootstrap resampling across held-out colleges, the estimated KNN-minus-linear MAE difference was -0.45 percentage points, with a 95% confidence interval from -0.79 to -0.11. The interval favors KNN, but the absolute gain is modest: nonlinear ML added measurable predictive value without changing the overall conclusion that prediction became substantially harder in the pandemic-affected cohort.

Insert `figures/main_results.svg` here. The 45-degree line represents perfect prediction; points farther from it have larger errors.

## Failure analysis

Failure analysis focuses on residuals, defined as observed minus predicted transfer rate. Positive residuals indicate underprediction, and negative residuals indicate overprediction.

First, colleges were divided using the test cohort's first quartile of cohort size. Small-college rates can be noisier because their denominators are smaller, and suppression or missingness may be more common. Contrary to that concern, small colleges did not have worse test errors. Linear-model MAE was 4.01 percentage points in the smallest quartile and 4.03 among larger colleges. KNN MAE was 3.27 percentage points in the smallest quartile and 3.68 among larger colleges. Thus, small institutional size was not the main source of failure in this test cohort.

Second, the held-out 2019-20 cohort is substantively important because students' first year crossed the onset of COVID-19. Residuals were predominantly negative, meaning both models overpredicted realized four-year transfer. The linear model overpredicted by 3.62 percentage points on average among larger colleges and 4.01 points in the smallest quartile; KNN overpredicted by approximately 2.89 points in both groups. Large negative residuals occurred at colleges including Mendocino, Butte, Oxnard, and Rio Hondo, while KNN substantially underpredicted San Diego Miramar. These examples illustrate heterogeneous distribution shift and omitted institutional context rather than college rankings.

Finally, large errors may reflect omitted institutional context, measurement noise, suppression, changes in reporting, or genuine departures from statewide historical relationships. These errors are diagnostic prompts, not evidence that a college performed well or poorly because of a particular predictor.

## Limitations and conclusion

This project has four central limitations. First, its ecological unit prevents inference about individual students. Second, five mature cohort years provide limited information about temporal stability and constrain model complexity. Third, missingness and suppression may be nonrandom, especially for small colleges. Fourth, the models omit policy, geography, student composition, transfer capacity, and other institutional factors that may explain residual variation.

Within those limits, the design provides an honest test of whether a compact early-momentum signal generalizes to a later cohort and whether a simple nonlinear learner adds predictive value. The indicators did predict meaningful cross-college variation: the held-out models explained approximately 41% to 50% of outcome variance. KNN produced a modest but statistically supported improvement over linear regression, reducing MAE by about 0.45 percentage points. However, errors nearly doubled relative to validation and were systematically optimistic for the 2019-20 cohort, showing that a model can rank institutions moderately well while missing a cohort-wide shock.
