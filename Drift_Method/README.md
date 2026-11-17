# 📈 Drift Method for Time Series Forecasting

The **Drift Method** is a simple statistical forecasting technique that extends the Naive Method by extrapolating the average change observed in historical data. It is useful as a transparent baseline when a time series has a reasonably consistent upward or downward direction.

This repository section contains a detailed, hands-on Python notebook covering the intuition, mathematical formulation, implementation, evaluation, and practical limitations of the Drift Method.

## 📚 Contents

- [1. Introduction](#1-introduction)
- [2. Mathematical formulation](#2-mathematical-formulation)
- [3. How the method works](#3-how-the-method-works)
- [4. Worked example](#4-worked-example)
- [5. Drift vs Naive vs Average](#5-drift-vs-naive-vs-average)
- [6. Evaluation](#6-evaluation)
- [7. Advantages](#7-advantages)
- [8. Limitations](#8-limitations)
- [9. Practical considerations](#9-practical-considerations)
- [10. Common mistakes](#10-common-mistakes)
- [11. Applications](#11-applications)
- [12. Notebook contents](#12-notebook-contents)
- [13. Key takeaways](#14-key-takeaways)
- [14. References](#15-references)

## 1. Introduction

The Drift Method is a baseline forecasting approach that assumes the average change between the first and last observations in the training series will continue into future periods.

It can be viewed as a trend-aware extension of Naive forecasting:

- **Naive:** future values remain equal to the last observed value.
- **Drift:** future values move by a constant amount per period, based on the average historical change.

The method is easy to understand and implement. However, it assumes that the historical average change is representative of the future. It does not explicitly model seasonality, nonlinear patterns, or sudden structural changes.

## 2. Mathematical formulation

Let:

- $y_1$ = first observation in the training series
- $y_T$ = last observation in the training series
- $T$ = number of observations in the training series
- $h$ = forecast horizon, or number of steps ahead
- $d$ = estimated average change per time step

The estimated drift is:

$$
d = \frac{y_T - y_1}{T - 1}
$$

The forecast $h$ steps ahead is:

$$
\hat{y}_{T+h} = y_T + h\left(\frac{y_T - y_1}{T - 1}\right)
$$

The denominator is $T-1$ because $T$ observations contain $T-1$ time intervals.

### Understanding the equation

1. Calculate the total change between the first and last training observations.
2. Divide the total change by the number of intervals.
3. This gives the average change per time step.
4. Multiply the drift by the forecast horizon.
5. Add the projected change to the last observed value.

A positive drift produces an upward forecast. A negative drift produces a downward forecast. A zero drift produces the same forecast as the Naive Method.

## 3. How the method works

Suppose monthly demand has generally increased over the historical period.

The Drift Method does not fit a complex trend model. Instead, it calculates one average change using the endpoints of the training data and extends that change into the future.

For example, if demand moves from 100 units to 160 units over 11 intervals, the estimated drift is:

$$
d = \frac{160 - 100}{11} \approx 5.45
$$

The method estimates an increase of approximately 5.45 units per period. Starting from the last observed value, it adds this amount for each future step.

## 4. Worked example

Consider the following historical observations:

| Period | Actual demand |
|---|---:|
| 1 | 100 |
| 2 | 110 |
| 3 | 120 |
| 4 | 130 |
| 5 | 140 |

There are five observations and therefore four intervals.

**Step 1: Calculate the drift**

$$
d = \frac{140 - 100}{5 - 1} = 10
$$

**Step 2: Calculate future forecasts**

One period ahead:

$$
\hat{y}_{6} = 140 + (1 \times 10) = 150
$$

Two periods ahead:

$$
\hat{y}_{7} = 140 + (2 \times 10) = 160
$$

Three periods ahead:

$$
\hat{y}_{8} = 140 + (3 \times 10) = 170
$$

| Forecast period | Forecast |
|---|---:|
| 6 | 150 |
| 7 | 160 |
| 8 | 170 |

The forecast increases by 10 units at every step because the estimated drift is constant.

## 5. Drift vs Naive vs Average

These three methods are useful baseline models, but they make different assumptions.

| Method | Forecast logic | Main assumption |
|---|---|---|
| Naive | Repeats the last observed value | No change in future |
| Average | Repeats the historical mean | Future level is represented by the overall average |
| Drift | Extends the average change between the first and last training observations | Historical average change continues |

### Key distinction

- **Naive:** no upward or downward movement.
- **Average:** forecasts a constant level based on all training observations.
- **Drift:** forecasts a straight-line movement based on average historical change.

When the estimated drift is zero, the Drift forecast is identical to the Naive forecast.

## 6. Evaluation

Forecasts should be evaluated on observations that were not used to calculate the drift. Use a chronological train/test split: train on earlier observations and evaluate on later observations.

Common evaluation metrics include:

### Mean Absolute Error (MAE)

$$
MAE = \frac{1}{n}\sum_{t=1}^{n} |y_t - \hat{y}_t|
$$

Measures the average absolute error in the original units.

### Root Mean Squared Error (RMSE)

$$
RMSE = \sqrt{\frac{1}{n}\sum_{t=1}^{n}(y_t - \hat{y}_t)^2}
$$

Penalizes larger errors more strongly than MAE.

### Mean Absolute Percentage Error (MAPE)

$$
MAPE = \frac{100}{n}\sum_{t=1}^{n}\left|\frac{y_t - \hat{y}_t}{y_t}\right|
$$

Expresses error as a percentage. It is undefined when actual values are zero, so zero handling must be explicit.

### Weighted Mean Absolute Percentage Error (WMAPE)

$$
WMAPE = 100 \times \frac{\sum_{t=1}^{n} |y_t - \hat{y}_t|}{\sum_{t=1}^{n} |y_t|}
$$

Measures total absolute error relative to total actual volume. It is often useful when evaluating demand across observations with different volumes.

Lower values indicate better accuracy for all four metrics. Always interpret metrics in the context of the series and business objective.

## 7. Advantages

- **Simple and intuitive:** easy to explain and implement.
- **Trend-aware baseline:** unlike Naive forecasting, it projects an average upward or downward movement.
- **Computationally efficient:** requires very little computation.
- **Transparent:** the forecast can be traced directly to the first and last training values.
- **Useful benchmark:** helps assess whether more complex forecasting models add value.

## 8. Limitations

- **Assumes constant drift:** the average change is projected at the same rate throughout the forecast horizon.
- **No explicit seasonality:** recurring weekly, monthly, or yearly patterns are not captured.
- **Endpoint sensitivity:** unusual first or last observations can have a large impact on the estimated drift.
- **Structural changes:** historical trends may no longer apply after changes in pricing, promotions, supply, or customer behavior.
- **Long-horizon risk:** linear extrapolation can produce unrealistic forecasts as the horizon increases.
- **No nonlinear trend modeling:** accelerating, decelerating, or changing trends are not represented.

## 9. Practical considerations

1. **Use chronological data:** observations must be sorted by time.
2. **Check regular frequency:** the method assumes equally spaced time intervals. Missing periods should be investigated and handled appropriately.
3. **Avoid data leakage:** calculate the drift using only the training data, never the test data.
4. **Inspect endpoint values:** investigate unusual values at the beginning or end of the training window.
5. **Check seasonality:** if strong seasonal patterns exist, compare against a Seasonal Naive method or a model that explicitly handles seasonality.
6. **Choose the forecast horizon carefully:** long-range linear extrapolation may become unreliable.
7. **Review business constraints:** for non-negative quantities such as demand, a downward trend can mathematically produce negative forecasts. Any clipping or business constraint should be explicit and justified.
8. **Use backtesting:** evaluate across multiple rolling forecast origins when possible, rather than relying on a single holdout period.

## 10. Common mistakes

- Dividing by $T$ instead of $T-1$ when calculating the drift from $T$ observations.
- Using future or test observations to estimate the drift.
- Confusing Drift with Naive forecasting.
- Assuming that a historical trend will continue indefinitely.
- Ignoring outliers or structural breaks at the endpoints.
- Applying the method to strongly seasonal data without a suitable seasonal benchmark.
- Comparing models using different training or test periods.

## 11. Applications

The Drift Method can be useful for:

- Establishing a simple trend-based forecasting baseline.
- Forecasting a series with a reasonably stable linear direction.
- Comparing advanced forecasting approaches against a transparent benchmark.
- Quickly producing a reference forecast when a lightweight method is needed.

It is not generally a strong choice when demand is highly intermittent, strongly seasonal, volatile, or affected by frequent structural changes.

## 12. Notebook contents

The accompanying notebook, `Drift_Method_Forecasting.ipynb`, includes:

1. Introduction and mathematical intuition.
2. Synthetic monthly demand data generation.
3. Chronological train/test split.
4. Drift calculation.
5. Reusable Python forecasting function.
6. Actual vs forecast visualization.
7. Comparison with Naive and Average methods.
8. MAE, RMSE, MAPE, and WMAPE evaluation.
9. Forecast error inspection.
10. Future horizon forecasting.
11. Practical considerations and common mistakes.
12. Hands-on exercises and key takeaways.

Drift Method Implementation: [Notebook Link](https://github.com/ashay-thamankar/time_series_models/blob/main/Drift_Method/Drift_Method_Forecasting.ipynb)

The notebook uses synthetic data so it can be run independently without downloading a dataset.

## 13. Key takeaways

- The Drift Method is a trend-aware extension of the Naive Method.
- It calculates average change per time interval from the first and last training observations.
- It extends this change linearly into future periods.
- It is a useful, transparent baseline for series with a reasonably consistent trend.
- It does not explicitly model seasonality, nonlinear patterns, or structural changes.
- Always evaluate it on a chronological holdout and compare it with other baseline methods.

## 14. References

- [Forecasting: Principles and Practice – Simple methods](https://otexts.com/fpp3/simple-methods.html)
- [Forecasting: Principles and Practice – Forecast accuracy](https://otexts.com/fpp3/accuracy.html)
- [Pandas documentation](https://pandas.pydata.org/docs/)
- [NumPy documentation](https://numpy.org/doc/)
- [Scikit-learn model evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html)

---

Happy Learning! 🚀
