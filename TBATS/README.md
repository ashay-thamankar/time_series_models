# TBATS for Time Series Forecasting

## 1. Introduction

TBATS is a statistical forecasting method designed for complex seasonal time series, particularly series with **multiple seasonal periods**. Consider hourly retail orders: demand may vary by hour of day and also differ across weekdays and weekends. TBATS can represent both daily and weekly cycles in one model.

TBATS stands for:

- **T – Trigonometric seasonality**
- **B – Box-Cox transformation**
- **A – ARMA errors**
- **T – Trend**
- **S – Seasonal components**

It was introduced by De Livera, Hyndman, and Snyder (2011) to handle complex seasonal patterns, including multiple and non-integer seasonal periods.

## 2. Why TBATS?

Many business time series contain more than one seasonal pattern.

| Data frequency | Seasonal pattern 1 | Seasonal pattern 2 |
|---|---:|---:|
| Hourly retail demand | Daily: 24 | Weekly: 168 |
| Half-hourly electricity demand | Daily: 48 | Weekly: 336 |
| Daily demand | Weekly: 7 | Annual: approximately 365.25 |
| Monthly demand | Annual: 12 | Other recurring cycles, if enough data exists |

Seasonal periods are measured in **number of observations**, not calendar labels. For hourly data, a day contains 24 observations and a week contains 168.

TBATS is worth considering when multiple seasonal cycles are evident, seasonal shapes are complex, or seasonal periods are non-integer. It is not automatically the best model for every seasonal series.

## 3. TBATS components

### 3.1 Trigonometric seasonality

TBATS represents seasonal patterns using trigonometric (Fourier) terms. For a seasonal period \(m\), a seasonal pattern can be illustrated as:

$$
s_t = \\sum_{k=1}^{K}\\left[
a_k\\cos\\left(\\frac{2\\pi kt}{m}\\right)
+b_k\\sin\\left(\\frac{2\\pi kt}{m}\\right)
\\right]
$$

Where:
- \(m\): seasonal period.
- \(k\): harmonic number.
- \(K\): number of harmonics.
- \(a_k, b_k\): estimated coefficients.

A smaller number of harmonics produces a smoother seasonal representation; more harmonics can capture more detail but increase complexity. This equation illustrates the trigonometric idea, not the complete TBATS state-space specification.

### 3.2 Box-Cox transformation

The Box-Cox transformation can help stabilize variance when variation increases with the level of the series.

For positive observations:

$$
w_t =
\begin{cases}
\frac{y_t^\lambda - 1}{\lambda}, & \lambda \ne 0 \\
\log(y_t), & \lambda = 0
\end{cases}
$$

Here \(y_t\) is the original observation, \(w_t\) is transformed data, and \(\lambda\) is the transformation parameter. Box-Cox requires positive values.

### 3.3 ARMA errors

Even after modelling trend and seasonality, residual errors may remain autocorrelated. ARMA errors can capture this remaining temporal dependence.

$$
\\varepsilon_t =
\\sum_{i=1}^{p}\\phi_i\\varepsilon_{t-i}
+\\eta_t
+\\sum_{j=1}^{q}\\theta_j\\eta_{t-j}
$$

Here \(p\) and \(q\) are AR and MA orders, \(\phi_i\) and \(\theta_j\) are coefficients, and \(\eta_t\) is a white-noise innovation.

### 3.4 Trend and seasonal components

TBATS can include a trend and optionally damp it, so that trend growth does not necessarily continue indefinitely into the future.

A simplified conceptual decomposition is:

$$
y_t = \\ell_t + d_t + \\sum_i s_{i,t} + \\varepsilon_t
$$

where \(\ell_t\) represents level, \(d_t\) trend contribution, \(s_{i,t}\) seasonal contributions, and \(\varepsilon_t\) error. This is an intuitive decomposition, not the complete mathematical specification.

## 4. How TBATS works

A simplified workflow:

1. Identify the data frequency and plausible seasonal periods.
2. Consider whether Box-Cox transformation is appropriate.
3. Represent seasonal cycles with trigonometric terms.
4. Estimate level, trend, and seasonal states.
5. Model remaining autocorrelation using ARMA errors, if enabled.
6. Estimate model parameters and generate forecasts.

TBATS can be computationally expensive. Runtime depends on series length, number of seasonal periods, and model options.

## 5. Python implementation

### Installation

```bash
pip install tbats numpy pandas matplotlib scikit-learn statsmodels
```

The package can be sensitive to Python and numerical-library versions. If installation fails, check compatibility and try a clean virtual environment.

### Imports

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tbats import TBATS
from sklearn.metrics import mean_absolute_error, mean_squared_error
```

### Example data and seasonal periods

The companion notebook creates 12 weeks of synthetic hourly retail demand with:
- Daily seasonality: 24 observations.
- Weekly seasonality: 168 observations.
- Gradual trend and random noise.

Python Implementation: [Notebook Link](https://github.com/ashay-thamankar/time_series_models/blob/main/TBATS/TBATS_Forecasting.ipynb).

### Chronological data split

The notebook uses:
- First 10 weeks: training.
- Next week: validation.
- Final week: test.

Time-series data should not be randomly shuffled. A chronological split prevents future observations from leaking into model training.

### Fit TBATS

```python
estimator = TBATS(
    seasonal_periods=[24, 168],
    use_box_cox=False,
    use_trend=True,
    use_damped_trend=True,
    use_arma_errors=True,
    n_jobs=1
)

model = estimator.fit(train.to_numpy())
```

For hourly observations, `[24, 168]` specifies daily and weekly cycles.

### Forecast

```python
result = model.forecast(
    steps=len(validation),
    confidence_level=0.95
)
```

The package API may return a forecast alone or a forecast together with confidence intervals, depending on version. The notebook handles both forms.

## 6. Important TBATS parameters

| Parameter | Meaning |
|---|---|
| `seasonal_periods` | List of seasonal cycle lengths, in observations |
| `use_box_cox` | Enable or disable Box-Cox transformation |
| `use_trend` | Include a trend component |
| `use_damped_trend` | Allow trend to dampen over the forecast horizon |
| `use_arma_errors` | Model autocorrelation in errors |
| `n_jobs` | Number of parallel jobs |

Examples:
- Hourly daily: `24`
- Hourly weekly: `168`
- Half-hourly daily: `48`
- Daily weekly: `7`
- Daily annual: approximately `365.25`
- Monthly annual: `12`

Long seasonal cycles require enough history. A few months of daily data are not sufficient to estimate annual patterns reliably.

## 7. Forecast evaluation

Mean Absolute Error:

$$
MAE = \\frac{1}{n}\\sum_{t=1}^{n}|y_t-\\hat y_t|
$$

Root Mean Squared Error:

$$
RMSE = \\sqrt{\\frac{1}{n}\\sum_{t=1}^{n}(y_t-\\hat y_t)^2}
$$

Weighted Mean Absolute Percentage Error:

$$
WMAPE = \\frac{\\sum_{t=1}^{n}|y_t-\\hat y_t|}
{\\sum_{t=1}^{n}|y_t|}\\times100
$$

Bias (positive means over-forecasting under this convention):

$$
Bias = \\frac{\\sum_{t=1}^{n}(\\hat y_t-y_t)}
{\\sum_{t=1}^{n}|y_t|}\\times100
$$

Use consistent metric definitions across all candidate models.

## 8. Prediction intervals

A point forecast provides one expected value for each future timestamp. A prediction interval expresses forecast uncertainty. A nominal 95% interval is intended to contain the future observation with 95% coverage under the model assumptions.

Intervals often widen with forecast horizon. Check empirical coverage and interval width on held-out data.

## 9. Residual diagnostics

Residuals are:

$$
e_t = y_t-\\hat y_t
$$

Recommended checks:
1. Plot residuals over time.
2. Inspect residual autocorrelation (ACF).
3. Use the Ljung-Box test to check remaining autocorrelation.
4. Check for changing variance and outliers.
5. Review errors by hour and weekday.

Good residual diagnostics do not guarantee strong business forecast accuracy. Always evaluate on validation and test data.

## 10. Advantages

- Supports multiple seasonal periods.
- Uses trigonometric terms to represent complex seasonal shapes.
- Supports non-integer seasonal periods.
- Can include Box-Cox transformation.
- Can include trend and damped trend.
- Can model autocorrelated errors through ARMA errors.

## 11. Limitations

- Can be computationally expensive, especially for long series and many seasonal periods.
- Requires enough history to estimate long seasonal cycles.
- More complex than simple exponential-smoothing methods.
- Does not natively incorporate arbitrary external regressors such as price, promotions, weather, or holidays in the same way as regression-based forecasting approaches.
- Greater complexity does not guarantee better forecast accuracy.

## 12. TBATS vs other forecasting methods

| Model | Seasonal support | Multiple seasonal periods |
|---|---|---|
| ARIMA | Not explicitly seasonal | No |
| SARIMA | Seasonal | Commonly one seasonal period |
| Auto-ARIMA | Depends on seasonal configuration | Implementation-dependent |
| Holt-Winters | One seasonal period | Not designed for multiple cycles |
| TBATS | Yes | Yes |
| Prophet | Yes | Multiple seasonalities can be specified |

## 13. Supply-chain and retail applications

- Hourly store and e-commerce order demand.
- Warehouse picking, receiving, and dispatch workload.
- Daily/weekly resource consumption.
- Operational volume forecasting for staffing and capacity planning.

For item-location demand forecasting, check whether each series has enough history and stable seasonal structure. Intermittent demand, stockouts, assortment changes, and promotions may require other methods or additional explanatory variables.

## 14. Practical modelling checklist

1. Verify timestamp regularity and missing observations.
2. Identify seasonal periods in units of observations.
3. Ensure enough history for long seasonal cycles.
4. Use chronological train, validation, and test sets.
5. Compare with simple baselines and alternative models.
6. Evaluate MAE, RMSE, WMAPE, and bias.
7. Inspect residuals and prediction intervals.
8. Refit only after model selection and monitor performance.


## Key takeaway

TBATS is a useful candidate when multiple seasonal patterns matter. Its added complexity should be justified by out-of-sample performance, residual diagnostics, and business relevance.

## References

1. De Livera, A. M., Hyndman, R. J., & Snyder, R. D. (2011). *Forecasting time series with complex seasonal patterns using exponential smoothing*. Journal of the American Statistical Association.
2. Python TBATS package: https://github.com/intive-DataScience/tbats
3. Forecasting: Principles and Practice – Complex seasonality: https://otexts.com/fpp3/complexseasonality.html
