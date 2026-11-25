# Day 11: ARIMA for Time Series Forecasting

## 1. Introduction

**ARIMA (AutoRegressive Integrated Moving Average)** is a classical statistical method for univariate time-series forecasting. It models relationships between observations over time and incorporates patterns in previous forecast errors.

ARIMA is written as **ARIMA(p, d, q)**:

- **p – AutoRegressive (AR) order:** number of lagged observations included.
- **d – Integrated order:** number of differences applied to address non-stationarity.
- **q – Moving Average (MA) order:** number of lagged forecast innovations/errors included.

ARIMA is useful when a series has temporal autocorrelation and can be made reasonably stationary through differencing. Standard ARIMA is non-seasonal; seasonal extensions are SARIMA and SARIMAX.

---

## 2. Understanding the components

### 2.1 AutoRegressive (AR) — p

The AR component models the current value using previous observations.

An AR(1) model:

$$
Y_t = c + \phi_1Y_{t-1} + \varepsilon_t
$$

An AR(2) model:

$$
Y_t = c + \phi_1Y_{t-1} + \phi_2Y_{t-2} + \varepsilon_t
$$

Where:
- $Y_t$ is the current observation.
- $Y_{t-1}, Y_{t-2}$ are lagged observations.
- $c$ is a constant.
- $\phi_i$ are autoregressive coefficients.
- $\varepsilon_t$ is the current innovation.

**Intuition:** if current demand is related to demand in previous periods, the AR component can learn that dependence.

### 2.2 Integrated (I) — d

The integrated component refers to differencing the series. Differencing can remove trends or changing levels and help make a series stationary.

First-order differencing:

$$
\Delta Y_t = Y_t - Y_{t-1}
$$

Second-order differencing:

$$
\Delta^2Y_t = \Delta Y_t - \Delta Y_{t-1}
$$

Example:

```text
Original:          100, 110, 120, 130, 140
First difference:   10,  10,  10,  10
```

The original series has a linear upward trend. Its first differences are constant. Real-world data is noisier, so use plots and diagnostics rather than relying on this simple illustration.

The `d` value specifies the number of differences:
- `d=0`: no differencing.
- `d=1`: first difference.
- `d=2`: second difference.

Do not automatically choose `d=1` or `d=2`. If the original series is stationary, `d=0` may be suitable. Over-differencing can add noise and create artificial autocorrelation.

### 2.3 Moving Average (MA) — q

The MA component uses past forecast errors (innovations), not a simple moving average of observed values.

An MA(1) model:

$$
Y_t = \mu + \varepsilon_t + \theta_1\varepsilon_{t-1}
$$

An MA(2) model:

$$
Y_t = \mu + \varepsilon_t + \theta_1\varepsilon_{t-1}
+ \theta_2\varepsilon_{t-2}
$$

Where:
- $\varepsilon_t$ is the current innovation.
- $\varepsilon_{t-1}$ and $\varepsilon_{t-2}$ are previous innovations.
- $\theta_i$ are moving-average coefficients.

**Intuition:** the model can learn whether a recent unexpected demand spike or drop has a relationship with the current observation.

> The MA component in ARIMA is not the same as the Simple Moving Average forecasting method.

---

## 3. General ARIMA equation

Using the backshift operator:

$$
\phi(B)(1-B)^dY_t = c + \theta(B)\varepsilon_t
$$

Where:
- $B$ is the backshift operator: $BY_t=Y_{t-1}$.
- $(1-B)^d$ represents differencing.
- $\phi(B)$ is the AR polynomial.
- $\theta(B)$ is the MA polynomial.
- $\varepsilon_t$ is a white-noise innovation.

A common AR polynomial is:

$$
\phi(B)=1-\phi_1B-\phi_2B^2-\cdots-\phi_pB^p
$$

MA polynomial sign conventions differ across references and software.

### Example: ARIMA(1,1,1)

- `p=1`: one lag of the differenced series in the AR part.
- `d=1`: difference the series once.
- `q=1`: include one lagged innovation in the MA part.

The order is not a quality ranking. ARIMA(2,1,1) is not automatically better than ARIMA(1,1,1).

---

## 4. Stationarity

A stationary series has statistical properties that are broadly stable over time. In weak stationarity, the mean and variance are constant and covariance depends on the lag rather than the specific time.

- A series fluctuating around a stable average may be stationary.
- A persistent upward trend usually makes the level non-stationary.
- Changing seasonal patterns may require seasonal modeling.

Stationarity matters because the AR and MA relationships are modeled on a stationary series, or on a series made stationary through differencing.

### Augmented Dickey-Fuller (ADF) test

The ADF test is commonly used to investigate a unit root.

- **Null hypothesis:** the series has a unit root (non-stationary).
- **Alternative hypothesis:** the series is stationary.

A p-value below a chosen significance level (often 0.05) is evidence against the null. Interpret the result together with plots and domain knowledge. ADF results can be affected by sample size, structural breaks, and deterministic trend assumptions.

Python:

```python
from statsmodels.tsa.stattools import adfuller

result = adfuller(train.dropna(), autolag="AIC")
print("ADF statistic:", result[0])
print("p-value:", result[1])
```

---

## 5. ACF and PACF

### Autocorrelation Function (ACF)

ACF measures the correlation between a time series and its lagged values:
- Lag 1: correlation between $Y_t$ and $Y_{t-1}$.
- Lag 2: correlation between $Y_t$ and $Y_{t-2}$.

### Partial Autocorrelation Function (PACF)

PACF measures the relationship at a lag after accounting for the shorter lags in between.

### Common identification guidelines

| Pattern | Possible clue |
|---|---|
| PACF cuts off after lag p; ACF tails off | AR(p) |
| ACF cuts off after lag q; PACF tails off | MA(q) |
| Both ACF and PACF tail off | ARMA may be suitable |

These are guidelines, not absolute rules. Real series may not show clean cutoffs. Use ACF/PACF to narrow candidates, then compare them using time-based validation and residual diagnostics.

```python
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plot_acf(train.diff().dropna(), lags=24)
plot_pacf(train.diff().dropna(), lags=24, method="ywm")
```

---

## 6. ARIMA model-building workflow

1. Plot the historical series.
2. Check date frequency, missing values, and outliers.
3. Split chronologically into training, validation, and test periods.
4. Inspect trend and stationarity.
5. Use ADF and differencing plots to help choose `d`.
6. Inspect ACF/PACF of the stationary series to suggest `p` and `q`.
7. Fit a small set of plausible ARIMA candidates.
8. Compare validation forecasts and metrics.
9. Inspect residual diagnostics.
10. Refit the selected specification on development data.
11. Evaluate once on the untouched test set.
12. Refit on all observations for the production forecast.
13. Monitor errors and retrain when patterns change.

### Why chronological splitting?

A random split can place future observations in training and earlier observations in testing. This leaks future information into model development. Time-series evaluation must respect time order.

The notebook uses an illustrative 70% training, 15% validation, and 15% test split. Choose windows that reflect your forecast horizon and business cycle.

---

## 7. Python implementation with statsmodels

### Install libraries

```bash
pip install numpy pandas matplotlib statsmodels scikit-learn
```

### Fit a model

```python
from statsmodels.tsa.arima.model import ARIMA

# train should be a date-sorted pandas Series
model = ARIMA(train, order=(1, 1, 1), trend="t")
fitted_model = model.fit()

print(fitted_model.summary())
```

The `order` tuple is `(p, d, q)`. `trend` specifies deterministic trend terms. Its interpretation and valid choices depend on differencing. For some series `trend="n"` (no deterministic trend) may be more suitable than `trend="t"`. Compare reasonable specifications using validation forecasts.

### Generate forecasts

```python
forecast_result = fitted_model.get_forecast(steps=12)
forecast_mean = forecast_result.predicted_mean
prediction_interval = forecast_result.conf_int()

print(forecast_mean)
print(prediction_interval)
```

`get_forecast()` returns out-of-sample point forecasts and prediction intervals.

---

## 8. Selecting p, d, and q

### Choosing d

1. Plot the original series.
2. Run ADF as one diagnostic.
3. Difference once if the level appears non-stationary.
4. Re-check the differenced series.
5. Avoid additional differencing unless there is evidence it is needed.

### Choosing p and q

Use ACF and PACF as diagnostics, then test a limited set of candidates. Avoid a very large grid, particularly when fitting many item-location series.

### AIC and BIC

AIC and BIC balance model fit and model complexity. Lower values can indicate a preferable candidate among comparable models fitted to the same data and likelihood setup.

However:
- AIC/BIC are not direct forecast-accuracy metrics.
- Compare validation forecasts as well.
- Ensure candidate models are fitted on comparable observations.
- A lower AIC model can still have worse validation accuracy.

---

## 9. Forecast evaluation metrics

### Mean Absolute Error (MAE)

$$
MAE = \frac{1}{n}\sum_{t=1}^{n}|Y_t-\hat{Y}_t|
$$

MAE is measured in the same units as the target.

### Root Mean Squared Error (RMSE)

$$
RMSE = \sqrt{\frac{1}{n}\sum_{t=1}^{n}(Y_t-\hat{Y}_t)^2}
$$

RMSE penalizes larger errors more heavily than MAE.

### Weighted Mean Absolute Percentage Error (WMAPE)

$$
WMAPE = \frac{\sum_t|Y_t-\hat{Y}_t|}{\sum_t|Y_t|}\times100
$$

WMAPE is useful for aggregate evaluation when the denominator is non-zero. Across many products, high-volume series can dominate the result.

### Forecast bias

$$
Bias = \frac{1}{n}\sum_{t=1}^{n}(\hat{Y}_t-Y_t)
$$

- Positive bias: forecast is higher than actual on average.
- Negative bias: forecast is lower than actual on average.

Select metrics based on the business costs of over- and under-forecasting.

---

## 10. Residual diagnostics

Residuals are the differences between actual observations and fitted values:

$$
e_t = Y_t-\hat{Y}_t
$$

A reasonable model should leave residuals with:
- Mean close to zero.
- No obvious trend or repeating pattern.
- Little remaining autocorrelation.

### Ljung-Box test

The Ljung-Box test checks whether residual autocorrelations up to selected lags are jointly different from zero.

- Null hypothesis: no autocorrelation up to the selected lag.
- A small p-value suggests remaining autocorrelation.

```python
from statsmodels.stats.diagnostic import acorr_ljungbox

lb = acorr_ljungbox(
    fitted_model.resid.dropna(),
    lags=[10, 20],
    return_df=True
)
print(lb)
```

Residual diagnostics are necessary but not sufficient. Always evaluate forecasts on data not used for fitting.

---

## 11. Prediction intervals

ARIMA can produce point forecasts and prediction intervals:

```python
forecast_result = fitted_model.get_forecast(steps=12)
mean_forecast = forecast_result.predicted_mean
intervals = forecast_result.conf_int()
```

A 95% prediction interval is a range expected to contain a future observation with the stated coverage under the model assumptions. Intervals usually widen as the forecast horizon increases.

Intervals do not automatically capture all business uncertainty, including promotions, stockouts, supply constraints, structural breaks, and changes in customer behavior.

---

## 12. Advantages

1. **Interpretable structure:** AR, differencing, and MA components have distinct roles.
2. **Captures autocorrelation:** models temporal relationships beyond a simple level or trend.
3. **Useful univariate benchmark:** can be a strong baseline for suitable series.
4. **Prediction intervals:** provides uncertainty estimates under model assumptions.
5. **Mature ecosystem:** supported by established statistical libraries.
6. **Flexible:** represents a range of autoregressive and moving-average dynamics after differencing.

## 13. Limitations

1. **Stationarity:** the series generally needs to be stationary after differencing.
2. **Seasonality:** standard ARIMA does not explicitly include seasonal AR/MA terms.
3. **External drivers:** promotions, prices, events, and weather require an extension such as ARIMAX/SARIMAX.
4. **Order selection:** p, d, and q require diagnostics and validation.
5. **Outliers and structural changes:** unusual events or sudden shifts can affect estimation.
6. **Intermittent demand:** many-zero demand may suit Croston-family methods better.
7. **Scale and maintenance:** fitting one model per item-location can be demanding at scale.
8. **Long horizons:** forecasts may become less responsive and uncertainty increases.

---

## 14. ARIMA compared with other forecasting methods

| Model | Main pattern | Main idea |
|---|---|---|
| Naive | Last observation | Repeats latest value |
| SES | Level | Exponential smoothing of level |
| Holt | Level and trend | Smooths level and trend |
| Holt-Winters | Level, trend, seasonality | Adds seasonal component |
| ARIMA | Autocorrelation after differencing | Combines AR, I, and MA |
| SARIMA | Seasonal and non-seasonal autocorrelation | Adds seasonal AR, differencing, and MA |
| SARIMAX | Seasonal dynamics and external variables | Adds exogenous regressors |

---

## 15. Supply-chain and demand-forecasting applications

ARIMA can be considered for:
- Monthly demand for products with reasonably continuous sales.
- Stable item-location series with autocorrelation.
- Aggregate category or product-family demand.
- Shipment or inventory-related time series.
- Baseline comparisons against exponential smoothing and machine-learning models.

Be careful with:
- **Stockouts:** sales can be lower than unconstrained demand.
- **Promotions:** one-off spikes may not repeat.
- **New product introductions:** limited history may not support reliable estimation.
- **Intermittent demand:** many zeros may make standard ARIMA unsuitable.
- **Seasonality:** recurring patterns may require SARIMA/SARIMAX.
- **External information:** known promotions, holidays, prices, and weather may require exogenous regressors.

Choose the model at the level and horizon that match the planning decision. Always benchmark against simple baselines.

---

## 16. Common mistakes

- Fitting ARIMA without checking frequency and missing dates.
- Assuming every series needs first differencing.
- Applying too many differences.
- Treating ACF/PACF rules as absolute.
- Selecting only by training error or AIC.
- Randomly shuffling time-series observations.
- Repeatedly tuning on the test set.
- Ignoring residual autocorrelation.
- Ignoring prediction intervals and long-horizon uncertainty.
- Using standard ARIMA for strong seasonality without considering SARIMA.
- Expecting the model to infer future promotions or events that were not supplied.

---

## 17. Notebook contents

Python Implementation: [Notebook Link](https://github.com/ashay-thamankar/time_series_models/blob/main/ARIMA/ARIMA_Forecasting.ipynb) includes:

1. ARIMA introduction and mathematical intuition.
2. Synthetic monthly demand generation.
3. Chronological train, validation, and test split.
4. Stationarity checks using ADF.
5. First differencing.
6. ACF and PACF plots.
7. ARIMA fitting with statsmodels.
8. Validation forecasts and prediction intervals.
9. MAE, RMSE, WMAPE, and bias.
10. Candidate-order comparison.
11. Final test evaluation.
12. Residual plots and Ljung-Box diagnostics.
13. Future 12-month forecast.
14. Practical considerations.

## 18. References

- [statsmodels — ARIMA model](https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html)
- [Forecasting: Principles and Practice — ARIMA models](https://otexts.com/fpp3/arima.html)
- [statsmodels — ADF stationarity test](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.adfuller.html)
- [statsmodels — ARIMA examples](https://www.statsmodels.org/stable/examples/index.html)

---

## 19. Key takeaway

ARIMA combines three ideas:

- **AR:** use past observations.
- **I:** difference the series to address non-stationarity.
- **MA:** use past forecast errors.

Effective ARIMA forecasting is not just about finding an order that fits historical data. It requires chronological validation, residual diagnostics, out-of-sample evaluation, and understanding whether seasonality or external business drivers require an extended model.

**Next:** SARIMA and SARIMAX — extending ARIMA to seasonal patterns and external explanatory variables.
