# Day 12: SARIMA and SARIMAX for Time Series Forecasting

## Overview

This project explores two seasonal statistical forecasting models:

- **SARIMA (Seasonal AutoRegressive Integrated Moving Average):** extends ARIMA by modelling seasonal autocorrelation and seasonal differencing.
- **SARIMAX (Seasonal AutoRegressive Integrated Moving Average with eXogenous variables):** extends the seasonal ARIMA structure with external explanatory variables such as promotions, price, holidays, and weather.

The practical notebook uses a synthetic monthly demand dataset to demonstrate data exploration, chronological splitting, stationarity checks, model fitting, validation, evaluation, residual diagnostics, and future forecasting.

> The dataset is synthetic and intended for education. Model orders and external-variable assumptions should be selected based on the characteristics and information availability of a real forecasting problem.

---

## 1. Why do we need SARIMA and SARIMAX?

ARIMA is useful for non-seasonal time series. However, many business time series have recurring patterns:

- Retail sales may rise during holiday periods.
- Air-conditioning demand may increase during summer.
- Grocery demand may show weekly or annual cycles.
- Spare-parts demand may show recurring maintenance or seasonal effects.

A standard ARIMA model does not explicitly represent a seasonal cycle. SARIMA adds seasonal autoregressive, differencing, and moving-average terms.

SARIMAX adds another capability: it can use external variables that help explain the target series. For example, demand may depend on promotion activity, price, holidays, or weather in addition to its own history.

### Conceptual distinction

| Model | What it captures |
|---|---|
| ARIMA | Non-seasonal autoregressive, differencing, and moving-average structure |
| SARIMA | ARIMA structure plus seasonal AR, differencing, and MA terms |
| SARIMAX | Seasonal ARIMA structure plus exogenous regressors |

In `statsmodels`, the `SARIMAX` class implements the general seasonal state-space model. If no exogenous variables are supplied, it can be used to fit a SARIMA-type model. When exogenous variables are supplied, it models regression effects together with seasonal ARIMA errors.

---

## 2. SARIMA model

### 2.1 Notation

SARIMA is represented as:

$$
SARIMA(p,d,q) \\times (P,D,Q,s)
$$

The first group represents the non-seasonal terms:

| Parameter | Name | Meaning |
|---|---|---|
| $p$ | AR order | Number of non-seasonal autoregressive lags |
| $d$ | Differencing order | Number of non-seasonal differences |
| $q$ | MA order | Number of non-seasonal moving-average error lags |

The second group represents the seasonal terms:

| Parameter | Name | Meaning |
|---|---|---|
| $P$ | Seasonal AR order | Number of seasonal autoregressive lags |
| $D$ | Seasonal differencing order | Number of seasonal differences |
| $Q$ | Seasonal MA order | Number of seasonal moving-average error lags |
| $s$ | Seasonal period | Number of observations in one seasonal cycle |

Examples of seasonal periods:

| Data frequency | Common seasonal period |
|---|---:|
| Monthly with annual seasonality | 12 |
| Quarterly with annual seasonality | 4 |
| Weekly with yearly seasonality | Approximately 52 |
| Daily with weekly seasonality | 7 |

The correct period depends on the calendar, business definition, and data frequency. A weekly series may have holiday/calendar complications, and daily demand may contain more than one seasonal cycle.

### 2.2 Autoregressive (AR) component

The autoregressive component uses previous observations to explain the current value.

A simple AR(1) process is:

$$
y_t = c + \\phi_1 y_{t-1} + \\epsilon_t
$$

where:

- $y_t$ is the current observation.
- $c$ is a constant.
- $\\phi_1$ is the AR coefficient.
- $\\epsilon_t$ is the current random error.

An AR model assumes that past values contain information about the current value.

### 2.3 Integrated (I) component

The integrated component applies differencing to make a non-stationary series more stationary.

First-order differencing:

$$
\\Delta y_t = y_t-y_{t-1}
$$

Seasonal differencing:

$$
\\Delta_s y_t = y_t-y_{t-s}
$$

For monthly data with annual seasonality:

$$
\\Delta_{12}y_t = y_t-y_{t-12}
$$

If both first-order and seasonal differencing are applied, the transformed series is:

$$
(1-B)(1-B^{12})y_t
$$

where $B$ is the backshift operator, defined by $By_t=y_{t-1}$.

Differencing should be sufficient to remove non-stationary behaviour, but excessive differencing can introduce unnecessary noise and distort the series.

### 2.4 Moving-average (MA) component

The moving-average component uses past forecast errors (innovations), not a simple rolling average of observed values.

An MA(1) process is:

$$
y_t = \\mu + \\epsilon_t + \\theta_1\\epsilon_{t-1}
$$

where:

- $\\epsilon_t$ is the current innovation.
- $\\epsilon_{t-1}$ is the previous innovation.
- $\\theta_1$ is the MA coefficient.

### 2.5 Seasonal AR and MA components

Seasonal AR terms use observations from previous seasonal cycles. For monthly data, a seasonal AR(1) term can relate the current value to the value 12 months earlier.

Seasonal MA terms use errors from previous seasonal cycles. Seasonal differencing removes a recurring seasonal level or pattern when appropriate.

A compact operator representation is:

$$
\\Phi_P(B^s)\\phi_p(B)(1-B)^d(1-B^s)^D y_t
=
\\Theta_Q(B^s)\\theta_q(B)\\epsilon_t
$$

Here, $B$ is the backshift operator; the AR polynomials are represented by $\\phi$ and $\\Phi$, while the MA polynomials are represented by $\\theta$ and $\\Theta$.

---

## 3. SARIMAX model

SARIMAX adds external regressors to the seasonal ARIMA framework.

A conceptual regression-with-seasonal-ARIMA-errors representation is:

$$
y_t = \\beta_1x_{1,t}+\\beta_2x_{2,t}+\\cdots+\\beta_kx_{k,t}+u_t
$$

where:

- $y_t$ is the target variable.
- $x_{i,t}$ are exogenous variables.
- $\\beta_i$ are estimated regression coefficients.
- $u_t$ follows a seasonal ARIMA process.

For example, demand can be influenced by:

| Exogenous variable | Possible relationship with demand |
|---|---|
| Promotion | Promotional periods may increase sales |
| Price | Price increases may reduce demand |
| Holiday indicator | Certain holidays may increase or shift demand |
| Temperature | Weather may influence seasonal product demand |
| Marketing spend | Campaigns may influence customer purchases |
| Store opening days | Availability of selling days may affect total sales |

The coefficient signs and magnitudes are learned from the data. They should be interpreted with care: association does not automatically imply causation, and correlated regressors can make individual coefficients unstable.

### 3.1 A critical requirement: future exogenous values

SARIMAX requires exogenous values for every forecasted period.

If a model is trained with `promotion`, `price_index`, and `temperature`, then forecasting the next 12 months requires 12 future rows containing all three variables.

Potential sources include:

- Approved future promotion calendars.
- Planned pricing or price scenarios.
- Calendar features known in advance.
- Weather forecasts or weather scenarios.
- Forecasts of external drivers.

Do not use future realised values during backtesting if they would not have been known at the forecast origin. That creates data leakage and can make the model appear much more accurate than it would be in production.

The prediction interval from SARIMAX generally does not automatically include all uncertainty from separately forecast future regressors. Scenario analysis or simulation may be needed when regressor uncertainty is material.

---

## 4. Dataset used in the notebook

The notebook generates 120 monthly observations from January 2015 onward. The synthetic demand series contains:

1. A gradual upward trend.
2. A recurring annual seasonal pattern.
3. Random noise.
4. Promotion activity.
5. A price index.
6. A temperature-like variable.

The target is `demand`. The SARIMAX example uses:

```python
exog_cols = ["promotion", "price_index", "temperature"]
```

The data is synthetic so that the complete workflow can be run without downloading external datasets.

---

## 5. Modelling workflow

### Step 1: Explore the series

Plot the target over time. Check:

- Trend.
- Seasonal behaviour.
- Outliers.
- Missing observations.
- Sudden level shifts.
- Changes in variance.

For seasonal data, inspect averages by month or season and compare observations one seasonal cycle apart.

### Step 2: Split chronologically

The notebook uses:

- 70% training.
- 15% validation.
- 15% test.

The training set estimates model parameters. The validation set supports model comparison and order selection. The test set is kept untouched until the final evaluation.

Never randomly shuffle time-series observations. A rolling-origin evaluation is often preferable when multiple forecast origins are available.

### Step 3: Assess stationarity and differencing

The notebook uses the Augmented Dickey-Fuller (ADF) test as one diagnostic.

- Null hypothesis: the series has a unit root.
- A small p-value provides evidence against the unit-root null.

Use the test together with plots and domain understanding. ADF does not identify the correct seasonal order by itself.

Compare the original series, first differences, and seasonal differences. For monthly annual seasonality, seasonal differencing uses lag 12.

### Step 4: Inspect ACF and PACF

- ACF: autocorrelation at different lags.
- PACF: partial autocorrelation after accounting for shorter lags.

These plots can help suggest AR and MA orders. Seasonal spikes at 12, 24, and 36 may provide evidence of annual dependence in monthly data.

ACF and PACF are diagnostic aids, not automatic order-selection algorithms. Interpret them on an appropriately transformed series.

### Step 5: Fit SARIMA

The notebook starts with:

```python
order = (1, 1, 1)
seasonal_order = (1, 1, 1, 12)
```

This is:

$$
SARIMA(1,1,1)\\times(1,1,1,12)
$$

The model is fitted using:

```python
from statsmodels.tsa.statespace.sarimax import SARIMAX

model = SARIMAX(
    y_train,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 12),
    trend="n",
    enforce_stationarity=False,
    enforce_invertibility=False
)

result = model.fit(disp=False)
```

The `SARIMAX` class is used here without an `exog` argument, so this is a seasonal ARIMA model without external regressors.

The settings `enforce_stationarity=False` and `enforce_invertibility=False` allow the optimizer to estimate a wider range of parameters. They do not guarantee that the resulting model is statistically appropriate; inspect the fitted model and diagnostics.

### Step 6: Forecast the validation period

```python
forecast_obj = result.get_forecast(steps=len(validation))
forecast = forecast_obj.predicted_mean
confidence_interval = forecast_obj.conf_int()
```

`get_forecast()` returns forecasts beyond the end of the fitted sample. `conf_int()` returns prediction intervals.

### Step 7: Fit SARIMAX with external regressors

```python
exog_cols = ["promotion", "price_index", "temperature"]

model = SARIMAX(
    y_train,
    exog=train[exog_cols],
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 12),
    trend="n",
    enforce_stationarity=False,
    enforce_invertibility=False
)

result = model.fit(disp=False)
```

For validation:

```python
forecast_obj = result.get_forecast(
    steps=len(validation),
    exog=validation[exog_cols]
)
```

The future exogenous DataFrame must have the correct number of rows, matching columns, and correct time alignment.

### Step 8: Evaluate models

The notebook calculates four metrics.

**Mean Absolute Error (MAE)**

$$
MAE=\\frac{1}{n}\\sum_{t=1}^{n}|y_t-\\hat{y}_t|
$$

MAE is easy to interpret because it is in the same unit as the target.

**Root Mean Squared Error (RMSE)**

$$
RMSE=\\sqrt{\\frac{1}{n}\\sum_{t=1}^{n}(y_t-\\hat{y}_t)^2}
$$

RMSE penalizes large errors more heavily than MAE.

**Weighted Mean Absolute Percentage Error (WMAPE)**

$$
WMAPE=\\frac{\\sum_{t=1}^{n}|y_t-\\hat{y}_t|}{\\sum_{t=1}^{n}|y_t|}
$$

The notebook reports WMAPE as a ratio. Multiply by 100 to display it as a percentage.

**Mean Bias**

$$
Bias=\\frac{1}{n}\\sum_{t=1}^{n}(\\hat{y}_t-y_t)
$$

- Positive bias: average over-forecasting.
- Negative bias: average under-forecasting.
- Bias near zero: little average directional error, but individual errors may still be large.

Choose metrics based on the business use case. Percentage metrics need care when actual values are zero or close to zero.

### Step 9: Compare candidate orders

The notebook includes a small example comparing a few seasonal orders. AIC can be used as an in-sample information criterion, while validation accuracy measures out-of-sample performance.

Do not select a model solely by the lowest AIC. Consider:

- Validation accuracy.
- Forecast horizon.
- Residual diagnostics.
- Model stability.
- Complexity and maintainability.
- Business interpretability.

### Step 10: Residual diagnostics

Residuals are the differences between observations and fitted values.

Inspect:

- Residuals over time.
- Residual histogram.
- Residual ACF.
- Residuals versus fitted values.
- Ljung-Box test.

The Ljung-Box test evaluates whether a set of residual autocorrelations is jointly different from zero. A small p-value suggests that some autocorrelation remains. It is one diagnostic among several, not a complete model-quality score.

### Step 11: Refit and test

After choosing a model using validation data:

1. Combine training and validation data.
2. Refit the selected model.
3. Forecast the untouched test horizon.
4. Evaluate the test forecast once.

The notebook demonstrates this process for both illustrative specifications. In a real project, use the validation comparison to select the model that will be evaluated on the test set.

### Step 12: Forecast future periods

SARIMA can forecast future periods using the fitted target history alone.

SARIMAX requires a future exogenous-variable scenario:

```python
future_forecast = fitted_model.get_forecast(
    steps=12,
    exog=future_exog
)
```

The notebook uses illustrative assumptions for future promotions, price index, and temperature. Replace these assumptions with actual planning inputs or explicit scenarios in a real project.

---

## 6. SARIMA vs SARIMAX

| Feature | SARIMA | SARIMAX |
|---|---|---|
| Non-seasonal AR terms | Yes | Yes |
| Non-seasonal differencing | Yes | Yes |
| Non-seasonal MA terms | Yes | Yes |
| Seasonal AR terms | Yes | Yes |
| Seasonal differencing | Yes | Yes |
| Seasonal MA terms | Yes | Yes |
| External regressors | No | Yes |
| Needs future exogenous data | No | Yes |
| Useful when | Seasonality and autocorrelation explain much of the series | External drivers add useful information |

### When to use SARIMA

- The series has a stable seasonal pattern.
- The target's own history provides most of the useful forecasting information.
- There are no reliable external drivers.
- A seasonal statistical baseline is required.

### When to use SARIMAX

- Promotions influence demand.
- Price changes affect sales.
- Holiday or calendar effects are important.
- Weather or other external drivers are meaningful.
- Future regressor values are available or can be forecast.

SARIMAX is not automatically better. External variables can add noise, create multicollinearity, or cause leakage if they are not handled correctly.

---

## 7. Applications in supply-chain forecasting

### Retail demand

- Seasonal demand patterns.
- Promotions and discount effects.
- Price changes.
- Holiday calendars.
- Weather-sensitive categories.

### Inventory planning

- Seasonal replenishment requirements.
- Demand forecasts for products with recurring patterns.
- Scenario forecasts based on planned promotions or price changes.

### Spare-parts forecasting

- Seasonal maintenance patterns.
- Planned maintenance schedules.
- Calendar-driven demand.

For highly intermittent, zero-heavy item-level demand, standard SARIMA may not be the best first choice. Croston-family methods, aggregation, or models designed for intermittent demand may be more suitable.

---

## 8. Limitations and practical cautions

1. **Seasonal history requirement:** Seasonal models need enough observations to estimate recurring patterns. Very short histories may not contain enough complete seasonal cycles.
2. **Single seasonal period:** Classical SARIMA is generally designed around one seasonal period. Data with multiple seasonalities may need a different modelling approach.
3. **Structural changes:** Product launches, assortment changes, supply disruptions, and changing customer behaviour can make historical patterns less relevant.
4. **External-variable availability:** SARIMAX requires future regressor values. Unavailable or inaccurate future inputs can reduce forecast quality.
5. **Regressor uncertainty:** Standard prediction intervals may not capture uncertainty from separately forecast exogenous variables.
6. **Intermittent demand:** A large number of zeros and irregular non-zero demand can make standard seasonal ARIMA models unsuitable.
7. **Model complexity:** Large seasonal orders and many regressors can make estimation slow or unstable.
8. **Calendar effects:** Month lengths, moving holidays, trading days, and leap years may require explicit calendar features.

---

## 9. Common mistakes

- Choosing a seasonal period without checking the data frequency.
- Randomly splitting time-series data.
- Applying too much differencing.
- Selecting orders only from ACF/PACF or AIC.
- Ignoring residual autocorrelation.
- Misaligning exogenous variables with the target dates.
- Using future realised regressors during historical backtesting.
- Assuming that adding more regressors will always improve accuracy.
- Treating a prediction interval as if it includes all uncertainty in future external drivers.

---

## 10. Notebook structure

Python Implementation: [Notebook Link](https://github.com/ashay-thamankar/time_series_models/blob/main/SARIMA_SARIMAX/SARIMA_and_SARIMAX_Forecasting.ipynb) includes:

1. Library imports and synthetic data generation.
2. Time-series visualization and seasonal profile.
3. Chronological train-validation-test split.
4. ADF test and differencing.
5. ACF and PACF analysis.
6. SARIMA model fitting and validation.
7. SARIMAX model fitting with exogenous variables.
8. Validation metric comparison.
9. Candidate seasonal order comparison.
10. Residual analysis and Ljung-Box test.
11. Final test evaluation.
12. Twelve-month future forecasts and exogenous scenario construction.
13. Supply-chain considerations.

## Key takeaway

**SARIMA models recurring seasonal patterns using seasonal AR, differencing, and MA components. SARIMAX adds external explanatory variables to that seasonal time-series structure.**

For demand forecasting, SARIMA is a useful seasonal baseline. SARIMAX can provide additional value when drivers such as promotions, prices, holidays, and weather contain reliable information and are available for the full forecast horizon.

## References

- [Statsmodels SARIMAX documentation](https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html)
- [Statsmodels SARIMAX example](https://www.statsmodels.org/stable/examples/notebooks/generated/statespace_sarimax_stata.html)
- [Forecasting: Principles and Practice — Seasonal ARIMA](https://otexts.com/fpp3/seasonal-arima.html)
