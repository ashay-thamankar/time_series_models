# Auto-ARIMA for Time Series Forecasting

## Overview

Auto-ARIMA automates the search for suitable ARIMA model orders. Rather than manually trying combinations of autoregressive, differencing, and moving-average orders, it evaluates candidate models and selects one using a criterion such as AIC or BIC.

This project uses Python's `pmdarima` library and a synthetic monthly demand dataset. The notebook demonstrates non-seasonal and seasonal Auto-ARIMA, chronological evaluation, residual diagnostics, and future forecasting.

> Auto-ARIMA automates order selection; it does not guarantee the best possible forecast. Validation and diagnostics remain essential.

## 1. ARIMA recap

ARIMA is represented as:

$$
ARIMA(p,d,q)
$$

- **AR (p):** uses past observations.
- **I (d):** applies differencing to address non-stationarity.
- **MA (q):** uses past forecast errors (innovations), not a simple moving average of observations.

An AR(1) example:

$$
y_t=c+\\phi_1y_{t-1}+\\epsilon_t
$$

First-order differencing:

$$
\\Delta y_t=y_t-y_{t-1}
$$

An MA(1) example:

$$
y_t=\\mu+\\epsilon_t+\\theta_1\\epsilon_{t-1}
$$

## 2. Seasonal ARIMA

For seasonal data:

$$
SARIMA(p,d,q)\\times(P,D,Q,m)
$$

| Parameter | Meaning |
|---|---|
| p, d, q | Non-seasonal AR, differencing, MA orders |
| P, D, Q | Seasonal AR, differencing, MA orders |
| m | Number of observations in one seasonal cycle |

Typical seasonal periods:
- Monthly data with annual seasonality: `m=12`.
- Quarterly data with annual seasonality: `m=4`.
- Daily data with weekly seasonality: `m=7`.
- Hourly data with daily seasonality: `m=24`.

The seasonal period must reflect the actual data cycle. It is not the forecast horizon.

## 3. What Auto-ARIMA does

Auto-ARIMA automates the search for suitable values of the ARIMA orders. A simplified workflow is:

1. Receive the time series and search configuration.
2. Assess differencing requirements using stationarity tests and, for seasonal models, seasonal tests.
3. Generate candidate model orders within configured bounds.
4. Fit candidate models.
5. Compare them using the selected information criterion.
6. Return the selected fitted model.

With `stepwise=True`, `pmdarima` explores a sequence of nearby candidate configurations rather than fitting every possible combination. This is often faster, but is not an exhaustive global search. With `stepwise=False`, a broader grid search can be performed at potentially high computational cost.

## 4. Information criteria

AIC is:

$$
AIC=2k-2\\log(L)
$$

where `k` is the number of estimated parameters and `L` is the maximized likelihood.

BIC is:

$$
BIC=k\\log(n)-2\\log(L)
$$

where `n` is the number of observations.

Lower AIC or BIC is preferred among comparable candidate models under the same fitting framework. These are model-selection criteria, not direct measures of future forecast accuracy. A model with the lowest AIC may not have the lowest validation WMAPE or MAE.

## 5. Dataset

The notebook generates 120 synthetic monthly demand observations from January 2015 onward, with an upward trend, annual seasonality, and random noise. It is educational data, not a real business dataset.

## 6. Modelling workflow

### Step 1: Explore the series
Inspect trend, seasonality, outliers, missing values, variance changes, and structural breaks. A monthly seasonal profile can help reveal annual patterns.

### Step 2: Split chronologically
The notebook uses 70% training, 15% validation, and 15% test. Do not randomly shuffle time-series observations. Rolling-origin or expanding-window evaluation is often more robust than one holdout.

### Step 3: Fit non-seasonal Auto-ARIMA

```python
from pmdarima import auto_arima

model = auto_arima(
    y_train,
    seasonal=False,
    start_p=0, start_q=0,
    max_p=4, max_q=4,
    d=None, max_d=2,
    information_criterion="aic",
    stepwise=True,
    trace=True,
    error_action="ignore",
    suppress_warnings=True,
    with_intercept="auto"
)
```

`seasonal=False` restricts the search to non-seasonal ARIMA models.

### Step 4: Fit seasonal Auto-ARIMA

```python
seasonal_model = auto_arima(
    y_train,
    seasonal=True,
    m=12,
    start_p=0, start_q=0,
    max_p=3, max_q=3,
    start_P=0, start_Q=0,
    max_P=2, max_Q=2,
    d=None, D=None,
    max_d=2, max_D=1,
    information_criterion="aic",
    stepwise=True,
    trace=True,
    error_action="ignore",
    suppress_warnings=True,
    with_intercept="auto"
)
```

The order bounds keep the example computationally manageable. Selected orders depend on the training data and search configuration.

### Step 5: Inspect selected orders

```python
print(seasonal_model.order)
print(seasonal_model.seasonal_order)
print(seasonal_model.summary())
```

Record the data period, seasonal period, search bounds, criterion, library version, and validation results to make experiments reproducible.

### Step 6: Forecast validation data

```python
validation_forecast = seasonal_model.predict(
    n_periods=len(validation)
)
```

Compare the forecast with actual validation observations.

### Step 7: Evaluate accuracy

**MAE**

$$
MAE=\\frac{1}{n}\\sum_{t=1}^{n}|y_t-\\hat{y}_t|
$$

**RMSE**

$$
RMSE=\\sqrt{\\frac{1}{n}\\sum_{t=1}^{n}(y_t-\\hat{y}_t)^2}
$$

**WMAPE**

$$
WMAPE=\\frac{\\sum_{t=1}^{n}|y_t-\\hat{y}_t|}{\\sum_{t=1}^{n}|y_t|}
$$

The notebook reports WMAPE as a ratio; multiply by 100 for percentage form. Percentage metrics require care when actual values are zero or close to zero.

**Mean bias**

$$
Bias=\\frac{1}{n}\\sum_{t=1}^{n}(\\hat{y}_t-y_t)
$$

Positive bias indicates average over-forecasting; negative bias indicates average under-forecasting.

### Step 8: Residual diagnostics

Inspect residuals over time, their distribution, residual ACF, and residuals versus fitted values. The Ljung-Box test checks whether a group of residual autocorrelations is jointly different from zero. A small p-value suggests remaining autocorrelation. It is one diagnostic, not a complete quality score.

### Step 9: Refit and test

After selecting the approach using validation:
1. Combine training and validation data.
2. Refit the final model.
3. Forecast the untouched test horizon.
4. Evaluate the test forecast once.

The notebook reruns seasonal Auto-ARIMA on the combined history.

### Step 10: Forecast future periods

```python
forecast, confidence_interval = final_model.predict(
    n_periods=12,
    return_conf_int=True,
    alpha=0.05
)
```

Prediction intervals quantify uncertainty under the fitted model assumptions. They do not include every possible uncertainty source, such as structural breaks or unexpected business events.

## 7. Important `auto_arima()` parameters

| Parameter | Purpose |
|---|---|
| `seasonal` | Enables/disables seasonal search |
| `m` | Seasonal cycle length |
| `start_p`, `start_q` | Starting non-seasonal AR and MA orders |
| `max_p`, `max_q` | Maximum non-seasonal AR and MA orders |
| `start_P`, `start_Q` | Starting seasonal AR and MA orders |
| `max_P`, `max_Q` | Maximum seasonal AR and MA orders |
| `d`, `D` | Non-seasonal and seasonal differencing; `None` allows automatic selection |
| `max_d`, `max_D` | Maximum differencing orders |
| `information_criterion` | AIC, BIC, or another supported criterion |
| `stepwise` | Uses stepwise search when enabled |
| `trace` | Prints search progress |
| `error_action` | Controls candidate-fit error handling |
| `with_intercept` | Controls intercept/drift inclusion, depending on configuration |

Supported options and defaults can vary by `pmdarima` version. Consult the installed version's documentation.

## 8. Auto-ARIMA vs manual ARIMA

| Aspect | Manual ARIMA | Auto-ARIMA |
|---|---|---|
| Order selection | Analyst chooses orders | Procedure searches candidate orders |
| Effort | More experimentation | Less manual search |
| Control | Direct order choice | Search bounds, tests, criteria |
| Computation | Depends on candidates | Can be substantial for seasonal search |
| Guaranteed best forecast? | No | No |
| Validation required? | Yes | Yes |

Auto-ARIMA automates order selection, not the complete forecasting problem. Data preparation, leakage prevention, feature decisions, validation, diagnostics, and business interpretation remain necessary.

## 9. Supply-chain applications

- Seasonal retail demand forecasting.
- Statistical baselines for product-category or item-location demand.
- Automated initial order selection across multiple time series.
- Benchmarking against machine-learning forecasting methods.

For highly intermittent, zero-heavy demand, standard Auto-ARIMA may not be suitable; Croston-family methods may be better candidates. For thousands of series, consider runtime, minimum history, search failures, refresh frequency, and fallback models.

## 10. Limitations and common mistakes

1. Assuming automatic means optimal.
2. Choosing the wrong seasonal period.
3. Using too little history to estimate seasonal behaviour.
4. Ignoring residual diagnostics.
5. Searching overly broad order ranges and incurring high computation.
6. Allowing future observations to influence fitting or order selection.
7. Relying on one validation window; rolling-origin evaluation is often stronger.
8. Expecting Auto-ARIMA to anticipate structural breaks, product launches, or supply disruptions.
9. Assuming univariate Auto-ARIMA automatically discovers external drivers such as promotions or price.

## Key takeaway

**Auto-ARIMA automates the search for ARIMA/SARIMA model orders using a configured search strategy and selection criterion.** It reduces manual order-selection effort, but forecasts must still be validated on unseen data and checked through residual diagnostics.

Python Implementation: [Notebook Link](https://github.com/ashay-thamankar/time_series_models/blob/main/Auto_ARIMA/Auto_ARIMA_Forecasting.ipynb).

## References

- [pmdarima documentation](https://alkaline-ml.com/pmdarima/)
- [auto_arima API](https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.auto_arima.html)
- [Forecasting: Principles and Practice — ARIMA](https://otexts.com/fpp3/arima.html)
