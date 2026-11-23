# Holt's Linear Trend Method

## Forecasting time series with level and trend

Holt's Linear Trend method, also known as **Holt's double exponential smoothing**, is an extension of Simple Exponential Smoothing (SES). While SES estimates only the current level of a time series, Holt's method estimates both:

1. **Level** – the estimated value of the series at the current time.
2. **Trend** – the estimated rate and direction of change.

This makes Holt's method useful when a time series has a reasonably consistent upward or downward trend but does not have a strong seasonal pattern.

This folder contains a detailed, executable Jupyter Notebook with a from-scratch implementation, parameter experiments, train-validation-test evaluation, a comparison with SES, and a Statsmodels implementation.

---

## 1. What is Holt's Linear Trend method?

Consider a product whose monthly demand has been increasing gradually. A model that only estimates the average demand level may fail to reflect this growth. Holt's method addresses this by maintaining a separate estimate of the current level and the current trend.

At every time step, the model:
- Updates the level using the latest observation and the previous level-plus-trend projection.
- Updates the trend using the change in the estimated level.
- Uses the latest level and trend to forecast future periods.

The method is called *double exponential smoothing* because it smooths two components: level and trend. It is not the same as applying SES twice in an arbitrary sequence.

### When is it appropriate?

Holt's method is a reasonable candidate when:
- The series has a visible trend.
- The trend changes gradually rather than through frequent abrupt jumps.
- There is no strong seasonal pattern that needs to be modelled.
- Demand is reasonably regular and has enough observations to estimate a trend.

It is not designed specifically for intermittent demand or strong seasonality.

---

## 2. Components of Holt's method

### 2.1 Level

The level represents the estimated current value of the time series after accounting for the trend.

### 2.2 Trend

The trend represents the estimated change per time period. It can be positive (upward trend), negative (downward trend), or close to zero (little trend).

The trend is expressed in the same units as the series per time period. For monthly data, it represents estimated change per month.

---

## 3. Mathematical equations

Holt's additive linear trend method uses two smoothing equations and one forecast equation.

### 3.1 Level equation

$$L_t = \alpha Y_t + (1-\alpha)(L_{t-1}+T_{t-1})$$

The new level is a weighted combination of:
- The latest actual observation, $Y_t$.
- The previous period's projected level, $L_{t-1}+T_{t-1}$.

### 3.2 Trend equation

$$T_t = \beta(L_t-L_{t-1}) + (1-\beta)T_{t-1}$$

The new trend is a weighted combination of:
- The latest change in the estimated level, $L_t-L_{t-1}$.
- The previous trend estimate, $T_{t-1}$.

### 3.3 Forecast equation

$$\hat{Y}_{t+h|t}=L_t+hT_t$$

The forecast is the latest estimated level plus the trend multiplied by the forecast horizon.

For example:
- One period ahead: $L_t+T_t$
- Two periods ahead: $L_t+2T_t$
- Three periods ahead: $L_t+3T_t$

### 3.4 Meaning of the variables

| Symbol | Meaning |
|---|---|
| $Y_t$ | Actual observation at time $t$ |
| $L_t$ | Estimated level at time $t$ |
| $T_t$ | Estimated trend at time $t$ |
| $L_{t-1}$ | Previous estimated level |
| $T_{t-1}$ | Previous estimated trend |
| $\alpha$ | Level smoothing parameter |
| $\beta$ | Trend smoothing parameter |
| $h$ | Number of periods ahead |
| $\hat{Y}_{t+h|t}$ | Forecast made at time $t$ for time $t+h$ |

Both $\alpha$ and $\beta$ are generally constrained between 0 and 1 in the standard formulation.

---

## 4. Understanding alpha and beta

Holt's method has two smoothing parameters because it updates two separate components.

### Alpha (α): level smoothing

Alpha controls how quickly the estimated level responds to new observations.

- **Low alpha:** More smoothing. The level changes gradually and reacts less to individual observations.
- **High alpha:** Less smoothing. The level responds more strongly to recent observations.

### Beta (β): trend smoothing

Beta controls how quickly the estimated trend responds to changes in the level.

- **Low beta:** The trend estimate changes gradually.
- **High beta:** The trend estimate reacts more quickly to recent changes in the level.

The best values depend on the time series. A high alpha or beta is not automatically better. Select parameters using a time-ordered validation procedure and a metric that reflects the forecasting objective.

---

## 5. Worked numerical example

Assume the following starting values:

- Initial level: $L_0=100$
- Initial trend: $T_0=5$
- Actual observations: 108, 111, 115
- Level smoothing: $\alpha=0.4$
- Trend smoothing: $\beta=0.3$

### Step 1: Update level for the first observation

$$L_1=0.4(108)+0.6(100+5)$$

$$L_1=43.2+63=106.2$$

### Step 2: Update trend

$$T_1=0.3(106.2-100)+0.7(5)$$

$$T_1=1.86+3.5=5.36$$

### Step 3: Forecast the next period

$$\hat{Y}_{2|1}=L_1+T_1$$

$$\hat{Y}_{2|1}=106.2+5.36=111.56$$

The forecast for the next period is **111.56 units**.

When the next actual observation becomes available, the level and trend are updated again. The same process continues through the series.

> The initial level and trend are starting-state choices. In practical implementations, they may be estimated from the data or optimized together with the smoothing parameters.

---

## 6. How the method works step by step

1. Choose or estimate the initial level and trend.
2. Use the current state to create a one-step-ahead forecast.
3. Observe the actual value for the current period.
4. Update the level using alpha.
5. Update the trend using beta.
6. Repeat for every observation.
7. Use the final level and trend to forecast future periods.

The order matters during evaluation: the one-step-ahead fitted forecast for a period must be generated using information available before that period's actual value is observed. Updating the state first and then calling it a forecast for the same observation would introduce data leakage.

---

## 7. Multi-step forecasting

Once the final level and trend have been estimated, the basic Holt forecast is:

$$\hat{Y}_{T+h|T}=L_T+hT_T$$

Suppose the final level is 150 and the estimated trend is 2.5 units per month.

| Horizon | Forecast |
|---|---:|
| 1 month ahead | 152.5 |
| 2 months ahead | 155.0 |
| 3 months ahead | 157.5 |
| 4 months ahead | 160.0 |

The forecast extends the latest trend linearly.

### Important limitation

The basic, undamped method assumes the estimated trend continues at the same rate for every future period. This can produce unrealistic forecasts at long horizons, especially when growth cannot continue indefinitely.

A **damped trend** variant reduces the trend's contribution over longer horizons. It can be useful when the trend is expected to flatten gradually.

---

## 8. Holt's method vs Simple Exponential Smoothing

| Feature | SES | Holt's Linear Trend |
|---|---|---|
| Components | Level | Level + trend |
| Smoothing parameters | Alpha | Alpha and beta |
| Trend modelling | No explicit trend | Yes |
| Multi-step forecast | Flat | Linear extension of trend |
| Suitable data | No clear trend or seasonality | Trend without strong seasonality |

SES is appropriate when the series fluctuates around a relatively stable level. Holt's method is a natural next step when a trend is present.

## 9. Holt's method vs Holt-Winters

| Method | Components | Suitable pattern |
|---|---|---|
| SES | Level | No clear trend or seasonality |
| Holt's Linear Trend | Level + trend | Trend, no strong seasonality |
| Holt-Winters | Level + trend + seasonality | Trend and seasonality |

Holt-Winters extends the model by adding a seasonal component. It supports additive or multiplicative seasonality depending on the data.

---

## 10. Python implementation

The accompanying notebook demonstrates two implementations.

### 10.1 From scratch

The notebook implements the level and trend updates directly:

```python
def holt_manual(y, alpha=0.4, beta=0.3,
                initial_level=None, initial_trend=None):

    y = np.asarray(y, dtype=float)

    level = float(y[0] if initial_level is None else initial_level)
    trend = float(y[1] - y[0] if initial_trend is None else initial_trend)

    fitted = np.full(len(y), np.nan)

    for t in range(1, len(y)):
        # Forecast before observing the current actual
        fitted[t] = level + trend

        previous_level = level

        # Update level
        level = alpha * y[t] + (1 - alpha) * (level + trend)

        # Update trend
        trend = beta * (level - previous_level) + (1 - beta) * trend

    return fitted, level, trend
```

This simplified code is intended to make the update sequence clear. The notebook contains a fuller version that also records the level and trend at every time step.

### 10.2 Using Statsmodels

Statsmodels provides Holt's method through `statsmodels.tsa.holtwinters.Holt`.

```python
from statsmodels.tsa.holtwinters import Holt

model = Holt(
    train,
    initialization_method="estimated",
    damped_trend=False
)

fit = model.fit(optimized=True)

forecast = fit.forecast(steps=12)
```

For a damped trend:

```python
damped_model = Holt(
    train,
    damped_trend=True,
    initialization_method="estimated"
)

damped_fit = damped_model.fit(optimized=True)

damped_forecast = damped_fit.forecast(steps=12)
```

`optimized=True` asks Statsmodels to estimate suitable smoothing parameters and initial states for the fitted data. Exact results may differ from a manual implementation because initialization and optimization procedures can differ.

---

## 11. Model evaluation

The notebook uses a chronological train-validation-test split.

- **Training set:** Used to fit the model.
- **Validation set:** Used to select alpha and beta.
- **Test set:** Kept separate until final evaluation.

Random shuffling should generally be avoided for time series because it can allow future observations to influence model selection or evaluation.

### Metrics covered

**Mean Absolute Error (MAE)**

$$MAE=\frac{1}{n}\sum_{t=1}^{n}|Y_t-\hat{Y}_t|$$

Measures the average absolute forecast error in the original units.

**Root Mean Squared Error (RMSE)**

$$RMSE=\sqrt{\frac{1}{n}\sum_{t=1}^{n}(Y_t-\hat{Y}_t)^2}$$

Penalizes larger errors more strongly than MAE.

**Weighted Mean Absolute Percentage Error (WMAPE)**

$$WMAPE=\frac{\sum_{t=1}^{n}|Y_t-\hat{Y}_t|}{\sum_{t=1}^{n}|Y_t|}$$

Expresses total absolute error relative to total actual demand. If the denominator is zero, WMAPE is undefined.

**Mean Error (Bias)**

$$Bias=\frac{1}{n}\sum_{t=1}^{n}(\hat{Y}_t-Y_t)$$

A positive value indicates overforecasting on average; a negative value indicates underforecasting.

No single metric should be used in isolation. In supply-chain forecasting, error magnitude and bias both matter.

---

## 12. Advantages

- Models both level and trend.
- Relatively easy to understand and implement.
- Computationally efficient.
- Can adapt to gradual changes in level and trend.
- Requires fewer inputs than many regression or machine-learning approaches.
- Useful as a statistical baseline for trend-driven demand.

## 13. Limitations

- Basic Holt's method does not model seasonality.
- Linear trend extrapolation may become unrealistic over long horizons.
- Sudden changes in demand may not be captured immediately.
- Results depend on parameter values and initialization.
- It may be unsuitable for intermittent demand with many zero periods.
- A visible trend in a short history may not persist into the future.

---

## 14. Supply-chain applications

Holt's method can be considered for:

- Products with gradually increasing demand.
- Products experiencing a steady decline in demand.
- Regularly consumed spare parts with a trend.
- Item-location combinations with sufficient non-intermittent history.
- Baseline forecasts for products without strong seasonal behaviour.

### Example

Suppose demand for a spare part has increased gradually over the past two years. SES may estimate the average level but produce a flat forecast. Holt's method can estimate the current level and the rate of increase, then extend that trend into the planning horizon.

However, if the increase is caused by a temporary promotion, a one-time project, a price change, or a supply disruption, simply extending the historical trend may be misleading. Relevant business context and explanatory variables should be considered.

---

## 15. Practical modelling checklist

Before selecting Holt's method:

1. Plot the historical series.
2. Check whether a trend is present.
3. Check for weekly, monthly, or annual seasonality.
4. Identify outliers, missing periods, promotions, and structural changes.
5. Use a chronological validation strategy.
6. Tune alpha and beta on validation data, not the final test set.
7. Compare against simple baselines such as Naive and SES.
8. Review MAE, RMSE, WMAPE, and bias.
9. Inspect residuals for remaining patterns.
10. Check whether long-horizon forecasts are operationally plausible.
11. Evaluate across multiple products and locations before using the method at scale.

---

## 16. Notebook contents

Python Implementation: [Notebook Link](https://github.com/ashay-thamankar/time_series_models/blob/main/Holts_Linear_Trend/Holts_Linear_Trend_Forecasting.ipynb) includes:

1. Introduction and learning objectives.
2. Synthetic monthly demand data with an upward trend.
3. Level, trend, and forecast equations.
4. A worked numerical example.
5. Holt's method implemented from scratch.
6. One-step-ahead fitted values.
7. Level and trend visualizations.
8. Multi-step forecasting.
9. Alpha and beta experiments.
10. Chronological train-validation-test split.
11. Validation-based parameter selection.
12. Test-set evaluation using MAE, RMSE, WMAPE, and bias.
13. Comparison with Simple Exponential Smoothing.
14. Statsmodels implementation.
15. Damped vs undamped trend.
16. Residual analysis.
17. Supply-chain considerations, advantages, limitations, and exercises.

---

## 20. Key takeaway

**Simple Exponential Smoothing estimates the level of a time series. Holt's Linear Trend extends it by estimating both level and trend.**

It is a useful method when demand follows a gradual upward or downward trajectory without strong seasonality. As with any forecasting method, validate it against suitable baselines and ensure its assumptions match the behaviour of the data.

### References

- Hyndman, R. J. and Athanasopoulos, G. *Forecasting: Principles and Practice* – Exponential smoothing.
- Statsmodels documentation – Holt's method and Holt-Winters exponential smoothing.
