# Theta Method for Time Series Forecasting

## 1. Introduction

The **Theta method** is a statistical time-series forecasting technique that modifies the curvature of a series using a parameter called **theta (θ)**. Forecasts from different theta-transformed representations are combined to produce the final forecast.

The method became widely known after its strong performance in the M3 forecasting competition. It is computationally efficient, relatively interpretable, and useful as a benchmark for many forecasting problems.

The classical two-theta formulation combines:
- A **linear-trend component**, which extrapolates the overall direction of the series.
- A **Simple Exponential Smoothing (SES) component**, which captures a smoothed level.

The classical method combines forecasts from θ = 0 and θ = 2 with equal weights.

The Day 10 notebook contains a from-scratch educational implementation, a seasonal `statsmodels` implementation, benchmark comparisons, chronological evaluation, and future forecasting.

---

## 2. What problem does Theta solve?

A time series can contain:
1. **Level:** the typical magnitude of the observations.
2. **Trend:** long-term upward or downward movement.
3. **Seasonality:** a recurring pattern at a fixed frequency.
4. **Noise:** random variation.

SES is useful when the series has a relatively stable level. Holt adds a trend component. Holt-Winters models level, trend, and seasonality.

The Theta method takes a different approach: it modifies the curvature of the series and combines forecasts from two theta lines.

It can be useful as:
- A statistical benchmark.
- A candidate for relatively smooth series.
- A method for series with a gradual trend.
- A comparison against SES, Holt, and Holt-Winters.

It is not guaranteed to be the best model for every series.

---

## 3. Intuition behind theta lines

Let the original time series be:

$$Y_1,Y_2,\ldots,Y_T$$

A theta line modifies the second differences (curvature) of the original series. A useful representation is:

$$Z_t(\theta)=\theta Y_t+(1-\theta)(a+bt)$$

where:
- $Z_t(\theta)$ = transformed series for theta value θ.
- $Y_t$ = observed value at time t.
- $a+bt$ = fitted linear trend.
- $a$ = regression intercept.
- $b$ = regression slope.
- $\theta$ = theta parameter.

The classical two-theta method uses two specific values.

### θ = 0: Linear trend component

$$Z_t(0)=a+bt$$

The curvature is removed, leaving a straight-line representation. Its forecast is obtained by extrapolating the fitted linear trend.

### θ = 2: Curvature-amplified component

$$Z_t(2)=2Y_t-(a+bt)$$

This component has twice the original curvature relative to the fitted linear trend. It is forecast using Simple Exponential Smoothing.

---

## 4. Classical two-theta forecasting equations

### 4.1 Linear trend component (θ = 0)

Fit a linear regression:

$$Y_t=a+bt+\epsilon_t$$

For a forecast horizon h:

$$\hat{Y}_{T+h}^{(0)}=a+b(T+h)$$

This component extrapolates the fitted trend.

### 4.2 Theta component (θ = 2)

Construct the theta-2 series:

$$Z_t(2)=2Y_t-(a+bt)$$

Fit Simple Exponential Smoothing to this transformed series. The SES level update is:

$$L_t=\alpha Z_t(2)+(1-\alpha)L_{t-1}$$

where:
- $L_t$ = smoothed level.
- $\alpha$ = smoothing parameter between 0 and 1.
- $Z_t(2)$ = current theta-2 observation.

The SES forecast is flat over the forecast horizon:

$$\hat{Y}_{T+h}^{(2)}=L_T$$

### 4.3 Combine the component forecasts

The classical Theta forecast is:

$$\hat{Y}_{T+h}=\frac{1}{2}\hat{Y}_{T+h}^{(0)}+\frac{1}{2}\hat{Y}_{T+h}^{(2)}$$

In words:

**Final Theta Forecast = 50% × Linear Trend Forecast + 50% × SES Forecast of the θ = 2 Series**

The equal weighting is an important part of the classical two-theta formulation.

---

## 5. Simple numerical example

Suppose the two components produce the following forecasts for the next month:
- θ = 0 linear-trend forecast = 150 units.
- θ = 2 SES forecast = 140 units.

Then:

$$\hat{Y}_{T+1}=0.5(150)+0.5(140)=145$$

The final Theta forecast is **145 units**.

The linear-trend component carries the extrapolated direction of the series. The SES component contributes a smoothed estimate of the level. This is an illustrative example; actual component values are estimated from historical observations.

---

## 6. Role of the smoothing parameter

The SES component uses a smoothing parameter α:

$$L_t=\alpha Z_t+(1-\alpha)L_{t-1}$$

| Alpha | Behaviour |
|---|---|
| Low α | More smoothing; reacts slowly to new observations |
| High α | More responsive to recent observations |
| α close to 1 | Places substantial weight on the latest observation |

The value of α can be estimated by software or selected using a validation period. Do not select α by looking at the final test period.

---

## 7. Seasonality and seasonal Theta

The classical two-theta formulation is commonly introduced for non-seasonal series. For seasonal data, a seasonal adjustment can be applied:

1. Estimate the seasonal pattern.
2. Remove the seasonal effect from the observations.
3. Apply the Theta method to the adjusted series.
4. Forecast the adjusted series.
5. Reapply the seasonal pattern to the forecasts.

For monthly data with annual seasonality, the seasonal period is 12. For quarterly data, it is usually 4. For daily data with a weekly pattern, it may be 7.

The `statsmodels` implementation provides a convenient interface:

```python
from statsmodels.tsa.forecasting.theta import ThetaModel

model = ThetaModel(
    train,
    period=12,
    deseasonalize=True
)
fit = model.fit()
forecast = fit.forecast(steps=12)
```

The seasonal period must match the data frequency and business pattern. A wrong period can produce misleading forecasts.

### Implementation note

The classical two-theta explanation and individual software implementations are related, but implementation details can differ, including seasonal adjustment, parameter estimation, initialization, and optimization. The notebook therefore presents both a transparent educational two-component implementation and the `statsmodels` seasonal `ThetaModel`. Do not assume they will produce identical results in every setting.

---

## 8. Python implementation from scratch

The notebook implements the classical two-component construction.

### Step 1: Fit a linear trend

```python
t = np.arange(len(y))
slope, intercept = np.polyfit(t, y.values, 1)
fitted_trend = intercept + slope * t
```

### Step 2: Construct the theta-2 series

```python
theta_2_series = 2 * y.values - fitted_trend
```

### Step 3: Fit SES to the theta-2 series

```python
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

ses_fit = SimpleExpSmoothing(
    pd.Series(theta_2_series),
    initialization_method="estimated"
).fit(optimized=True)
```

### Step 4: Forecast both components

```python
future_t = np.arange(len(y), len(y) + horizon)
theta_0_forecast = intercept + slope * future_t
theta_2_forecast = ses_fit.forecast(horizon)
```

### Step 5: Combine forecasts

```python
theta_forecast = (
    0.5 * theta_0_forecast
    + 0.5 * np.asarray(theta_2_forecast)
)
```

The complete function and example are available in `Theta_Method_Forecasting.ipynb`.

---

## 9. Using statsmodels ThetaModel

`statsmodels` provides `ThetaModel`:

```python
from statsmodels.tsa.forecasting.theta import ThetaModel

theta_model = ThetaModel(
    train,
    period=12,
    deseasonalize=True
)

theta_fit = theta_model.fit()
forecast = theta_fit.forecast(steps=12)
```

Common configuration concepts:

| Argument | Meaning |
|---|---|
| `endog` | Historical time series |
| `period` | Seasonal cycle length |
| `deseasonalize` | Whether to deseasonalize the data |
| `use_test` | Whether a test is used for selecting the seasonal decomposition approach |
| `theta` | Theta parameter exposed by the implementation |

Check the installed `statsmodels` documentation for the exact API and defaults, as library versions can change.

---

## 10. Models compared

| Model | Main idea | Suitable when |
|---|---|---|
| Naive | Repeat the latest observation | Simple baseline |
| SES | Smooth the level | No strong trend or seasonality |
| Holt | Smooth level and trend | Trend exists, seasonality absent |
| Classical Theta | Combine linear trend and SES-based component | Smooth series and benchmark forecasting |
| Seasonal Theta | Theta forecasting with seasonal adjustment | Trend and recurring seasonal pattern |

A more complex model should demonstrate measurable improvement over sensible baselines before being selected for production.

---

## 11. Forecast evaluation

The notebook uses chronological training, validation, and test periods.

- **Training set:** used to estimate model parameters.
- **Validation set:** used to compare models and select configurations.
- **Test set:** held out for final evaluation.

### MAE — Mean Absolute Error

$$MAE=\frac{1}{n}\sum_{t=1}^{n}|Y_t-\hat{Y}_t|$$

MAE measures average absolute error in the original demand units.

### RMSE — Root Mean Squared Error

$$RMSE=\sqrt{\frac{1}{n}\sum_{t=1}^{n}(Y_t-\hat{Y}_t)^2}$$

RMSE penalizes large errors more heavily than MAE.

### WMAPE — Weighted Mean Absolute Percentage Error

$$WMAPE=\frac{\sum_{t=1}^{n}|Y_t-\hat{Y}_t|}{\sum_{t=1}^{n}|Y_t|}\times100$$

WMAPE is useful for aggregating absolute errors across periods with different demand volumes. It is undefined if the denominator is zero.

### Mean Error (Bias)

$$Bias=\frac{1}{n}\sum_{t=1}^{n}(\hat{Y}_t-Y_t)$$

- Positive bias: average over-forecasting.
- Negative bias: average under-forecasting.

For intermittent or zero-heavy demand, MAPE can be undefined or unstable. MAE, RMSE, WMAPE (when its denominator is non-zero), and bias are often more useful.

---

## 12. Advantages

1. **Relatively simple:** combines a linear trend component and exponential smoothing.
2. **Interpretable:** the two components have understandable roles.
3. **Computationally efficient:** suitable for large collections of univariate series.
4. **Useful benchmark:** a strong statistical method to include in model comparisons.
5. **Seasonal implementation available:** seasonal adjustment can be incorporated.
6. **No external regressors required:** can be applied when only historical observations are available.

## 13. Limitations

1. **Not universally best:** performance depends on the characteristics of the series.
2. **Trend extrapolation risk:** the linear component may continue a trend that later changes.
3. **Seasonality requires care:** the seasonal period and adjustment approach must be appropriate.
4. **Limited external information:** the basic method does not directly model promotions, price, weather, holidays, or other explanatory variables.
5. **Structural breaks:** sudden business changes may reduce accuracy.
6. **Implementation differences:** initialization and seasonal handling can vary across libraries.

---

## 14. Supply-chain and retail applications

Potential applications include:
- Monthly demand forecasting for products with a gradual trend.
- Baseline forecasting for item-location combinations with reasonably smooth demand.
- Benchmarking statistical models against machine-learning models.
- Forecasting product families with recurring seasonal patterns when seasonal adjustment is suitable.
- Creating a baseline before adding promotion, price, weather, or event effects.

For highly intermittent demand, consider Croston, SBA, or TSB. For strong trend and seasonality, compare against Holt-Winters. For explanatory drivers, regression-based or machine-learning models may be more appropriate.

---

## 15. Common mistakes

- Randomly shuffling time-series data before splitting.
- Tuning the model against the test set.
- Using an incorrect seasonal period.
- Comparing models on different evaluation windows.
- Assuming a seasonal Theta model and the classical non-seasonal two-theta implementation are identical.
- Assuming Theta will always outperform SES, Holt, or Holt-Winters.
- Ignoring forecast bias and business impact while selecting a model.

---

## 16. Notebook contents

Python Implementation: [Notebook Link](https://github.com/ashay-thamankar/time_series_models/blob/main/Theta_Method/Theta_Method_Forecasting.ipynb) includes:

1. Introduction and learning objectives.
2. Mathematical intuition and equations.
3. Synthetic monthly demand generation.
4. Time-series exploration and decomposition.
5. Chronological train-validation-test split.
6. Classical two-theta implementation from scratch.
7. `statsmodels` seasonal ThetaModel.
8. Naive, SES, and Holt benchmarks.
9. MAE, RMSE, WMAPE, and bias evaluation.
10. Validation and final test comparison.
11. Future 12-month forecast.
12. Advantages, limitations, and supply-chain applications.

---

## 17. Key takeaway

The Theta method combines a linear-trend component with an exponentially smoothed component. It offers a practical and efficient statistical forecasting approach and is valuable as a benchmark for univariate time-series forecasting.

Understand the data pattern, handle seasonality appropriately, and evaluate Theta alongside simple baselines and other suitable methods using chronological validation.

---

## References and further reading

- Assimakopoulos, V. and Nikolopoulos, K. (2000). *The theta model: a decomposition approach to forecasting.*
- M3 Competition — research context for the method's popularity.
- [statsmodels ThetaModel documentation](https://www.statsmodels.org/stable/generated/statsmodels.tsa.forecasting.theta.ThetaModel.html)
- [Statsmodels Time Series Analysis](https://www.statsmodels.org/stable/tsa.html)
