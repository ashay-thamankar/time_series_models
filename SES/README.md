# Simple Exponential Smoothing (SES)

## 1. Introduction

Simple Exponential Smoothing (SES) is a statistical time-series forecasting method used to predict future values when a series has a relatively stable level and does not show a clear trend or seasonality.

SES gives more importance to recent observations while gradually reducing the influence of older observations. It achieves this by updating a smoothed estimate of the series level whenever a new actual value becomes available.

For example, SES can be used to forecast monthly demand for a product whose sales fluctuate around a relatively consistent average, without a strong upward or downward trend or a repeating seasonal pattern.

### When should we use SES?

SES is a suitable starting point when:

- The time series has a relatively stable level.
- There is no clear trend.
- There is no significant seasonal pattern.
- Recent observations should influence the forecast more than older observations.

SES is not designed to explicitly model trend or seasonality. If these patterns are present, methods such as Holt's Linear Trend or Holt-Winters may be more appropriate.

---

## 2. Learning Objectives

By working through this notebook, you will learn how to:

- Understand the intuition behind Simple Exponential Smoothing.
- Explain the role of the smoothing parameter, alpha.
- Calculate smoothed levels manually.
- Build an SES forecasting function using Python.
- Generate one-step-ahead fitted forecasts.
- Fit an SES model using `statsmodels`.
- Compare different alpha values.
- Select alpha using a validation dataset.
- Evaluate forecasts using common error metrics.
- Generate forecasts for future periods.
- Understand the strengths and limitations of SES.

---

## 3. Understanding the Intuition

Consider a product with the following monthly demand:

| Month | Actual Demand |
|---|---:|
| January | 100 |
| February | 110 |
| March | 105 |
| April | 115 |
| May | 120 |

The demand changes from month to month, but there is no obvious repeating seasonal pattern or strong trend in this short example.

A simple average would give equal importance to every observation. SES instead assigns more weight to recent observations and less weight to older ones.

The model maintains a value called the **smoothed level**. Whenever a new actual observation becomes available, the level is updated using that observation and the previous level estimate.

The updated level becomes the forecast for the next period.

### Key intuition

- A low alpha gives more weight to the previous smoothed level. The forecast changes slowly.
- A high alpha gives more weight to the latest actual observation. The forecast reacts more quickly.
- Alpha controls how quickly the model responds to changes in the time series.

---

## 4. Mathematical Formula

Simple Exponential Smoothing uses one main component: the level.

### 4.1 Level equation

The smoothed level at time `t` is:

$$
L_t = \alpha Y_t + (1-\alpha)L_{t-1}
$$

Where:

- $L_t$ = updated smoothed level at time `t`
- $Y_t$ = actual observation at time `t`
- $L_{t-1}$ = previous period's smoothed level
- $\alpha$ = smoothing parameter, where $0 < \alpha \leq 1$

The equation combines the latest actual observation with the previous smoothed level.

### 4.2 Forecast equation

For any future horizon `h`:

$$
\hat{Y}_{t+h|t} = L_t
$$

This means that SES produces a **flat forecast** for all future periods. Once the model has estimated the latest level, that level is used as the forecast until a new actual observation becomes available.

### 4.3 Role of alpha

The smoothing parameter alpha determines how much weight is assigned to the latest actual observation.

| Alpha | Behaviour |
|---:|---|
| Close to 0 | Strong smoothing; forecast changes slowly |
| Around 0.3 | Moderate smoothing |
| Around 0.6 | More responsive to recent observations |
| Close to 1 | Very responsive; forecast approaches the latest actual value |

Alpha is generally selected using a validation dataset or estimated by minimizing an in-sample objective. It should not be selected based only on how well the model fits the training data.

---

## 5. Step-by-Step Numerical Example

Suppose the actual demand values are:

`[100, 110, 105, 115, 120]`

Assume:

- Initial level: $L_1 = 100$
- Smoothing parameter: $\alpha = 0.30$

The level equation is:

$$
L_t = 0.30Y_t + 0.70L_{t-1}
$$

### Calculations

**Period 1**

The initial level is set to 100:

$$
L_1 = 100
$$

**Period 2**

Actual demand is 110:

$$
L_2 = 0.30(110) + 0.70(100) = 103
$$

**Period 3**

Actual demand is 105:

$$
L_3 = 0.30(105) + 0.70(103) = 103.6
$$

**Period 4**

Actual demand is 115:

$$
L_4 = 0.30(115) + 0.70(103.6) = 107.02
$$

**Period 5**

Actual demand is 120:

$$
L_5 = 0.30(120) + 0.70(107.02) = 110.914
$$

The final smoothed level is approximately **110.91**.

Therefore, the forecast for each future period is:

| Forecast Period | Forecast |
|---|---:|
| Period 6 | 110.91 |
| Period 7 | 110.91 |
| Period 8 | 110.91 |

The forecast remains constant because SES models only the level, not trend or seasonality.

---

## 6. Initialization

SES needs an initial level before it can start updating.

A simple initialization approach is to use the first observation:

$$
L_1 = Y_1
$$

Other approaches are possible. For example, the initial level can be estimated by optimizing the model's objective function.

The choice of initialization can affect early fitted values, especially for short time series. Many statistical libraries provide options for estimating the initial level automatically.

In the notebook, the manual implementation uses the first observation as the initial level so that the calculations are easy to follow.

---

## 7. Python Implementation from Scratch

The notebook implements SES manually using Python.

The core logic is:

```python
def simple_exponential_smoothing(data, alpha):
    data = np.asarray(data, dtype=float)

    level = data[0]
    fitted = [np.nan]

    for actual in data[1:]:
        # Forecast made before observing the current actual value
        fitted.append(level)

        # Update the level after observing the actual value
        level = alpha * actual + (1 - alpha) * level

    return np.array(fitted), level
```

### Understanding the implementation

1. Convert the input observations into a NumPy array.
2. Initialize the level using the first observation.
3. For each subsequent period, record the previous level as the one-step-ahead fitted forecast.
4. Once the actual value is observed, update the level.
5. Return the fitted forecasts and the final level.

The ordering matters. A one-step-ahead forecast for period `t` must be produced using information available before observing the actual value at period `t`.

The first fitted value is `NaN` because the simple implementation initializes the level using the first observation and does not create a forecast for that initial point.

---

## 8. Understanding Alpha Through Experiments

A useful way to understand SES is to fit the model with different alpha values.

For example:

- `alpha = 0.10`: places relatively little weight on the latest actual value.
- `alpha = 0.30`: gives moderate weight to the latest actual value.
- `alpha = 0.60`: responds more strongly to recent changes.
- `alpha = 0.90`: reacts very quickly to the latest observation.

When alpha is low, short-term fluctuations are smoothed out, but the model may respond slowly when the underlying level changes.

When alpha is high, the model responds quickly, but it may also follow random fluctuations more closely.

There is no universally best alpha. Its suitability depends on the behaviour of the series and the forecasting objective.

---

## 9. Train-Test Split and Model Evaluation

Forecasting evaluation should respect the chronological order of observations.

The notebook uses a chronological split:

- Earlier observations are used for training.
- Later observations are held out for testing.
- The model is fitted using training data.
- Forecasts are generated for the test horizon.
- Predictions are compared with the actual test values.

Do not randomly shuffle time-series observations before splitting. Doing so can allow future information to enter the training dataset.

### 9.1 Mean Absolute Error (MAE)

$$
MAE = \frac{1}{n}\sum_{t=1}^{n}|Y_t-\hat{Y}_t|
$$

MAE measures the average absolute difference between actual and forecast values. It is expressed in the same unit as the target variable.

### 9.2 Root Mean Squared Error (RMSE)

$$
RMSE = \sqrt{\frac{1}{n}\sum_{t=1}^{n}(Y_t-\hat{Y}_t)^2}
$$

RMSE penalizes larger errors more heavily because the errors are squared before averaging.

### 9.3 Weighted Mean Absolute Percentage Error (WMAPE)

$$
WMAPE = \frac{\sum_{t=1}^{n}|Y_t-\hat{Y}_t|}{\sum_{t=1}^{n}|Y_t|}\times 100
$$

WMAPE expresses total absolute error relative to total actual demand. It is often useful for demand forecasting when observations have different volumes.

WMAPE is undefined when the sum of absolute actual values is zero. The notebook handles this case explicitly.

### 9.4 Mean Error / Bias

$$
Bias = \frac{1}{n}\sum_{t=1}^{n}(\hat{Y}_t-Y_t)
$$

This definition uses forecast minus actual:

- Positive bias indicates forecasts are higher than actuals on average.
- Negative bias indicates forecasts are lower than actuals on average.
- A value close to zero indicates low average signed error, but does not necessarily mean forecast accuracy is good.

A model can have a low bias while still having large absolute errors, so bias should be considered alongside MAE, RMSE, or WMAPE.

### 9.5 Why the notebook includes MAPE carefully

Mean Absolute Percentage Error is:

$$
MAPE = \frac{100}{n}\sum_{t=1}^{n}\left|\frac{Y_t-\hat{Y}_t}{Y_t}\right|
$$

MAPE is undefined when an actual value is zero and can become extremely large when actual values are close to zero. It should therefore be used cautiously, particularly for intermittent demand.

The notebook's synthetic SES example uses positive demand values, but this does not make MAPE universally suitable for all forecasting datasets.

---

## 10. Selecting Alpha Using a Validation Dataset

A common mistake is to try several alpha values and choose the one with the lowest error on the test dataset.

The test dataset should be kept separate for the final evaluation.

A better process is:

1. Split the original time series into training and test periods.
2. Reserve the latest portion of the training data as a validation window.
3. Fit SES using the earlier training observations for each candidate alpha.
4. Forecast the validation window.
5. Compare validation errors, such as MAE or RMSE.
6. Select the alpha that performs best for the chosen evaluation metric.
7. Refit the model using the full training dataset with the selected alpha.
8. Evaluate once on the untouched test dataset.

The notebook demonstrates this process with a small grid of candidate alpha values.

For a robust production workflow, use rolling-origin evaluation across multiple validation windows rather than relying on a single split.

---

## 11. Using Statsmodels

Python's `statsmodels` library provides a ready-to-use implementation through `SimpleExpSmoothing`.

Example:

```python
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

model = SimpleExpSmoothing(
    train,
    initialization_method="estimated"
)

fitted_model = model.fit(
    smoothing_level=0.30,
    optimized=False
)

forecast = fitted_model.forecast(steps=len(test))
```

### Important parameters

- `initialization_method`: controls how the initial level is initialized. The example uses estimated initialization.
- `smoothing_level`: the alpha parameter.
- `optimized`: when set to `False`, the supplied smoothing parameter is used. When set to `True`, the model estimates the smoothing parameter by optimization.

The notebook demonstrates both a fixed-alpha model and a model where alpha is optimized.

For reproducible comparisons, make sure that the initialization and optimization settings are understood and documented.

---

## 12. Generating Future Forecasts

After selecting the model and fitting it on the available historical observations, SES can generate forecasts for future periods.

If the final estimated level is $L_T$, then:

$$
\hat{Y}_{T+1|T} = L_T
$$

$$
\hat{Y}_{T+2|T} = L_T
$$

$$
\cdots
$$

$$
\hat{Y}_{T+h|T} = L_T
$$

For example, if the final level is 110.91, the forecast for the next 12 months is 110.91 for every month.

This flat forecast is an important characteristic of SES. It is useful when the expected demand level is stable, but it may be unsuitable when demand is expected to grow, decline, or follow a seasonal pattern.

---

## 13. Residual Analysis

After fitting a forecasting model, it is useful to inspect the residuals.

Residual:

$$
e_t = Y_t - \hat{Y}_t
$$

Residual analysis can help identify patterns that the model has not captured.

Check for:

- **Non-zero average:** may indicate systematic under-forecasting or over-forecasting.
- **Trend in residuals:** may suggest that the series has a trend that SES does not model.
- **Seasonal pattern:** may suggest that seasonal structure is missing.
- **Autocorrelation:** may indicate that information remains in the residual sequence.
- **Large outliers:** may indicate unusual events, data issues, or exceptional demand.

A good forecasting model should ideally leave residuals without systematic structure. Residuals are not expected to be exactly zero.

---

## 14. Advantages of SES

- **Simple to understand:** uses one level component and one smoothing parameter.
- **Easy to implement:** can be built with a few lines of Python.
- **Low computational cost:** suitable for fitting many stable time series.
- **Responsive to recent observations:** alpha controls how quickly the forecast adapts.
- **Useful baseline:** provides a benchmark for more advanced forecasting models.
- **No need to store all historical observations for updating:** the level can be updated recursively.

---

## 15. Limitations of SES

- **No explicit trend component:** it cannot project a continuing upward or downward trend.
- **No seasonality component:** it does not capture repeating weekly, monthly, or yearly patterns.
- **Flat multi-step forecasts:** all future periods receive the same forecast.
- **Sensitive to alpha selection:** an unsuitable alpha can lead to forecasts that are too reactive or too slow.
- **Not a complete solution for intermittent demand:** frequent zero-demand periods may require methods designed for intermittent demand, such as Croston, SBA, or TSB.
- **Unexpected events are not anticipated:** promotions, disruptions, price changes, and other external factors are not modeled directly.

---

## 16. Practical Supply-Chain Applications

SES can be a useful baseline for products with relatively stable demand.

Examples include:

- Forecasting monthly demand for a mature product with a stable sales pattern.
- Estimating a baseline demand level for products without strong trend or seasonality.
- Supporting replenishment planning when demand is relatively consistent.
- Creating a benchmark before testing more complex forecasting methods.
- Forecasting stable consumption of regularly used materials or components.

### Example: Stable spare-parts demand

Suppose a spare part has monthly demand that fluctuates around a stable level. There is no clear upward trend and no repeating seasonal pattern.

SES can smooth short-term fluctuations and provide a baseline forecast for future periods.

However, if the spare part has long periods of zero demand followed by occasional orders, an intermittent-demand method may be more suitable.

Likewise, if demand is increasing over time, Holt's Linear Trend may be a better choice.

---

## 17. SES vs Holt's Linear Trend vs Holt-Winters

These methods belong to the exponential smoothing family, but they model different time-series components.

| Method | Components | Suitable pattern | Forecast behaviour |
|---|---|---|---|
| Simple Exponential Smoothing | Level | Stable level, no clear trend or seasonality | Flat forecast |
| Holt's Linear Trend | Level + Trend | Trend, no seasonality | Trend continues into the forecast horizon |
| Holt-Winters | Level + Trend + Seasonality | Trend and seasonality | Forecast includes trend and seasonal pattern |

### Which method should you choose?

- Use **SES** when the series has a stable level.
- Use **Holt's Linear Trend** when a trend is present but seasonality is not.
- Use **Holt-Winters** when both trend and seasonality are present.

Always validate the selected method using an appropriate time-series validation strategy.

---

## 18. Common Mistakes to Avoid

1. **Using SES on strongly trending data:** SES does not explicitly model trend.
2. **Ignoring seasonality:** a flat forecast may be inappropriate for seasonal demand.
3. **Selecting alpha using the test dataset:** this leads to optimistic evaluation.
4. **Randomly splitting time-series observations:** preserve chronological order.
5. **Confusing fitted values with future forecasts:** fitted values are historical one-step-ahead predictions; future forecasts use the final estimated level.
6. **Ignoring initialization:** initial-level choices can affect early fitted values.
7. **Using MAPE without checking zeros:** MAPE is undefined when actual demand is zero.
8. **Assuming a more complex model is always better:** compare models using out-of-sample performance and business relevance.

---

## 19. Notebook Contents

Python Implementation: [Notebook Link](https://github.com/ashay-thamankar/time_series_models/blob/main/SES/Simple_Exponential_Smoothing_Forecasting.ipynb)

The accompanying notebook, `Simple_Exponential_Smoothing_Forecasting.ipynb`, includes:

1. Introduction and learning objectives.
2. Synthetic monthly demand generation.
3. SES equations and alpha intuition.
4. Step-by-step manual numerical example.
5. Manual implementation using NumPy.
6. Historical one-step-ahead fitted forecasts.
7. Comparison of different alpha values.
8. Chronological train-test split.
9. SES implementation using `statsmodels`.
10. Fixed-alpha and optimized-alpha models.
11. Forecast evaluation using MAE, RMSE, WMAPE, MAPE, and bias.
12. Alpha selection using a validation window.
13. Future multi-step forecasts.
14. Residual analysis.
15. Advantages, limitations, and supply-chain applications.
16. Comparison with Holt's and Holt-Winters methods.
17. References.

---

## 20. Key Takeaways

- Simple Exponential Smoothing is designed for time series with a relatively stable level and no clear trend or seasonality.
- It updates the estimated level using the latest observation and the previous smoothed level.
- Alpha controls the balance between responsiveness and smoothness.
- SES produces a flat forecast across all future horizons.
- Alpha should be selected using training and validation data, while the test dataset should remain untouched until final evaluation.
- SES is a useful baseline, but trend, seasonality, intermittent demand, and external drivers may require other forecasting methods.

**The key idea:** SES uses recent observations to continuously update the estimated demand level and uses that latest level as the forecast for future periods.

---

## 21. References and Further Reading

- Hyndman, R. J., & Athanasopoulos, G. *Forecasting: Principles and Practice* — Exponential smoothing and simple exponential smoothing: https://otexts.com/fpp3/ses.html
- Statsmodels documentation — `SimpleExpSmoothing`: https://www.statsmodels.org/stable/generated/statsmodels.tsa.holtwinters.SimpleExpSmoothing.html
- Statsmodels documentation — Exponential smoothing examples: https://www.statsmodels.org/stable/examples/notebooks/generated/exponential_smoothing.html

---

If you find this useful, feel free to explore the notebook, run the experiments, and share your feedback!
