# Day 9: Holt-Winters Additive and Multiplicative Forecasting

## 1. Introduction

Holt-Winters, also known as **triple exponential smoothing**, extends Holt's Linear Trend method by modelling three components:

1. **Level** — the current baseline of the series.
2. **Trend** — the direction and rate of change.
3. **Seasonality** — a repeating pattern at a fixed frequency.

It is useful when a time series contains both a trend and recurring seasonal behaviour. The two common seasonal formulations are additive and multiplicative.

## 2. Additive vs Multiplicative Seasonality

### Additive seasonality

Use additive seasonality when the seasonal effect remains approximately constant in absolute units, even when the overall series level changes.

Example: December demand is about 200 units higher than the baseline whether normal demand is 1,000 or 2,000 units.

Conceptually:

$$Y_t = Level_t + Trend_t + Seasonal_t + Error_t$$

The seasonal component is expressed in the same units as the observations and is typically centred around zero over a full cycle.

### Multiplicative seasonality

Use multiplicative seasonality when the seasonal effect is proportional to the series level.

Example: December demand is consistently about 20% above baseline. As baseline demand grows, the absolute December uplift grows too.

Conceptually:

$$Y_t = (Level_t + Trend_t) \\times Seasonal_t \\times Error_t$$

The seasonal component is a factor: 1.20 represents approximately 20% above baseline; 0.80 represents approximately 20% below baseline. Seasonal factors are typically centred around one.

| Characteristic | Additive | Multiplicative |
|---|---|---|
| Seasonal amplitude | Roughly constant in units | Changes with series level |
| Seasonal component | Absolute effect | Relative factor |
| Typical seasonal centre | 0 | 1 |
| Example | +200 units each December | 1.20 × baseline each December |
| Data restriction | Can accommodate zero/negative values in many cases | Standard formulation requires strictly positive data |

**Practical rule:** Start with additive seasonality when seasonal swings remain similar in absolute size. Test multiplicative seasonality when seasonal swings grow or shrink with the level. Confirm the choice through time-ordered validation and business context.

## 3. Components and seasonal period

- **Level ($L_t$):** estimated baseline at time $t$.
- **Trend ($T_t$):** estimated movement per time step.
- **Seasonal component ($S_t$):** seasonal effect or factor.
- **Seasonal period ($m$):** number of observations in one complete cycle.

Typical seasonal periods:
- Monthly data with annual seasonality: 12.
- Quarterly data with annual seasonality: 4.
- Daily data with weekly seasonality: 7.
- Weekly data with yearly seasonality: often approximately 52, with calendar effects requiring care.

The seasonal period must match the data frequency and business process.

## 4. Mathematical formulation

Notation:
- $Y_t$: actual observation
- $L_t$: level
- $T_t$: trend
- $S_t$: seasonal component
- $m$: seasonal period
- $\\alpha$: level smoothing parameter
- $\\beta$: trend smoothing parameter
- $\\gamma$: seasonal smoothing parameter
- $h$: forecast horizon

### 4.1 Holt-Winters Additive

**Level**

$$L_t = \\alpha(Y_t-S_{t-m})+(1-\\alpha)(L_{t-1}+T_{t-1})$$

The observation is adjusted by subtracting the seasonal effect from the corresponding period in the previous cycle. The adjusted observation is combined with the previous level and trend.

**Trend**

$$T_t = \\beta(L_t-L_{t-1})+(1-\\beta)T_{t-1}$$

The trend is updated using the change in level and the previous trend estimate.

**Seasonality**

$$S_t = \\gamma(Y_t-L_t)+(1-\\gamma)S_{t-m}$$

The seasonal effect is updated using the difference between the observation and the current level.

**Forecast**

$$\\hat{Y}_{t+h|t}=L_t+hT_t+S_{t-m+((h-1)\\bmod m)+1}$$

The forecast combines projected level, trend and the seasonal effect corresponding to the future period.

### 4.2 Holt-Winters Multiplicative

**Level**

$$L_t=\\alpha\\left(\\frac{Y_t}{S_{t-m}}\\right)+(1-\\alpha)(L_{t-1}+T_{t-1})$$

The observation is seasonally adjusted by dividing by its seasonal factor.

**Trend**

$$T_t=\\beta(L_t-L_{t-1})+(1-\\beta)T_{t-1}$$

The trend update has the same form as in the additive seasonal method.

**Seasonality**

$$S_t=\\gamma\\left(\\frac{Y_t}{L_t}\\right)+(1-\\gamma)S_{t-m}$$

The seasonal factor is updated using the ratio of the observation to the current level.

**Forecast**

$$\\hat{Y}_{t+h|t}=(L_t+hT_t)\\times S_{t-m+((h-1)\\bmod m)+1}$$

The projected level and trend are multiplied by the seasonal factor for the forecast period.

These are the standard formulations with additive trend. Damped-trend variants are also available.

## 5. Smoothing parameters: Alpha, Beta and Gamma

| Parameter | Component | Role |
|---|---|---|
| Alpha ($\\alpha$) | Level | Controls how quickly level responds to new observations |
| Beta ($\\beta$) | Trend | Controls how quickly trend adapts |
| Gamma ($\\gamma$) | Seasonality | Controls how quickly seasonal effects adapt |

- Higher alpha: more responsive level.
- Lower alpha: smoother level, slower reaction.
- Higher beta: trend responds more quickly to recent changes.
- Lower beta: trend changes more gradually.
- Higher gamma: seasonal effects adapt more quickly.
- Lower gamma: seasonal effects remain smoother.

Parameters should be estimated or selected using validation data, not the final test period.

## 6. Worked example: additive component updates

Assume:
- Previous level = 100
- Previous trend = 5
- Matching seasonal effect = 10
- Current actual demand = 120
- Alpha = 0.4
- Beta = 0.3
- Gamma = 0.2

Level:

$$L_t=0.4(120-10)+0.6(100+5)=107$$

Trend:

$$T_t=0.3(107-100)+0.7(5)=5.6$$

Seasonal component:

$$S_t=0.2(120-107)+0.8(10)=10.6$$

The updated level is 107, trend is 5.6 and seasonal effect is 10.6. A future forecast combines the updated level and trend with the seasonal component for the relevant future month.

## 7. Python implementation

The notebook provides both manual implementations and implementations using `statsmodels`.

### Install dependencies

```bash
pip install numpy pandas matplotlib statsmodels scikit-learn
```

### Manual additive update logic

```python
fitted[i] = level + trend + seasonal[i % m]
old_level = level

level = alpha * (y[i] - seasonal[i % m]) + (1-alpha) * (level + trend)
trend = beta * (level - old_level) + (1-beta) * trend
seasonal[i % m] = gamma * (y[i] - level) + (1-gamma) * seasonal[i % m]
```

### Manual multiplicative update logic

```python
fitted[i] = (level + trend) * seasonal[i % m]
old_level = level

level = alpha * (y[i] / seasonal[i % m]) + (1-alpha) * (level + trend)
trend = beta * (level - old_level) + (1-beta) * trend
seasonal[i % m] = gamma * (y[i] / level) + (1-gamma) * seasonal[i % m]
```

The notebook includes complete functions, initialization, fitted values, multi-step forecasting and input checks. The manual code is an educational implementation of the standard equations. Initialization affects the results, so the manual and library outputs need not match exactly.

## 8. Using Statsmodels

Import:

```python
from statsmodels.tsa.holtwinters import ExponentialSmoothing
```

### Additive model

```python
model = ExponentialSmoothing(
    train_data,
    trend="add",
    seasonal="add",
    seasonal_periods=12,
    initialization_method="estimated"
)
fit = model.fit(optimized=True)
forecast = fit.forecast(12)
```

### Multiplicative model

```python
model = ExponentialSmoothing(
    train_data,
    trend="add",
    seasonal="mul",
    seasonal_periods=12,
    initialization_method="estimated"
)
fit = model.fit(optimized=True)
forecast = fit.forecast(12)
```

### Damped trend

A linear trend extrapolates at the same rate at every forecast horizon. This can create unrealistic long-range forecasts. A damped trend gradually reduces the trend contribution.

```python
model = ExponentialSmoothing(
    train_data,
    trend="add",
    damped_trend=True,
    seasonal="add",
    seasonal_periods=12,
    initialization_method="estimated"
)
fit = model.fit(optimized=True)
forecast = fit.forecast(12)
```

## 9. Notebook workflow

Python Implementation: [Notebook Link](https://github.com/ashay-thamankar/time_series_models/blob/main/Holt_Winter_Additive_Multiplicative/Holts_Winters_Additive_and_Multiplicative_Forecasting.ipynb)

The notebook creates two synthetic monthly datasets with 120 observations (10 years):

- Additive data: increasing baseline + fixed seasonal effects + noise.
- Multiplicative data: increasing baseline × seasonal factors + noise.

It then:
1. Plots and inspects both datasets.
2. Explains the equations and smoothing parameters.
3. Demonstrates a numerical component update.
4. Implements both models from scratch.
5. Plots fitted values and seasonal components.
6. Splits data chronologically into training, validation and test sets.
7. Tunes alpha, beta and gamma on validation data.
8. Evaluates the selected manual models on the untouched test set.
9. Fits additive and multiplicative `statsmodels` models.
10. Compares manual and library implementations.
11. Forecasts the next 12 months.
12. Compares damped and undamped trends.
13. Inspects residuals and discusses practical applications.

## 10. Evaluation metrics

**MAE**

$$MAE=\\frac{1}{n}\\sum_{t=1}^{n}|Y_t-\\hat{Y}_t|$$

Average absolute error in demand units.

**RMSE**

$$RMSE=\\sqrt{\\frac{1}{n}\\sum_{t=1}^{n}(Y_t-\\hat{Y}_t)^2}$$

Penalizes large errors more heavily than MAE.

**WMAPE**

$$WMAPE=\\frac{\\sum_{t=1}^{n}|Y_t-\\hat{Y}_t|}{\\sum_{t=1}^{n}|Y_t|}$$

Total absolute error relative to total actual demand. High-volume observations can dominate this aggregate metric.

**Mean bias**

$$Bias=\\frac{1}{n}\\sum_{t=1}^{n}(\\hat{Y}_t-Y_t)$$

- Positive: overforecasting on average.
- Negative: underforecasting on average.

MAPE can be undefined or unstable when actual demand is zero or close to zero, so it is not used as the primary metric in this notebook.

## 11. Validation strategy

The notebook uses a chronological split:
- First 70%: training.
- Next 15%: validation.
- Final 15%: test.

The validation set selects smoothing parameters. The test set is reserved for final evaluation. A stronger production evaluation can use rolling-origin or expanding-window validation across several forecast origins.

Do not randomly shuffle time-series observations for ordinary forecasting evaluation; it can leak future information into training.

## 12. Supply-chain applications

**Additive candidate:** A product receives roughly the same extra unit demand each festive season, even as its baseline demand changes.

**Multiplicative candidate:** Demand rises by a relatively stable percentage during a seasonal period, so absolute seasonal peaks become larger as baseline demand grows.

Potential use cases include retail, apparel, food and beverage, consumer products and location-level demand with stable recurring seasonality.

Holt-Winters does not directly model promotions, prices, stockouts, holidays or causal drivers. These may require event adjustments, regression, SARIMAX or machine-learning approaches.

## 13. Advantages and limitations

### Advantages
- Models level, trend and seasonality together.
- Interpretable and computationally efficient.
- Useful benchmark for recurring seasonal demand.
- Additive and multiplicative variants support different seasonal behaviours.

### Limitations
- Requires a meaningful seasonal frequency and sufficient history.
- Standard multiplicative seasonality requires positive observations.
- May struggle with changing or irregular seasonal patterns.
- Does not automatically incorporate external drivers.
- Undamped trend can become unrealistic at long horizons.
- Usually not the first choice for highly intermittent demand.

## 14. Holt vs Holt-Winters

| Feature | Holt Linear Trend | Holt-Winters Additive | Holt-Winters Multiplicative |
|---|---|---|---|
| Level | Yes | Yes | Yes |
| Trend | Yes | Yes | Yes |
| Seasonality | No | Yes | Yes |
| Seasonal behaviour | — | Constant absolute effect | Proportional effect |

Use Holt's Linear Trend when there is a trend but no meaningful seasonal pattern. Use Holt-Winters when both trend and seasonality are present.

## 15. Common mistakes

1. Choosing the wrong seasonal period.
2. Using multiplicative seasonality with zero or negative observations.
3. Randomly splitting the data.
4. Tuning parameters on the test set.
5. Using too little history; at least two full cycles are useful for initialization.
6. Assuming the seasonal pattern is stable without checking.
7. Ignoring long-horizon trend uncertainty.
8. Ignoring promotions, stockouts and structural breaks.

## 16. Key takeaways

- Holt-Winters is triple exponential smoothing: level + trend + seasonality.
- Additive seasonality fits approximately constant absolute seasonal effects.
- Multiplicative seasonality fits seasonal effects proportional to the series level.
- Alpha, beta and gamma update level, trend and seasonality.
- Use chronological validation and an untouched test set.
- Select based on data behaviour, forecast accuracy, bias and business context.

## References

- [Statsmodels — ExponentialSmoothing](https://www.statsmodels.org/stable/generated/statsmodels.tsa.holtwinters.ExponentialSmoothing.html)
- [Forecasting: Principles and Practice — Exponential Smoothing](https://otexts.com/fpp3/expsmooth.html)
- [Forecasting: Principles and Practice — Holt-Winters' Seasonal Method](https://otexts.com/fpp3/holt-winters.html)