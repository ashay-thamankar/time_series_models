# 📈 Naive and Seasonal Naive Forecasting

**Naive Forecasting** and **Seasonal Naive Forecasting** methods are simple, yet they play an important role in forecasting. They are commonly used as baseline models to evaluate whether more advanced forecasting techniques actually provide better results.

Both methods rely entirely on historical observations, without requiring complex model training or parameter optimization.

---

## 📚 Table of Contents

1. [Introduction](#-introduction)
2. [Naive Forecasting](#-1-naive-forecasting)
   - What is Naive Forecasting?
   - Mathematical Intuition
   - Example
   - Advantages
   - Limitations
   - Practical Use Cases
3. [Seasonal Naive Forecasting](#-2-seasonal-naive-forecasting)
   - What is Seasonal Naive Forecasting?
   - Mathematical Intuition
   - Example
   - Advantages
   - Limitations
   - Practical Use Cases
4. [Naive vs Seasonal Naive](#-3-naive-vs-seasonal-naive)
5. [Model Evaluation](#-4-model-evaluation)
6. [Python Implementation](#-5-python-implementation)
7. [Key Takeaways](#-6-key-takeaways)
8. [References](#-7-references)

---

## 🎯 Introduction

Time series forecasting involves predicting future values using historical observations collected over time.

Before exploring advanced techniques such as ARIMA, SARIMA, Exponential Smoothing, XGBoost, and LSTM, it is important to understand the basic forecasting methods.

**Why start with Naive and Seasonal Naive?**

- They are easy to understand and implement.
- They provide baseline forecasting results.
- They require minimal computational resources.
- They help evaluate whether advanced models provide meaningful improvements.
- They introduce the fundamentals of forecast generation and evaluation.

We will explore both methods using mathematical intuition, numerical examples, and Python implementations.

---

# 📌 1. Naive Forecasting

## 1.1 What is Naive Forecasting?

Naive forecasting is one of the simplest time series forecasting techniques.

It assumes that the next observation will be equal to the most recently observed actual value.

In simple terms:

**Tomorrow's forecast = Today's actual value.**

For example, if today's demand is 120 units, the forecast for tomorrow will also be 120 units.

The same value is carried forward for every future forecast period.

### 💡 Key Intuition

The model assumes that the most recent observation is the best estimate of future values.

It does not attempt to identify trends, seasonal patterns, or other underlying relationships in the data.

## 1.2 Mathematical Intuition

Let:

- $y_t$ = Actual observation at time $t$
- $T$ = Last observed time period
- $h$ = Forecast horizon
- $\hat{y}_{T+h|T}$ = Forecast for period $T+h$, using information available up to $T$

The Naive forecasting equation is:

$$
\hat{y}_{T+h|T} = y_T
$$

This means that every future forecast is equal to the last available actual observation.

### Understanding the equation

Suppose the last observed demand is 130 units.

- One-step-ahead forecast = 130
- Two-step-ahead forecast = 130
- Three-step-ahead forecast = 130

The forecast remains constant regardless of the forecast horizon.

## 1.3 📊 Numerical Example

Consider the following monthly demand:

| Month | Actual Demand |
|---|---:|
| January | 100 |
| February | 110 |
| March | 105 |
| April | 120 |
| May | 125 |
| June | 130 |

We want to forecast demand for July, August, and September.

Since the last observed demand in June is 130 units:

| Month | Naive Forecast |
|---|---:|
| July | 130 |
| August | 130 |
| September | 130 |

**Observation:** The model simply carries forward the last observed value for all future periods.

## 1.4 ⚙️ How Does It Work?

The Naive forecasting process involves three simple steps:

1. Identify the last available actual observation.
2. Use that observation as the forecast for the next period.
3. Repeat the same value for the remaining forecast horizon.

No model fitting, parameter optimization, or training process is required.

## 1.5 ✅ Advantages

- Very easy to understand and implement.
- No model training or parameter tuning required.
- Computationally inexpensive.
- Useful as a baseline for evaluating advanced forecasting models.
- Can be effective for relatively stable time series.

## 1.6 ❌ Limitations

- Does not capture trends.
- Does not capture seasonality.
- Forecasts remain constant for all future periods.
- Sensitive to unusual changes in the last observation.
- Can perform poorly when the time series has strong trend or seasonal patterns.

## 1.7 🏭 Practical Use Cases

- Baseline demand forecasting.
- Short-term forecasting when values are relatively stable.
- Benchmarking advanced forecasting models.
- Quick forecasting when limited historical data is available.

---

# 📌 2. Seasonal Naive Forecasting

## 2.1 What is Seasonal Naive Forecasting?

Seasonal Naive forecasting is an extension of the Naive method.

Instead of using the most recent observation, it uses the observation from the corresponding season in the previous seasonal cycle.

In simple terms:

**This month's forecast = Same month's actual value from last year.**

For example, if we are forecasting July 2026, the model uses July 2025 actual demand, assuming monthly seasonality.

### 💡 Key Intuition

The model assumes that seasonal patterns repeat over time.

If demand increases every December due to seasonal shopping, the Seasonal Naive method uses the previous December's demand to forecast the next December.

It repeats the most recently observed seasonal cycle for future periods.

## 2.2 Mathematical Intuition

Let:

- $y_t$ = Actual observation at time $t$
- $T$ = Last observed time period
- $h$ = Forecast horizon
- $m$ = Seasonal period

The Seasonal Naive forecasting equation is:

$$
\hat{y}_{T+h|T} = y_{T+h-m\lceil h/m\rceil}
$$

Where:

- $m$ represents the number of observations in one seasonal cycle.
- $\lceil h/m\rceil$ ensures that the forecast uses the corresponding observation from the latest available seasonal cycle.

### Understanding the equation

Suppose we have monthly data with a seasonal period of 12.

For the first forecasted month:

$$
\hat{y}_{T+1|T} = y_{T+1-12}
$$

The forecast uses the corresponding month from the previous year.

For future periods, the same seasonal cycle is repeated.

### Common Seasonal Periods

| Data Frequency | Seasonal Period |
|---|---:|
| Hourly | 24 (daily seasonality) |
| Daily | 7 (weekly seasonality) |
| Weekly | 52 (yearly seasonality) |
| Monthly | 12 (yearly seasonality) |
| Quarterly | 4 (yearly seasonality) |

*Note: These are common choices, not universal rules. The appropriate seasonal period depends on the data and the forecasting problem.*

## 2.3 📊 Numerical Example

Consider monthly demand for two years:

| Month | 2025 Actual | 2026 Actual |
|---|---:|---:|
| January | 100 | 110 |
| February | 120 | 125 |
| March | 130 | 140 |
| April | 150 | 155 |
| May | 170 | 180 |
| June | 200 | 210 |
| July | 220 | - |
| August | 230 | - |

We want to forecast July and August 2026.

The seasonal period is 12 because we are working with monthly data and assuming yearly seasonality.

Therefore:

- July 2026 forecast = July 2025 actual = 220
- August 2026 forecast = August 2025 actual = 230

| Month | Seasonal Naive Forecast |
|---|---:|
| July 2026 | 220 |
| August 2026 | 230 |

**Observation:** Unlike the Naive method, the Seasonal Naive method preserves the recurring seasonal pattern.

## 2.4 ⚙️ How Does It Work?

1. Identify the seasonal frequency of the time series.
2. Determine the last available seasonal cycle.
3. Extract the observations from that cycle.
4. Repeat the seasonal pattern to generate future forecasts.

For example, with monthly data and a seasonal period of 12, the last 12 observations are repeated for future forecasting periods.

## 2.5 ✅ Advantages

- Captures recurring seasonal patterns.
- Simple to understand and implement.
- Does not require model training.
- Useful for seasonal demand forecasting.
- Provides a strong baseline for seasonal time series.

## 2.6 ❌ Limitations

- Assumes seasonal patterns repeat consistently.
- Does not independently capture changes in trend.
- Requires sufficient historical data to identify the seasonal cycle.
- May perform poorly when seasonal patterns change over time.
- Does not account for external factors affecting demand.

## 2.7 🏭 Practical Use Cases

- Retail demand forecasting.
- Electricity consumption forecasting.
- Monthly sales forecasting.
- Seasonal inventory planning.
- Weekly or monthly business demand patterns.

---

# 📊 3. Naive vs Seasonal Naive

Both methods are simple forecasting techniques, but their forecasting assumptions are different.

| Aspect | Naive | Seasonal Naive |
|---|---|---|
| Forecast basis | Last observation | Corresponding seasonal observation |
| Trend handling | No | No |
| Seasonality handling | No | Repeats historical seasonality |
| Complexity | Very low | Very low |
| Training required | No | No |
| Parameters required | None | Seasonal period |
| Best suited for | Stable series | Seasonal series |

### When should we use each method?

**Use Naive Forecasting when:**

- The time series does not exhibit strong seasonality.
- Recent observations provide a reasonable baseline for future values.
- You need a simple benchmark.

**Use Seasonal Naive Forecasting when:**

- The time series exhibits recurring seasonal patterns.
- Historical seasonal cycles are reasonably consistent.
- You want to establish a seasonal benchmark before trying advanced models.

---

# 📏 4. Model Evaluation

Generating forecasts is only one part of forecasting.

We also need to evaluate how closely the predicted values match the actual observations.

Some commonly used forecasting evaluation metrics are:

## 4.1 Mean Absolute Error (MAE)

MAE measures the average absolute difference between actual and forecast values.

$$
\text{MAE} = \frac{1}{n}\sum_{t=1}^{n}|y_t - \hat{y}_t|
$$

**Interpretation:** Lower MAE indicates better forecasting accuracy.

## 4.2 Root Mean Squared Error (RMSE)

RMSE measures the square root of the average squared forecasting errors.

$$
\text{RMSE} = \sqrt{\frac{1}{n}\sum_{t=1}^{n}(y_t - \hat{y}_t)^2}
$$

**Interpretation:** Larger errors receive more weight because the errors are squared.

## 4.3 Mean Absolute Percentage Error (MAPE)

MAPE expresses forecasting errors as a percentage of actual values.

$$
\text{MAPE} = \frac{100}{n}\sum_{t=1}^{n}\left|\frac{y_t - \hat{y}_t}{y_t}\right|
$$

**Interpretation:** Lower MAPE indicates better forecasting accuracy.

**Limitation:** MAPE is undefined when actual values are zero and can become misleading when actual values are very close to zero.

## 4.4 Weighted Mean Absolute Percentage Error (WMAPE)

WMAPE calculates total absolute forecasting error relative to total actual demand.

$$
\text{WMAPE} = \frac{\sum_{t=1}^{n}|y_t - \hat{y}_t|}{\sum_{t=1}^{n}|y_t|} \times 100
$$

**Interpretation:** Lower WMAPE indicates better forecasting accuracy.

WMAPE is particularly useful in demand forecasting when evaluating multiple products with different demand volumes.

**Important:** All models must be evaluated on the same test period using the same evaluation metrics to ensure a fair comparison.

---

# 💻 5. Python Implementation

Theory becomes more useful when we implement the models and observe how they behave on actual data.

For this exercise, both models are implemented using Python, NumPy, and pandas.

### 📓 Notebook 1 – Naive Forecast

Explore the complete Python implementation, including:

- Sample time series dataset.
- Exploratory data analysis.
- Train-test split.
- Naive forecasting implementation.
- Forecast visualization.
- Evaluation using MAE, RMSE, MAPE, and WMAPE.
- Interpretation of forecasting results.

🔗 **GitHub Notebook:** [Naive Forecast Notebook](https://github.com/ashay-thamankar/time_series_models/blob/main/Naive_and_Seasonal_Naive/Naive_Forecast.ipynb)

### 📓 Notebook 2 – Seasonal Naive Forecast

Explore the complete Python implementation, including:

- Seasonal time series dataset.
- Identifying seasonal frequency.
- Exploratory data analysis.
- Train-test split.
- Seasonal Naive forecasting implementation.
- Forecast visualization.
- Evaluation using MAE, RMSE, MAPE, and WMAPE.
- Interpretation of forecasting results.

🔗 **GitHub Notebook:** [Seasonal Naive Forecast](https://github.com/ashay-thamankar/time_series_models/blob/main/Naive_and_Seasonal_Naive/Seasonal_Naive_Forecast.ipynb)

---

# 🎯 6. Key Takeaways

- Naive forecasting uses the last observed value to predict all future periods.
- Seasonal Naive forecasting uses the corresponding observation from the previous seasonal cycle.
- Both methods are computationally inexpensive and do not require model training.
- Naive forecasting is useful for stable time series, while Seasonal Naive is useful for recurring seasonal patterns.
- Both methods provide important benchmarks for evaluating advanced forecasting techniques.
- Model selection should depend on the characteristics of the time series rather than the complexity of the algorithm.

**Remember:** A more complex forecasting model is not necessarily a better forecasting model. It should demonstrate measurable improvement over a suitable baseline.

---

# 📚 7. References

1. **Forecasting: Principles and Practice** – Rob J Hyndman and George Athanasopoulos  
   https://otexts.com/fpp3/

2. **StatsForecast Documentation**  
   https://nixtlaverse.nixtla.io/statsforecast/

3. **sktime Documentation**  
   https://www.sktime.net/