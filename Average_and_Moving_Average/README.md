# 📈 Average and Moving Average Forecasting

## 1. Introduction

Before moving towards advanced time series forecasting algorithms, it is important to understand simple statistical forecasting methods.

**Average and Moving Average** are two fundamental forecasting techniques that use historical observations to estimate future values.

These methods are easy to understand, computationally inexpensive, and useful as baseline models when evaluating more advanced forecasting approaches.

Although they are simple, they introduce important forecasting concepts such as historical averages, rolling windows, smoothing, and forecast evaluation.

### 📚 Topics Covered

- Average (Mean) Forecasting
- Simple Moving Average (SMA)
- Choosing the Moving Average window
- Trailing and Centered Moving Averages
- Multi-step forecasting using Moving Average
- Weighted Moving Average (WMA)
- Model evaluation and practical considerations

---

## 2. 📊 Average (Mean) Forecasting

### 2.1 What is the Average Method?

The Average Method forecasts future values using the arithmetic mean of all historical observations available in the training data.

It assumes that future observations will be close to the average level observed in the past.

Therefore, every future time step receives the same forecast value.

### 2.2 Mathematical Formula

The forecast for any future time period is:

$$
\hat{y}_{T+h} = \bar{y}_T
$$

The historical average is calculated as:

$$
\bar{y}_T = \frac{1}{T}\sum_{t=1}^{T} y_t
$$

Where:

- $\hat{y}_{T+h}$ = Forecast for future time period $T+h$
- $T$ = Last time period in the training data
- $h$ = Forecast horizon
- $\bar{y}_T$ = Average of historical observations
- $y_t$ = Actual observation at time $t$

### 2.3 Worked Example

Consider the following monthly demand:

| Month | Actual Demand |
|---|---:|
| January | 100 |
| February | 120 |
| March | 110 |
| April | 130 |
| May | 140 |

The average demand is:

$$
\bar{y}_T = \frac{100+120+110+130+140}{5}
$$

$$
\bar{y}_T = \frac{600}{5} = 120
$$

Therefore, the forecast for the next three months will be:

| Month | Forecast |
|---|---:|
| June | 120 |
| July | 120 |
| August | 120 |

The model predicts a constant demand of 120 units for every future period.

### 2.4 ⚙️ How the Average Model Works

1. Collect historical observations for a time series.
2. Calculate the arithmetic mean using the training data.
3. Use the calculated mean as the forecast for every future time period.
4. Compare the forecast with actual observations when they become available.
5. Recalculate the average when the model is retrained with new observations.

### 2.5 💡 Intuition

Imagine a product whose demand fluctuates around a relatively stable level.

Some months have higher demand, while others have lower demand. However, there is no consistent upward or downward trend.

In this situation, the historical average can represent the typical demand level.

The Average Method works best when:

- The series has a relatively stable mean.
- There is no strong upward or downward trend.
- There is no pronounced seasonal pattern.
- Random fluctuations occur around a consistent level.

### 2.6 ✅ Advantages

- Very easy to understand and implement.
- Computationally inexpensive.
- Provides a useful baseline for comparing advanced forecasting models.
- Reduces the impact of random fluctuations through averaging.
- Can perform reasonably well for stable time series.

### 2.7 ⚠️ Limitations

- Does not capture trend.
- Does not capture seasonality.
- Gives equal importance to old and recent observations.
- Can be affected by outliers.
- May become less representative when the underlying demand level changes.
- Produces a constant forecast for all future periods.

### 2.8 🎯 When to Use It

- As a baseline before implementing complex forecasting algorithms.
- For stable demand with no clear trend or seasonal pattern.
- When a quick and transparent estimate is required.
- To determine whether a more advanced model provides meaningful improvement over a simple historical baseline.

---

## 3. 📉 Simple Moving Average (SMA)

### 3.1 What is a Moving Average?

A **Simple Moving Average (SMA)** forecasts the next value using the average of a fixed number of the most recent observations.

Unlike the Average Method, which considers all historical observations in the training data, the Moving Average uses only a selected window of recent observations.

As new observations become available, the window moves forward. The oldest observation leaves the window, and the newest observation enters it.

### 3.2 Mathematical Formula

For a moving-average window of size $k$, the one-step-ahead forecast is:

$$
\hat{y}_{T+1} = \frac{1}{k}\sum_{i=0}^{k-1}y_{T-i}
$$

Expanded:

$$
\hat{y}_{T+1} =
\frac{y_T+y_{T-1}+\cdots+y_{T-k+1}}{k}
$$

Where:

- $\hat{y}_{T+1}$ = Forecast for the next time period
- $y_T$ = Most recent actual observation
- $k$ = Number of observations in the moving window
- $y_{T-i}$ = Historical observation included in the window

### 3.3 Worked Example

Suppose the last five weeks of demand are:

| Week | Demand |
|---|---:|
| Week 1 | 100 |
| Week 2 | 120 |
| Week 3 | 110 |
| Week 4 | 130 |
| Week 5 | 140 |

We want to calculate a **3-period Moving Average**.

Only the latest three observations are considered:

$$
\hat{y}_{6} = \frac{110+130+140}{3}
$$

$$
\hat{y}_{6} = \frac{380}{3} = 126.67
$$

Therefore, the forecast for Week 6 is approximately **126.67 units**.

When the actual demand for Week 6 becomes available, it can be included in the next window to calculate the forecast for Week 7.

### 3.4 ⚙️ How the Moving Average Model Works

1. Select a window size, such as 3, 4, 7 or 12 observations.
2. Take the most recent $k$ actual observations.
3. Calculate their arithmetic mean.
4. Use this mean as the next-period forecast.
5. When a new actual observation becomes available, move the window forward.
6. Repeat the process to generate updated forecasts.

### 3.5 💡 Intuition

A Moving Average gives more relevance to recent history by excluding older observations outside the selected window.

For example:

- A 3-period Moving Average reacts relatively quickly to recent changes.
- A 6-period Moving Average smooths more fluctuations.
- A 12-period Moving Average produces a smoother series but may react slowly when demand changes.

The window size controls the balance between responsiveness and smoothness.

### 3.6 ✅ Advantages

- Simple to calculate and explain.
- Reduces short-term random fluctuations.
- Focuses on recent observations instead of the entire history.
- Easy to update when new actual values become available.
- Useful for relatively stable series with limited trend and seasonality.

### 3.7 ⚠️ Limitations

- The choice of window size can significantly affect forecast accuracy.
- A short window may be sensitive to noise.
- A long window may react slowly to changes.
- Every observation within the window receives equal importance.
- Does not explicitly model trend or seasonality.
- Multi-step forecasting requires assumptions about future observations.

### 3.8 🎯 When to Use It

- For short-term forecasting when recent demand is more representative than older demand.
- For smoothing noisy time series.
- For products with relatively stable demand levels.
- As a baseline for comparison with advanced forecasting models.

---

## 4. 🔍 Choosing the Moving Average Window

The window size $k$ is one of the most important decisions when applying a Moving Average.

### 4.1 Small Window

A smaller window uses fewer recent observations.

**Characteristics:**

- Reacts quickly to recent changes.
- Retains more short-term variation.
- Can be sensitive to random noise and outliers.
- May produce less stable forecasts.

Example: 3-period Moving Average.

### 4.2 Large Window

A larger window uses more historical observations.

**Characteristics:**

- Produces smoother estimates.
- Reduces the impact of short-term fluctuations.
- Reacts more slowly to sudden changes.
- May include older observations that no longer represent current demand.

Example: 12-period Moving Average.

### 4.3 How to Select the Window

The window size should not be selected only because it produces a smoother-looking line.

A practical approach is:

1. Select a few reasonable candidate windows based on data frequency and business context.
2. Train each candidate using the same training period.
3. Evaluate them on a separate validation period.
4. Compare error metrics such as MAE, RMSE or WMAPE.
5. Select the window that performs best on validation data.
6. Evaluate the selected approach on an untouched test period.

For monthly data, candidate windows might include 3, 6 and 12 months.

These are examples, not universal rules.

---

## 5. 🔄 Trailing vs. Centered Moving Average

Moving averages can be calculated in different ways depending on the objective.

### 5.1 Trailing Moving Average

A Trailing Moving Average uses the current observation and the preceding observations.

For a window size of $k$:

$$
MA_t = \frac{1}{k}\sum_{i=0}^{k-1}y_{t-i}
$$

It uses only observations available up to time $t$.

**Applications:**

- Operational forecasting
- Demand monitoring
- Regularly updated analysis

A trailing moving average is suitable for forecasting because it does not require future observations.

### 5.2 Centered Moving Average

A Centered Moving Average places the average around the middle of the selected time window.

For an odd window size:

$$
CMA_t =
\frac{1}{k}
\sum_{i=-(k-1)/2}^{(k-1)/2}y_{t+i}
$$

It uses observations both before and after time $t$.

**Applications:**

- Historical trend estimation
- Time-series decomposition
- Exploratory data analysis

**Important:** A centered moving average uses future observations relative to its center point. Therefore, it should not be used as a real-time forecasting input when those future observations are unavailable. Doing so can introduce data leakage.

For even window sizes, centering generally requires an additional averaging step between adjacent moving averages.

---

## 6. 🔮 Multi-Step Forecasting with a Moving Average

A Moving Average naturally provides a one-step-ahead forecast when recent actual observations are available.

To forecast multiple future periods, a recursive approach can be used.

### 6.1 Recursive Forecasting

For the first future period:

$$
\hat{y}_{T+1} =
\frac{1}{k}\sum_{i=0}^{k-1}y_{T-i}
$$

For the second future period, the unavailable actual value is replaced with the previous forecast:

$$
\hat{y}_{T+2} =
\frac{y_{T-1}+\cdots+y_T+\hat{y}_{T+1}}{k}
$$

More generally, each next forecast uses the latest $k$ values from the combined sequence of actual observations and previously generated forecasts.

### 6.2 Important Consideration

In multi-step forecasting, future actual values are not known.

Therefore, the model must use its own earlier predictions to generate later predictions.

This can cause forecasts to move towards a relatively stable level.

A basic Moving Average does not independently project a trend or seasonal pattern into the future.

---

## 7. ⚖️ Weighted Moving Average (WMA)

### 7.1 What is a Weighted Moving Average?

A **Weighted Moving Average (WMA)** is an extension of the Simple Moving Average.

Instead of assigning equal importance to every observation in the window, it assigns different weights to observations.

In many forecasting applications, more recent observations receive higher weights because they may better represent the current demand level.

### 7.2 Mathematical Formula

For a window containing $k$ observations:

$$
\hat{y}_{T+1} =
\frac{\sum_{i=0}^{k-1}w_i y_{T-i}}
{\sum_{i=0}^{k-1}w_i}
$$

Where:

- $w_i$ = Weight assigned to an observation
- $y_{T-i}$ = Corresponding historical observation
- $k$ = Number of observations in the window

If the weights are normalized so that their sum is 1:

$$
\sum_{i=0}^{k-1}w_i=1
$$

The formula becomes:

$$
\hat{y}_{T+1} = \sum_{i=0}^{k-1}w_i y_{T-i}
$$

### 7.3 Worked Example

Suppose the last three demand observations are:

| Period | Demand | Weight |
|---|---:|---:|
| Oldest | 100 | 0.20 |
| Previous | 120 | 0.30 |
| Latest | 140 | 0.50 |

The weights sum to 1:

$$
0.20+0.30+0.50=1
$$

The weighted forecast is:

$$
\hat{y}_{T+1} =
(100 \times 0.20)+(120 \times 0.30)+(140 \times 0.50)
$$

$$
\hat{y}_{T+1}=20+36+70=126
$$

The WMA forecast is **126 units**.

For comparison, the Simple Moving Average is:

$$
SMA = \frac{100+120+140}{3}=120
$$

The WMA forecast is higher because the latest observation receives the greatest weight.

### 7.4 ✅ Advantages

- Allows recent observations to have greater influence.
- More flexible than a Simple Moving Average.
- Can respond more quickly to changes when higher weights are assigned to recent data.

### 7.5 ⚠️ Limitations

- Weights must be selected or estimated.
- Poorly selected weights can reduce forecast accuracy.
- Does not automatically identify trend or seasonality.
- Still depends on the selected window.

---

## 8. 📋 Average vs. Moving Average vs. Weighted Moving Average

| Feature | Average Method | Simple Moving Average | Weighted Moving Average |
|---|---|---|---|
| Historical observations | All training observations | Most recent $k$ observations | Most recent $k$ observations |
| Weighting | Equal across all history | Equal within the window | Different weights within the window |
| Updates with new actuals | When recalculated/retrained | When the window moves | When the window moves |
| Main purpose | Constant-level baseline | Recent-level estimate and smoothing | Recent-level estimate with unequal importance |
| Main parameters | Training history | Window size $k$ | Window size and weights |
| Trend modelling | No | No | No |
| Seasonality modelling | No | No | No |

### Key Difference

- **Average Method:** Uses the mean of all available training observations.
- **Simple Moving Average:** Uses the mean of a fixed number of recent observations.
- **Weighted Moving Average:** Uses a weighted mean of a fixed number of recent observations.

---

## 9. 📏 Model Evaluation

Forecasting models should be evaluated on observations that were not used to calculate their forecasts.

A time series should be split chronologically rather than randomly.

### 9.1 Train-Test Split

A typical workflow is:

1. Use earlier observations as the training set.
2. Calculate the model parameters or historical averages using only training data.
3. Generate forecasts for the test period.
4. Compare forecasts with actual test values.
5. Calculate evaluation metrics.

For window selection, use a validation period within the historical data. Keep the final test period untouched until the model and window size have been selected.

### 9.2 Evaluation Metrics

Let:

- $A_t$ = Actual value at time $t$
- $F_t$ = Forecast value at time $t$
- $n$ = Number of evaluated observations

#### Mean Absolute Error (MAE)

MAE measures the average absolute difference between actual and forecast values.

$$
MAE = \frac{1}{n}\sum_{t=1}^{n}|A_t-F_t|
$$

**Interpretation:** Lower MAE indicates smaller average absolute forecast errors. It is expressed in the same units as the target variable.

#### Root Mean Squared Error (RMSE)

RMSE gives greater importance to larger errors because the errors are squared before averaging.

$$
RMSE =
\sqrt{
\frac{1}{n}
\sum_{t=1}^{n}(A_t-F_t)^2
}
$$

**Interpretation:** Lower RMSE is better. It is useful when large forecast errors should receive additional penalty.

#### Mean Absolute Percentage Error (MAPE)

MAPE expresses absolute errors as percentages of actual values.

$$
MAPE =
\frac{100}{n}
\sum_{t=1}^{n}
\left|\frac{A_t-F_t}{A_t}\right|
$$

**Interpretation:** Lower MAPE is better.

**Limitation:** MAPE is undefined when actual values are zero and can become extremely large when actual values are close to zero. It should be used carefully for intermittent or low-volume demand.

#### Weighted Mean Absolute Percentage Error (WMAPE)

WMAPE divides the sum of absolute errors by the sum of absolute actual values.

$$
WMAPE =
100 \times
\frac{\sum_{t=1}^{n}|A_t-F_t|}
{\sum_{t=1}^{n}|A_t|}
$$

**Interpretation:** Lower WMAPE is better. It gives greater influence to observations with larger actual volumes.

WMAPE is often useful in demand-forecasting settings where the impact of errors should reflect demand volume.

If the sum of absolute actual values is zero, WMAPE is undefined.

### 9.3 Which Metric Should You Use?

There is no single best metric for every forecasting problem.

- **MAE:** When you want an easily interpretable average error in business units.
- **RMSE:** When large forecast errors should receive additional penalty.
- **MAPE:** When percentage errors are meaningful and actual values are safely away from zero.
- **WMAPE:** When you want an aggregate percentage error weighted by actual demand volume.

For supply-chain demand forecasting, WMAPE is commonly useful for aggregate volume-weighted evaluation. However, it should be considered alongside other metrics and business context.

---

## 10. 🏭 Practical Considerations in Demand Forecasting

When applying these methods to real demand data, consider the following:

### 10.1 Trend

If demand is consistently increasing or decreasing, an Average or Moving Average may lag behind the actual demand level.

### 10.2 Seasonality

If demand repeats a pattern every week, month, quarter or year, a basic average may hide that pattern.

A seasonal forecasting method may be more appropriate.

### 10.3 Outliers and Promotions

Unusual demand caused by promotions, one-time orders, stockpiling or data errors can influence the average.

Investigate unusual observations before deciding whether they should be retained, adjusted or treated separately.

### 10.4 Intermittent Demand

For products with many zero-demand periods and occasional non-zero demand, simple averages can produce forecasts that do not reflect the underlying demand occurrence pattern.

Intermittent-demand methods such as Croston's Method and its variants may be more suitable.

### 10.5 Data Leakage

All calculations used to create a forecast must rely only on information available at the forecast origin.

Do not calculate historical averages, moving averages or transformations using future test observations.

### 10.6 Forecast Horizon

A method that performs well for one-step-ahead forecasting may not perform equally well for a longer horizon.

Evaluate forecasts at the horizon that matches the actual business requirement.

---

## 11. ⚠️ Common Mistakes to Avoid

- Using the full dataset to calculate an average before splitting into training and test sets.
- Randomly splitting time-series observations.
- Selecting a moving-average window based on test-set performance.
- Assuming a larger window is always better.
- Using centered moving averages as real-time forecasting inputs without accounting for future data.
- Ignoring zero actual values when calculating MAPE.
- Comparing models using different test periods.
- Treating a moving average as a method that explicitly models trend or seasonality.
- Using recursive multi-step forecasting without understanding that earlier predictions become inputs to later predictions.

---

## 12. 📌 Advantages and Limitations: Summary

| Method | Advantages | Limitations |
|---|---|---|
| Average | Simple, transparent, inexpensive, useful baseline | Constant forecast; ignores trend, seasonality and changes in demand level |
| Simple Moving Average | Smooths noise; uses recent observations; easy to update | Window-sensitive; equal weights; can lag; no explicit trend or seasonality |
| Weighted Moving Average | Gives more influence to selected observations; flexible | Requires weight selection; can be sensitive to poor weights; no explicit trend or seasonality |

---

## 13. 🎯 Key Takeaways

1. The **Average Method** forecasts every future period using the mean of all historical training observations.
2. The **Simple Moving Average** uses a fixed window of the most recent observations.
3. The **Weighted Moving Average** assigns different weights to observations within the selected window.
4. A smaller moving-average window is generally more responsive but can be noisier.
5. A larger window generally produces smoother estimates but may react slowly to changes.
6. Trailing moving averages use only past and current observations, while centered moving averages can use future observations.
7. Moving averages are useful baselines and smoothing techniques, but they do not explicitly model trend or seasonality.
8. Use chronological validation and test periods to evaluate forecasting performance fairly.

---

## 14. 💻 Notebook Implementation

The accompanying notebook demonstrates these concepts using synthetic time-series data.

It includes:

- Average forecasting with a worked example
- Simple Moving Average calculations
- Comparison of different window sizes
- Trailing and centered moving averages
- Recursive multi-step Moving Average forecasting
- Weighted Moving Average calculations
- A chronological train-validation-test workflow
- Forecast visualizations and evaluation metrics
- Exercises to reinforce the concepts

GitHub Notebook: [Notebook Link]()

---

## 15. 📖 References and Further Learning

- [Forecasting: Principles and Practice](https://otexts.com/fpp3/) — Rob J. Hyndman and George Athanasopoulos
- [Pandas Rolling Window Documentation](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rolling.html)
- [NumPy Mean Documentation](https://numpy.org/doc/stable/reference/generated/numpy.mean.html)
- [Scikit-learn Model Evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html)

---

**Happy Learning! 🚀**