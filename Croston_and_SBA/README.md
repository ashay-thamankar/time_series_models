# 📦 Croston's Method and Syntetos–Boylan Approximation (SBA)

## 1. Introduction

Many time series contain frequent zero-demand periods and occasional non-zero demand. This pattern is called **intermittent demand** and is common in spare parts, maintenance inventory, slow-moving products, and service-parts planning.

Traditional methods such as moving averages and exponential smoothing can struggle because they treat the sequence as a continuous stream of values. Croston's method addresses this by modelling two components separately:

1. **Demand size:** How much is demanded when demand occurs?
2. **Inter-demand interval:** How many periods typically pass between demand occurrences?

The **Syntetos–Boylan Approximation (SBA)** is a bias-adjusted version of Croston's method.

---

## 2. What is intermittent demand?

An example monthly demand series might look like this:

| Month | Demand |
|---|---:|
| Jan | 0 |
| Feb | 0 |
| Mar | 12 |
| Apr | 0 |
| May | 0 |
| Jun | 8 |
| Jul | 0 |
| Aug | 0 |
| Sep | 0 |
| Oct | 15 |

Characteristics:

- Many periods have zero demand.
- Non-zero demand occurs irregularly.
- Demand quantities can vary between occurrences.
- The timing and size of demand are both important.

A zero should represent a true no-demand period. Missing records and stockout-censored sales should not automatically be treated as zero demand.

---

## 3. Croston's Method

### 3.1 Intuition

Croston's method separates intermittent demand into two sequences:

- The non-zero demand quantities.
- The intervals between successive non-zero demand occurrences.

It applies simple exponential smoothing separately to these two components. The estimated demand rate is then calculated as the estimated demand size divided by the estimated inter-demand interval.

### 3.2 Mathematical formulation

Let:

- \(z_i\) = size of the \(i\)-th non-zero demand occurrence.
- \(p_i\) = interval between non-zero demand occurrences.
- \(\alpha\) = smoothing parameter, where \(0 < \alpha \leq 1\).
- \(\hat{z}_i\) = smoothed demand-size estimate.
- \(\hat{p}_i\) = smoothed inter-demand interval estimate.

When a non-zero demand occurs, update the estimates:

$$
\hat{z}_i = \alpha z_i + (1-\alpha)\hat{z}_{i-1}
$$

$$
\hat{p}_i = \alpha p_i + (1-\alpha)\hat{p}_{i-1}
$$

The Croston forecast per period is:

$$
\hat{y}_{t+h} = \frac{\hat{z}_t}{\hat{p}_t},
\qquad h=1,2,\ldots
$$

The forecast is constant across future periods until new observations are received and the estimates are updated.

### 3.3 Worked example

Suppose non-zero demand observations are:

- Demand sizes: 10, 14, 12 units
- Inter-demand intervals: 2, 3 periods
- Smoothing parameter: \(\alpha = 0.2\)

For illustration, initialize:

$$
\hat{z}_1 = 10,\qquad \hat{p}_1 = 2
$$

Update with the second occurrence:

$$
\hat{z}_2 = 0.2(14) + 0.8(10) = 10.8
$$

$$
\hat{p}_2 = 0.2(3) + 0.8(2) = 2.2
$$

Update with the third occurrence (assuming the next observed interval is 2 periods):

$$
\hat{z}_3 = 0.2(12) + 0.8(10.8) = 11.04
$$

$$
\hat{p}_3 = 0.2(2) + 0.8(2.2) = 2.16
$$

Therefore:

$$
\hat{y} = \frac{11.04}{2.16} \approx 5.11
$$

The estimated average demand rate is approximately 5.11 units per period.

*Note:* Initialization conventions differ across implementations. The notebooks document the convention used: initialize at the first non-zero observation, with the first interval equal to the number of elapsed periods from the beginning through that occurrence. Later intervals count periods since the previous non-zero occurrence.

### 3.4 Role of the smoothing parameter

- A smaller \(\alpha\) gives more weight to previous estimates and produces smoother updates.
- A larger \(\alpha\) gives more weight to the latest non-zero demand and interval.
- Alpha should be selected using a chronological validation period, not the final test set.

### 3.5 Advantages

- Specifically designed for intermittent demand.
- Separately models demand size and demand timing.
- Simple and computationally efficient.
- Provides a useful baseline for slow-moving items.

### 3.6 Limitations

- Standard Croston's method is biased under common theoretical assumptions.
- It does not explicitly model seasonality or trend.
- Its forecast is constant across future periods.
- It does not directly model changes in the probability of demand occurrence.
- Results can depend on initialization and smoothing parameter.

---

## 4. Syntetos–Boylan Approximation (SBA)

### 4.1 Why SBA?

Croston's method is widely used for intermittent demand, but it has a known tendency to overestimate the underlying demand rate under the assumptions considered in the literature.

SBA was proposed as a bias-adjusted approximation. It retains Croston's estimates of demand size and inter-demand interval, then applies a correction factor.

### 4.2 Mathematical formulation

The SBA forecast is:

$$
\hat{y}_{t+h}^{SBA}
=
\left(1-\frac{\alpha}{2}\right)
\frac{\hat{z}_t}{\hat{p}_t}
$$

Where:

- \(\alpha\) = smoothing parameter.
- \(\hat{z}_t\) = smoothed non-zero demand size.
- \(\hat{p}_t\) = smoothed inter-demand interval.
- \(\left(1-\frac{\alpha}{2}\right)\) = SBA correction factor.

The term \(\frac{\hat{z}_t}{\hat{p}_t}\) is the Croston forecast. SBA multiplies it by a factor below one for \(0 < \alpha \leq 1\).

### 4.3 Worked example

Suppose:

- Croston demand-size estimate = 20 units.
- Croston interval estimate = 4 periods.
- Smoothing parameter = 0.2.

First calculate the Croston forecast:

$$
\hat{y}^{Croston} = \frac{20}{4} = 5
$$

Calculate the SBA correction factor:

$$
1-\frac{0.2}{2} = 0.9
$$

Calculate the SBA forecast:

$$
\hat{y}^{SBA} = 0.9 \times 5 = 4.5
$$

Therefore:

- Croston forecast = 5 units per period.
- SBA forecast = 4.5 units per period.

SBA produces a lower forecast because of its bias correction. This does not mean it will always have lower forecast error on every individual item.

### 4.4 Advantages

- Retains Croston's separate treatment of demand size and interval.
- Applies a correction intended to reduce Croston's bias.
- Simple to implement.
- Useful as a benchmark for intermittent-demand forecasting.

### 4.5 Limitations

- The correction does not guarantee improved accuracy for every series.
- It does not explicitly model trend, seasonality, or changing demand occurrence probability.
- It produces a constant rate forecast across the future horizon.
- It remains sensitive to data quality, initialization, and alpha.

---

## 5. Croston vs. SBA

| Aspect | Croston | SBA |
|---|---|---|
| Demand size | Exponentially smoothed | Same as Croston |
| Inter-demand interval | Exponentially smoothed | Same as Croston |
| Forecast | \(\hat{z}/\hat{p}\) | \((1-\alpha/2)(\hat{z}/\hat{p})\) |
| Bias adjustment | No SBA correction | Applies correction factor |
| Future forecast | Constant rate | Constant rate |
| Main purpose | Intermittent-demand baseline | Bias-adjusted Croston alternative |

**Key difference:** SBA is not a completely different smoothing algorithm. It uses Croston's estimates and adjusts the resulting forecast using the factor \(1-\alpha/2\).

---

## 6. Croston and SBA compared with other forecasting methods

- **Naive:** Repeats the last observed value. It may produce zero forecasts if the last period has zero demand.
- **Average:** Uses the average of all historical periods, including zeros.
- **Moving Average:** Uses a fixed window of recent periods, including zeros.
- **Croston:** Separately smooths non-zero demand size and inter-demand interval.
- **SBA:** Applies a bias-reducing correction to the Croston forecast.
- **TSB:** Separately updates the probability of demand occurrence and the average non-zero demand size. It can adapt when demand occurrence becomes more or less frequent.

Croston and SBA are useful methods to understand before moving to TSB and other intermittent-demand approaches.

---

## 7. Evaluation considerations

Intermittent demand contains many zeros, so metric selection matters.

### Mean Absolute Error (MAE)

$$
MAE = \frac{1}{n}\sum_{t=1}^{n}|y_t-\hat{y}_t|
$$

Measures average absolute forecast error.

### Root Mean Squared Error (RMSE)

$$
RMSE = \sqrt{\frac{1}{n}\sum_{t=1}^{n}(y_t-\hat{y}_t)^2}
$$

Penalizes larger errors more strongly.

### Weighted Absolute Percentage Error (WMAPE)

$$
WMAPE = \frac{\sum_{t=1}^{n}|y_t-\hat{y}_t|}
{\sum_{t=1}^{n}|y_t|}
$$

WMAPE can be useful for volume-weighted evaluation, but it is undefined when total actual demand is zero. It can also hide poor performance on low-volume items.

### Important metric note

MAPE is generally unsuitable when actual values contain zeros because division by zero occurs. Consider MAE, RMSE, WMAPE (when defined), service-level measures, or inventory-cost-based metrics according to the business objective.

Use chronological train, validation, and test splits. Do not randomly shuffle time-series observations.

---

## 8. Practical considerations

1. **Data granularity:** Choose a consistent daily, weekly, or monthly interval.
2. **True zeros:** Distinguish genuine zero demand from missing data.
3. **Stockouts:** Recorded sales during stockouts may understate true demand.
4. **Initialization:** Different implementations can use different initial values and interval conventions. Document the convention when comparing results.
5. **Smoothing parameter:** Tune alpha using validation data.
6. **Forecast horizon:** Croston and SBA generate a constant per-period rate; aggregate forecasts over the required planning horizon when appropriate.
7. **Item segmentation:** Evaluate results across demand patterns and item groups rather than relying only on an overall metric.
8. **Business context:** Forecast accuracy should be considered alongside inventory availability, service levels, lead times, and holding costs.

---


### Croston notebook

- Synthetic intermittent-demand generation.
- Demand visualization.
- Croston's algorithm implementation.
- Demand-size and interval estimates.
- Fitted-rate visualization.
- Future forecasting.
- Baseline comparison.
- Holdout evaluation.
- Alpha experimentation.
- Takeaways.

Python Implementation: [Notebook Link](https://github.com/ashay-thamankar/time_series_models/blob/main/Croston_and_SBA/Croston_Method_Forecasting.ipynb)

### SBA notebook

- Synthetic intermittent-demand generation.
- SBA algorithm implementation.
- Croston and SBA forecast comparison.
- Correction-factor explanation.
- Fitted-rate visualization.
- Future forecasting.
- Holdout evaluation.
- Alpha experimentation.
- Takeaways.

Python Implementation: [Notebook Link](https://github.com/ashay-thamankar/time_series_models/blob/main/Croston_and_SBA/SBA_Method_Forecasting.ipynb)


## 9. References and further learning

- [Forecasting: Principles and Practice](https://otexts.com/fpp3/)
- [Croston's method – Forecasting: Principles and Practice](https://otexts.com/fpp3/counts.html)
- [Syntetos and Boylan (2005), The accuracy of intermittent demand estimates](https://doi.org/10.1016/j.ijpe.2004.07.001)
- [Pandas documentation](https://pandas.pydata.org/docs/)
- [NumPy documentation](https://numpy.org/doc/)

---

## 10. Key takeaways

- Intermittent demand has many zero periods and occasional non-zero demand.
- Croston's method separately smooths non-zero demand size and inter-demand intervals.
- Its forecast is the ratio of the two smoothed estimates.
- SBA applies the correction factor \(1-\alpha/2\) to reduce Croston's bias under its theoretical assumptions.
- Neither method guarantees the best result for every intermittent-demand series.
- Always validate using chronological data splits and metrics appropriate for the business problem.

**Happy Learning! 🚀**
