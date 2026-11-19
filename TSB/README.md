# TSB (Teunter–Syntetos–Babai) Method for Intermittent Demand Forecasting

## 📌 Introduction

Intermittent demand occurs when a product experiences many periods with zero demand and occasional periods with non-zero demand. This pattern is common in:

- Spare parts and service parts
- Slow-moving inventory
- Maintenance and repair materials
- Low-volume products
- Products with irregular purchasing patterns

Traditional forecasting methods can struggle with intermittent demand because the demand pattern contains many zeros. Croston's method addresses this by estimating demand size and the interval between demand occurrences separately. The **Teunter–Syntetos–Babai (TSB) method** builds on this idea by estimating the probability of demand occurrence directly.

The key distinction is that **TSB updates the demand-occurrence probability in every period, including periods with zero demand**. As a result, its forecast can gradually decrease when demand becomes less frequent. This makes TSB particularly relevant when a product may be approaching obsolescence.

This repository includes a standalone Jupyter Notebook with a from-scratch Python implementation.

---

## 🎯 Learning Objectives

By working through the notebook, you will learn how to:

- Understand intermittent demand and its characteristics.
- Explain the two components maintained by TSB.
- Implement TSB using Python, NumPy, and Pandas.
- Generate and visualize synthetic intermittent-demand data.
- Calculate one-step-ahead fitted forecasts.
- Generate multi-period forecasts.
- Evaluate forecasts using MAE, RMSE, and WMAPE.
- Understand the impact of the smoothing parameters.
- Compare TSB with Croston's method and the SBA method.

---

## 1. Understanding the TSB Method

TSB separates intermittent demand into two components:

1. **Demand occurrence probability** – How likely is demand to occur in a period?
2. **Demand size** – How much demand is expected when demand occurs?

The expected demand per period is calculated by multiplying these two estimates.

### 1.1 Demand Occurrence Probability

Let:

- `y_t` = actual demand in period `t`
- `I_t` = demand occurrence indicator
- `p_hat_t` = smoothed probability of demand occurrence
- `beta` = smoothing parameter for occurrence probability

The occurrence indicator is:

- `I_t = 1` when demand is non-zero.
- `I_t = 0` when demand is zero.

The probability estimate is updated in every period:

$$
\hat{p}_t = \beta I_t + (1-\beta)\hat{p}_{t-1}
$$

**What happens?**

- When demand occurs, the occurrence probability moves upward.
- When demand is zero, the occurrence probability moves downward.
- The size of the adjustment depends on `beta`.

This is the most important feature distinguishing TSB from Croston and SBA.

### 1.2 Smoothed Demand Size

Let:

- `z_hat_t` = smoothed non-zero demand size
- `alpha` = smoothing parameter for demand size

When demand is non-zero:

$$
\hat{z}_t = \alpha y_t + (1-\alpha)\hat{z}_{t-1}
$$

When demand is zero:

$$
\hat{z}_t = \hat{z}_{t-1}
$$

The demand-size estimate is updated only when a non-zero demand observation occurs.

### 1.3 TSB Forecast

The TSB forecast is the product of the two smoothed components:

$$
\boxed{\hat{y}_{t+1} = \hat{p}_t \times \hat{z}_t}
$$

In words:

**TSB Forecast = Demand Occurrence Probability × Smoothed Non-Zero Demand Size**

The result is the expected demand per period.

---

## 2. Understanding the Smoothing Parameters

TSB uses two smoothing parameters.

| Parameter | Purpose | Higher value | Lower value |
|---|---|---|---|
| `alpha` | Smooths non-zero demand size | Reacts faster to new demand sizes | Produces a smoother size estimate |
| `beta` | Smooths occurrence probability | Reacts faster to recent zero/non-zero observations | Changes occurrence probability more gradually |

Both parameters generally lie between 0 and 1.

Their values should be selected using validation data or rolling-origin backtesting. There is no single combination that works best for every intermittent-demand series.

---

## 3. Worked Example

Consider the following demand observations:

| Period | Actual Demand |
|---:|---:|
| 1 | 0 |
| 2 | 12 |
| 3 | 0 |
| 4 | 0 |
| 5 | 20 |
| 6 | 0 |

Assume the initial estimates are:

- Initial occurrence probability = 0.50
- Initial demand size = 12
- `alpha = 0.20`
- `beta = 0.10`

### Period 1: Demand = 0

Occurrence probability:

$$
\hat{p}_1 = 0.10(0) + 0.90(0.50) = 0.45
$$

Demand size remains unchanged:

$$
\hat{z}_1 = 12
$$

The updated expected demand rate is:

$$
0.45 \times 12 = 5.40
$$

### Period 2: Demand = 12

Occurrence probability:

$$
\hat{p}_2 = 0.10(1) + 0.90(0.45) = 0.505
$$

Demand size:

$$
\hat{z}_2 = 0.20(12) + 0.80(12) = 12
$$

Updated expected demand rate:

$$
0.505 \times 12 = 6.06
$$

### Period 3: Demand = 0

Occurrence probability decreases:

$$
\hat{p}_3 = 0.10(0) + 0.90(0.505) = 0.4545
$$

Demand size remains 12.

Updated expected demand rate:

$$
0.4545 \times 12 = 5.454
$$

The same process continues for every period. Notice how zero-demand periods reduce the occurrence probability, while the demand-size estimate remains unchanged.

---

## 4. Python Implementation

The notebook implements TSB from scratch rather than relying on a pre-built forecasting library.

### Main implementation steps

1. Convert the demand series into a NumPy array.
2. Initialize the occurrence probability and non-zero demand size.
3. Calculate the one-step-ahead forecast using the current estimates.
4. Update occurrence probability in every period.
5. Update demand size only when actual demand is non-zero.
6. Store the estimates and return the final state.

### Core Python logic

```python
for actual in demand:
    # Forecast using estimates available before observing this period
    forecast = p_hat * z_hat

    # Demand occurrence indicator
    occurrence = 1 if actual > 0 else 0

    # Update occurrence probability every period
    p_hat = beta * occurrence + (1 - beta) * p_hat

    # Update demand size only when demand is non-zero
    if actual > 0:
        z_hat = alpha * actual + (1 - alpha) * z_hat
```

The notebook also includes input validation, fitted-value generation, final-state forecasting, visualizations, and evaluation functions.

### Initialization note

Initialization can affect the resulting forecast, especially for short or sparse series. In the educational notebook:

- The initial demand-size estimate is the first non-zero demand observation.
- The initial occurrence probability is the first-period occurrence indicator. If the first observation is zero, the overall occurrence rate is used.

Other initialization approaches can be used, so document and validate the convention when comparing implementations.

---

## 5. One-Step-Ahead Fitted Forecasts vs Future Forecasts

It is important to distinguish these two outputs.

### One-step-ahead fitted forecasts

For each historical period, the notebook calculates a forecast using the estimates available before observing that period's actual demand. The estimates are then updated using the actual observation.

This avoids using the current observation to forecast itself.

### Future multi-step forecasts

At the end of the training series, the latest occurrence probability and demand-size estimate are used to calculate the forecast.

When no future observations are available to update the model, the TSB point forecast remains constant across the forecast horizon:

$$
\hat{y}_{T+h} = \hat{p}_T \times \hat{z}_T
$$

for future horizon `h = 1, 2, ...`.

---

## 6. Forecast Evaluation

The notebook uses a chronological train-test split:

- Earlier observations are used to estimate the model.
- Later observations are held out for evaluation.
- The final TSB state from the training period is used to forecast the holdout horizon.

The following metrics are calculated:

### Mean Absolute Error (MAE)

$$
MAE = \frac{1}{n}\sum_{t=1}^{n}|y_t-\hat{y}_t|
$$

MAE measures the average absolute forecast error.

### Root Mean Squared Error (RMSE)

$$
RMSE = \sqrt{\frac{1}{n}\sum_{t=1}^{n}(y_t-\hat{y}_t)^2}
$$

RMSE penalizes larger errors more heavily.

### Weighted Mean Absolute Percentage Error (WMAPE)

$$
WMAPE = \frac{\sum_{t=1}^{n}|y_t-\hat{y}_t|}{\sum_{t=1}^{n}|y_t|}\times 100
$$

WMAPE expresses total absolute error relative to total actual demand.

**Why is MAPE not included?**

Intermittent-demand series contain many zero actual values. MAPE divides each absolute error by the actual value, so it is undefined when actual demand is zero and can become misleading when actual demand is very small. MAE, RMSE, and WMAPE are more practical choices here, although every metric should be interpreted in the context of the business objective.

---

## 7. TSB vs Croston's Method vs SBA

All three methods are designed for intermittent demand, but they estimate demand frequency differently.

| Aspect | Croston's Method | SBA Method | TSB Method |
|---|---|---|---|
| Demand-size estimate | Smoothed non-zero demand size | Same as Croston | Smoothed non-zero demand size |
| Frequency component | Smoothed interval between non-zero demands | Same as Croston | Smoothed probability of demand occurrence |
| Frequency update | When demand occurs | When demand occurs | Every period |
| Bias adjustment | No SBA correction | Applies a correction factor | No SBA correction factor |
| Effect of consecutive zero periods | Interval estimate is not updated until demand occurs again | Same as Croston | Occurrence probability declines over zero periods |
| Useful characteristic | Separately estimates size and interval | Bias-adjusted Croston forecast | Can adapt when demand becomes less frequent |

### When can TSB be useful?

TSB can be particularly useful when the probability of demand occurrence changes over time. For example, if a spare part has repeated periods with no demand, its estimated occurrence probability gradually decreases.

This makes TSB relevant for products that may be approaching obsolescence.

However, a declining forecast is not proof of obsolescence. Business information such as product lifecycle, installed base, replacement plans, and inventory policy should also be considered.

---

## 8. Advantages of TSB

- **Handles intermittent demand:** Designed for series with many zero-demand periods.
- **Separates occurrence and size:** Models how often demand occurs independently from how much is demanded.
- **Updates every period:** The occurrence probability responds to both zero and non-zero observations.
- **Can adapt to declining demand frequency:** Consecutive zero-demand periods gradually reduce the occurrence probability.
- **Computationally efficient:** The method is straightforward to implement and maintain.

---

## 9. Limitations of TSB

- **No explicit trend or seasonality:** Standard TSB does not directly model trend, seasonal patterns, or causal drivers.
- **Parameter sensitivity:** Results depend on the choice of `alpha` and `beta`.
- **Initialization sensitivity:** Initial probability and demand-size estimates can influence forecasts.
- **Constant future point forecast:** Without new observations, the standard multi-step point forecast remains constant.
- **Obsolescence is not identified with certainty:** A declining occurrence probability may indicate changing demand, but it cannot independently confirm that a product is obsolete.

---

## 10. Practical Considerations for Supply Chain Forecasting

When applying TSB to real demand data:

1. **Choose the right time bucket:** Daily, weekly, or monthly demand should match the replenishment and planning process.
2. **Use the appropriate forecasting level:** Item-location combinations are often useful for intermittent-demand analysis.
3. **Tune both parameters:** Use a validation period or rolling-origin backtesting to select `alpha` and `beta`.
4. **Compare with other methods:** Evaluate TSB against Croston, SBA, and simple baselines.
5. **Select suitable metrics:** Avoid relying on MAPE when actual demand contains zeros.
6. **Consider business context:** Lead time, lifecycle, minimum order quantity, installed base, and obsolescence can affect the final planning decision.
7. **Review very long zero-demand runs:** Check whether they reflect true demand inactivity, data issues, stockouts, or product discontinuation.

A forecasting model should support the planning decision, not replace business understanding.

---

## 11. Repository Contents


The notebook includes:
- Synthetic intermittent-demand data generation
- Demand pattern exploration
- TSB implementation from scratch
- Step-by-step update example
- Fitted forecast visualization
- Occurrence-probability and demand-size plots
- Chronological train-test evaluation
- Smoothing-parameter experiments
- Comparison with Croston and SBA
- Key takeaways

Python Implementation: [Notebook Link](https://github.com/ashay-thamankar/time_series_models/blob/main/TSB/TSB_Method_Forecasting.ipynb)

---

## 💡 Key Takeaways

- TSB separates demand occurrence probability from non-zero demand size.
- Occurrence probability is updated in every period, including zero-demand periods.
- Demand size is updated only when demand is non-zero.
- The forecast is the product of these two estimates.
- TSB can adapt when demand becomes less frequent, making it useful for some slow-moving and potentially obsolescent products.

**The key idea: TSB does not just estimate how much demand occurs; it also keeps learning how likely demand is to occur in each period.**

---

## 📚 References

1. Teunter, R. H., Syntetos, A. A., & Babai, M. Z. (2011). *Intermittent demand: Linking forecasting to inventory obsolescence*. European Journal of Operational Research.
2. Hyndman, R. J., & Athanasopoulos, G. *Forecasting: Principles and Practice*. https://otexts.com/fpp3/
3. Syntetos, A. A., & Boylan, J. E. (2005). *The accuracy of intermittent demand estimates*. International Journal of Forecasting.
