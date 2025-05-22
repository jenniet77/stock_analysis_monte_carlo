# Monte Carlo Simulation for Target-Based Investing: SMCI Stock Analysis

🔗 App link: https://upachimontecarlo.streamlit.app/ (try simulating other tickers)

## Introduction
This report presents a detailed implementation of Monte Carlo simulation for target-based investing, using Super Micro Computer Inc. (SMCI) stock as a case study. The target-based investing strategy involves setting specific entry and exit points based on historical price percentiles, providing a systematic approach to investment decisions.

Monte Carlo simulation is a computational technique that uses random sampling to obtain numerical results. In the context of stock investing, it allows us to model the uncertainty of future price movements by generating thousands of possible price paths based on historical volatility and returns. This approach provides a probabilistic framework for evaluating investment strategies and making data-driven decisions.

## Strategy and Rules
Our target-based investing strategy follows these specific rules:
- Entry point: Buy at P10 – the 10th percentile of 1-year historical price data
- Partial exit: Sell at P50 – take profits at the 50th percentile (median)
- Final exit: Sell at P75 – exit all holdings at the 75th percentile
- Time horizon: 90 days from entry

<br>This strategy aims to:
- Enter positions at relatively low prices (P10)
- Take partial profits at median prices (P50)
- Maximize returns by setting a realistic upper target (P75)
- Limit exposure time to 90 days to manage opportunity cost
<br>By using historical percentiles rather than arbitrary price levels, this approach adapts to each stock's specific price distribution and volatility characteristics.

## Key Historical Metrics
- Mean of log returns (μ): Average daily logarithmic return
- Standard deviation (σ): Volatility measure
- Current Price (S0): Latest closing price

## Monte Carlo Simulation Methodology
Our simulation is based on the geometric Brownian motion model, which is widely used to model stock price movements. The model assumes that stock prices follow a log-normal distribution and can be described by the following stochastic differential equation:

![Screen Shot 2025-05-22 at 4 55 41 pm](https://github.com/user-attachments/assets/08f652e6-512a-42f8-9c95-9c859297e81c)

Where:
**St** is the stock price at time t
**μ** is the mean of log returns (drift coefficient)
**σ** is the standard deviation of log returns (volatility)
**Δt** is the time step (1/252 for daily steps, assuming 252 trading days in a year)
**Z** is a random standard normal value (representing random market movements)

This equation captures both the deterministic trend component (μ - 0.5σ²) and the random shock component (σ · √Δt · Z) of stock price movements. By generating multiple price paths using this equation, we can estimate the probability distribution of future stock prices and evaluate the likelihood of reaching our target prices within the specified time horizon.

## Simulation Results
We ran 1,000 simulations of SMCI's stock price over a 90-day period. The results provide insights into the potential future price movements and the probability of reaching our target levels.
![monte_carlo_simulation](https://github.com/user-attachments/assets/94fa3da9-c8b4-4f6c-803f-dddd28fa3b04)
![final_price_distribution](https://github.com/user-attachments/assets/fdf3517c-0de0-4d33-9a14-5e01212b9486)
Probability of reaching P50 ($41.88): 42.00% | 
Probability of reaching P75 ($56.62): 0.00%

### Interpretation
- Moderate Probability of Reaching P50: There is a 43.30% probability of the stock price reaching the P50 level ($41.88) within the 90-day time horizon. This suggests a reasonable chance of achieving partial profits if entering at the P10 level.
- Low Probability of Reaching P75: The simulation indicates a 0.00% probability of reaching the P75 level ($56.62) within 90 days. This suggests that the final exit target may be too ambitious given the current market conditions and historical volatility of SMCI.
- Current Price Considerations: The current price ($41.65) is already very close to the P50 level ($41.88), which means the stock is currently trading near its median historical price. This suggests that waiting for a pullback to the P10 level ($30.03) before entering would be consistent with our strategy.

**⚠️ Disclaimer: This project is intended solely for educational and case study purposes. It does not reflect current market conditions and should not be considered financial advice.**

## References
Hull, J. C. (2017). Options, Futures, and Other Derivatives (10th ed.). Pearson.
Glasserman, P. (2003). Monte Carlo Methods in Financial Engineering. Springer.
Yahoo Finance. (n.d.). Super Micro Computer, Inc. (SMCI) Stock Data. https://finance.yahoo.com/quote/SMCI/
Python Software Foundation. (n.d.). Python Language Reference, version 3.9. https://www.python.org/
McKinney, W. (2018). Python for Data Analysis (2nd ed.). O'Reilly Media.
