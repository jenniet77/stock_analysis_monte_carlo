import streamlit as st
from yahoo_fin import stock_info as si
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from matplotlib.lines import Line2D


def monte_carlo_simulation(ticker='SMCI'):
    st.title('Monte Carlo Simulation for Target-Based Investing')

    # Get data
    st.subheader('1. Historical Data Analysis')

    # Get today's date
    end_date = datetime.datetime.now() - datetime.timedelta(days=1)
    # Calculate the start date (1 year ago)
    start_date = end_date - datetime.timedelta(days=365)

    # Add a stock ticker input field
    ticker = st.text_input('Enter Stock Ticker:', value=ticker)

    if st.button('Run Simulation'):
        try:
            with st.spinner(f'Fetching historical data for {ticker}...'):
                # Download the data
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
                }

                data = si.get_data(
                    ticker,
                    start_date=start_date,
                    end_date=end_date,
                    headers=headers
                )

                if data.empty:
                    st.error(f"No data found for ticker {ticker}. Please check the symbol and try again.")
                    return

                # Display the dataframe
                st.write("Historical Data:")
                st.dataframe(data.head())

                # Calculate metrics and historical percentiles
                prices = data['adjclose'].values
                log_returns = np.log(prices[1:] / prices[:-1])

                # Calculate the mean and standard deviation of log returns
                mu = np.mean(log_returns)
                sigma = np.std(log_returns)

                # Create metrics display
                col1, col2 = st.columns(2)
                col1.metric("Mean of Log Returns (μ)", f"{mu:.6f}")
                col2.metric("Standard Deviation (σ)", f"{sigma:.6f}")

                # Calculate the percentiles for entry and exit points
                P10 = np.percentile(prices, 10)
                P50 = np.percentile(prices, 50)
                P75 = np.percentile(prices, 75)

                # Display target prices
                st.subheader('2. Target Prices')
                col1, col2, col3 = st.columns(3)
                col1.metric("Entry Point (P10)", f"${P10:.2f}")
                col2.metric("Partial Exit (P50)", f"${P50:.2f}")
                col3.metric("Final Exit (P75)", f"${P75:.2f}")

                # Plot the historical prices with percentile lines
                st.subheader('3. Historical Prices with Target Levels')
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(data.index, data['adjclose'], label=f'{ticker} Adj Close')
                ax.axhline(y=P10, color='g', linestyle='-')
                ax.axhline(y=P50, color='y', linestyle='-')
                ax.axhline(y=P75, color='r', linestyle='-')

                # Create custom legend
                legend_elements = [
                    Line2D([0], [0], color='blue', lw=2, label=f'{ticker} Adj Close'),
                    Line2D([0], [0], color='g', lw=2, label=f'P10 (Entry): ${P10:.2f}'),
                    Line2D([0], [0], color='y', lw=2, label=f'P50 (Partial Exit): ${P50:.2f}'),
                    Line2D([0], [0], color='r', lw=2, label=f'P75 (Final Exit): ${P75:.2f}')
                ]
                ax.legend(handles=legend_elements)

                ax.set_title(f'{ticker} Historical Prices with Target Levels')
                ax.set_xlabel('Date')
                ax.set_ylabel('Price ($)')
                ax.grid(True)
                st.pyplot(fig)

                # Run the Monte Carlo simulation
                st.subheader('4. Monte Carlo Simulation')

                # Define simulation parameters
                days = 90  # time horizon
                dt = 1 / 252  # daily time step (1/252 trading days in a year)
                num_simulations = 1000  # number of simulations (reduced for streamlit)

                # Get the most recent price as the starting point
                S0 = prices[-1]

                # Progress bar for simulation
                progress_bar = st.progress(0)
                status_text = st.empty()

                # Initialize the array to store all simulation paths
                simulation_df = pd.DataFrame()

                # Run the Monte Carlo simulation
                for i in range(num_simulations):
                    # Update progress bar
                    progress = (i + 1) / num_simulations
                    progress_bar.progress(progress)
                    status_text.text(f"Simulation progress: {int(progress * 100)}%")

                    # Generate array of random standard normal values
                    Z = np.random.standard_normal(days)

                    # Initialize price array
                    price_path = np.zeros(days)
                    price_path[0] = S0

                    # Generate the price path
                    for t in range(1, days):
                        price_path[t] = price_path[t - 1] * np.exp(
                            (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z[t])

                    # Store the simulation path
                    simulation_df[f'Sim_{i}'] = price_path

                status_text.text("Simulation completed!")

                # Calculate final prices for each simulation
                final_prices = simulation_df.iloc[-1].values

                # Calculate the probability of reaching the targets
                prob_P50 = np.mean(final_prices >= P50)
                prob_P75 = np.mean(final_prices >= P75)

                # Display probabilities
                st.subheader('5. Simulation Results')
                col1, col2 = st.columns(2)
                col1.metric("Probability of reaching P50", f"{prob_P50:.2%}")
                col2.metric("Probability of reaching P75", f"{prob_P75:.2%}")

                # Plot the Monte Carlo simulation paths
                st.subheader('6. Monte Carlo Simulation Paths')
                fig, ax = plt.subplots(figsize=(10, 6))

                # Plot a subset of paths for clarity
                subset_size = min(100, num_simulations)
                subset_indices = np.random.choice(num_simulations, subset_size, replace=False)
                for i in subset_indices:
                    ax.plot(simulation_df[f'Sim_{i}'], 'b-', alpha=0.1)

                # Plot the percentile lines
                ax.axhline(y=P10, color='g', linestyle='-')
                ax.axhline(y=P50, color='y', linestyle='-')
                ax.axhline(y=P75, color='r', linestyle='-')
                ax.axhline(y=S0, color='k', linestyle='--')

                # Create custom legend
                legend_elements = [
                    Line2D([0], [0], color='blue', lw=2, alpha=0.5, label='Simulation Paths'),
                    Line2D([0], [0], color='g', lw=2, label=f'P10 (Entry): ${P10:.2f}'),
                    Line2D([0], [0], color='y', lw=2, label=f'P50 (Partial Exit): ${P50:.2f}'),
                    Line2D([0], [0], color='r', lw=2, label=f'P75 (Final Exit): ${P75:.2f}'),
                    Line2D([0], [0], color='k', linestyle='--', lw=2, label=f'Current Price: ${S0:.2f}')
                ]
                ax.legend(handles=legend_elements)

                ax.set_title(f'Monte Carlo Simulation - {ticker} Stock Price Over {days} Days')
                ax.set_xlabel('Days')
                ax.set_ylabel('Price ($)')
                ax.grid(True)
                st.pyplot(fig)

                # Plot the histogram of final prices
                st.subheader('7. Distribution of Final Prices')
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.hist(final_prices, bins=50, alpha=0.7)
                ax.axvline(x=P10, color='g', linestyle='-')
                ax.axvline(x=P50, color='y', linestyle='-')
                ax.axvline(x=P75, color='r', linestyle='-')
                ax.axvline(x=S0, color='k', linestyle='--')

                # Create custom legend
                legend_elements = [
                    Line2D([0], [0], color='g', lw=2, label=f'P10 (Entry): ${P10:.2f}'),
                    Line2D([0], [0], color='y', lw=2, label=f'P50 (Partial Exit): ${P50:.2f}'),
                    Line2D([0], [0], color='r', lw=2, label=f'P75 (Final Exit): ${P75:.2f}'),
                    Line2D([0], [0], color='k', linestyle='--', lw=2, label=f'Current Price: ${S0:.2f}')
                ]
                ax.legend(handles=legend_elements)

                ax.set_title(f'Distribution of Simulated {ticker} Prices After {days} Days')
                ax.set_xlabel('Price ($)')
                ax.set_ylabel('Frequency')
                ax.grid(True)
                st.pyplot(fig)

                # Interpretation
                st.subheader('8. Interpretation')

                interpretation = ""
                if prob_P50 > 0.7:
                    interpretation += f"- **High confidence (P50)**: There is a high probability ({prob_P50:.2%}) of reaching the P50 target, suggesting strong confidence in partial profit-taking.\n\n"
                elif prob_P50 > 0.5:
                    interpretation += f"- **Moderate confidence (P50)**: There is a moderate probability ({prob_P50:.2%}) of reaching the P50 target for partial profit-taking.\n\n"
                else:
                    interpretation += f"- **Low confidence (P50)**: There is a relatively low probability ({prob_P50:.2%}) of reaching the P50 target, suggesting caution with this stock.\n\n"

                if prob_P75 > 0.5:
                    interpretation += f"- **High confidence (P75)**: There is a good probability ({prob_P75:.2%}) of reaching the P75 target, suggesting strong potential for significant profit.\n\n"
                elif prob_P75 > 0.3:
                    interpretation += f"- **Moderate confidence (P75)**: There is a moderate probability ({prob_P75:.2%}) of reaching the P75 target for maximum profit-taking.\n\n"
                else:
                    interpretation += f"- **Low confidence (P75)**: There is a low probability ({prob_P75:.2%}) of reaching the P75 target, suggesting that full profit target might be challenging.\n\n"

                st.markdown(interpretation)

                # Recommendation
                st.subheader('9. Investment Recommendation')

                if prob_P50 > 0.6 and prob_P75 > 0.4:
                    recommendation = "**Strong Buy Recommendation**\n\nThe simulation shows high probabilities of reaching both partial and full profit targets within the 90-day time horizon. This stock presents a favorable risk-reward ratio and should be prioritized for entry when price reaches the P10 level."
                elif prob_P50 > 0.5:
                    recommendation = "**Moderate Buy Recommendation**\n\nThe simulation shows a reasonable probability of reaching at least the partial profit target. Consider allocating a moderate position when price reaches the P10 level, with a plan to take partial profits at P50."
                else:
                    recommendation = "**Cautious Approach Recommended**\n\nThe simulation shows relatively low probabilities of reaching profit targets. Consider alternative investments with better probability profiles or reduce position size if entering at P10 level."

                st.markdown(recommendation)

        except Exception as e:
            st.error(f"An error occurred: {e}")


if __name__ == "__main__":
    monte_carlo_simulation()