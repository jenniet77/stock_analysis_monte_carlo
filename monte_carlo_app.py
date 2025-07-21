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
                st.dataframe(data.tail())

                # Calculate metrics and historical percentiles
                prices = data['adjclose'].values
                log_returns = np.log(prices[1:] / prices[:-1])

                # Calculate the mean and standard deviation of log returns
                mu = np.mean(log_returns)
                sigma = np.std(log_returns)

                # Get current price (latest adjusted close)
                current_price = prices[-1]

                # Calculate what percentile the current price represents
                current_percentile = (np.sum(prices <= current_price) / len(prices)) * 100

                # Create metrics display
                col1, col2, col3 = st.columns(3)
                col1.metric("Mean of Log Returns (μ)", f"{mu:.6f}")
                col2.metric("Standard Deviation (σ)", f"{sigma:.6f}")
                col3.metric("Current Price", f"${current_price:.2f}", f"{current_percentile:.1f}th percentile")

                # Calculate the HISTORICAL percentiles for entry and exit points
                historical_P10 = np.percentile(prices, 10)
                historical_P50 = np.percentile(prices, 50)
                historical_P75 = np.percentile(prices, 75)

                # Display target prices
                st.subheader('2. Historical Target Prices (Strategy Reference)')
                col1, col2, col3 = st.columns(3)
                col1.metric("Entry Point (P10)", f"${historical_P10:.2f}")
                col2.metric("Partial Exit (P50)", f"${historical_P50:.2f}")
                col3.metric("Final Exit (P75)", f"${historical_P75:.2f}")

                # Plot the historical prices with percentile lines
                st.subheader('3. Historical Prices with Target Levels')
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(data.index, data['adjclose'], label=f'{ticker} Adj Close')
                ax.axhline(y=historical_P10, color='g', linestyle='-')
                ax.axhline(y=historical_P50, color='y', linestyle='-')
                ax.axhline(y=historical_P75, color='r', linestyle='-')

                # Create custom legend
                legend_elements = [
                    Line2D([0], [0], color='blue', lw=2, label=f'{ticker} Adj Close'),
                    Line2D([0], [0], color='g', lw=2, label=f'P10 (Entry): ${historical_P10:.2f}'),
                    Line2D([0], [0], color='y', lw=2, label=f'P50 (Partial Exit): ${historical_P50:.2f}'),
                    Line2D([0], [0], color='r', lw=2, label=f'P75 (Final Exit): ${historical_P75:.2f}')
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
                num_simulations = 10000  # number of simulations (reduced for streamlit)

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
                    Z = np.random.normal(0,1,size=(days))

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
                prob_historical_P50 = np.mean(final_prices >= historical_P50)
                prob_historical_P75 = np.mean(final_prices >= historical_P75)

                # Display probabilities
                st.subheader('5. Simulation Results')
                col1, col2 = st.columns(2)
                col1.metric("Probability of reaching Historical P50", f"{prob_historical_P50:.2%}")
                col2.metric("Probability of reaching Historical P75", f"{prob_historical_P75:.2%}")

                # Calculate PREDICTIVE percentiles from simulation results
                predictive_P10 = np.percentile(final_prices, 10)
                predictive_P25 = np.percentile(final_prices, 25)
                predictive_P50 = np.percentile(final_prices, 50)
                predictive_P75 = np.percentile(final_prices, 75)
                predictive_P90 = np.percentile(final_prices, 90)

                # Plot the Monte Carlo simulation paths
                st.subheader('6. Monte Carlo Simulation Paths')
                fig, ax = plt.subplots(figsize=(10, 6))

                # Plot a subset of paths for clarity
                subset_size = min(100, num_simulations)
                subset_indices = np.random.choice(num_simulations, subset_size, replace=False)
                for i in subset_indices:
                    ax.plot(simulation_df[f'Sim_{i}'], 'b-', alpha=0.1)

                # Plot the percentile lines
                ax.axhline(y=historical_P10, color='g', linestyle='--')
                ax.axhline(y=historical_P50, color='y', linestyle='--')
                ax.axhline(y=historical_P75, color='r', linestyle='--')
                ax.axhline(y=predictive_P10, color='g', linestyle='-')
                ax.axhline(y=predictive_P50, color='y', linestyle='-')
                ax.axhline(y=predictive_P75, color='r', linestyle='-')
                ax.axhline(y=S0, color='k', linestyle='--')

                # Create custom legend
                legend_elements = [
                    Line2D([0], [0], color='blue', lw=2, alpha=0.5, label='Simulation Paths'),
                    Line2D([0], [0], color='g', linestyle='--', label=f'P10 (Historical): ${historical_P10:.2f}'),
                    Line2D([0], [0], color='g', lw=2, label=f'P10 (Predictive): ${predictive_P10:.2f}'),
                    Line2D([0], [0], color='y', linestyle='--', label=f'P50 (Historical): ${historical_P50:.2f}'),
                    Line2D([0], [0], color='y', lw=2, label=f'P50 (Predictive): ${predictive_P50:.2f}'),
                    Line2D([0], [0], color='r', linestyle='--', label=f'P75 (Historical): ${historical_P75:.2f}'),
                    Line2D([0], [0], color='r', lw=2, label=f'P75 (Predictive): ${predictive_P75:.2f}'),
                    Line2D([0], [0], color='k', linestyle='--', lw=2, label=f'Current Price: ${S0:.2f}')
                ]
                ax.legend(handles=legend_elements)

                ax.set_title(f'Monte Carlo Simulation - {ticker} Stock Price Over {days} Days')
                ax.set_xlabel('Days')
                ax.set_ylabel('Price ($)')
                ax.grid(True)
                st.pyplot(fig)

                # Plot the histogram of final prices
                st.subheader('7. Distribution of Final Prices with Predictive Targets')
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.hist(final_prices, bins=50, alpha=0.7)
                ax.axvline(x=historical_P10, color='g', linestyle='--')
                ax.axvline(x=historical_P50, color='y', linestyle='--')
                ax.axvline(x=historical_P75, color='r', linestyle='--')
                ax.axvline(x=predictive_P10, color='g', linestyle='-')
                ax.axvline(x=predictive_P50, color='y', linestyle='-')
                ax.axvline(x=predictive_P75, color='r', linestyle='-')
                ax.axvline(x=S0, color='k', linestyle='--')

                # Create custom legend
                legend_elements = [
                    Line2D([0], [0], color='g', linestyle='--', label=f'P10 (Historical): ${historical_P10:.2f}'),
                    Line2D([0], [0], color='g', lw=2, label=f'P10 (Predictive): ${predictive_P10:.2f}'),
                    Line2D([0], [0], color='y', linestyle='--', label=f'P50 (Historical): ${historical_P50:.2f}'),
                    Line2D([0], [0], color='y', lw=2, label=f'P50 (Predictive): ${predictive_P50:.2f}'),
                    Line2D([0], [0], color='r', linestyle='--', label=f'P75 (Historical): ${historical_P75:.2f}'),
                    Line2D([0], [0], color='r', lw=2, label=f'P75 (Predictive): ${predictive_P75:.2f}'),
                    Line2D([0], [0], color='k', linestyle='--', lw=2, label=f'Current Price: ${S0:.2f}')
                ]
                ax.legend(handles=legend_elements)

                ax.set_title(f'Distribution of Simulated {ticker} Prices After {days} Days')
                ax.set_xlabel('Price ($)')
                ax.set_ylabel('Frequency')
                ax.grid(True)
                st.pyplot(fig)

                # Display predictive percentiles in a nice format
                st.subheader('8. Predictive Price Targets from Monte Carlo Simulation')
                st.write("These percentiles represent the price levels that the corresponding percentage of simulations reached or exceeded:")
                
                col1, col2, col3, col4, col5 = st.columns(5)
                col1.metric("P10 (90% reach)", f"${predictive_P10:.2f}")
                col2.metric("P25 (75% reach)", f"${predictive_P25:.2f}")
                col3.metric("P50 (50% reach)", f"${predictive_P50:.2f}")
                col4.metric("P75 (25% reach)", f"${predictive_P75:.2f}")
                col5.metric("P90 (10% reach)", f"${predictive_P90:.2f}")

                # Calculate probabilities for predictive targets
                prob_predictive_P50 = 0.50  # By definition, 50% of simulations reach P50
                prob_predictive_P75 = 0.25  # By definition, 25% of simulations reach P75

                 # Interpretation based on predictive model
                st.subheader('9. Interpretation of Predictive Results')
                
                interpretation = f"""
                **Predictive Analysis Summary:**
                
                - **Expected Median Price (P50)**: ${predictive_P50:.2f} - This is the price level that 50% of simulations reached or exceeded
                - **Conservative Target (P75)**: ${predictive_P75:.2f} - This is the price level that 25% of simulations reached or exceeded
                - **Optimistic Target (P90)**: ${predictive_P90:.2f} - This is the price level that only 10% of simulations reached or exceeded
                
                **Price Movement Expectations:**
                - Starting Price: ${S0:.2f}
                - Expected change to P50: {((predictive_P50/S0 - 1) * 100):+.1f}%
                - Expected change to P75: {((predictive_P75/S0 - 1) * 100):+.1f}%
                
                **Risk Assessment:**
                - Downside Risk (P10): ${predictive_P10:.2f} ({((predictive_P10/S0 - 1) * 100):+.1f}%)
                - Upside Potential (P90): ${predictive_P90:.2f} ({((predictive_P90/S0 - 1) * 100):+.1f}%)
                """
                
                st.markdown(interpretation)
                
                # Investment recommendation based on predictive model
                st.subheader('10. Investment Recommendation')
                
                expected_return_p50 = (predictive_P50/S0 - 1) * 100
                expected_return_p75 = (predictive_P75/S0 - 1) * 100
                downside_risk = (predictive_P10/S0 - 1) * 100
                
                if expected_return_p50 > 10 and downside_risk > -20:
                    recommendation = f"""
                    **Strong Buy Recommendation**
                    
                    The Monte Carlo simulation suggests favorable risk-reward characteristics:
                    - Expected median return: {expected_return_p50:+.1f}%
                    - Conservative target return: {expected_return_p75:+.1f}%
                    - Limited downside risk: {downside_risk:+.1f}%
                    
                    Consider this stock for target-based investing with:
                    - Entry strategy: Current price levels
                    - Partial profit taking: Around ${predictive_P50:.2f} (50% probability)
                    - Full exit target: Around ${predictive_P75:.2f} (25% probability)
                    """
                elif expected_return_p50 > 0:
                    recommendation = f"""
                    **Moderate Buy Recommendation**
                    
                    The simulation shows modest positive expectations:
                    - Expected median return: {expected_return_p50:+.1f}%
                    - Conservative target return: {expected_return_p75:+.1f}%
                    - Downside risk: {downside_risk:+.1f}%
                    
                    Consider a moderate position with careful risk management.
                    """
                else:
                    recommendation = f"""
                    **Cautious Approach Recommended**
                    
                    The simulation indicates challenging conditions:
                    - Expected median return: {expected_return_p50:+.1f}%
                    - Potential downside: {downside_risk:+.1f}%
                    
                    Consider waiting for better entry opportunities or exploring alternative investments.
                    """
                
                st.markdown(recommendation)
                
                # Additional statistics
                st.subheader('11. Additional Statistics')
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Mean Final Price", f"${np.mean(final_prices):.2f}")
                col2.metric("Std Dev Final Price", f"${np.std(final_prices):.2f}")
                col3.metric("Min Simulated Price", f"${np.min(final_prices):.2f}")
                col4.metric("Max Simulated Price", f"${np.max(final_prices):.2f}")
        
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.write("Please check the stock ticker and try again.")

if __name__ == "__main__":
    monte_carlo_simulation()