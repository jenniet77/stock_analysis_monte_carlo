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
    
    # Add sidebar controls for display options
    st.sidebar.header("Display Options")
    display_mode = st.sidebar.selectbox(
        "Choose percentile display:",
        ["Predictive Only (PNew)", "Historical Only (POld)", "Both for Comparison"],
        index=0  # Default to Predictive Only
    )

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

                # Create metrics display
                col1, col2 = st.columns(2)
                col1.metric("Mean of Log Returns (μ)", f"{mu:.6f}")
                col2.metric("Standard Deviation (σ)", f"{sigma:.6f}")

                # Calculate the historical percentiles
                POld10 = np.percentile(prices, 10)
                POld50 = np.percentile(prices, 50)
                POld75 = np.percentile(prices, 75)

                # Plot the historical prices with historical percentile lines
                st.subheader('2. Historical Prices with Historical Target Levels')
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(data.index, data['adjclose'], label=f'{ticker} Adj Close')
                ax.axhline(y=POld10, color='g', linestyle='-')
                ax.axhline(y=POld50, color='y', linestyle='-')
                ax.axhline(y=POld75, color='r', linestyle='-')

                # Create custom legend
                legend_elements = [
                    Line2D([0], [0], color='blue', lw=2, label=f'{ticker} Adj Close'),
                    Line2D([0], [0], color='g', lw=2, label=f'POld10 (Historical): ${POld10:.2f}'),
                    Line2D([0], [0], color='y', lw=2, label=f'POld50 (Historical): ${POld50:.2f}'),
                    Line2D([0], [0], color='r', lw=2, label=f'POld75 (Historical): ${POld75:.2f}')
                ]
                ax.legend(handles=legend_elements)

                ax.set_title(f'{ticker} Historical Prices with Historical Target Levels')
                ax.set_xlabel('Date')
                ax.set_ylabel('Price ($)')
                ax.grid(True)
                st.pyplot(fig)

                # Run the Monte Carlo simulation
                st.subheader('3. Monte Carlo Simulation')

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

                # Calculate the predictive percentiles from simulation results
                PNew10 = np.percentile(final_prices, 10)
                PNew50 = np.percentile(final_prices, 50)
                PNew75 = np.percentile(final_prices, 75)

                # Display target prices based on selected mode
                st.subheader('4. Target Prices')
                
                if display_mode == "Predictive Only (PNew)":
                    col1, col2, col3 = st.columns(3)
                    col1.metric("PNew10 (Predicted 10th %ile)", f"${PNew10:.2f}")
                    col2.metric("PNew50 (Predicted 50th %ile)", f"${PNew50:.2f}")
                    col3.metric("PNew75 (Predicted 75th %ile)", f"${PNew75:.2f}")
                    
                elif display_mode == "Historical Only (POld)":
                    col1, col2, col3 = st.columns(3)
                    col1.metric("POld10 (Historical 10th %ile)", f"${POld10:.2f}")
                    col2.metric("POld50 (Historical 50th %ile)", f"${POld50:.2f}")
                    col3.metric("POld75 (Historical 75th %ile)", f"${POld75:.2f}")
                    
                else:  # Both for Comparison
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Historical Targets (POld)**")
                        st.metric("POld10 (Historical 10th %ile)", f"${POld10:.2f}")
                        st.metric("POld50 (Historical 50th %ile)", f"${POld50:.2f}")
                        st.metric("POld75 (Historical 75th %ile)", f"${POld75:.2f}")
                    
                    with col2:
                        st.write("**Predictive Targets (PNew)**")
                        st.metric("PNew10 (Predicted 10th %ile)", f"${PNew10:.2f}")
                        st.metric("PNew50 (Predicted 50th %ile)", f"${PNew50:.2f}")
                        st.metric("PNew75 (Predicted 75th %ile)", f"${PNew75:.2f}")

                # Calculate probabilities for both historical and predictive targets
                prob_POld50 = np.mean(final_prices >= POld50)
                prob_POld75 = np.mean(final_prices >= POld75)
                prob_PNew50 = 0.50  # By definition
                prob_PNew75 = 0.25  # By definition
                prob_profit = np.mean(final_prices > S0)  # Probability of any profit

                # Display probabilities based on selected mode
                st.subheader('5. Simulation Results')
                
                if display_mode == "Predictive Only (PNew)":
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Probability of Profit", f"{prob_profit:.2%}")
                    col2.metric("Expected Return", f"{((np.mean(final_prices)/S0 - 1)*100):+.1f}%")
                    col3.metric("Simulations with Gain", f"{int(prob_profit * num_simulations)}/{num_simulations}")
                    
                    st.write("**Predictive Target Achievement (by definition):**")
                    col1, col2 = st.columns(2)
                    col1.write(f"• 50% of simulations reach PNew50 (${PNew50:.2f})")
                    col2.write(f"• 25% of simulations reach PNew75 (${PNew75:.2f})")
                    
                elif display_mode == "Historical Only (POld)":
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Probability of Profit", f"{prob_profit:.2%}")
                    col2.metric("Prob. of reaching POld50", f"{prob_POld50:.2%}")
                    col3.metric("Prob. of reaching POld75", f"{prob_POld75:.2%}")
                    
                else:  # Both for Comparison
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Probability of Profit", f"{prob_profit:.2%}")
                    col2.metric("Prob. of reaching POld50", f"{prob_POld50:.2%}")
                    col3.metric("Prob. of reaching POld75", f"{prob_POld75:.2%}")
                    
                    st.write("**Note:** PNew percentiles represent predicted price levels by definition:")
                    col1, col2 = st.columns(2)
                    col1.write(f"• 50% of simulations reach PNew50 (${PNew50:.2f})")
                    col2.write(f"• 25% of simulations reach PNew75 (${PNew75:.2f})")

                # Plot the Monte Carlo simulation paths
                st.subheader('6. Monte Carlo Simulation Paths')
                fig, ax = plt.subplots(figsize=(10, 6))

                # Plot a subset of paths for clarity
                subset_size = min(100, num_simulations)
                subset_indices = np.random.choice(num_simulations, subset_size, replace=False)
                for i in subset_indices:
                    ax.plot(simulation_df[f'Sim_{i}'], 'b-', alpha=0.1)

                # Plot percentile lines based on display mode
                legend_elements = [Line2D([0], [0], color='blue', lw=2, alpha=0.5, label='Simulation Paths')]
                
                if display_mode == "Predictive Only (PNew)":
                    ax.axhline(y=PNew10, color='g', linestyle='-')
                    ax.axhline(y=PNew50, color='y', linestyle='-')
                    ax.axhline(y=PNew75, color='r', linestyle='-')
                    legend_elements.extend([
                        Line2D([0], [0], color='g', lw=2, label=f'PNew10 (Predicted): ${PNew10:.2f}'),
                        Line2D([0], [0], color='y', lw=2, label=f'PNew50 (Predicted): ${PNew50:.2f}'),
                        Line2D([0], [0], color='r', lw=2, label=f'PNew75 (Predicted): ${PNew75:.2f}')
                    ])
                    
                elif display_mode == "Historical Only (POld)":
                    ax.axhline(y=POld10, color='g', linestyle='-')
                    ax.axhline(y=POld50, color='y', linestyle='-')
                    ax.axhline(y=POld75, color='r', linestyle='-')
                    legend_elements.extend([
                        Line2D([0], [0], color='g', lw=2, label=f'POld10 (Historical): ${POld10:.2f}'),
                        Line2D([0], [0], color='y', lw=2, label=f'POld50 (Historical): ${POld50:.2f}'),
                        Line2D([0], [0], color='r', lw=2, label=f'POld75 (Historical): ${POld75:.2f}')
                    ])
                    
                else:  # Both for Comparison
                    # Historical lines (dashed)
                    ax.axhline(y=POld10, color='g', linestyle='--', alpha=0.7)
                    ax.axhline(y=POld50, color='y', linestyle='--', alpha=0.7)
                    ax.axhline(y=POld75, color='r', linestyle='--', alpha=0.7)
                    # Predictive lines (solid)
                    ax.axhline(y=PNew10, color='g', linestyle='-')
                    ax.axhline(y=PNew50, color='y', linestyle='-')
                    ax.axhline(y=PNew75, color='r', linestyle='-')
                    legend_elements.extend([
                        Line2D([0], [0], color='g', linestyle='--', lw=2, alpha=0.7, label=f'POld10 (Historical): ${POld10:.2f}'),
                        Line2D([0], [0], color='y', linestyle='--', lw=2, alpha=0.7, label=f'POld50 (Historical): ${POld50:.2f}'),
                        Line2D([0], [0], color='r', linestyle='--', lw=2, alpha=0.7, label=f'POld75 (Historical): ${POld75:.2f}'),
                        Line2D([0], [0], color='g', lw=2, label=f'PNew10 (Predicted): ${PNew10:.2f}'),
                        Line2D([0], [0], color='y', lw=2, label=f'PNew50 (Predicted): ${PNew50:.2f}'),
                        Line2D([0], [0], color='r', lw=2, label=f'PNew75 (Predicted): ${PNew75:.2f}')
                    ])

                ax.axhline(y=S0, color='k', linestyle='--')
                legend_elements.append(Line2D([0], [0], color='k', linestyle='--', lw=2, label=f'Current Price: ${S0:.2f}'))
                
                ax.legend(handles=legend_elements)
                ax.set_title(f'Monte Carlo Simulation - {ticker} Stock Price Over {days} Days')
                ax.set_xlabel('Days')
                ax.set_ylabel('Price ($)')
                ax.grid(True)
                st.pyplot(fig)

                # Plot the histogram of final prices
                st.subheader('7. Distribution of Final Prices')
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.hist(final_prices, bins=50, alpha=0.7, color='lightblue', label='Simulated Prices')
                
                # Plot lines based on display mode
                legend_elements = [Line2D([0], [0], color='lightblue', lw=8, alpha=0.7, label='Simulated Price Distribution')]
                
                if display_mode == "Predictive Only (PNew)":
                    ax.axvline(x=PNew10, color='g', linestyle='-', linewidth=2)
                    ax.axvline(x=PNew50, color='y', linestyle='-', linewidth=2)
                    ax.axvline(x=PNew75, color='r', linestyle='-', linewidth=2)
                    legend_elements.extend([
                        Line2D([0], [0], color='g', lw=2, label=f'PNew10 (Predicted): ${PNew10:.2f}'),
                        Line2D([0], [0], color='y', lw=2, label=f'PNew50 (Predicted): ${PNew50:.2f}'),
                        Line2D([0], [0], color='r', lw=2, label=f'PNew75 (Predicted): ${PNew75:.2f}')
                    ])
                    
                elif display_mode == "Historical Only (POld)":
                    ax.axvline(x=POld10, color='g', linestyle='-', linewidth=2)
                    ax.axvline(x=POld50, color='y', linestyle='-', linewidth=2)
                    ax.axvline(x=POld75, color='r', linestyle='-', linewidth=2)
                    legend_elements.extend([
                        Line2D([0], [0], color='g', lw=2, label=f'POld10 (Historical): ${POld10:.2f}'),
                        Line2D([0], [0], color='y', lw=2, label=f'POld50 (Historical): ${POld50:.2f}'),
                        Line2D([0], [0], color='r', lw=2, label=f'POld75 (Historical): ${POld75:.2f}')
                    ])
                    
                else:  # Both for Comparison
                    # Historical lines (dashed)
                    ax.axvline(x=POld10, color='g', linestyle='--', alpha=0.7, linewidth=2)
                    ax.axvline(x=POld50, color='y', linestyle='--', alpha=0.7, linewidth=2)
                    ax.axvline(x=POld75, color='r', linestyle='--', alpha=0.7, linewidth=2)
                    # Predictive lines (solid)
                    ax.axvline(x=PNew10, color='g', linestyle='-', linewidth=2)
                    ax.axvline(x=PNew50, color='y', linestyle='-', linewidth=2)
                    ax.axvline(x=PNew75, color='r', linestyle='-', linewidth=2)
                    legend_elements.extend([
                        Line2D([0], [0], color='g', linestyle='--', lw=2, alpha=0.7, label=f'POld10 (Historical): ${POld10:.2f}'),
                        Line2D([0], [0], color='y', linestyle='--', lw=2, alpha=0.7, label=f'POld50 (Historical): ${POld50:.2f}'),
                        Line2D([0], [0], color='r', linestyle='--', lw=2, alpha=0.7, label=f'POld75 (Historical): ${POld75:.2f}'),
                        Line2D([0], [0], color='g', lw=2, label=f'PNew10 (Predicted): ${PNew10:.2f}'),
                        Line2D([0], [0], color='y', lw=2, label=f'PNew50 (Predicted): ${PNew50:.2f}'),
                        Line2D([0], [0], color='r', lw=2, label=f'PNew75 (Predicted): ${PNew75:.2f}')
                    ])
                
                ax.axvline(x=S0, color='k', linestyle=':', linewidth=2)
                legend_elements.append(Line2D([0], [0], color='k', linestyle=':', lw=2, label=f'Current Price: ${S0:.2f}'))
                
                ax.legend(handles=legend_elements, loc='upper right')
                ax.set_title(f'Distribution of Simulated {ticker} Prices After {days} Days')
                ax.set_xlabel('Price ($)')
                ax.set_ylabel('Frequency')
                ax.grid(True)
                st.pyplot(fig)

                # Interpretation based on display mode
                st.subheader('8. Analysis and Interpretation')

                interpretation = ""
                
                if display_mode == "Predictive Only (PNew)":
                    # Focus on predictive analysis
                    mean_return = (np.mean(final_prices) / S0 - 1) * 100
                    interpretation += f"- **Expected Return**: The model predicts an average return of {mean_return:+.1f}% over {days} days.\n\n"
                    interpretation += f"- **Price Targets**: Based on simulation results:\n"
                    interpretation += f"  • 90% of scenarios stay above ${PNew10:.2f} (PNew10)\n"
                    interpretation += f"  • 50% of scenarios reach or exceed ${PNew50:.2f} (PNew50)\n"
                    interpretation += f"  • 25% of scenarios reach or exceed ${PNew75:.2f} (PNew75)\n\n"
                    
                elif display_mode == "Historical Only (POld)":
                    # Focus on historical comparison
                    interpretation += f"- **Historical Target Analysis**: Based on historical patterns vs. simulation results:\n"
                    interpretation += f"  • {prob_POld50:.2%} probability of reaching historical median (${POld50:.2f})\n"
                    interpretation += f"  • {prob_POld75:.2%} probability of reaching historical 75th percentile (${POld75:.2f})\n\n"
                    
                else:  # Both for Comparison
                    # Compare historical vs predictive targets
                    st.write("**Historical vs. Predictive Target Analysis:**")
                    
                    direction_P50 = "higher" if PNew50 > POld50 else "lower" if PNew50 < POld50 else "similar to"
                    direction_P75 = "higher" if PNew75 > POld75 else "lower" if PNew75 < POld75 else "similar to"
                    
                    st.write(f"• **PNew50 vs POld50**: Model predicts ${PNew50:.2f} vs historical ${POld50:.2f} - "
                            f"Predictive target is {direction_P50} historical ({((PNew50/POld50-1)*100):+.1f}%)")
                    st.write(f"• **PNew75 vs POld75**: Model predicts ${PNew75:.2f} vs historical ${POld75:.2f} - "
                            f"Predictive target is {direction_P75} historical ({((PNew75/POld75-1)*100):+.1f}%)")

                    # Analysis based on historical target probabilities
                    if prob_POld50 > 0.7:
                        interpretation += f"- **High confidence (Historical P50)**: There is a high probability ({prob_POld50:.2%}) of reaching the historical P50 target, suggesting strong confidence in reaching historical median levels.\n\n"
                    elif prob_POld50 > 0.5:
                        interpretation += f"- **Moderate confidence (Historical P50)**: There is a moderate probability ({prob_POld50:.2%}) of reaching the historical P50 target.\n\n"
                    else:
                        interpretation += f"- **Low confidence (Historical P50)**: There is a relatively low probability ({prob_POld50:.2%}) of reaching the historical P50 target, suggesting current conditions may be different from historical patterns.\n\n"

                    if prob_POld75 > 0.5:
                        interpretation += f"- **High confidence (Historical P75)**: There is a good probability ({prob_POld75:.2%}) of reaching the historical P75 target, suggesting strong potential for significant gains.\n\n"
                    elif prob_POld75 > 0.3:
                        interpretation += f"- **Moderate confidence (Historical P75)**: There is a moderate probability ({prob_POld75:.2%}) of reaching the historical P75 target.\n\n"
                    else:
                        interpretation += f"- **Low confidence (Historical P75)**: There is a low probability ({prob_POld75:.2%}) of reaching the historical P75 target, suggesting that historical high targets might be challenging under current model assumptions.\n\n"

                # Analysis of profit potential (common to all modes)
                interpretation += f"- **Overall Profit Potential**: There is a {prob_profit:.2%} probability of any profit (price > ${S0:.2f}), "
                
                if prob_profit > 0.6:
                    interpretation += "indicating favorable conditions for positive returns.\n\n"
                elif prob_profit > 0.4:
                    interpretation += "indicating moderate potential for positive returns.\n\n"
                else:
                    interpretation += "indicating challenging conditions with higher risk of losses.\n\n"

                st.markdown(interpretation)

                # Investment Recommendation based on display mode
                st.subheader('9. Investment Recommendation')

                recommendation = ""
                
                if display_mode == "Predictive Only (PNew)":
                    # Focus on predictive targets
                    mean_return = (np.mean(final_prices) / S0 - 1) * 100
                    
                    if prob_profit > 0.6 and mean_return > 10:
                        recommendation += "**Strong Buy Recommendation**\n\n"
                        recommendation += f"The model predicts favorable outcomes with {prob_profit:.2%} probability of profit and {mean_return:+.1f}% expected return. "
                    elif prob_profit > 0.5:
                        recommendation += "**Moderate Buy Recommendation**\n\n"
                        recommendation += f"The model shows reasonable upside potential with {prob_profit:.2%} probability of profit. "
                    else:
                        recommendation += "**Cautious Approach Recommended**\n\n"
                        recommendation += f"Model suggests {prob_profit:.2%} probability of profit with potential risks. "

                    recommendation += "**Suggested Strategy (Predictive Model):**\n"
                    recommendation += f"• **Current price**: ${S0:.2f}\n"
                    recommendation += f"• **Conservative target**: PNew50 at ${PNew50:.2f} (50% probability)\n"
                    recommendation += f"• **Aggressive target**: PNew75 at ${PNew75:.2f} (25% probability)\n"
                    recommendation += f"• **Stop consideration**: Below PNew10 at ${PNew10:.2f} (10th percentile)\n"
                    
                elif display_mode == "Historical Only (POld)":
                    # Focus on historical comparison
                    if prob_POld50 > 0.6 and prob_POld75 > 0.4:
                        recommendation += "**Strong Buy Recommendation**\n\n"
                        recommendation += f"High probability of reaching historical targets (POld50: {prob_POld50:.2%}, POld75: {prob_POld75:.2%}). "
                    elif prob_POld50 > 0.5:
                        recommendation += "**Moderate Buy Recommendation**\n\n"
                        recommendation += f"Reasonable probability ({prob_POld50:.2%}) of reaching historical median targets. "
                    else:
                        recommendation += "**Cautious Approach Recommended**\n\n"
                        recommendation += f"Lower probabilities of reaching historical targets suggest challenging conditions. "

                    recommendation += "**Suggested Strategy (Historical Analysis):**\n"
                    recommendation += f"• **Entry consideration**: Below historical POld10 at ${POld10:.2f}\n"
                    recommendation += f"• **Partial target**: POld50 at ${POld50:.2f} ({prob_POld50:.2%} probability)\n"
                    recommendation += f"• **Full target**: POld75 at ${POld75:.2f} ({prob_POld75:.2%} probability)\n"
                    
                else:  # Both for Comparison
                    # Combined analysis
                    if prob_POld50 > 0.6 and prob_POld75 > 0.4:
                        recommendation += "**Strong Buy Recommendation**\n\n"
                        recommendation += f"High probabilities of reaching historical targets (POld50: {prob_POld50:.2%}, POld75: {prob_POld75:.2%}) with predictive targets at PNew50: ${PNew50:.2f} and PNew75: ${PNew75:.2f}. "
                    elif prob_POld50 > 0.5:
                        recommendation += "**Moderate Buy Recommendation**\n\n"
                        recommendation += f"Reasonable probability ({prob_POld50:.2%}) of reaching historical targets with predictive guidance. "
                    else:
                        recommendation += "**Cautious Approach Recommended**\n\n"
                        recommendation += f"Mixed signals between historical and predictive analysis. "

                    recommendation += "**Suggested Strategy (Combined Analysis):**\n"
                    recommendation += f"• **Entry consideration**: Current ${S0:.2f} vs PNew10: ${PNew10:.2f}\n"
                    recommendation += f"• **Targets**: PNew50 (${PNew50:.2f}) vs POld50 (${POld50:.2f})\n"
                    recommendation += f"• **Stretch targets**: PNew75 (${PNew75:.2f}) vs POld75 (${POld75:.2f})\n"

                # Common risk assessment
                recommendation += f"\n**Risk Assessment**: {prob_profit:.2%} probability of any profit over {days} days.\n"

                st.markdown(recommendation)

        except Exception as e:
            st.error(f"An error occurred: {e}")


if __name__ == "__main__":
    monte_carlo_simulation()
