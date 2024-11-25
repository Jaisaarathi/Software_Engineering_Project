import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="FinTrack: Smart Portfolio Management", layout="wide")

# Caching stock data
@st.cache_data
def get_stock_data(ticker, start_date, end_date):
    try:
        data = yf.download(ticker, start=start_date, end=end_date)['Adj Close']
        return data
    except Exception as e:
        st.error(f"Error fetching stock data for {ticker}: {e}")
        return None

# Portfolio metrics functions
def portfolio_return(weights, returns):
    return np.dot(weights, returns)

def portfolio_risk(weights, covariance_matrix):
    return np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))

# Streamlit App
st.title("FinTrack: Smart Portfolio Management & Analytics")

# Sidebar Inputs
st.sidebar.header("Portfolio Inputs")
num_stocks = st.sidebar.slider("Number of Stocks", min_value=2, max_value=10, value=5)

tickers = []
quantities = []
start_dates = []

for i in range(num_stocks):
    col1, col2, col3 = st.sidebar.columns([2, 1, 1])
    with col1:
        tickers.append(st.text_input(f"Stock {i + 1} Ticker", value="AAPL" if i == 0 else ""))
    with col2:
        quantities.append(st.number_input(f"Qty {i + 1}", min_value=1, value=10, step=1))
    with col3:
        start_dates.append(st.date_input(f"Start Date {i + 1}", value=pd.to_datetime("2022-01-01")))

if st.sidebar.button("Analyze Portfolio"):
    tickers = [ticker.upper() for ticker in tickers if ticker.strip()]

    if len(tickers) != num_stocks:
        st.error("Provide valid tickers for all stocks.")
    else:
        end_date = pd.Timestamp.today()
        stock_data_list = [get_stock_data(ticker, start_dates[i], end_date) for i, ticker in enumerate(tickers)]
        missing_data = [tickers[i] for i, data in enumerate(stock_data_list) if data is None]

        if missing_data:
            st.error(f"The following tickers returned no data: {', '.join(missing_data)}. Please check and try again.")
        else:
            stock_data = pd.concat(stock_data_list, axis=1, keys=tickers).dropna()

            # Calculate Portfolio Return & Risk
            returns = stock_data.pct_change().mean() * 252  # Annualized return (252 trading days)
            covariance_matrix = stock_data.pct_change().cov() * 252  # Annualized covariance matrix

            portfolio_value = np.dot(quantities, stock_data.iloc[-1])  # Portfolio value at the last date
            portfolio_weights = (np.array(quantities) * stock_data.iloc[-1]) / portfolio_value  # Normalized portfolio weights

            port_return = portfolio_return(portfolio_weights, returns)
            port_risk = portfolio_risk(portfolio_weights, covariance_matrix)

            # Convert Portfolio Return and Risk to Percentage
            port_return_percentage = port_return * 100  # Portfolio return in percentage
            port_risk_percentage = port_risk * 100  # Portfolio risk (volatility) in percentage

            # Visualizations
            st.subheader("Portfolio Composition in Quantities")
            quantities_series = pd.Series(quantities, index=tickers)
            plt.figure(figsize=(10, 5))
            quantities_series.plot(kind='bar', title="Portfolio Composition (Quantities)")
            plt.xlabel("Stocks")
            plt.ylabel("Quantity")
            st.pyplot(plt)

            st.subheader("Portfolio Composition in Percentages")
            opt_weights_series = pd.Series(np.ones(len(tickers)) / len(tickers), index=tickers)  # Equal weight for visualization
            plt.figure(figsize=(8, 8))
            plt.pie(opt_weights_series, labels=opt_weights_series.index, autopct='%1.1f%%', startangle=90, colors=plt.cm.tab20.colors)
            plt.title("Portfolio Weights")
            plt.axis('equal')
            st.pyplot(plt)

            st.subheader("Stock Price Trends")
            stock_data_normalized = stock_data / stock_data.iloc[0]
            plt.figure(figsize=(12, 6))
            for ticker in tickers:
                plt.plot(stock_data_normalized.index, stock_data_normalized[ticker], label=ticker)
            plt.title("Stock Price Trends")
            plt.legend()
            st.pyplot(plt)

            st.subheader("Portfolio Risk vs. Return")
            num_portfolios = 5000
            all_weights = np.zeros((num_portfolios, len(tickers)))
            ret_array = np.zeros(num_portfolios)
            risk_array = np.zeros(num_portfolios)
            sharpe_array = np.zeros(num_portfolios)

            for i in range(num_portfolios):
                weights = np.random.random(len(tickers))
                weights /= np.sum(weights)

                all_weights[i, :] = weights
                ret_array[i] = np.dot(weights, returns)
                risk_array[i] = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
                sharpe_array[i] = (ret_array[i] - 0.015) / risk_array[i]  # Assuming risk-free rate of 1.5%

            plt.figure(figsize=(10, 6))
            scatter = plt.scatter(risk_array, ret_array, c=sharpe_array, cmap='viridis')
            plt.colorbar(scatter, label="Sharpe Ratio")
            plt.title("Portfolio Risk vs Return")
            plt.xlabel("Risk (Standard Deviation)")
            plt.ylabel("Return")
            st.pyplot(plt)

            # Additional Visualizations

            # 1. Cumulative Returns of the Portfolio
            st.subheader("Cumulative Returns")
            cum_returns = (1 + stock_data.pct_change()).cumprod()
            portfolio_cum_return = (1 + stock_data.pct_change().dot(np.array(quantities) / sum(quantities))).cumprod()
            plt.figure(figsize=(12, 6))
            for ticker in tickers:
                plt.plot(cum_returns.index, cum_returns[ticker], label=f"{ticker} Cumulative Return")
            plt.plot(cum_returns.index, portfolio_cum_return, label="Portfolio Cumulative Return", linewidth=3, color='black')
            plt.title("Cumulative Returns of Stocks & Portfolio")
            plt.legend()
            st.pyplot(plt)

            # 2. Correlation Heatmap of the Stocks
            st.subheader("Correlation Heatmap")
            correlation_matrix = stock_data.pct_change().corr()
            plt.figure(figsize=(8, 6))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
            plt.title("Correlation Heatmap of Stock Returns")
            st.pyplot(plt)

            # 3. Rolling 60-Day Volatility of the Portfolio
            st.subheader("Rolling 60-Day Volatility of the Portfolio")
            rolling_volatility = stock_data.pct_change().rolling(window=60).std() * np.sqrt(252)
            portfolio_volatility = (rolling_volatility.dot(np.array(quantities) / sum(quantities)))
            plt.figure(figsize=(12, 6))
            plt.plot(rolling_volatility.index, portfolio_volatility, label="Portfolio Rolling Volatility", color='black')
            plt.title("Rolling 60-Day Volatility of Portfolio")
            plt.legend()
            st.pyplot(plt)

            # Portfolio Metrics
            st.subheader("Portfolio Return & Risk")
            st.metric("Portfolio Return", f"{port_return_percentage:.2f}%")
            st.metric("Portfolio Risk", f"{port_risk_percentage:.2f}%")
