import yfinance as yf
import numpy as np
import pandas as pd
from itertools import product
import quantstats as qs
from bisect import bisect_left
import sys

# Function to update loading bar
def update_loading_bar(progress, total):
    bar_length = 40
    block = int(bar_length * progress / total)
    text = f"\rLoading: [{'#' * block}{'-' * (bar_length - block)}] {progress}/{total}"
    sys.stdout.write(text)
    sys.stdout.flush()

# Fetch all S&P 500 stock tickers
def get_sp500_tickers():
    sp500_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    sp500_table = pd.read_html(sp500_url, header=0)[0]
    return sp500_table["Symbol"].tolist()

# Fetch fundamental data and stock prices
def fetch_data(tickers):
    stock_data = []
    total = len(tickers)

    for idx, ticker in enumerate(tickers, 1):
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            hist = stock.history(period="24y")
            stock_data.append({
                "ticker": ticker,
                "price": hist['Close'],
                "pe_ratio": info.get("trailingPE", np.nan),
                "dividend_yield": info.get("dividendYield", 0),
                "market_cap": info.get("marketCap", np.nan),
            })
        except Exception as e:
            print(f"\nError fetching data for {ticker}: {e}")
        
        # Update loading bar
        update_loading_bar(idx, total)

    print()  # Move to the next line after loading completes

    # Sort stocks by market cap in descending order
    stock_data.sort(key=lambda x: x["market_cap"] or 0, reverse=True)
    return stock_data

# Rebalance stocks based on weights and calculate portfolio returns
def calculate_portfolio_returns(weights, stock_data):
    total_weights = sum(weights)
    weighted_prices = None

    for i, stock in enumerate(stock_data):
        price_data = stock["price"]
        if len(price_data) > 1:
            weight = weights[i] / total_weights
            weighted_price = price_data.pct_change() * weight

            if weighted_prices is None:
                weighted_prices = weighted_price
            else:
                weighted_prices += weighted_price

    return weighted_prices.dropna()

# Binary search for the optimal weights
def binary_search_optimization(stock_data, low=0.01, high=1.0, granularity=0.01):
    num_stocks = len(stock_data)
    weights = [1 / num_stocks] * num_stocks
    best_weights = weights[:]
    best_cagr = -np.inf

    # Iteratively adjust weights
    for i in range(num_stocks):
        while high - low > granularity:
            mid = (low + high) / 2
            weights[i] = mid
            portfolio_returns = calculate_portfolio_returns(weights, stock_data)
            cagr = qs.stats.cagr(portfolio_returns)

            if cagr > best_cagr:
                best_cagr = cagr
                best_weights = weights[:]
                low = mid
            else:
                high = mid

    return best_weights, best_cagr

# Analyze portfolio statistics
def analyze_portfolio(portfolio_returns):
    print("Portfolio Performance Metrics:")
    print(f"Sharpe Ratio: {qs.stats.sharpe(portfolio_returns):.2f}")
    print(f"Sortino Ratio: {qs.stats.sortino(portfolio_returns):.2f}")
    print(f"Max Drawdown: {qs.stats.max_drawdown(portfolio_returns):.2%}")
    print(f"Calmar Ratio: {qs.stats.calmar(portfolio_returns):.2f}")
    print(f"Information Ratio: {qs.stats.information_ratio(portfolio_returns):.2f}")
    print(f"CAGR: {qs.stats.cagr(portfolio_returns):.2%}")

# Main execution
stock_tickers = get_sp500_tickers()
print("Fetching data for S&P 500 stocks...")
stock_data = fetch_data(stock_tickers)
best_weights, best_cagr = binary_search_optimization(stock_data)

# Calculate portfolio returns for the best weights
portfolio_returns = calculate_portfolio_returns(best_weights, stock_data)

# Output results
print("Optimal Portfolio Weights:")
for i, stock in enumerate(stock_data):
    print(f"{stock['ticker']}: {best_weights[i]:.2%}")
print(f"Maximum CAGR: {best_cagr:.2%}")

# Analyze portfolio performance
analyze_portfolio(portfolio_returns)

# Generate a performance report
qs.reports.html(portfolio_returns, output="portfolio_performance.html")
print("Performance report saved as 'portfolio_performance.html'")
