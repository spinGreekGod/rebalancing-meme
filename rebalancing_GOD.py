import yfinance as yf
import numpy as np
import pandas as pd
from itertools import product
import quantstats as qs
from bisect import bisect_left
import sys
import pandas as pd

def update_loading_bar(progress, total):
    bar_length = 40
    block = int(bar_length * progress / total)
    text = f"\rLoading: [{'#' * block}{'-' * (bar_length - block)}] {progress}/{total}"
    sys.stdout.write(text)
    sys.stdout.flush()

def get_sp500_tickers():

    # Fetch S&P 500 tickers from Wikipedia
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    sp500_table = pd.read_html(url, header=0)[0]
    tickers = sp500_table["Symbol"].tolist()

    # Removing specific tickers individually
    tickers.remove("AAPL")
    tickers.remove("ABNB")
    tickers.remove("ADBE")
    tickers.remove("ADI")
    tickers.remove("ADP")
    tickers.remove("ADSK")
    tickers.remove("AEP")
    tickers.remove("AMAT")
    tickers.remove("AMD")
    tickers.remove("AMGN")
    tickers.remove("AMZN")
    tickers.remove("ANSS")
    tickers.remove("AVGO")
    tickers.remove("BIIB")
    tickers.remove("BKNG")
    tickers.remove("BKR")
    tickers.remove("CDNS")
    tickers.remove("CDW")
    tickers.remove("CEG")
    tickers.remove("CHTR")
    tickers.remove("CMCSA")
    tickers.remove("COST")
    tickers.remove("CPRT")
    tickers.remove("CRWD")
    tickers.remove("CSCO")
    tickers.remove("CSGP")
    tickers.remove("CSX")
    tickers.remove("CTAS")
    tickers.remove("CTSH")
    tickers.remove("DXCM")
    tickers.remove("EA")
    tickers.remove("EXC")
    tickers.remove("FANG")
    tickers.remove("FAST")
    tickers.remove("FTNT")
    tickers.remove("GEHC")
    tickers.remove("GILD")
    tickers.remove("GOOG")
    tickers.remove("GOOGL")
    tickers.remove("HON")
    tickers.remove("IDXX")
    tickers.remove("INTC")
    tickers.remove("INTU")
    tickers.remove("ISRG")
    tickers.remove("KDP")
    tickers.remove("KHC")
    tickers.remove("KLAC")
    tickers.remove("LIN")
    tickers.remove("LRCX")
    tickers.remove("LULU")
    tickers.remove("MAR")
    tickers.remove("MCHP")
    tickers.remove("MDLZ")
    tickers.remove("META")
    tickers.remove("MNST")
    tickers.remove("MRNA")
    tickers.remove("MSFT")
    tickers.remove("MU")
    tickers.remove("NFLX")
    tickers.remove("NVDA")
    tickers.remove("NXPI")
    tickers.remove("ODFL")
    tickers.remove("ON")
    tickers.remove("ORLY")
    tickers.remove("PANW")
    tickers.remove("PAYX")
    tickers.remove("PCAR")
    tickers.remove("PEP")
    tickers.remove("PYPL")
    tickers.remove("QCOM")
    tickers.remove("REGN")
    tickers.remove("ROP")
    tickers.remove("ROST")
    tickers.remove("SBUX")
    tickers.remove("SMCI")
    tickers.remove("SNPS")
    tickers.remove("TMUS")
    tickers.remove("TSLA")
    tickers.remove("TTWO")
    tickers.remove("TXN")
    tickers.remove("VRSK")
    tickers.remove("VRTX")
    tickers.remove("WBD")
    tickers.remove("XEL")

    return tickers



def fetch_data(tickers):
    stock_data = []
    total = len(tickers)

    for idx, ticker in enumerate(tickers, 1):
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            hist = stock.history(period="max")
            stock_data.append({
                "ticker": ticker,
                "price": hist['Close'],
                "pe_ratio": info.get("trailingPE", np.nan),
                "dividend_yield": info.get("dividendYield", 0),
                "market_cap": info.get("marketCap", np.nan),
            })
        except Exception as e:
            print(f"\nError fetching data for {ticker}: {e}")
        
        update_loading_bar(idx, total)

    stock_data.sort(key=lambda x: x["market_cap"] or 0, reverse=True)
    return stock_data

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
def calculate_portfolio_beta(weights, stock_data, benchmark_ticker="^GSPC"):
    benchmark_data = yf.download(benchmark_ticker, period="24y")['Close'].pct_change().dropna()
    portfolio_returns = calculate_portfolio_returns(weights, stock_data)
    covariance_matrix = np.cov(portfolio_returns, benchmark_data)
    beta = covariance_matrix[0, 1] / covariance_matrix[1, 1]
    return beta

def binary_search_optimization(stock_data, low=0.01, high=1.0, granularity=0.01):
    num_stocks = len(stock_data)
    weights = [1 / num_stocks] * num_stocks
    best_weights = weights[:]
    best_sharpe = -np.inf

    for i in range(num_stocks):
        while high - low > granularity:
            mid = (low + high) / 2
            weights[i] = mid
            portfolio_returns = calculate_portfolio_returns(weights, stock_data)
            sharpe_ratio = qs.stats.sharpe(portfolio_returns)

            if sharpe_ratio > best_sharpe:
                best_sharpe = sharpe_ratio
                best_weights = weights[:]
                low = mid
            else:
                high = mid

    return best_weights, best_sharpe

def analyze_portfolio(portfolio_returns):
    print("Portfolio Performance Metrics:")
    print(f"Sharpe Ratio: {qs.stats.sharpe(portfolio_returns):.2f}")
    print(f"Sortino Ratio: {qs.stats.sortino(portfolio_returns):.2f}")
    print(f"Max Drawdown: {qs.stats.max_drawdown(portfolio_returns):.2%}")
    print(f"Calmar Ratio: {qs.stats.calmar(portfolio_returns):.2f}")
    print(f"Information Ratio: {qs.stats.information_ratio(portfolio_returns):.2f}")
    print(f"CAGR (Compound Annual Growth Rate): {qs.stats.cagr(portfolio_returns):.2%}")
    print(f"Volatility (Annualized): {qs.stats.volatility(portfolio_returns):.2%}")
    print(f"Skewness: {qs.stats.skew(portfolio_returns):.2f}")
    print(f"Kurtosis: {qs.stats.kurtosis(portfolio_returns):.2f}")
    print(f"Tail Ratio: {qs.stats.tail_ratio(portfolio_returns):.2f}")
    print(f"Value at Risk (5%): {qs.stats.value_at_risk(portfolio_returns, cutoff=0.05):.2%}")
    print(f"Conditional Value at Risk (5%): {qs.stats.cvar(portfolio_returns, cutoff=0.05):.2%}")
    print(f"Recovery Factor: {qs.stats.recovery_factor(portfolio_returns):.2f}")
    print(f"Stability: {qs.stats.stability(portfolio_returns):.2f}")
    print(f"Gain-to-Pain Ratio: {qs.stats.gain_to_pain_ratio(portfolio_returns):.2f}")
    print(f"Alpha (vs SPY): {qs.stats.alpha(portfolio_returns):.2f}")
    print(f"Beta (vs SPY): {qs.stats.beta(portfolio_returns):.2f}")


stock_tickers = get_sp500_tickers()
print("Fetching data for S&P 500 stocks...")
stock_data = fetch_data(stock_tickers)
best_weights, best_sharpe = binary_search_optimization(stock_data)

portfolio_returns = calculate_portfolio_returns(best_weights, stock_data)

print("Optimal Portfolio Weights:")
for i, stock in enumerate(stock_data):
    print(f"{stock['ticker']}: {best_weights[i]:.2%}")
print(f"Maximum Sharpe Ratio: {best_sharpe:.2f}")

analyze_portfolio(portfolio_returns)
qs.reports.full(portfolio_returns)
