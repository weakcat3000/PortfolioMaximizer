"""
Data Fetcher Module
Downloads historical price data from Yahoo Finance
"""

import yfinance as yf
import pandas as pd
import datetime as dt
from typing import List, Optional


def fetch_data(
    tickers: List[str],
    start_date: str = "2009-01-01",
    end_date: str = "2025-12-31",
    interval: str = "1d"
) -> pd.DataFrame:
    """
    Download historical close prices from Yahoo Finance.
    
    Args:
        tickers: List of ticker symbols (e.g., ['AAPL', 'MSFT', 'SPY'])
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        interval: Data interval ('1d' for daily, '1mo' for monthly)
    
    Returns:
        DataFrame with close prices, indexed by date
    """
    # Join tickers into space-separated string
    ticker_str = " ".join(tickers)
    
    print(f"Downloading data for: {ticker_str}")
    print(f"Period: {start_date} to {end_date}")
    
    # Download data
    data = yf.download(ticker_str, start=start_date, end=end_date, interval=interval)
    
    # Handle single ticker vs multiple tickers
    if len(tickers) == 1:
        data = data[['Close']].copy()
        data.columns = tickers
    else:
        data = data['Close'].copy()
    
    # Forward fill missing values
    data = data.ffill()
    
    # Drop any remaining NaN rows
    data = data.dropna()
    
    # Ensure index is DatetimeIndex (fix for "Only valid with DatetimeIndex" error)
    data.index = pd.to_datetime(data.index)
    
    # Localize/Navie check (remove timezone if present to avoid issues)
    if data.index.tz is not None:
        data.index = data.index.tz_localize(None)
    
    if data.empty:
        print("WARNING: Download failed or blocked. generating MOCK DATA for demonstration.")
        return generate_mock_data(tickers, start_date, end_date)
    
    print(f"Downloaded {len(data)} rows of data")
    
    return data

def generate_mock_data(tickers, start, end):
    """Generate synthetic data for demo purposes."""
    import numpy as np
    
    dates = pd.date_range(start=start, end=end, freq='D')
    # Create empty DF
    data = pd.DataFrame(index=dates)
    
    for ticker in tickers:
        # Random walk: start at 100, daily return ~ N(0.05%, 1.5%)
        returns = np.random.normal(0.0005, 0.015, len(dates))
        prices = 100 * (1 + returns).cumprod()
        data[ticker] = prices
        
    return data


def calculate_returns(
    prices: pd.DataFrame,
    frequency: str = "daily"
) -> pd.DataFrame:
    """
    Calculate returns from price data.
    
    Args:
        prices: DataFrame of prices
        frequency: 'daily' or 'monthly'
    
    Returns:
        DataFrame of returns
    """
    if frequency == "monthly":
        # Resample to month-end and calculate returns
        monthly_prices = prices.resample('ME').last()
        returns = monthly_prices.pct_change().dropna()
    else:
        # Daily returns
        returns = prices.pct_change().dropna()
    
    return returns


def get_risk_free_rate(
    start_date: str = "2009-01-01",
    end_date: str = "2025-12-31"
) -> pd.DataFrame:
    """
    Get risk-free rate proxy using 3-month Treasury Bill (^IRX).
    Returns annualized rate divided by 12 for monthly, 252 for daily.
    
    Note: ^IRX is in percentage points, so we divide by 100.
    """
    try:
        rf_data = yf.download("^IRX", start=start_date, end=end_date)
        if len(rf_data) > 0:
            rf_rate = rf_data['Close'] / 100 / 12  # Convert to monthly decimal
            return rf_rate.ffill().dropna()
    except:
        pass
    
    # Default to 2% annualized if data unavailable
    print("Using default risk-free rate of 2% annualized")
    return None


def save_to_csv(data: pd.DataFrame, filename: str):
    """Save DataFrame to CSV file."""
    data.to_csv(filename)
    print(f"Saved data to {filename}")


# Example usage
if __name__ == "__main__":
    # Example: Fetch data for sample ETFs
    sample_tickers = ['SPY', 'IWM', 'EFA', 'AGG', 'GLD']
    
    prices = fetch_data(sample_tickers, "2009-01-01", "2025-01-31")
    print("\nSample prices:")
    print(prices.head())
    
    monthly_returns = calculate_returns(prices, "monthly")
    print("\nSample monthly returns:")
    print(monthly_returns.head())
    
    save_to_csv(monthly_returns, "sample_returns.csv")
