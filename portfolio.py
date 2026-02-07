"""
Portfolio Module
Handles portfolio construction and return calculations
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional


class Portfolio:
    """
    Portfolio class for constructing and analyzing investment portfolios.
    """
    
    def __init__(
        self,
        name: str,
        allocations: Dict[str, float],
        returns_data: pd.DataFrame,
        rebalance_frequency: str = "monthly"
    ):
        """
        Initialize a portfolio.
        
        Args:
            name: Portfolio name
            allocations: Dictionary of {ticker: weight} (weights should sum to 1)
            returns_data: DataFrame of asset returns
            rebalance_frequency: 'daily', 'monthly', or 'quarterly'
        """
        self.name = name
        self.allocations = allocations
        self.returns_data = returns_data
        self.rebalance_frequency = rebalance_frequency
        
        # Validate allocations
        self._validate_allocations()
        
        # Calculate portfolio returns
        self.portfolio_returns = self._calculate_portfolio_returns()
    
    def _validate_allocations(self):
        """Validate that allocations sum to 1 and tickers exist in data."""
        total_weight = sum(self.allocations.values())
        if not np.isclose(total_weight, 1.0, atol=0.01):
            raise ValueError(f"Allocations must sum to 1.0, got {total_weight}")
        
        missing_tickers = set(self.allocations.keys()) - set(self.returns_data.columns)
        if missing_tickers:
            raise ValueError(f"Missing tickers in returns data: {missing_tickers}")
    
    def _calculate_portfolio_returns(self) -> pd.Series:
        """
        Calculate portfolio returns based on allocations.
        
        This applies fixed weights (implicitly rebalancing at the frequency
        of the return series).
        """
        # Get relevant columns
        tickers = list(self.allocations.keys())
        weights = np.array([self.allocations[t] for t in tickers])
        
        # Calculate weighted returns
        asset_returns = self.returns_data[tickers]
        portfolio_returns = (asset_returns * weights).sum(axis=1)
        portfolio_returns.name = self.name
        
        return portfolio_returns
    
    def get_cumulative_returns(self) -> pd.Series:
        """Calculate cumulative returns (growth of $1)."""
        cumulative = (1 + self.portfolio_returns).cumprod()
        cumulative.name = self.name
        return cumulative
    
    def get_allocation_df(self) -> pd.DataFrame:
        """Return allocations as a DataFrame for display."""
        df = pd.DataFrame([
            {"Asset": ticker, "Weight": weight, "Weight %": f"{weight*100:.1f}%"}
            for ticker, weight in self.allocations.items()
        ])
        return df
    
    def summary(self) -> Dict:
        """Get portfolio summary statistics."""
        returns = self.portfolio_returns
        cumulative = self.get_cumulative_returns()
        
        # Annualized metrics (assuming monthly returns)
        n_periods = len(returns)
        years = n_periods / 12
        
        total_return = cumulative.iloc[-1] - 1
        annualized_return = (1 + total_return) ** (1/years) - 1
        annualized_volatility = returns.std() * np.sqrt(12)
        
        return {
            "Portfolio": self.name,
            "Total Return": f"{total_return*100:.2f}%",
            "Annualized Return": f"{annualized_return*100:.2f}%",
            "Annualized Volatility": f"{annualized_volatility*100:.2f}%",
            "Number of Periods": n_periods,
            "Years": f"{years:.1f}"
        }


def create_benchmark_portfolio(
    returns_data: pd.DataFrame,
    benchmark_ticker: str = "SPY"
) -> Portfolio:
    """Create a 100% benchmark portfolio."""
    return Portfolio(
        name=f"Benchmark ({benchmark_ticker})",
        allocations={benchmark_ticker: 1.0},
        returns_data=returns_data
    )


# Example client profile portfolios
def get_sample_allocations(profile: str) -> Dict[str, float]:
    """
    Get sample allocations based on client profile.
    These are starting points - users should customize based on their analysis.
    """
    profiles = {
        "conservative": {
            # Richard Tan - Capital Preservation & Income
            # 30% Equities, 50% Bonds, 10% Gold, 10% Cash equivalent
            "VTI": 0.15,   # Total US Stock Market
            "VEA": 0.10,   # Developed International
            "VNQ": 0.05,   # Real Estate
            "BND": 0.30,   # Total Bond Market
            "TIP": 0.10,   # Treasury Inflation-Protected
            "LQD": 0.10,   # Investment Grade Corporate Bonds
            "GLD": 0.10,   # Gold
            "SHY": 0.10,   # Short-term Treasury (cash proxy)
        },
        "balanced": {
            # Sophia Lim - Growth with Moderate Risk
            # 60% Equities, 30% Bonds, 10% Alternatives
            "VTI": 0.25,   # Total US Stock Market
            "VGT": 0.10,   # Technology (her industry)
            "VEA": 0.15,   # Developed International
            "VWO": 0.05,   # Emerging Markets
            "VNQ": 0.05,   # Real Estate
            "BND": 0.20,   # Total Bond Market
            "LQD": 0.10,   # Corporate Bonds
            "GLD": 0.05,   # Gold
            "DBC": 0.05,   # Commodities
        },
        "aggressive": {
            # David Lee - High Growth & Alternative Assets
            # 80% Equities, 10% Alternatives, 10% Bonds
            "VTI": 0.20,   # Total US Stock Market
            "QQQ": 0.20,   # Nasdaq-100 (Tech heavy)
            "VGT": 0.10,   # Technology Sector
            "VWO": 0.10,   # Emerging Markets
            "IWM": 0.10,   # Small Cap Growth
            "VNQ": 0.05,   # Real Estate
            "ARKK": 0.05,  # Innovation/Disruptive Tech (if available)
            "GLD": 0.05,   # Gold
            "BND": 0.10,   # Bonds (minimal)
            "DBC": 0.05,   # Commodities
        }
    }
    
    if profile.lower() not in profiles:
        raise ValueError(f"Unknown profile: {profile}. Choose from: {list(profiles.keys())}")
    
    return profiles[profile.lower()]


if __name__ == "__main__":
    # Example usage with dummy data
    import numpy as np
    
    # Create sample returns data
    np.random.seed(42)
    dates = pd.date_range("2020-01-31", periods=24, freq="ME")
    
    sample_returns = pd.DataFrame({
        "VTI": np.random.normal(0.01, 0.04, 24),
        "BND": np.random.normal(0.003, 0.01, 24),
        "GLD": np.random.normal(0.005, 0.03, 24),
        "SPY": np.random.normal(0.01, 0.04, 24),
    }, index=dates)
    
    # Create a simple portfolio
    portfolio = Portfolio(
        name="Sample Portfolio",
        allocations={"VTI": 0.6, "BND": 0.3, "GLD": 0.1},
        returns_data=sample_returns
    )
    
    print("Portfolio Allocations:")
    print(portfolio.get_allocation_df())
    print("\nPortfolio Summary:")
    for k, v in portfolio.summary().items():
        print(f"  {k}: {v}")
