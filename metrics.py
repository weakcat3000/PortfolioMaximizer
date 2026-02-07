"""
Performance Metrics Module
Calculates all required portfolio performance metrics
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Optional, Tuple


class PerformanceMetrics:
    """
    Calculate comprehensive performance metrics for a portfolio.
    """
    
    def __init__(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
        risk_free_rate: float = 0.02,
        periods_per_year: int = 12  # Monthly returns
    ):
        """
        Initialize with return series.
        
        Args:
            portfolio_returns: Series of portfolio returns
            benchmark_returns: Series of benchmark returns
            risk_free_rate: Annualized risk-free rate (default 2%)
            periods_per_year: Number of periods per year (12 for monthly, 252 for daily)
        """
        # Align the series
        aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
        
        if aligned.empty or len(aligned) < 2:
            self.portfolio_returns = pd.Series(dtype=float)
            self.benchmark_returns = pd.Series(dtype=float)
            self.beta = 0.0
            self.alpha = 0.0
            self.r_squared = 0.0
            self.excess_returns = pd.Series(dtype=float)
            self.benchmark_excess = pd.Series(dtype=float)
            self.rf_annual = risk_free_rate
            self.rf_periodic = risk_free_rate / periods_per_year
            self.periods_per_year = periods_per_year
            return

        self.portfolio_returns = aligned.iloc[:, 0]
        self.benchmark_returns = aligned.iloc[:, 1]
        
        self.rf_annual = risk_free_rate
        self.rf_periodic = risk_free_rate / periods_per_year
        self.periods_per_year = periods_per_year
        
        # Calculate excess returns
        self.excess_returns = self.portfolio_returns - self.rf_periodic
        self.benchmark_excess = self.benchmark_returns - self.rf_periodic
        
        # Calculate beta and alpha
        self._calculate_regression()
    
    def _calculate_regression(self):
        """Run regression of portfolio excess returns on benchmark excess returns."""
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            self.benchmark_excess, self.excess_returns
        )
        self.beta = slope
        self.alpha = intercept * self.periods_per_year  # Annualize alpha
        self.r_squared = r_value ** 2
    
    # ==================== Core Metrics ====================
    
    def sharpe_ratio(self) -> float:
        """
        Sharpe Ratio = (Rp - Rf) / σp
        Measures excess return per unit of total risk.
        """
        mean_return = self.portfolio_returns.mean() * self.periods_per_year
        volatility = self.portfolio_returns.std() * np.sqrt(self.periods_per_year)
        
        if volatility == 0:
            return 0.0
        
        return (mean_return - self.rf_annual) / volatility
    
    def jensens_alpha(self) -> float:
        """
        Jensen's Alpha = Rp - [Rf + β(Rm - Rf)]
        Measures excess return above CAPM expected return.
        Already annualized in _calculate_regression.
        """
        return self.alpha
    
    def treynor_ratio(self) -> float:
        """
        Treynor Ratio = (Rp - Rf) / β
        Measures excess return per unit of systematic risk.
        """
        mean_return = self.portfolio_returns.mean() * self.periods_per_year
        
        if self.beta == 0:
            return 0.0
        
        return (mean_return - self.rf_annual) / self.beta
    
    def sortino_ratio(self) -> float:
        """
        Sortino Ratio = (Rp - Rf) / Downside Deviation
        Like Sharpe but only penalizes downside volatility.
        """
        mean_return = self.portfolio_returns.mean() * self.periods_per_year
        
        # Calculate downside deviation (only negative returns)
        negative_returns = self.portfolio_returns[self.portfolio_returns < 0]
        if len(negative_returns) == 0:
            return float('inf')
        
        downside_deviation = np.sqrt((negative_returns ** 2).mean()) * np.sqrt(self.periods_per_year)
        
        if downside_deviation == 0:
            return float('inf')
        
        return (mean_return - self.rf_annual) / downside_deviation
    
    def information_ratio(self) -> float:
        """
        Information Ratio = (Rp - Rb) / Tracking Error
        Measures active return per unit of active risk.
        """
        active_returns = self.portfolio_returns - self.benchmark_returns
        mean_active_return = active_returns.mean() * self.periods_per_year
        tracking_error = active_returns.std() * np.sqrt(self.periods_per_year)
        
        if tracking_error == 0:
            return 0.0
        
        return mean_active_return / tracking_error
    
    def maximum_drawdown(self) -> Tuple[float, pd.Timestamp, pd.Timestamp]:
        """
        Maximum Drawdown = Max peak-to-trough decline.
        Returns (max_dd, peak_date, trough_date)
        """
        cumulative = (1 + self.portfolio_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        max_dd = drawdown.min()
        trough_date = drawdown.idxmin()
        peak_date = cumulative[:trough_date].idxmax()
        
        return max_dd, peak_date, trough_date
    
    # ==================== Additional Metrics ====================
    
    def annualized_return(self) -> float:
        """Calculate annualized return."""
        total_return = (1 + self.portfolio_returns).prod() - 1
        years = len(self.portfolio_returns) / self.periods_per_year
        return (1 + total_return) ** (1/years) - 1
    
    def annualized_volatility(self) -> float:
        """Calculate annualized volatility."""
        return self.portfolio_returns.std() * np.sqrt(self.periods_per_year)
    
    def calmar_ratio(self) -> float:
        """Calmar Ratio = Annualized Return / |Max Drawdown|"""
        max_dd, _, _ = self.maximum_drawdown()
        ann_return = self.annualized_return()
        
        if max_dd == 0:
            return float('inf')
        
        return ann_return / abs(max_dd)
    
    def get_beta(self) -> float:
        """Get portfolio beta."""
        return self.beta
    
    def get_all_metrics(self) -> Dict:
        """
        Get all performance metrics as a dictionary.
        """
        max_dd, peak_date, trough_date = self.maximum_drawdown()
        
        return {
            # Required metrics
            "Sharpe Ratio": self.sharpe_ratio(),
            "Jensen's Alpha": self.jensens_alpha(),
            "Treynor Ratio": self.treynor_ratio(),
            "Sortino Ratio": self.sortino_ratio(),
            "Information Ratio": self.information_ratio(),
            "Maximum Drawdown": max_dd,
            "Max DD Peak Date": peak_date,
            "Max DD Trough Date": trough_date,
            
            # Additional metrics
            "Annualized Return": self.annualized_return(),
            "Annualized Volatility": self.annualized_volatility(),
            "Beta": self.beta,
            "R-Squared": self.r_squared,
            "Calmar Ratio": self.calmar_ratio(),
        }
    
    def get_metrics_df(self) -> pd.DataFrame:
        """Get metrics as a formatted DataFrame."""
        metrics = self.get_all_metrics()
        
        # Format the values
        formatted = []
        for key, value in metrics.items():
            if isinstance(value, pd.Timestamp):
                formatted.append({"Metric": key, "Value": value.strftime("%Y-%m-%d")})
            elif "Drawdown" in key or "Return" in key or "Volatility" in key or "Alpha" in key:
                formatted.append({"Metric": key, "Value": f"{value*100:.2f}%"})
            elif isinstance(value, float):
                formatted.append({"Metric": key, "Value": f"{value:.4f}"})
            else:
                formatted.append({"Metric": key, "Value": str(value)})
        
        return pd.DataFrame(formatted)


def compare_portfolios(
    portfolios: Dict[str, pd.Series],
    benchmark_returns: pd.Series,
    risk_free_rate: float = 0.02
) -> pd.DataFrame:
    """
    Compare multiple portfolios on all metrics.
    
    Args:
        portfolios: Dictionary of {name: returns_series}
        benchmark_returns: Benchmark return series
        risk_free_rate: Annualized risk-free rate
    
    Returns:
        DataFrame comparing all portfolios
    """
    results = []
    
    for name, returns in portfolios.items():
        metrics = PerformanceMetrics(returns, benchmark_returns, risk_free_rate)
        all_metrics = metrics.get_all_metrics()
        all_metrics["Portfolio"] = name
        results.append(all_metrics)
    
    # Add benchmark
    benchmark_metrics = PerformanceMetrics(benchmark_returns, benchmark_returns, risk_free_rate)
    bench_all = benchmark_metrics.get_all_metrics()
    bench_all["Portfolio"] = "Benchmark (SPY)"
    results.append(bench_all)
    
    df = pd.DataFrame(results)
    
    # Reorder columns
    cols = ["Portfolio"] + [c for c in df.columns if c != "Portfolio"]
    df = df[cols]
    
    return df


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    dates = pd.date_range("2010-01-31", periods=120, freq="ME")  # 10 years
    
    # Create sample returns
    benchmark = pd.Series(np.random.normal(0.008, 0.04, 120), index=dates, name="SPY")
    portfolio = pd.Series(np.random.normal(0.010, 0.035, 120), index=dates, name="Portfolio")
    
    # Calculate metrics
    metrics = PerformanceMetrics(portfolio, benchmark)
    
    print("Performance Metrics:")
    print(metrics.get_metrics_df().to_string(index=False))
    print(f"\nBeta: {metrics.get_beta():.3f}")
