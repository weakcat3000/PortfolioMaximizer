"""
Visualization Module
Creates charts and plots for portfolio analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Dict, List, Optional
import os


def set_style():
    """Set consistent plotting style."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12


def plot_cumulative_returns(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    portfolio_name: str = "Portfolio",
    title: str = "Cumulative Returns: Portfolio vs Benchmark",
    save_path: Optional[str] = None
):
    """
    Plot cumulative returns comparison.
    """
    set_style()
    
    # Calculate cumulative returns
    port_cum = (1 + portfolio_returns).cumprod()
    bench_cum = (1 + benchmark_returns).cumprod()
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    ax.plot(port_cum.index, port_cum.values, label=portfolio_name, linewidth=2, color='#2E86AB')
    ax.plot(bench_cum.index, bench_cum.values, label='SPY (Benchmark)', linewidth=2, color='#A23B72', linestyle='--')
    
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Growth of $1')
    ax.legend(loc='upper left', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Format x-axis
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.xticks(rotation=45)
    
    # Add final values annotation
    port_final = port_cum.iloc[-1]
    bench_final = bench_cum.iloc[-1]
    ax.annotate(f'${port_final:.2f}', xy=(port_cum.index[-1], port_final), 
                xytext=(10, 0), textcoords='offset points', fontsize=11, color='#2E86AB')
    ax.annotate(f'${bench_final:.2f}', xy=(bench_cum.index[-1], bench_final), 
                xytext=(10, 0), textcoords='offset points', fontsize=11, color='#A23B72')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    return fig


def plot_drawdown(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    portfolio_name: str = "Portfolio",
    title: str = "Drawdown Analysis",
    save_path: Optional[str] = None
):
    """
    Plot drawdown comparison.
    """
    set_style()
    
    def calculate_drawdown(returns):
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown
    
    port_dd = calculate_drawdown(portfolio_returns)
    bench_dd = calculate_drawdown(benchmark_returns)
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    ax.fill_between(port_dd.index, port_dd.values, 0, alpha=0.3, color='#2E86AB', label=portfolio_name)
    ax.plot(port_dd.index, port_dd.values, color='#2E86AB', linewidth=1)
    
    ax.fill_between(bench_dd.index, bench_dd.values, 0, alpha=0.3, color='#A23B72', label='SPY (Benchmark)')
    ax.plot(bench_dd.index, bench_dd.values, color='#A23B72', linewidth=1, linestyle='--')
    
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Drawdown')
    ax.legend(loc='lower left', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    
    # Format x-axis
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.xticks(rotation=45)
    
    # Add max drawdown annotations
    port_max_dd = port_dd.min()
    bench_max_dd = bench_dd.min()
    ax.axhline(y=port_max_dd, color='#2E86AB', linestyle=':', alpha=0.5)
    ax.axhline(y=bench_max_dd, color='#A23B72', linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    return fig


def plot_allocation_pie(
    allocations: Dict[str, float],
    title: str = "Portfolio Allocation",
    save_path: Optional[str] = None
):
    """
    Plot portfolio allocation as a pie chart.
    """
    set_style()
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    labels = list(allocations.keys())
    sizes = list(allocations.values())
    
    # Color palette
    colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
    
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, autopct='%1.1f%%',
        colors=colors, startangle=90,
        explode=[0.02] * len(labels),
        shadow=False
    )
    
    # Style the text
    for text in texts:
        text.set_fontsize(11)
    for autotext in autotexts:
        autotext.set_fontsize(10)
        autotext.set_fontweight('bold')
    
    ax.set_title(title, fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    return fig


def plot_rolling_metrics(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    window: int = 12,  # 12 months = 1 year
    portfolio_name: str = "Portfolio",
    save_path: Optional[str] = None
):
    """
    Plot rolling Sharpe ratio and volatility.
    """
    set_style()
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    rf_monthly = 0.02 / 12  # Assume 2% annual risk-free rate
    
    # Rolling Sharpe Ratio
    port_rolling_sharpe = (portfolio_returns.rolling(window).mean() - rf_monthly) / portfolio_returns.rolling(window).std() * np.sqrt(12)
    bench_rolling_sharpe = (benchmark_returns.rolling(window).mean() - rf_monthly) / benchmark_returns.rolling(window).std() * np.sqrt(12)
    
    axes[0].plot(port_rolling_sharpe.index, port_rolling_sharpe.values, label=portfolio_name, linewidth=2, color='#2E86AB')
    axes[0].plot(bench_rolling_sharpe.index, bench_rolling_sharpe.values, label='SPY', linewidth=2, color='#A23B72', linestyle='--')
    axes[0].axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    axes[0].axhline(y=1, color='green', linestyle=':', alpha=0.5, label='Sharpe = 1')
    axes[0].set_title(f'Rolling {window}-Month Sharpe Ratio', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Sharpe Ratio')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)
    
    # Rolling Volatility
    port_rolling_vol = portfolio_returns.rolling(window).std() * np.sqrt(12)
    bench_rolling_vol = benchmark_returns.rolling(window).std() * np.sqrt(12)
    
    axes[1].plot(port_rolling_vol.index, port_rolling_vol.values, label=portfolio_name, linewidth=2, color='#2E86AB')
    axes[1].plot(bench_rolling_vol.index, bench_rolling_vol.values, label='SPY', linewidth=2, color='#A23B72', linestyle='--')
    axes[1].set_title(f'Rolling {window}-Month Volatility (Annualized)', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Volatility')
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    
    for ax in axes:
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    return fig


def plot_annual_returns(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    portfolio_name: str = "Portfolio",
    save_path: Optional[str] = None
):
    """
    Plot annual returns as bar chart.
    """
    set_style()
    
    # Calculate annual returns
    port_annual = (1 + portfolio_returns).resample('YE').prod() - 1
    bench_annual = (1 + benchmark_returns).resample('YE').prod() - 1
    
    # Align indices
    years = port_annual.index.year
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    x = np.arange(len(years))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, port_annual.values, width, label=portfolio_name, color='#2E86AB', alpha=0.8)
    bars2 = ax.bar(x + width/2, bench_annual.values, width, label='SPY', color='#A23B72', alpha=0.8)
    
    ax.set_title('Annual Returns Comparison', fontsize=16, fontweight='bold')
    ax.set_xlabel('Year')
    ax.set_ylabel('Annual Return')
    ax.set_xticks(x)
    ax.set_xticklabels(years)
    ax.legend(loc='upper left')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1%}',
                   xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3 if height >= 0 else -12),
                   textcoords="offset points",
                   ha='center', va='bottom' if height >= 0 else 'top',
                   fontsize=8, rotation=90)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    return fig


def plot_metrics_comparison(
    metrics_df: pd.DataFrame,
    save_path: Optional[str] = None
):
    """
    Plot bar chart comparing key metrics between portfolio and benchmark.
    """
    set_style()
    
    # Select key metrics for visualization
    key_metrics = ['Sharpe Ratio', 'Sortino Ratio', 'Information Ratio', 'Treynor Ratio']
    
    port_metrics = metrics_df[metrics_df['Portfolio'] != 'Benchmark (SPY)'].iloc[0]
    bench_metrics = metrics_df[metrics_df['Portfolio'] == 'Benchmark (SPY)'].iloc[0]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(key_metrics))
    width = 0.35
    
    port_values = [port_metrics[m] for m in key_metrics]
    bench_values = [bench_metrics[m] for m in key_metrics]
    
    bars1 = ax.bar(x - width/2, port_values, width, label=port_metrics['Portfolio'], color='#2E86AB', alpha=0.8)
    bars2 = ax.bar(x + width/2, bench_values, width, label='SPY', color='#A23B72', alpha=0.8)
    
    ax.set_title('Performance Metrics Comparison', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(key_metrics)
    ax.legend()
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    return fig


def generate_all_charts(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    allocations: Dict[str, float],
    portfolio_name: str,
    output_dir: str = "charts"
):
    """
    Generate all charts and save to output directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nGenerating charts in '{output_dir}/'...")
    
    # 1. Cumulative Returns
    plot_cumulative_returns(
        portfolio_returns, benchmark_returns, portfolio_name,
        save_path=os.path.join(output_dir, "01_cumulative_returns.png")
    )
    
    # 2. Drawdown
    plot_drawdown(
        portfolio_returns, benchmark_returns, portfolio_name,
        save_path=os.path.join(output_dir, "02_drawdown.png")
    )
    
    # 3. Allocation Pie
    plot_allocation_pie(
        allocations, f"{portfolio_name} - Asset Allocation",
        save_path=os.path.join(output_dir, "03_allocation.png")
    )
    
    # 4. Rolling Metrics
    plot_rolling_metrics(
        portfolio_returns, benchmark_returns, 12, portfolio_name,
        save_path=os.path.join(output_dir, "04_rolling_metrics.png")
    )
    
    # 5. Annual Returns
    plot_annual_returns(
        portfolio_returns, benchmark_returns, portfolio_name,
        save_path=os.path.join(output_dir, "05_annual_returns.png")
    )
    
    print(f"\n[OK] All charts saved to '{output_dir}/'")


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    dates = pd.date_range("2010-01-31", periods=120, freq="ME")
    
    benchmark = pd.Series(np.random.normal(0.008, 0.04, 120), index=dates, name="SPY")
    portfolio = pd.Series(np.random.normal(0.010, 0.035, 120), index=dates, name="Portfolio")
    
    allocations = {"VTI": 0.4, "BND": 0.3, "VEA": 0.15, "GLD": 0.15}
    
    plot_cumulative_returns(portfolio, benchmark, "Sample Portfolio")
    plot_allocation_pie(allocations)
