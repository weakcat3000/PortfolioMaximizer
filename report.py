"""
Report Generator Module
Exports portfolio analysis results to CSV and Excel
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import os
from datetime import datetime


def create_summary_report(
    portfolio_name: str,
    client_profile: str,
    allocations: Dict[str, float],
    metrics: Dict,
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series
) -> Dict[str, pd.DataFrame]:
    """
    Create all DataFrames needed for the report.
    
    Returns dictionary of DataFrames for different sections.
    """
    # 1. Client Profile Summary
    profile_data = {
        "Item": ["Portfolio Name", "Client Profile", "Analysis Date", "Data Period"],
        "Value": [
            portfolio_name,
            client_profile,
            datetime.now().strftime("%Y-%m-%d"),
            f"{portfolio_returns.index[0].strftime('%Y-%m')} to {portfolio_returns.index[-1].strftime('%Y-%m')}"
        ]
    }
    profile_df = pd.DataFrame(profile_data)
    
    # 2. Allocation Table
    allocation_df = pd.DataFrame([
        {"Asset": ticker, "Weight": weight, "Weight (%)": f"{weight*100:.1f}%"}
        for ticker, weight in sorted(allocations.items(), key=lambda x: -x[1])
    ])
    
    # 3. Performance Metrics
    metrics_formatted = []
    for key, value in metrics.items():
        if isinstance(value, pd.Timestamp):
            formatted_value = value.strftime("%Y-%m-%d")
        elif key in ["Maximum Drawdown", "Annualized Return", "Annualized Volatility", "Jensen's Alpha"]:
            formatted_value = f"{value*100:.2f}%"
        elif isinstance(value, float):
            formatted_value = f"{value:.4f}"
        else:
            formatted_value = str(value)
        metrics_formatted.append({"Metric": key, "Value": formatted_value})
    
    metrics_df = pd.DataFrame(metrics_formatted)
    
    # 4. Monthly Returns Table
    monthly_df = pd.DataFrame({
        "Date": portfolio_returns.index,
        "Portfolio Return": portfolio_returns.values,
        "Benchmark Return": benchmark_returns.values,
        "Excess Return": portfolio_returns.values - benchmark_returns.values
    })
    monthly_df['Date'] = monthly_df['Date'].dt.strftime('%Y-%m')
    
    # Format as percentages for display
    for col in ["Portfolio Return", "Benchmark Return", "Excess Return"]:
        monthly_df[f"{col} (%)"] = monthly_df[col].apply(lambda x: f"{x*100:.2f}%")
    
    # 5. Annual Returns Summary
    port_annual = (1 + portfolio_returns).resample('YE').prod() - 1
    bench_annual = (1 + benchmark_returns).resample('YE').prod() - 1
    
    annual_df = pd.DataFrame({
        "Year": port_annual.index.year,
        "Portfolio Return (%)": (port_annual.values * 100).round(2),
        "Benchmark Return (%)": (bench_annual.values * 100).round(2),
        "Excess Return (%)": ((port_annual.values - bench_annual.values) * 100).round(2),
        "Outperformed": port_annual.values > bench_annual.values
    })
    
    # 6. Cumulative performance
    port_cum = (1 + portfolio_returns).cumprod()
    bench_cum = (1 + benchmark_returns).cumprod()
    
    cumulative_df = pd.DataFrame({
        "Date": portfolio_returns.index,
        "Portfolio Value ($)": port_cum.values.round(4),
        "Benchmark Value ($)": bench_cum.values.round(4)
    })
    cumulative_df['Date'] = cumulative_df['Date'].dt.strftime('%Y-%m')
    
    return {
        "Profile": profile_df,
        "Allocations": allocation_df,
        "Performance Metrics": metrics_df,
        "Monthly Returns": monthly_df[["Date", "Portfolio Return", "Benchmark Return", "Excess Return"]],
        "Annual Returns": annual_df,
        "Cumulative Growth": cumulative_df
    }


def export_to_excel(
    report_data: Dict[str, pd.DataFrame],
    filename: str = "portfolio_report.xlsx"
):
    """
    Export all report data to an Excel file with multiple sheets.
    """
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        for sheet_name, df in report_data.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    
    print(f"[OK] Report exported to: {filename}")


def export_to_csv(
    report_data: Dict[str, pd.DataFrame],
    output_dir: str = "report_csv"
):
    """
    Export all report data to separate CSV files.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for name, df in report_data.items():
        filename = os.path.join(output_dir, f"{name.lower().replace(' ', '_')}.csv")
        df.to_csv(filename, index=False)
        print(f"  Saved: {filename}")
    
    print(f"[OK] All CSV files exported to: {output_dir}/")


def print_report_summary(
    portfolio_name: str,
    metrics: Dict,
    allocations: Dict[str, float]
):
    """
    Print a formatted summary to console.
    """
    print("\n" + "="*70)
    print(f"  PORTFOLIO ANALYSIS REPORT: {portfolio_name}")
    print("="*70)
    
    print("\n[ASSET ALLOCATION]")
    print("-"*40)
    for ticker, weight in sorted(allocations.items(), key=lambda x: -x[1]):
        bar = "#" * int(weight * 30)
        print(f"  {ticker:6s} {weight*100:5.1f}% {bar}")
    
    print("\n[KEY PERFORMANCE METRICS]")
    print("-"*40)
    
    key_metrics = [
        ("Annualized Return", metrics.get("Annualized Return", 0), True),
        ("Annualized Volatility", metrics.get("Annualized Volatility", 0), True),
        ("Maximum Drawdown", metrics.get("Maximum Drawdown", 0), True),
        ("Sharpe Ratio", metrics.get("Sharpe Ratio", 0), False),
        ("Sortino Ratio", metrics.get("Sortino Ratio", 0), False),
        ("Treynor Ratio", metrics.get("Treynor Ratio", 0), False),
        ("Information Ratio", metrics.get("Information Ratio", 0), False),
        ("Jensen's Alpha", metrics.get("Jensen's Alpha", 0), True),
        ("Beta", metrics.get("Beta", 0), False),
    ]
    
    for name, value, is_pct in key_metrics:
        if is_pct:
            print(f"  {name:25s}: {value*100:>8.2f}%")
        else:
            print(f"  {name:25s}: {value:>8.4f}")
    
    print("\n" + "="*70)


def generate_investment_objectives_check(
    client_profile: str,
    metrics: Dict
) -> pd.DataFrame:
    """
    Check if portfolio meets client's investment objectives.
    """
    checks = []
    
    ann_return = metrics.get("Annualized Return", 0)
    ann_vol = metrics.get("Annualized Volatility", 0)
    max_dd = abs(metrics.get("Maximum Drawdown", 0))
    sharpe = metrics.get("Sharpe Ratio", 0)
    
    if client_profile.lower() == "conservative":
        # Richard Tan: Max DD < 10%
        checks.append({
            "Objective": "Maximum Drawdown < 10%",
            "Target": "< 10%",
            "Actual": f"{max_dd*100:.2f}%",
            "Met": "YES" if max_dd < 0.10 else "NO"
        })
        checks.append({
            "Objective": "Lower Volatility than SPY",
            "Target": "< SPY Vol",
            "Actual": f"{ann_vol*100:.2f}%",
            "Met": "Check vs SPY"
        })
    
    elif client_profile.lower() == "balanced":
        # Sophia Lim: Max DD < 20%, Vol < 15%
        checks.append({
            "Objective": "Maximum Drawdown < 20%",
            "Target": "< 20%",
            "Actual": f"{max_dd*100:.2f}%",
            "Met": "YES" if max_dd < 0.20 else "NO"
        })
        checks.append({
            "Objective": "Annual Volatility < 15%",
            "Target": "< 15%",
            "Actual": f"{ann_vol*100:.2f}%",
            "Met": "YES" if ann_vol < 0.15 else "NO"
        })
    
    elif client_profile.lower() == "aggressive":
        # David Lee: Sharpe > 1
        checks.append({
            "Objective": "Sharpe Ratio > 1",
            "Target": "> 1.0",
            "Actual": f"{sharpe:.4f}",
            "Met": "YES" if sharpe > 1.0 else "NO"
        })
        checks.append({
            "Objective": "Target 2x SPY Returns",
            "Target": "2x SPY",
            "Actual": f"{ann_return*100:.2f}%",
            "Met": "Check vs SPY"
        })
    
    return pd.DataFrame(checks)


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    dates = pd.date_range("2010-01-31", periods=60, freq="ME")
    
    portfolio_returns = pd.Series(np.random.normal(0.01, 0.035, 60), index=dates)
    benchmark_returns = pd.Series(np.random.normal(0.008, 0.04, 60), index=dates)
    
    allocations = {"VTI": 0.4, "BND": 0.3, "VEA": 0.15, "GLD": 0.15}
    
    metrics = {
        "Sharpe Ratio": 0.85,
        "Sortino Ratio": 1.2,
        "Treynor Ratio": 0.07,
        "Information Ratio": 0.35,
        "Jensen's Alpha": 0.02,
        "Maximum Drawdown": -0.15,
        "Annualized Return": 0.10,
        "Annualized Volatility": 0.12,
        "Beta": 0.9
    }
    
    # Create and print report
    print_report_summary("Sample Portfolio", metrics, allocations)
    
    # Generate report data
    report_data = create_summary_report(
        "Sample Portfolio",
        "Balanced",
        allocations,
        metrics,
        portfolio_returns,
        benchmark_returns
    )
    
    # Export
    export_to_excel(report_data, "sample_report.xlsx")
