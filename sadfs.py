import numpy as np
import pandas as pd
from pandas_datareader.famafrench import FamaFrenchReader
from sklearn.covariance import LedoitWolf
from typing import Tuple
import warnings

def retrieve_industry_data(start_date='2014-12', end_date='2024-11'):
    """Retrieve and clean industry return data"""
    try:
        ff_data = FamaFrenchReader('48_Industry_Portfolios', start=start_date, end=end_date).read()[0]
        cleaned_data = ff_data.replace([-99.99, -999], np.nan).dropna() / 100
        return cleaned_data
    except Exception as e:
        raise RuntimeError(f"Failed to retrieve industry data: {str(e)}")

def get_ledoit_wolf_estimates(returns: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Ledoit-Wolf shrinkage estimates for mean and covariance
    """
    T, N = returns.shape
    
    # Ledoit-Wolf shrinkage for covariance
    lw = LedoitWolf()
    shrunk_cov = lw.fit(returns).covariance_
    
    # Simple shrinkage for mean returns
    shrinkage_factor = min(0.9, (N + 2)/(2 * T))
    shrunk_mean = returns.mean() * (1 - shrinkage_factor)
    
    return shrunk_mean, shrunk_cov

def optimize_with_short(returns: pd.DataFrame, target_risk: float = 0.15) -> Tuple[pd.Series, dict]:
    """
    Optimize portfolio with short-selling using Ledoit-Wolf shrinkage
    """
    T, N = returns.shape
    if T < 2 * N:
        warnings.warn(f"Sample size {T} may be insufficient for {N} assets")
    
    # Get shrinkage estimates
    shrunk_mean, shrunk_cov = get_ledoit_wolf_estimates(returns)
    
    # Compute optimal weights
    try:
        inv_cov = np.linalg.inv(shrunk_cov)
    except np.linalg.LinAlgError:
        inv_cov = np.linalg.pinv(shrunk_cov)
        warnings.warn("Using pseudo-inverse due to singular covariance matrix")
    
    # Compute efficient portfolio weights
    weights = inv_cov @ shrunk_mean
    weights = weights / np.sum(np.abs(weights))  # Normalize
    
    # Scale to target risk
    port_vol = np.sqrt(weights @ shrunk_cov @ weights)
    weights = weights * (target_risk / port_vol)
    
    final_weights = pd.Series(weights, index=returns.columns)
    
    # Compute metrics
    metrics = {
        'expected_return': final_weights @ shrunk_mean,
        'realized_risk': np.sqrt(final_weights @ shrunk_cov @ final_weights),
        'sharpe_ratio': (final_weights @ shrunk_mean) / np.sqrt(final_weights @ shrunk_cov @ final_weights),
        'long_exposure': np.sum(np.maximum(weights, 0)),
        'short_exposure': abs(np.sum(np.minimum(weights, 0))),
        'active_positions': np.sum(np.abs(weights) > 1e-4)
    }
    
    return final_weights, metrics

def optimize_no_short(returns: pd.DataFrame, target_risk: float = 0.15) -> Tuple[pd.Series, dict]:
    """
    Optimize portfolio without short-selling using Ledoit-Wolf shrinkage
    """
    T, N = returns.shape
    if T < 2 * N:
        warnings.warn(f"Sample size {T} may be insufficient for {N} assets")
    
    # Get shrinkage estimates
    shrunk_mean, shrunk_cov = get_ledoit_wolf_estimates(returns)
    
    # Compute initial weights based on relative Sharpe ratios
    vols = np.sqrt(np.diag(shrunk_cov))
    sharpe_ratios = shrunk_mean / vols
    
    # Only consider assets with positive Sharpe ratios
    positive_sharpe_mask = sharpe_ratios > 0
    weights = np.zeros(N)
    
    if np.sum(positive_sharpe_mask) > 0:
        weights[positive_sharpe_mask] = sharpe_ratios[positive_sharpe_mask]
        weights = weights / np.sum(weights)  # Normalize
        
        # Scale to target risk
        port_vol = np.sqrt(weights @ shrunk_cov @ weights)
        weights = weights * (target_risk / port_vol)
    else:
        warnings.warn("No positive Sharpe ratios found, defaulting to equal weights")
        weights = np.ones(N) / N
    
    final_weights = pd.Series(weights, index=returns.columns)
    
    # Compute metrics
    metrics = {
        'expected_return': final_weights @ shrunk_mean,
        'realized_risk': np.sqrt(final_weights @ shrunk_cov @ final_weights),
        'sharpe_ratio': (final_weights @ shrunk_mean) / np.sqrt(final_weights @ shrunk_cov @ final_weights),
        'max_weight': np.max(weights),
        'active_positions': np.sum(weights > 1e-4)
    }
    
    return final_weights, metrics

if __name__ == "__main__":
    # Get data
    returns = retrieve_industry_data()
    
    # Run both optimizations
    print("Optimizing portfolios...")
    weights_short, metrics_short = optimize_with_short(returns)
    weights_no_short, metrics_no_short = optimize_no_short(returns)
    
    # Print results for short-selling portfolio
    print("\nPortfolio with Short-Selling:")
    print("Metrics:")
    for key, value in metrics_short.items():
        print(f"{key}: {value:.4f}")
    
    print("\nTop 5 Long Positions:")
    print(weights_short[weights_short > 0].sort_values(ascending=False).head())
    print("\nTop 5 Short Positions:")
    print(weights_short[weights_short < 0].sort_values().head())
    
    # Print results for long-only portfolio
    print("\nLong-Only Portfolio:")
    print("Metrics:")
    for key, value in metrics_no_short.items():
        print(f"{key}: {value:.4f}")
    
    print("\nTop 5 Positions:")
    print(weights_no_short[weights_no_short > 0].sort_values(ascending=False).head())