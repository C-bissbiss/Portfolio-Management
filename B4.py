import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LassoCV
from scipy.optimize import minimize
import pandas_datareader as web
import warnings

# Suppress warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Retrieve 48 industry portfolios from Kenneth French's website
def retrieve_full_industry_data():
    industry_reader = web.famafrench.FamaFrenchReader('48_Industry_Portfolios', start='2014-12', end='2024-12')
    portfolios_data = industry_reader.read()[0]  # First table contains the return data
    portfolios_data = portfolios_data.replace([-99.99, -999], np.nan).dropna()
    return portfolios_data / 100  # Convert percentage to decimal

# Compute the estimated maximum Sharpe ratio (θ) using equation (2.32)
def estimate_max_sharpe(returns, rf_rate):
    mean_returns = returns.mean()
    cov_matrix = estimate_cov_matrix(returns)
    inv_cov = np.linalg.pinv(cov_matrix)  # Use pseudo-inverse for numerical stability
    
    # Compute θ as in equation (2.32): θ̂ = μ' Σ⁻¹ μ
    theta_hat = mean_returns.T @ inv_cov @ mean_returns
    
    # Compute the response variable rc = (1 + θ) / sqrt(θ)
    rc = (1 + theta_hat) / np.sqrt(theta_hat)
    
    return theta_hat, rc

# Estimate shrinkage covariance matrix using Ledoit-Wolf
def estimate_cov_matrix(returns):
    lw = LedoitWolf()
    return lw.fit(returns).covariance_

# Apply LASSO regression with cross-validation for sparse selection of industries
def lasso_feature_selection(returns, rc):
    """
    Uses LASSO cross-validation to select industries based on the transformed response `rc`.
    """
    lasso = LassoCV(cv=5).fit(returns.T, np.full(returns.shape[1], rc))  # Response is `rc`
    selected = np.where(lasso.coef_ != 0)[0]  # Select non-zero coefficients
    
    # If no industries are selected, fallback to the top 5 industries
    return returns.columns[selected] if len(selected) > 0 else returns.columns[:5]

# Calculate portfolio Sharpe ratio
def sharpe_ratio(weights, expected_returns, cov_matrix, rf_rate):
    port_return = np.dot(weights, expected_returns)
    port_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) + 1e-6  # Prevent divide-by-zero
    return (port_return - rf_rate) / port_risk

# Monte Carlo optimization with MAXSER using LASSO selection
def monte_carlo_maxser(returns, rf_rate, risk_constraint, n_simulations=5000, allow_short_selling=True):
    best_sharpe = -np.inf
    best_selection, best_weights = None, None
    
    # Step 1: Compute max Sharpe ratio estimate θ and response rc
    theta_hat, rc = estimate_max_sharpe(returns, rf_rate)
    
    # Step 2: Use LASSO cross-validation to select industries based on rc
    selected_industries = lasso_feature_selection(returns, rc)
    if len(selected_industries) == 0:
        selected_industries = returns.columns[:5]  # Default selection

    subset_returns = returns[selected_industries]
    expected_returns = subset_returns.mean()
    cov_matrix = estimate_cov_matrix(subset_returns)
    n_assets = len(expected_returns)

    # Objective function (negative Sharpe ratio to minimize)
    def neg_sharpe(weights):
        return -sharpe_ratio(weights, expected_returns, cov_matrix, rf_rate)

    # Constraints and bounds
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Sum of weights = 1
        {'type': 'ineq', 'fun': lambda x: risk_constraint - np.sqrt(np.dot(x.T, np.dot(cov_matrix, x)))}  # Risk constraint
    ]
    bounds = [(-1, 1) if allow_short_selling else (0, 1) for _ in range(n_assets)]

    # Initial weights
    initial_weights = np.ones(n_assets) / n_assets

    # Optimize
    result = minimize(neg_sharpe, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)

    # Store best result if successful
    if result.success:
        best_sharpe = -result.fun
        best_selection = selected_industries
        best_weights = result.x

    return best_selection, best_weights, best_sharpe

# Main execution
portfolios_data = retrieve_full_industry_data()
risk_constraint = 0.04  # As per the authors
rf_rate = 0.4 / 100  # Example risk-free rate

# Find best portfolios using Monte Carlo with MAXSER and LASSO
best_industries_short, best_weights_short, best_sharpe_short = monte_carlo_maxser(portfolios_data, rf_rate, risk_constraint, 5000, True)
best_industries_noshort, best_weights_noshort, best_sharpe_noshort = monte_carlo_maxser(portfolios_data, rf_rate, risk_constraint, 5000, False)

# Display results
print("Best Portfolio With Short Selling (MAXSER + LASSO):")
print(pd.DataFrame({'Industry': best_industries_short, 'Weight': best_weights_short}).set_index('Industry'))
print(f"Sharpe Ratio: {best_sharpe_short:.4f}\n")

print("Best Portfolio Without Short Selling (MAXSER + LASSO):")
print(pd.DataFrame({'Industry': best_industries_noshort, 'Weight': best_weights_noshort}).set_index('Industry'))
print(f"Sharpe Ratio: {best_sharpe_noshort:.4f}\n")
