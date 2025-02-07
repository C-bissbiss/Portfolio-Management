import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LassoCV
from scipy.optimize import minimize
import pandas_datareader.famafrench as ff
import warnings
import scipy.special as sp

def beta_function(a, b, x):
    """
    Implements the incomplete beta function B_x(a,b) as defined in the paper.
    
    Parameters:
    - a, b (float): Shape parameters
    - x (float): Upper limit of integration
    
    Returns:
    - float: Value of the incomplete beta function
    """
    return sp.betainc(a, b, x) * sp.beta(a, b)

def retrieve_full_industry_data(start_date='2014-12', end_date='2024-12'):
    """
    Retrieves 48 industry portfolio return data from Kenneth French's website.
    
    Parameters:
    - start_date (str): Start date in YYYY-MM format
    - end_date (str): End date in YYYY-MM format
    
    Returns:
    - DataFrame: Industry return data (in decimal format)
    """
    try:
        industry_data = ff.FamaFrenchReader('48_Industry_Portfolios', 
                                          start=start_date, 
                                          end=end_date).read()[0]
        industry_data = industry_data.replace([-99.99, -999], np.nan)
        if industry_data.isna().any().any():
            warnings.warn("Missing values detected in industry data")
        return industry_data.dropna() / 100
    except Exception as e:
        raise RuntimeError(f"Failed to retrieve industry data: {str(e)}")

def estimate_max_sharpe(returns):
    """
    Computes the adjusted estimate of the maximum Sharpe ratio (θ̂) using equation (2.32)
    and the transformed response (r̂c).
    
    Parameters:
    - returns (DataFrame): Industry return data
    
    Returns:
    - tuple: (θ̂_a, r̂c) - Adjusted max Sharpe ratio and transformed response
    
    Raises:
    - ValueError: If input data is invalid
    """
    if not isinstance(returns, pd.DataFrame):
        raise ValueError("Returns must be a pandas DataFrame")
        
    T, N = returns.shape
    if T <= N:
        raise ValueError(f"Insufficient samples (T={T}) for number of assets (N={N})")
    
    try:
        # Step 1: Compute plug-in estimator θ̂s
        mean_returns = returns.mean()
        cov_matrix = estimate_cov_matrix(returns)
        inv_cov = np.linalg.pinv(cov_matrix)
        theta_s = mean_returns @ inv_cov @ mean_returns
        
        if T > N + 2:
            # Step 2: Compute the bias correction term using equation (2.32)
            x = theta_s / (1 + theta_s)
            beta_x = beta_function(N/2, (T-N)/2, x)
            
            # Compute the numerator and denominator separately for clarity
            numerator = 2 * (theta_s)**(N/2) * (1 + theta_s)**(-(T-2)/2)
            denominator = T * beta_x
            
            # Combine terms according to equation (2.32)
            bias_correction = numerator / denominator
            theta_hat = ((T - N - 2) * theta_s - N) / T + bias_correction
        else:
            warnings.warn("Sample size T too small for bias correction; using raw estimate")
            theta_hat = theta_s
            
        # Compute transformed response rc = (1 + θ̂) / √θ̂
        rc = (1 + theta_hat) / np.sqrt(max(theta_hat, 1e-10))
        return theta_hat, rc
        
    except np.linalg.LinAlgError as e:
        raise RuntimeError(f"Linear algebra computation failed: {str(e)}")

def estimate_cov_matrix(returns):
    """
    Computes the shrinkage covariance matrix using the Ledoit-Wolf estimator.
    
    Parameters:
    - returns (DataFrame): Industry return data
    
    Returns:
    - ndarray: Shrinkage covariance matrix
    """
    try:
        lw = LedoitWolf()
        return lw.fit(returns).covariance_
    except Exception as e:
        raise RuntimeError(f"Covariance estimation failed: {str(e)}")

def lasso_feature_selection(returns, rc, min_features=5):
    """
    Uses LASSO cross-validation to select industries based on the transformed response rc.
    
    Parameters:
    - returns (DataFrame): Industry return data
    - rc (float): Transformed response variable
    - min_features (int): Minimum number of features to select
    
    Returns:
    - Index: Selected industry names
    """
    try:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaled_returns = scaler.fit_transform(returns)
        
        # Use LassoCV with transformed response rc
        lasso = LassoCV(cv=5, max_iter=50000, random_state=42)
        lasso.fit(scaled_returns, np.full(returns.shape[0], rc))
        
        # Select non-zero coefficients
        selected = np.where(lasso.coef_ != 0)[0]
        
        if len(selected) < min_features:
            warnings.warn(f"LASSO selected fewer than {min_features} features, using top performers")
            selected = returns.mean().nlargest(min_features).index
        else:
            selected = returns.columns[selected]
            
        return selected
        
    except Exception as e:
        warnings.warn(f"LASSO selection failed: {str(e)}, using top performers")
        return returns.mean().nlargest(min_features).index

def solve_maxser_portfolio(returns, rf_rate=0.004, allow_short_selling=True, min_features=5):
    """
    Solves the MAXSER portfolio optimization problem following the paper's methodology.
    
    Parameters:
    - returns (DataFrame): Industry return data
    - rf_rate (float): Risk-free rate (default: 0.4%)
    - allow_short_selling (bool): Whether to allow short positions
    - min_features (int): Minimum number of features to select
    
    Returns:
    - tuple: (selected_industries, optimal_weights, max_sharpe_ratio)
    """
    if not isinstance(returns, pd.DataFrame):
        raise ValueError("Returns must be a pandas DataFrame")
    if rf_rate < 0:
        raise ValueError("Risk-free rate must be non-negative")
    
    try:
        # Step 1: Compute max Sharpe ratio estimate and response
        theta_hat, rc = estimate_max_sharpe(returns)
        
        # Step 2: Select industries using LASSO with cross-validation
        selected_industries = lasso_feature_selection(returns, rc, min_features)
        subset_returns = returns[selected_industries]
        
        # Step 3: Solve for the MAXSER portfolio weights
        expected_returns = subset_returns.mean()
        cov_matrix = estimate_cov_matrix(subset_returns)
        n_assets = len(selected_industries)
        
        def neg_sharpe(weights):
            """Negative Sharpe ratio for optimization"""
            portfolio_return = weights @ expected_returns
            portfolio_risk = np.sqrt(weights @ cov_matrix @ weights)
            return -(portfolio_return - rf_rate) / max(portfolio_risk, 1e-10)
        
        # Set up optimization constraints
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]  # Sum of weights = 1
        bounds = [(-1, 1) if allow_short_selling else (0, 1) for _ in range(n_assets)]
        
        # Solve optimization problem
        result = minimize(
            neg_sharpe,
            x0=np.ones(n_assets) / n_assets,  # Start with equal weights
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        if not result.success:
            warnings.warn("Optimization failed, using equal weights")
            weights = np.ones(n_assets) / n_assets
            sharpe = -neg_sharpe(weights)
        else:
            weights = result.x
            sharpe = -result.fun
            
        return selected_industries, weights, sharpe
        
    except Exception as e:
        raise RuntimeError(f"Portfolio optimization failed: {str(e)}")

def main():
    """Main execution function demonstrating the MAXSER methodology"""
    try:
        # Set parameters
        start_date = '2014-12'
        end_date = '2024-12'
        rf_rate = 0.004  # 0.4%
        
        # Retrieve data
        print("Retrieving industry data...")
        portfolios_data = retrieve_full_industry_data(start_date, end_date)
        
        # Format settings for pandas
        pd.set_option('display.float_format', lambda x: '%.6f' % x)
        
        # Optimize portfolios with and without short-selling
        print("\nOptimizing portfolios...")
        results = {}
        for allow_short in [True, False]:
            label = "WITH" if allow_short else "WITHOUT"
            industries, weights, sharpe = solve_maxser_portfolio(
                portfolios_data, 
                rf_rate=rf_rate,
                allow_short_selling=allow_short
            )
            
            # Create DataFrame with formatted weights
            portfolio_df = pd.DataFrame({
                'Weight': [f'{w * 100:.6f}' for w in weights]
            }, index=industries)
            
            results[label] = {
                'portfolio': portfolio_df,
                'sharpe': sharpe
            }
            
        # Display results
        for label, result in results.items():
            print(f"\nMAXSER Optimal Portfolio {label} Short-Selling:")
            print(result['portfolio'])
            print(f"Maximum Sharpe Ratio Achieved: {result['sharpe']:.6f}")
            
    except Exception as e:
        print(f"Error in main execution: {str(e)}")

if __name__ == "__main__":
    main()