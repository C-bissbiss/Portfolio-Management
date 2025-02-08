import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import Lasso
from scipy.optimize import minimize
import pandas_datareader.famafrench as ff
import warnings

def retrieve_full_industry_data(start_date='2014-12', end_date='2024-12'):
    """
    Retrieves 48 industry portfolio return data from Kenneth French's website.
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

def risk_based_lasso_cv(returns, target_risk, allow_short_selling=True):
    """
    Modified LASSO cross-validation with direct risk targeting.
    """
    # Define a more granular lambda grid
    lambda_grid = np.logspace(-10, 0, 100)
    
    best_lambda = None
    min_risk_diff = float('inf')
    
    # Calculate covariance matrix once
    cov_matrix = np.cov(returns, rowvar=False)
    
    for lam in lambda_grid:
        try:
            lasso = Lasso(alpha=lam, 
                         positive=not allow_short_selling,
                         max_iter=100000,
                         tol=1e-8)
            
            # Fit on raw returns
            lasso.fit(returns, returns.mean(axis=0))
            
            # Skip if all coefficients are too close to zero
            if np.max(np.abs(lasso.coef_)) < 1e-10:
                continue
                
            # Normalize weights
            weights = lasso.coef_ / np.sum(np.abs(lasso.coef_))
            
            # Calculate portfolio risk
            portfolio_risk = np.sqrt(weights @ cov_matrix @ weights)
            
            # Update if this lambda gives risk closer to target
            risk_diff = abs(portfolio_risk - target_risk)
            if risk_diff < min_risk_diff:
                min_risk_diff = risk_diff
                best_lambda = lam
                
        except Exception as e:
            continue
            
    if best_lambda is None:
        raise ValueError("Could not find suitable lambda parameter")
        
    return best_lambda

def solve_maxser_portfolio(returns, target_risk=0.04, allow_short_selling=True):
    """
    Modified portfolio optimization with improved risk targeting.
    """
    # Calculate expected returns and covariance
    expected_returns = returns.mean()
    cov_matrix = LedoitWolf().fit(returns).covariance_
    
    try:
        # Get optimal lambda
        optimal_lambda = risk_based_lasso_cv(returns, target_risk, allow_short_selling)
        
        # Perform LASSO selection
        lasso = Lasso(alpha=optimal_lambda,
                     positive=not allow_short_selling,
                     max_iter=100000,
                     tol=1e-8)
        
        lasso.fit(returns, returns.mean(axis=0))
        
        # Get initial weights and selected assets
        initial_weights = lasso.coef_ / np.sum(np.abs(lasso.coef_))
        selected_indices = np.where(np.abs(initial_weights) > 1e-6)[0]
        
        if len(selected_indices) < 3:
            # If too few assets selected, include top assets by Sharpe ratio
            individual_sharpes = expected_returns / np.sqrt(np.diag(cov_matrix))
            top_sharpe_indices = np.argsort(individual_sharpes)[-5:]
            selected_indices = np.unique(np.concatenate([selected_indices, top_sharpe_indices]))
        
        # Extract selected assets data
        selected_returns = expected_returns.iloc[selected_indices]
        selected_cov = cov_matrix[np.ix_(selected_indices, selected_indices)]
        
        def objective(w):
            port_return = w @ selected_returns
            port_risk = np.sqrt(w @ selected_cov @ w)
            return -port_return / port_risk if port_risk > 0 else 0
            
        def risk_constraint(w):
            return np.sqrt(w @ selected_cov @ w) - target_risk
        
        # Set up constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'eq', 'fun': risk_constraint}
        ]
        
        # Set bounds based on short-selling allowance
        bounds = [(-1, 1) if allow_short_selling else (0, 1) 
                 for _ in range(len(selected_indices))]
        
        # Initial guess: normalized selected weights
        x0 = initial_weights[selected_indices]
        x0 = x0 / np.sum(np.abs(x0))
        
        # Optimize
        result = minimize(objective,
                        x0=x0,
                        method='SLSQP',
                        bounds=bounds,
                        constraints=constraints,
                        options={'maxiter': 10000, 'ftol': 1e-12})
        
        if not result.success:
            raise ValueError("Optimization failed to converge")
            
        weights = result.x
        portfolio_return = weights @ selected_returns
        portfolio_risk = np.sqrt(weights @ selected_cov @ weights)
        sharpe = portfolio_return / portfolio_risk
        
        # Verify the solution
        if abs(np.sum(weights) - 1) > 1e-4 or abs(portfolio_risk - target_risk) > 1e-4:
            raise ValueError("Solution does not satisfy constraints")
            
        return returns.columns[selected_indices], weights, sharpe
        
    except Exception as e:
        raise RuntimeError(f"Portfolio optimization failed: {str(e)}")

def main():
    """Main execution function with improved error handling"""
    try:
        # Set parameters
        start_date = '2014-12'
        end_date = '2024-12'
        target_risk = 0.04
        
        # Retrieve data
        print("Retrieving industry data...")
        portfolios_data = retrieve_full_industry_data(start_date, end_date)
        print(f"Data shape: {portfolios_data.shape}")
        
        # Format settings
        pd.set_option('display.float_format', lambda x: '%.4f' % x)
        
        # Optimize portfolios
        for allow_short in [True, False]:
            try:
                label = "WITH" if allow_short else "WITHOUT"
                print(f"\nOptimizing portfolio {label} short-selling...")
                
                industries, weights, sharpe = solve_maxser_portfolio(
                    portfolios_data, 
                    target_risk=target_risk,
                    allow_short_selling=allow_short
                )
                
                portfolio_df = pd.DataFrame({
                    'Asset': industries,
                    'Weight (%)': [f'{w * 100:.2f}%' for w in weights]
                })
                
                print("\nSelected Assets and Weights:")
                print(portfolio_df)
                print(f"\nPortfolio Statistics:")
                print(f"Number of assets: {len(weights)}")
                print(f"Sum of weights: {np.sum(weights):.4f}")
                print(f"Portfolio risk: {np.sqrt(weights @ np.cov(portfolios_data[industries].T) @ weights):.4f}")
                print(f"Sharpe ratio: {sharpe:.4f}")
                
            except Exception as e:
                print(f"Failed to optimize portfolio {label} short-selling: {str(e)}")
                continue
            
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()