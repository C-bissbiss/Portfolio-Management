# Import all packages for the project
import pandas as pd
import numpy as np
import pandas_datareader as web
import warnings
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import sympy as sp
import random
from sklearn.covariance import LedoitWolf

# Remove future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def retrieve_industry_data():
    """
    Retrieve 48 industry portfolios from Kenneth French website
    """
    # Instantiate the reader for 48 Industry Portfolios to capture the monthly data
    industry_reader = web.famafrench.FamaFrenchReader('48_Industry_Portfolios', start='2019-12', end='2024-12')

    # Read the data
    portfolios_data = industry_reader.read()[0]  # Assuming we need the first table [0]

    # Clean missing data
    portfolios_data = portfolios_data.replace([-99.99, -999], pd.NA)

    # Close the sessions
    industry_reader.close()

    return portfolios_data

# Retrieve the data
portfolios_data = retrieve_industry_data()

# Drop all industries except for Rtail, Whlsl, BusSv, Comps, and Cnstr
portfolios_data = portfolios_data[['Rtail', 'Whlsl', 'BusSv', 'Comps', 'Cnstr']]







#### ------------------------------------------------ PART A ------------------------------------------------ ####
# A) 1. Graph the "mean-variance locus" (without the risk-free asset and w/o short-selling constraint) of the 5 industries. Specify each industry in the chart. 

# Convert percentage returns to decimal
returns = portfolios_data / 100
returns = returns.dropna()

expected_returns = returns.mean()  # Mean returns vector
cov_matrix = returns.cov()  # Covariance matrix

# Number of assets
n_assets = len(expected_returns)

# Vector of ones
ones = np.ones(n_assets)

def calculate_mvp_weights(cov_matrix, allow_short_selling=False):
    """
    Calculate Minimum Variance Portfolio (MVP) weights using an analytical approach.

    Parameters:
    -----------
    cov_matrix : np.array
        Covariance matrix of asset returns
    allow_short_selling : bool, optional
        Whether to allow negative weights (default: False)

    Returns:
    --------
    np.array: MVP weights
    """
    num_assets = cov_matrix.shape[0]
    ones = np.ones(num_assets)

    # Compute the inverse of the covariance matrix
    inv_cov_matrix = np.linalg.inv(cov_matrix)

    # Calculate MVP parameters
    A = ones.T @ inv_cov_matrix @ ones  # Scalar value
    B = ones.T @ inv_cov_matrix @ expected_returns
    mvp_weights = (inv_cov_matrix @ ones) / A  # MVP weights formula

    if allow_short_selling:
        return mvp_weights
    else:
        # If short-selling is NOT allowed, use numerical optimization to enforce constraints
        from scipy.optimize import minimize

        # Constraints: sum of weights = 1
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bounds = [(0, 1) for _ in range(num_assets)]  # No short selling

        # Initial guess: Equal weights
        initial_weights = np.ones(num_assets) / num_assets

        # Optimization
        result = minimize(
            lambda w: w.T @ cov_matrix @ w,  # Minimize portfolio variance
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        if result.success:
            return result.x
        else:
            raise ValueError("Optimization failed: " + result.message)

def plot_efficient_frontier(returns, expected_returns, cov_matrix, mvp_weights):
    """
    Plot the efficient frontier with industries and MVP point
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Returns data
    expected_returns : pd.Series
        Expected returns for each asset
    cov_matrix : pd.DataFrame
        Covariance matrix
    mvp_weights : np.array
        MVP portfolio weights
    """
    # Calculate MVP return and risk
    mvp_return = np.dot(mvp_weights, expected_returns) 
    mvp_risk = np.sqrt(mvp_weights.T @ cov_matrix @ mvp_weights)
        
    # Convert to percentages for plotting
    mvp_return_plot = mvp_return * 100
    mvp_risk_plot = mvp_risk * 100
    
    plt.figure(figsize=(10, 6))
    plt.grid(True)
    
    # Plot individual industries
    for industry in expected_returns.index:
        plt.scatter(returns[industry].std()* 100, 
                   expected_returns[industry] * 100,
                   marker='X', s=200, label=industry)
    
    # Plot MVP point
    plt.scatter(mvp_risk_plot, mvp_return_plot, 
               color="red", marker="*", s=200, 
               label="Minimum Variance Portfolio")
    
    # Add MVP coordinates
    plt.annotate(f"(r = {mvp_return_plot:.2f}%, σ = {mvp_risk_plot:.2f}%)",
                (mvp_risk_plot, mvp_return_plot),
                xytext=(10, 10), textcoords='offset points',
                fontsize=8)
    
    # Format axes
    plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter())
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
    
    plt.xlim(0, 10)
    plt.ylim(-3, 3)
    
    # Labels and title
    plt.xlabel("Standard Deviation (%)", fontsize=12)
    plt.ylabel("Expected Return (%)", fontsize=12)
    plt.title("Efficient Frontier with MVP (With Short Selling)", 
             fontsize=15, fontweight='bold')
    plt.legend()
    
    plt.savefig("efficient_frontierA1.png", bbox_inches='tight', dpi=300)
    plt.show()

# Convert percentage returns to decimal
returns = portfolios_data / 100
returns = returns.dropna()

expected_returns = returns.mean()
cov_matrix = returns.cov()

# Calculate MVP weights
mvp_weights = calculate_mvp_weights(cov_matrix, allow_short_selling=True)

# Plot the efficient frontier
plot_efficient_frontier(returns, expected_returns, cov_matrix, mvp_weights)






# A) 2. Calculate tangency portfolio and CML (w/ risk-free rate and w/ short-selling)

rf_rate = 0.4/100

inv_cov_matrix = np.linalg.inv(cov_matrix)

# Calculate the Tangency Portfolio (Optimal Risky Portfolio)
def calculate_tangency_portfolio(expected_returns, cov_matrix, rf_rate, allow_short_selling=True):
    """
    Calculate the tangency portfolio with optional short-selling constraint
    
    Parameters:
    - expected_returns: Series/array of expected returns for assets
    - cov_matrix: Covariance matrix of asset returns
    - rf_rate: Risk-free rate
    - allow_short_selling: Boolean to allow or prohibit short selling
    
    Returns:
    - Dictionary containing portfolio weights, return, risk, and Sharpe ratio
    """
    
    def neg_sharpe_ratio(weights):
        """Negative Sharpe ratio for optimization (to be minimized)"""
        portfolio_return = np.dot(weights, expected_returns)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe_ratio = (portfolio_return - rf_rate) / portfolio_volatility
        return -sharpe_ratio
    
    # Constraints and bounds
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    
    if allow_short_selling:
        # No constraints on weights if short selling is allowed
        bounds = tuple((-np.inf, np.inf) for _ in range(n_assets))
    else:
        # Weights between 0 and infinity if no short selling
        bounds = tuple((0, np.inf) for _ in range(n_assets))
    
    # Initial guess: equal weights
    initial_weights = np.ones(n_assets) / n_assets
    
    # Optimize to find weights that maximize Sharpe ratio
    result = minimize(
        neg_sharpe_ratio, 
        initial_weights, 
        method='SLSQP', 
        bounds=bounds, 
        constraints=constraints
    )
    
    # Extract optimized weights
    optimal_weights = result.x
    
    # Calculate portfolio metrics
    portfolio_return = np.dot(optimal_weights, expected_returns)
    portfolio_volatility = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))
    sharpe_ratio = (portfolio_return - rf_rate) / portfolio_volatility
    
    return {
        'weights': optimal_weights,
        'expected_return': portfolio_return,
        'risk': portfolio_volatility,
        'sharpe_ratio': sharpe_ratio
    }

# Calculate the Tangency Portfolio
tangency_portfolio = calculate_tangency_portfolio(expected_returns, cov_matrix, rf_rate, allow_short_selling=True)
tangency_weights = tangency_portfolio['weights']
tangency_return = tangency_portfolio['expected_return']
tangency_risk = tangency_portfolio['risk']
tangency_sharpe_ratio = tangency_portfolio['sharpe_ratio']

# Remove the old MVP point
plt.figure(figsize=(10, 6))


# Mark each individual industry on the plot
for industry in expected_returns.index:
    plt.scatter(np.std(returns[industry]) * 100, expected_returns[industry] * 100, 
                marker='X', s=200, label=industry)

# Mark the tangency portfolio on the plot
plt.scatter(tangency_risk * 100, tangency_return * 100, color="blue", marker="*", s=200, label="Tangency Portfolio")

# Annotate the tangency portfolio point
plt.text(tangency_risk * 100, tangency_return * 100, 
         f"(r = {tangency_return*100:.2f}%), σ = {tangency_risk*100:.2f}%)", fontsize=8, ha='right')

# Format axes as percentages
plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter())
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())

plt.xlim(0, 10)
plt.ylim(-3, 3)

# Labels and title
plt.xlabel("Standard Deviation (%)", fontsize=12)
plt.ylabel("Expected Return (%)", fontsize=12)
plt.title("Tangency Portfolio (With Short Selling)", fontsize=15, fontweight='bold')
plt.legend()
plt.grid(True)
plt.savefig("efficient_frontier_cmlA2.png")
plt.show()







# A) 3. Describe the tangent portfolio and its characteristics

# Numerical verification function
def verify_sharpe_maximization(expected_returns, cov_matrix, rf_rate, optimal_weights, allow_short_selling=False):
    """
    Verify Sharpe ratio maximization with optional short-selling constraint
    
    Parameters:
    - expected_returns: Asset expected returns
    - cov_matrix: Covariance matrix
    - rf_rate: Risk-free rate
    - optimal_weights: Optimized portfolio weights
    - allow_short_selling: Boolean to control short-selling constraint
    
    Returns:
    - Dictionary with verification results
    """
    # Compute optimal portfolio metrics
    optimal_return = np.dot(optimal_weights, expected_returns)
    optimal_variance = np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights))
    optimal_sharpe = (optimal_return - rf_rate) / np.sqrt(optimal_variance)
    
    # Constraint verification
    assert np.isclose(np.sum(optimal_weights), 1.0, atol=1e-6), "Weights must sum to 1"
    
    # Verify short-selling constraint if needed
    if not allow_short_selling and np.any(optimal_weights < 0):
        raise ValueError("Negative weights detected when short-selling is prohibited")
    
    # Alternative portfolio generation
    max_alternative_sharpe = 0
    for _ in range(5000):
        # Constrain weight generation based on short-selling parameter
        if allow_short_selling:
            # Unrestricted alternative weights
            alt_weights = np.random.normal(0, 0.1, len(optimal_weights))
        else:
            # Non-negative weights
            alt_weights = np.abs(np.random.normal(0, 0.1, len(optimal_weights)))
        
        # Normalize weights
        alt_weights /= np.sum(alt_weights)
        
        # Compute alternative portfolio Sharpe ratio
        alt_return = np.dot(alt_weights, expected_returns)
        alt_variance = np.dot(alt_weights.T, np.dot(cov_matrix, alt_weights))
        alt_sharpe = (alt_return - rf_rate) / np.sqrt(alt_variance)
        
        max_alternative_sharpe = max(max_alternative_sharpe, alt_sharpe)
    
    # Verification results
    return {
        'optimal_sharpe_ratio': optimal_sharpe,
        'max_alternative_sharpe_ratio': max_alternative_sharpe,
        'is_globally_optimal': optimal_sharpe > max_alternative_sharpe
    }

# Compute tangency portfolio with short-selling
tangency_portfolio = calculate_tangency_portfolio(expected_returns, cov_matrix, rf_rate, allow_short_selling=True)
tangency_weights = tangency_portfolio['weights']
tangency_return = tangency_portfolio['expected_return']
tangency_risk = tangency_portfolio['risk']
tangency_sharpe_ratio = tangency_portfolio['sharpe_ratio']

# Verification with short-selling allowed
verification_results = verify_sharpe_maximization(expected_returns, cov_matrix, rf_rate, tangency_weights, allow_short_selling=True)

print("\n------ A.3 Tangency Portfolio Characteristics (W/ Short-Selling) ------")
for asset, weight in zip(expected_returns.index, tangency_weights):
   print(f"{asset}: {weight*100:.2f}%")
   
print(f"\nTangency Portfolio Expected Return: {tangency_return*100:.2f}%")
print(f"Tangency Portfolio Standard Deviation: {tangency_risk*100:.2f}%")
print(f"Tangency Portfolio Sharpe Ratio: {tangency_sharpe_ratio:.4f}")

print("\n------ Mathematical Verification ------")
print(f"Optimal Portfolio Sharpe Ratio: {verification_results['optimal_sharpe_ratio']:.4f}")
print(f"Maximum Alternative Sharpe Ratio: {verification_results['max_alternative_sharpe_ratio']:.4f}")
print(f"Globally Optimal: {verification_results['is_globally_optimal']}")

# Ensure weights sum to 1
print(f"\nSum of Tangency Portfolio Weights: {np.sum(tangency_weights):.4f}")

# Check for negative weights
if np.any(tangency_weights < 0):
   print("Note: Portfolio includes short positions")







# A) 4. Graph the "mean-variance locus" (without the risk-free asset and w/ short-selling constraint) of the 5 industries. Specify each industry in the chart. 

# Calculate MVP weights
mvp_weights = calculate_mvp_weights(cov_matrix, allow_short_selling=False)

# Calculate MVP return and risk
mvp_return = np.dot(mvp_weights, expected_returns)
mvp_risk = np.sqrt(mvp_weights.T @ cov_matrix @ mvp_weights)
sharpe_ratios = mvp_return / mvp_risk   # No risk-free asset here

# Convert values for plotting (to percentages)
mvp_return_plot = mvp_return * 100
mvp_risk_plot = mvp_risk  * 100
mvp_return_plot = mvp_return * 100
mvp_risk_plot = mvp_risk * 100

# Plot Efficient Frontier (with no short selling)
plt.figure(figsize=(10, 6))
plt.grid(True)

# Mark each industry (its own risk-return point)
for industry in expected_returns.index:
    plt.scatter(np.std(returns[industry]) * 100, expected_returns[industry] * 100, 
                marker='X', s=200, label=industry)

# Mark the MVP (for reference, even though the MVP might not be feasible under the no short-sale constraint)
plt.scatter(mvp_risk_plot, mvp_return_plot, color="red", marker="*", s=200, label="Minimum Variance Portfolio")
plt.text(mvp_risk_plot, mvp_return_plot, 
         f"(r = {mvp_return_plot:.2f}%), σ = {mvp_risk_plot:.2f}%)", fontsize=8, ha='right')

# Format axes as percentages
plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter())
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())

# Manually set axis limits to focus on the main cluster of portfolios
plt.xlim(0, 10)
plt.ylim(0, 3)

# Labels and title
plt.xlabel("Standard Deviation (%)", fontsize=12)
plt.ylabel("Expected Return (%)", fontsize=12)
plt.title("Efficient Frontier with MVP (No Short Selling)", fontsize=15, fontweight='bold')
plt.legend()
plt.savefig("efficient_frontierA4.png")
plt.show()






# A) 5. Graph the "mean-variance locus" (with the risk-free asset and w/ short-selling constraint) of the 5 industries. Specify each industry in the chart. 

# Calculate the Tangency Portfolio
tangency_portfolio = calculate_tangency_portfolio(expected_returns, cov_matrix, rf_rate, allow_short_selling=False)
tangency_weights = tangency_portfolio['weights']
tangency_return = tangency_portfolio['expected_return']
tangency_risk = tangency_portfolio['risk']
tangency_sharpe_ratio = tangency_portfolio['sharpe_ratio']

# Plotting the Efficient Frontier
plt.figure(figsize=(10, 6))

# Mark each individual industry on the plot
for industry in expected_returns.index:
    plt.scatter(np.std(returns[industry]) * 100, expected_returns[industry] * 100, 
                marker='X', s=200, label=industry)

# Mark the tangency portfolio on the plot
plt.scatter(tangency_risk * 100, tangency_return * 100, color="blue", marker="*", s=200, label="Tangency Portfolio")

# Annotate the tangency portfolio point
plt.text(tangency_risk * 100, tangency_return * 100, 
         f"(r = {tangency_return*100:.2f}%), σ = {tangency_risk*100:.2f}%)", fontsize=8, ha='right')

# Format axes as percentages
plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter())
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())

plt.xlim(0, 10)
plt.ylim(-3, 3)

# Labels and title
plt.xlabel("Standard Deviation (%)", fontsize=12)
plt.ylabel("Expected Return (%)", fontsize=12)
plt.title("Tangency Portfolio (Without Short Selling)", fontsize=15, fontweight='bold')
plt.legend()
plt.grid(True)
plt.savefig("efficient_frontier_cmlA5.png")
plt.show()






# A) 6.  Describe the tangent portfolio and its characteristics
def mathematical_sharpe_verification(expected_returns, cov_matrix, rf_rate):
    """
    Mathematically verify Sharpe ratio maximization
    
    Steps:
    1. Symbolic derivation of Sharpe ratio
    2. Compute gradient and Hessian
    3. Verify first and second-order optimality conditions
    """
    # Number of assets
    n = len(expected_returns)
    
    # Symbolic weight variables
    w = sp.symbols(f'w1:{n+1}')
    
    # Symbolic computation of portfolio return
    portfolio_return = sum(w[i] * expected_returns[i] for i in range(n))
    
    # Symbolic computation of portfolio variance
    portfolio_variance = sum(sum(w[i] * w[j] * cov_matrix[i][j] for j in range(n)) for i in range(n))
    
    # Sharpe ratio symbolically
    sharpe_ratio = (portfolio_return - rf_rate) / sp.sqrt(portfolio_variance)
    
    # Gradient computation
    gradient = [sp.diff(sharpe_ratio, wi) for wi in w]
    
    # Hessian computation
    hessian = [[sp.diff(sp.diff(sharpe_ratio, w[i]), w[j]) for j in range(n)] for i in range(n)]
    
    # Constraint: weights sum to 1
    constraint = sum(w) - 1
    
    return {
        'symbolic_sharpe_ratio': sharpe_ratio,
        'gradient': gradient,
        'hessian': hessian,
        'constraint': constraint
    }

# Verify Sharpe ratio maximization w/o short-selling
verification_results = verify_sharpe_maximization(expected_returns, cov_matrix, rf_rate, tangency_weights, allow_short_selling=False)

# Existing tangency portfolio calculation
tangency_portfolio = calculate_tangency_portfolio(expected_returns, cov_matrix, rf_rate, allow_short_selling=False)
tangency_weights = tangency_portfolio['weights']
tangency_return = tangency_portfolio['expected_return']
tangency_risk = tangency_portfolio['risk']
tangency_sharpe_ratio = tangency_portfolio['sharpe_ratio']

# Detailed Output
print("\n------ A.6 Tangency Portfolio Characteristics (W/o Short-Selling) ------")
for asset, weight in zip(expected_returns.index, tangency_weights):
    print(f"{asset}: {weight*100:.2f}%")
    
print(f"\nTangency Portfolio Expected Return: {tangency_return*100:.2f}%")
print(f"Tangency Portfolio Standard Deviation: {tangency_risk*100:.2f}%")
print(f"Tangency Portfolio Sharpe Ratio: {tangency_sharpe_ratio:.4f}")

# Mathematical Verification Results
print("\n------ Mathematical Verification ------")
print(f"Optimal Portfolio Sharpe Ratio: {verification_results['optimal_sharpe_ratio']:.4f}")
print(f"Maximum Alternative Sharpe Ratio: {verification_results['max_alternative_sharpe_ratio']:.4f}")
print(f"Globally Optimal: {verification_results['is_globally_optimal']}")

# Ensure weights sum to 1
print(f"\nSum of Tangency Portfolio Weights: {np.sum(tangency_weights):.4f}")











#### ------------------------------------------------ PART B ------------------------------------------------ ####
# B) 1. Resample with replacement the porfolio data and create a bootstrap sample of 60 months. Repeat the bootstrap 1000 times. And repeat A) 1-6 for each bootstrap sample.
# Set seed to 243
np.random.seed(243)

def bootstrap_sampling(data, n_iterations=1000, sample_size=60):
   bootstrap_samples = []
   for _ in range(n_iterations):
       sample = data.sample(n=sample_size, replace=True)
       bootstrap_samples.append(sample)
   return bootstrap_samples

# Perform bootstrap sampling
bootstrap_samples = bootstrap_sampling(returns, n_iterations=1000, sample_size=60)

# Initialize lists to store results for both short-selling scenarios
mvp_results_with_short = []
mvp_results_without_short = []
tangency_results_with_short = []
tangency_results_without_short = []

# Loop through each bootstrap sample
for sample in bootstrap_samples:
   # Calculate expected returns and covariance matrix
   expected_returns = sample.mean()
   cov_matrix = sample.cov()
   
   # MVP with short-selling
   mvp_weights_with_short = calculate_mvp_weights(cov_matrix, allow_short_selling=True)
   mvp_return_with_short = np.dot(mvp_weights_with_short, expected_returns)
   mvp_risk_with_short = np.sqrt(mvp_weights_with_short.T @ cov_matrix @ mvp_weights_with_short)
   mvp_results_with_short.append({
       'mvp_weights': mvp_weights_with_short,
       'mvp_return': mvp_return_with_short,
       'mvp_risk': mvp_risk_with_short,
       'sharpe_ratio': (mvp_return_with_short - rf_rate) / mvp_risk_with_short
   })
   
   # MVP without short-selling
   mvp_weights_without_short = calculate_mvp_weights(cov_matrix, allow_short_selling=False)
   mvp_return_without_short = np.dot(mvp_weights_without_short, expected_returns)
   mvp_risk_without_short = np.sqrt(mvp_weights_without_short.T @ cov_matrix @ mvp_weights_without_short)
   mvp_results_without_short.append({
       'mvp_weights': mvp_weights_without_short,
       'mvp_return': mvp_return_without_short,
       'mvp_risk': mvp_risk_without_short,
       'sharpe_ratio': (mvp_return_without_short - rf_rate) / mvp_risk_without_short
   })
   
   # Tangency with short-selling
   tangency_portfolio_with_short = calculate_tangency_portfolio(expected_returns, cov_matrix, rf_rate, allow_short_selling=True)
   tangency_results_with_short.append(tangency_portfolio_with_short)
   
   # Tangency without short-selling
   tangency_portfolio_without_short = calculate_tangency_portfolio(expected_returns, cov_matrix, rf_rate, allow_short_selling=False)
   tangency_results_without_short.append(tangency_portfolio_without_short)

# Convert results to DataFrames
def results_to_dataframe(results):
   return pd.DataFrame([{
       'portfolio_return': result['mvp_return'] if 'mvp_return' in result else result['expected_return'],
       'portfolio_risk': result['mvp_risk'] if 'mvp_risk' in result else result['risk'],
       'sharpe_ratio': result['sharpe_ratio']
   } for result in results])

mvp_df_with_short = results_to_dataframe(mvp_results_with_short)
mvp_df_without_short = results_to_dataframe(mvp_results_without_short)
tangency_df_with_short = results_to_dataframe(tangency_results_with_short)
tangency_df_without_short = results_to_dataframe(tangency_results_without_short)

# Plotting
plt.figure(figsize=(16, 12))
plt.subplot(2, 2, 1)
plt.hist(mvp_df_with_short['sharpe_ratio'], bins=30, color='blue', alpha=0.7)
plt.title('MVP Sharpe Ratios (With Short-Selling)', fontsize=14, fontweight='bold')
plt.xlabel('Sharpe Ratio')
plt.ylabel('Frequency')

plt.subplot(2, 2, 2)
plt.hist(mvp_df_without_short['sharpe_ratio'], bins=30, color='red', alpha=0.7)
plt.title('MVP Sharpe Ratios (Without Short-Selling)', fontsize=14, fontweight='bold')
plt.xlabel('Sharpe Ratio')
plt.ylabel('Frequency')

plt.subplot(2, 2, 3)
plt.hist(tangency_df_with_short['sharpe_ratio'], bins=30, color='green', alpha=0.7)
plt.title('Tangency Portfolio Sharpe Ratios (With Short-Selling)', fontsize=14, fontweight='bold')
plt.xlabel('Sharpe Ratio')
plt.ylabel('Frequency')

plt.subplot(2, 2, 4)
plt.hist(tangency_df_without_short['sharpe_ratio'], bins=30, color='orange', alpha=0.7)
plt.title('Tangency Portfolio Sharpe Ratios (Without Short-Selling)', fontsize=14, fontweight='bold')
plt.xlabel('Sharpe Ratio')
plt.ylabel('Frequency')

plt.tight_layout()
plt.savefig("bootstrap_comparisonB11.png")
plt.show()

# Print summary statistics
print("MVP With Short-Selling Summary:")
print(mvp_df_with_short.describe())
print("\nMVP Without Short-Selling Summary:")
print(mvp_df_without_short.describe())
print("\nTangency With Short-Selling Summary:")
print(tangency_df_with_short.describe())
print("\nTangency Without Short-Selling Summary:")
print(tangency_df_without_short.describe())


# Make another bundle of graphs where it shows the MPV and tangency portfolio (as in A1, A2, A4, A5) 
# from the bootstrap samples that is the most optimal
# Function to find the most optimal portfolio based on Sharpe ratio
def find_most_optimal_portfolio(results):
    """
    Find the most optimal portfolio based on the highest Sharpe ratio.
    
    Parameters:
    - results: List of dictionaries containing portfolio information
    
    Returns:
    - Dictionary with the most optimal portfolio details
    """
    optimal_portfolio = max(results, key=lambda x: x['sharpe_ratio'])
    return optimal_portfolio

# Find the most optimal portfolios
most_optimal_mvp_with_short = find_most_optimal_portfolio(mvp_results_with_short)
most_optimal_mvp_without_short = find_most_optimal_portfolio(mvp_results_without_short)
most_optimal_tangency_with_short = find_most_optimal_portfolio(tangency_results_with_short)
most_optimal_tangency_without_short = find_most_optimal_portfolio(tangency_results_without_short)
# Create subplots for the most optimal portfolios
fig, axs = plt.subplots(2, 2, figsize=(16, 12))

# Plotting the most optimal MVP with short-selling
axs[0, 0].set_title("Most Optimal MVP (With Short Selling)", fontsize=15, fontweight='bold')
for industry in expected_returns.index:
    axs[0, 0].scatter(np.std(returns[industry]) * 100, expected_returns[industry] * 100, marker='X', s=200, label=industry)
axs[0, 0].scatter(most_optimal_mvp_with_short['mvp_risk'] * 100, most_optimal_mvp_with_short['mvp_return'] * 100, color="red", marker="*", s=200, label="Most Optimal MVP (With Short Selling)")
axs[0, 0].text(most_optimal_mvp_with_short['mvp_risk'] * 100, most_optimal_mvp_with_short['mvp_return'] * 100, f"(r = {most_optimal_mvp_with_short['mvp_return']*100:.2f}%), σ = {most_optimal_mvp_with_short['mvp_risk']*100:.2f}%)", fontsize=8, ha='right')
axs[0, 0].xaxis.set_major_formatter(mtick.PercentFormatter())
axs[0, 0].yaxis.set_major_formatter(mtick.PercentFormatter())
axs[0, 0].set_xlim(0, 20)
axs[0, 0].set_ylim(-10, 10)
axs[0, 0].set_xlabel("Standard Deviation (%)", fontsize=12)
axs[0, 0].set_ylabel("Expected Return (%)", fontsize=12)
axs[0, 0].legend()
axs[0, 0].grid(True)

# Plotting the most optimal MVP without short-selling
axs[0, 1].set_title("Most Optimal MVP (Without Short Selling)", fontsize=15, fontweight='bold')
for industry in expected_returns.index:
    axs[0, 1].scatter(np.std(returns[industry]) * 100, expected_returns[industry] * 100, marker='X', s=200, label=industry)
axs[0, 1].scatter(most_optimal_mvp_without_short['mvp_risk'] * 100, most_optimal_mvp_without_short['mvp_return'] * 100, color="red", marker="*", s=200, label="Most Optimal MVP (Without Short Selling)")
axs[0, 1].text(most_optimal_mvp_without_short['mvp_risk'] * 100, most_optimal_mvp_without_short['mvp_return'] * 100, f"(r = {most_optimal_mvp_without_short['mvp_return']*100:.2f}%), σ = {most_optimal_mvp_without_short['mvp_risk']*100:.2f}%)", fontsize=8, ha='right')
axs[0, 1].xaxis.set_major_formatter(mtick.PercentFormatter())
axs[0, 1].yaxis.set_major_formatter(mtick.PercentFormatter())
axs[0, 1].set_xlim(0, 20)
axs[0, 1].set_ylim(-10, 10)
axs[0, 1].set_xlabel("Standard Deviation (%)", fontsize=12)
axs[0, 1].set_ylabel("Expected Return (%)", fontsize=12)
axs[0, 1].legend()
axs[0, 1].grid(True)

# Plotting the most optimal Tangency Portfolio with short-selling
axs[1, 0].set_title("Most Optimal Tangency Portfolio (With Short Selling)", fontsize=15, fontweight='bold')
for industry in expected_returns.index:
    axs[1, 0].scatter(np.std(returns[industry]) * 100, expected_returns[industry] * 100, marker='X', s=200, label=industry)
axs[1, 0].scatter(most_optimal_tangency_with_short['risk'] * 100, most_optimal_tangency_with_short['expected_return'] * 100, color="blue", marker="*", s=200, label="Most Optimal Tangency Portfolio (With Short Selling)")
axs[1, 0].text(most_optimal_tangency_with_short['risk'] * 100, most_optimal_tangency_with_short['expected_return'] * 100, f"(r = {most_optimal_tangency_with_short['expected_return']*100:.2f}%), σ = {most_optimal_tangency_with_short['risk']*100:.2f}%)", fontsize=8, ha='right')
axs[1, 0].xaxis.set_major_formatter(mtick.PercentFormatter())
axs[1, 0].yaxis.set_major_formatter(mtick.PercentFormatter())
axs[1, 0].set_xlim(0, 20)
axs[1, 0].set_ylim(-10, 10)
axs[1, 0].set_xlabel("Standard Deviation (%)", fontsize=12)
axs[1, 0].set_ylabel("Expected Return (%)", fontsize=12)
axs[1, 0].legend()
axs[1, 0].grid(True)

# Plotting the most optimal Tangency Portfolio without short-selling
axs[1, 1].set_title("Most Optimal Tangency Portfolio (Without Short Selling)", fontsize=15, fontweight='bold')
for industry in expected_returns.index:
    axs[1, 1].scatter(np.std(returns[industry]) * 100, expected_returns[industry] * 100, marker='X', s=200, label=industry)
axs[1, 1].scatter(most_optimal_tangency_without_short['risk'] * 100, most_optimal_tangency_without_short['expected_return'] * 100, color="blue", marker="*", s=200, label="Most Optimal Tangency Portfolio (Without Short Selling)")
axs[1, 1].text(most_optimal_tangency_without_short['risk'] * 100, most_optimal_tangency_without_short['expected_return'] * 100, f"(r = {most_optimal_tangency_without_short['expected_return']*100:.2f}%), σ = {most_optimal_tangency_without_short['risk']*100:.2f}%)", fontsize=8, ha='right')
axs[1, 1].xaxis.set_major_formatter(mtick.PercentFormatter())
axs[1, 1].yaxis.set_major_formatter(mtick.PercentFormatter())
axs[1, 1].set_xlim(0, 20)
axs[1, 1].set_ylim(-10, 10)
axs[1, 1].set_xlabel("Standard Deviation (%)", fontsize=12)
axs[1, 1].set_ylabel("Expected Return (%)", fontsize=12)
axs[1, 1].legend()
axs[1, 1].grid(True)

# Adjust layout and save the figure
plt.tight_layout()
plt.savefig("most_optimal_portfoliosB12.png")
plt.show() 








# B) 2. Repeat 1 but with a maximum of 3 of the 5 industries chosen
# Chose 3 random industries (columns) from the portfolios_data DataFrame

def bootstrap_sampling(data, n_iterations=1000, sample_size=60):
   bootstrap_samples = []
   for _ in range(n_iterations):
       sample = data.sample(n=sample_size, replace=True)
       bootstrap_samples.append(sample)
   return bootstrap_samples

# Perform bootstrap sampling
bootstrap_samples = bootstrap_sampling(returns, n_iterations=1000, sample_size=60)
# Initialize lists to store results for both short-selling scenarios
mvp_results_with_short = []
mvp_results_without_short = []
tangency_results_with_short = []
tangency_results_without_short = []

# Loop through each bootstrap sample
for sample in bootstrap_samples:
   # Calculate expected returns and covariance matrix
   expected_returns = sample.mean()
   cov_matrix = sample.cov()
   
   # MVP with short-selling
   mvp_weights_with_short = calculate_mvp_weights(cov_matrix, allow_short_selling=True)
   mvp_return_with_short = np.dot(mvp_weights_with_short, expected_returns)
   mvp_risk_with_short = np.sqrt(mvp_weights_with_short.T @ cov_matrix @ mvp_weights_with_short)
   mvp_results_with_short.append({
       'mvp_weights': mvp_weights_with_short,
       'mvp_return': mvp_return_with_short,
       'mvp_risk': mvp_risk_with_short,
       'sharpe_ratio': (mvp_return_with_short - rf_rate) / mvp_risk_with_short
   })
   
   # MVP without short-selling
   mvp_weights_without_short = calculate_mvp_weights(cov_matrix, allow_short_selling=False)
   mvp_return_without_short = np.dot(mvp_weights_without_short, expected_returns)
   mvp_risk_without_short = np.sqrt(mvp_weights_without_short.T @ cov_matrix @ mvp_weights_without_short)
   mvp_results_without_short.append({
       'mvp_weights': mvp_weights_without_short,
       'mvp_return': mvp_return_without_short,
       'mvp_risk': mvp_risk_without_short,
       'sharpe_ratio': (mvp_return_without_short - rf_rate) / mvp_risk_without_short
   })
   
   # Tangency with short-selling
   tangency_portfolio_with_short = calculate_tangency_portfolio(expected_returns, cov_matrix, rf_rate, allow_short_selling=True)
   tangency_results_with_short.append(tangency_portfolio_with_short)
   
   # Tangency without short-selling
   tangency_portfolio_without_short = calculate_tangency_portfolio(expected_returns, cov_matrix, rf_rate, allow_short_selling=False)
   tangency_results_without_short.append(tangency_portfolio_without_short)

# Convert results to DataFrames
def results_to_dataframe(results):
   return pd.DataFrame([{
       'portfolio_return': result['mvp_return'] if 'mvp_return' in result else result['expected_return'],
       'portfolio_risk': result['mvp_risk'] if 'mvp_risk' in result else result['risk'],
       'sharpe_ratio': result['sharpe_ratio']
   } for result in results])

mvp_df_with_short = results_to_dataframe(mvp_results_with_short)
mvp_df_without_short = results_to_dataframe(mvp_results_without_short)
tangency_df_with_short = results_to_dataframe(tangency_results_with_short)
tangency_df_without_short = results_to_dataframe(tangency_results_without_short)

# Plotting
plt.figure(figsize=(16, 12))
plt.subplot(2, 2, 1)
plt.hist(mvp_df_with_short['sharpe_ratio'], bins=30, color='blue', alpha=0.7)
plt.title('MVP Sharpe Ratios (With Short-Selling)', fontsize=14, fontweight='bold')
plt.xlabel('Sharpe Ratio')
plt.ylabel('Frequency')

plt.subplot(2, 2, 2)
plt.hist(mvp_df_without_short['sharpe_ratio'], bins=30, color='red', alpha=0.7)
plt.title('MVP Sharpe Ratios (Without Short-Selling)', fontsize=14, fontweight='bold')
plt.xlabel('Sharpe Ratio')
plt.ylabel('Frequency')

plt.subplot(2, 2, 3)
plt.hist(tangency_df_with_short['sharpe_ratio'], bins=30, color='green', alpha=0.7)
plt.title('Tangency Portfolio Sharpe Ratios (With Short-Selling)', fontsize=14, fontweight='bold')
plt.xlabel('Sharpe Ratio')
plt.ylabel('Frequency')

plt.subplot(2, 2, 4)
plt.hist(tangency_df_without_short['sharpe_ratio'], bins=30, color='orange', alpha=0.7)
plt.title('Tangency Portfolio Sharpe Ratios (Without Short-Selling)', fontsize=14, fontweight='bold')
plt.xlabel('Sharpe Ratio')
plt.ylabel('Frequency')

plt.tight_layout()
plt.savefig("bootstrap_comparisonB21.png")
plt.show()

# Print summary statistics
print("MVP With Short-Selling Summary:")
print(mvp_df_with_short.describe())
print("\nMVP Without Short-Selling Summary:")
print(mvp_df_without_short.describe())
print("\nTangency With Short-Selling Summary:")
print(tangency_df_with_short.describe())
print("\nTangency Without Short-Selling Summary:")
print(tangency_df_without_short.describe())


# Make another bundle of graphs where it shows the MPV and tangency portfolio (as in A1, A2, A4, A5) 
# from the bootstrap samples that is the most optimal
# Function to find the most optimal portfolio based on Sharpe ratio
def find_most_optimal_portfolio(results):
    """
    Find the most optimal portfolio based on the highest Sharpe ratio.
    
    Parameters:
    - results: List of dictionaries containing portfolio information
    
    Returns:
    - Dictionary with the most optimal portfolio details
    """
    optimal_portfolio = max(results, key=lambda x: x['sharpe_ratio'])
    return optimal_portfolio

# Find the most optimal portfolios
most_optimal_mvp_with_short = find_most_optimal_portfolio(mvp_results_with_short)
most_optimal_mvp_without_short = find_most_optimal_portfolio(mvp_results_without_short)
most_optimal_tangency_with_short = find_most_optimal_portfolio(tangency_results_with_short)
most_optimal_tangency_without_short = find_most_optimal_portfolio(tangency_results_without_short)
# Create subplots for the most optimal portfolios
fig, axs = plt.subplots(2, 2, figsize=(16, 12))

# Plotting the most optimal MVP with short-selling
axs[0, 0].set_title("Most Optimal MVP (With Short Selling)", fontsize=15, fontweight='bold')
for industry in expected_returns.index:
    axs[0, 0].scatter(np.std(returns[industry]) * 100, expected_returns[industry] * 100, marker='X', s=200, label=industry)
axs[0, 0].scatter(most_optimal_mvp_with_short['mvp_risk'] * 100, most_optimal_mvp_with_short['mvp_return'] * 100, color="red", marker="*", s=200, label="Most Optimal MVP (With Short Selling)")
axs[0, 0].text(most_optimal_mvp_with_short['mvp_risk'] * 100, most_optimal_mvp_with_short['mvp_return'] * 100, f"(r = {most_optimal_mvp_with_short['mvp_return']*100:.2f}%), σ = {most_optimal_mvp_with_short['mvp_risk']*100:.2f}%)", fontsize=8, ha='right')
axs[0, 0].xaxis.set_major_formatter(mtick.PercentFormatter())
axs[0, 0].yaxis.set_major_formatter(mtick.PercentFormatter())
axs[0, 0].set_xlim(0, 20)
axs[0, 0].set_ylim(-10, 10)
axs[0, 0].set_xlabel("Standard Deviation (%)", fontsize=12)
axs[0, 0].set_ylabel("Expected Return (%)", fontsize=12)
axs[0, 0].legend()
axs[0, 0].grid(True)

# Plotting the most optimal MVP without short-selling
axs[0, 1].set_title("Most Optimal MVP (Without Short Selling)", fontsize=15, fontweight='bold')
for industry in expected_returns.index:
    axs[0, 1].scatter(np.std(returns[industry]) * 100, expected_returns[industry] * 100, marker='X', s=200, label=industry)
axs[0, 1].scatter(most_optimal_mvp_without_short['mvp_risk'] * 100, most_optimal_mvp_without_short['mvp_return'] * 100, color="red", marker="*", s=200, label="Most Optimal MVP (Without Short Selling)")
axs[0, 1].text(most_optimal_mvp_without_short['mvp_risk'] * 100, most_optimal_mvp_without_short['mvp_return'] * 100, f"(r = {most_optimal_mvp_without_short['mvp_return']*100:.2f}%), σ = {most_optimal_mvp_without_short['mvp_risk']*100:.2f}%)", fontsize=8, ha='right')
axs[0, 1].xaxis.set_major_formatter(mtick.PercentFormatter())
axs[0, 1].yaxis.set_major_formatter(mtick.PercentFormatter())
axs[0, 1].set_xlim(0, 20)
axs[0, 1].set_ylim(-10, 10)
axs[0, 1].set_xlabel("Standard Deviation (%)", fontsize=12)
axs[0, 1].set_ylabel("Expected Return (%)", fontsize=12)
axs[0, 1].legend()
axs[0, 1].grid(True)

# Plotting the most optimal Tangency Portfolio with short-selling
axs[1, 0].set_title("Most Optimal Tangency Portfolio (With Short Selling)", fontsize=15, fontweight='bold')
for industry in expected_returns.index:
    axs[1, 0].scatter(np.std(returns[industry]) * 100, expected_returns[industry] * 100, marker='X', s=200, label=industry)
axs[1, 0].scatter(most_optimal_tangency_with_short['risk'] * 100, most_optimal_tangency_with_short['expected_return'] * 100, color="blue", marker="*", s=200, label="Most Optimal Tangency Portfolio (With Short Selling)")
axs[1, 0].text(most_optimal_tangency_with_short['risk'] * 100, most_optimal_tangency_with_short['expected_return'] * 100, f"(r = {most_optimal_tangency_with_short['expected_return']*100:.2f}%), σ = {most_optimal_tangency_with_short['risk']*100:.2f}%)", fontsize=8, ha='right')
axs[1, 0].xaxis.set_major_formatter(mtick.PercentFormatter())
axs[1, 0].yaxis.set_major_formatter(mtick.PercentFormatter())
axs[1, 0].set_xlim(0, 20)
axs[1, 0].set_ylim(-10, 10)
axs[1, 0].set_xlabel("Standard Deviation (%)", fontsize=12)
axs[1, 0].set_ylabel("Expected Return (%)", fontsize=12)
axs[1, 0].legend()
axs[1, 0].grid(True)

# Plotting the most optimal Tangency Portfolio without short-selling
axs[1, 1].set_title("Most Optimal Tangency Portfolio (Without Short Selling)", fontsize=15, fontweight='bold')
for industry in expected_returns.index:
    axs[1, 1].scatter(np.std(returns[industry]) * 100, expected_returns[industry] * 100, marker='X', s=200, label=industry)
axs[1, 1].scatter(most_optimal_tangency_without_short['risk'] * 100, most_optimal_tangency_without_short['expected_return'] * 100, color="blue", marker="*", s=200, label="Most Optimal Tangency Portfolio (Without Short Selling)")
axs[1, 1].text(most_optimal_tangency_without_short['risk'] * 100, most_optimal_tangency_without_short['expected_return'] * 100, f"(r = {most_optimal_tangency_without_short['expected_return']*100:.2f}%), σ = {most_optimal_tangency_without_short['risk']*100:.2f}%)", fontsize=8, ha='right')
axs[1, 1].xaxis.set_major_formatter(mtick.PercentFormatter())
axs[1, 1].yaxis.set_major_formatter(mtick.PercentFormatter())
axs[1, 1].set_xlim(0, 20)
axs[1, 1].set_ylim(-10, 10)
axs[1, 1].set_xlabel("Standard Deviation (%)", fontsize=12)
axs[1, 1].set_ylabel("Expected Return (%)", fontsize=12)
axs[1, 1].legend()
axs[1, 1].grid(True)

# Adjust layout and save the figure
plt.tight_layout()
plt.savefig("most_optimal_portfoliosB22.png")
plt.show() 










# B) 3. From all 48 industries, find up to 5 industries as a portfolio that has the highest Sharpe ratio (with short-selling and without short-selling portfolios)

# Retrieve 48 industry portfolios from Kenneth French website
def retrieve_full_industry_data():
    industry_reader = web.famafrench.FamaFrenchReader('48_Industry_Portfolios', start='2019-12', end='2024-12')
    portfolios_data = industry_reader.read()[0]  # First table contains the return data
    portfolios_data = portfolios_data.replace([-99.99, -999], pd.NA).dropna()
    industry_reader.close()
    return portfolios_data / 100  # Convert percentage to decimal

# Retrieve risk-free rate
def retrieve_risk_free_rate():
    rf_data = web.famafrench.FamaFrenchReader('F-F_Research_Data_Factors', start='2019-12', end='2024-12')
    rf = rf_data.read()[0]['RF'] / 100  # Convert to decimal
    return rf.mean()

# Calculate portfolio Sharpe ratio
def sharpe_ratio(weights, expected_returns, cov_matrix, rf_rate):
    port_return = np.dot(weights, expected_returns)
    port_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return (port_return - rf_rate) / port_risk

# Monte Carlo optimization
def monte_carlo_portfolio(returns, rf_rate, n_simulations=10000, allow_short_selling=True):
    best_sharpe = -np.inf
    best_selection = None
    best_weights = None
    
    industries = returns.columns
    
    for _ in range(n_simulations):
        selected_industries = random.sample(list(industries), 5)
        subset_returns = returns[selected_industries]
        expected_returns = subset_returns.mean()
        cov_matrix = subset_returns.cov()
        n_assets = len(expected_returns)
        
        # Objective function (negative Sharpe ratio to minimize)
        def neg_sharpe(weights):
            return -sharpe_ratio(weights, expected_returns, cov_matrix, rf_rate)
        
        # Constraints and bounds
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = [(-1, 1) if allow_short_selling else (0, 1) for _ in range(n_assets)]
        
        # Initial weights
        initial_weights = np.ones(n_assets) / n_assets
        
        # Optimize
        result = minimize(neg_sharpe, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
        
        if result.success and -result.fun > best_sharpe:
            best_sharpe = -result.fun
            best_selection = selected_industries
            best_weights = result.x
    
    return best_selection, best_weights, best_sharpe

# Main execution
portfolios_data = retrieve_full_industry_data()
rf_rate = retrieve_risk_free_rate()

# Find the best portfolios using Monte Carlo
best_industries_short, best_weights_short, best_sharpe_short = monte_carlo_portfolio(portfolios_data, rf_rate, 10000, True)
best_industries_noshort, best_weights_noshort, best_sharpe_noshort = monte_carlo_portfolio(portfolios_data, rf_rate, 10000, False)

# Display results
print("Best Portfolio With Short Selling:")
print(dict(zip(best_industries_short, best_weights_short)))
print(f"Sharpe Ratio: {best_sharpe_short:.4f}\n")

print("Best Portfolio Without Short Selling:")
print(dict(zip(best_industries_noshort, best_weights_noshort)))
print(f"Sharpe Ratio: {best_sharpe_noshort:.4f}\n")








# B) 4. Implement MAXSER approach to portfolio allocation for the 48 industries
# Apply shrinkage estimation to the covariance matrix
# Retrieve 48 industry portfolios from Kenneth French website
def retrieve_full_industry_data():
    industry_reader = web.famafrench.FamaFrenchReader('48_Industry_Portfolios', start='2014-12', end='2024-12')
    portfolios_data = industry_reader.read()[0]  # First table contains the return data
    portfolios_data = portfolios_data.replace([-99.99, -999], pd.NA).dropna()
    industry_reader.close()
    return portfolios_data / 100  # Convert percentage to decimal

def estimate_cov_matrix(returns):
    lw = LedoitWolf()
    return lw.fit(returns).covariance_

# Calculate portfolio Sharpe ratio
def sharpe_ratio(weights, expected_returns, cov_matrix, rf_rate):
    port_return = np.dot(weights, expected_returns)
    port_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return (port_return - rf_rate) / port_risk

# Monte Carlo optimization with MAXSER
def monte_carlo_maxser(returns, rf_rate, n_simulations=10000, allow_short_selling=True):
    best_sharpe = -np.inf
    best_selection = None
    best_weights = None
    
    industries = returns.columns
    
    for _ in range(n_simulations):
        selected_industries = random.sample(list(industries), 5)
        subset_returns = returns[selected_industries]
        expected_returns = subset_returns.mean()
        cov_matrix = estimate_cov_matrix(subset_returns)
        n_assets = len(expected_returns)
        
        # Objective function (negative Sharpe ratio to minimize)
        def neg_sharpe(weights):
            return -sharpe_ratio(weights, expected_returns, cov_matrix, rf_rate)
        
        # Constraints and bounds
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = [(-1, 1) if allow_short_selling else (0, 1) for _ in range(n_assets)]
        
        # Initial weights
        initial_weights = np.ones(n_assets) / n_assets
        
        # Optimize
        result = minimize(neg_sharpe, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
        
        if result.success and -result.fun > best_sharpe:
            best_sharpe = -result.fun
            best_selection = selected_industries
            best_weights = result.x
    
    return best_selection, best_weights, best_sharpe

# Main execution
portfolios_data = retrieve_full_industry_data()

# Find the best portfolios using Monte Carlo with MAXSER
best_industries_short, best_weights_short, best_sharpe_short = monte_carlo_maxser(portfolios_data, rf_rate, 10000, True)
best_industries_noshort, best_weights_noshort, best_sharpe_noshort = monte_carlo_maxser(portfolios_data, rf_rate, 10000, False)

# Display results
print("Best Portfolio With Short Selling (MAXSER):")
print(dict(zip(best_industries_short, best_weights_short)))
print(f"Sharpe Ratio: {best_sharpe_short:.4f}\n")

print("Best Portfolio Without Short Selling (MAXSER):")
print(dict(zip(best_industries_noshort, best_weights_noshort)))
print(f"Sharpe Ratio: {best_sharpe_noshort:.4f}\n")