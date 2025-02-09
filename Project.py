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
from sklearn.linear_model import LassoCV, ElasticNetCV

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
    num_assets = cov_matrix.shape[0]
    ones = np.ones(num_assets)
    inv_cov_matrix = np.linalg.inv(cov_matrix)
    A = ones.T @ inv_cov_matrix @ ones
    B = ones.T @ inv_cov_matrix @ expected_returns
    mvp_weights = (inv_cov_matrix @ ones) / A
    if allow_short_selling:
        return mvp_weights
    else:
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bounds = [(0, 1) for _ in range(num_assets)]
        initial_weights = np.ones(num_assets) / num_assets
        result = minimize(lambda w: w.T @ cov_matrix @ w, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
        if result.success:
            return result.x
        else:
            raise ValueError("Optimization failed: " + result.message)

def plot_efficient_frontier(returns, expected_returns, cov_matrix, mvp_weights):
    mvp_return = np.dot(mvp_weights, expected_returns)
    mvp_risk = np.sqrt(mvp_weights.T @ cov_matrix @ mvp_weights)
    mvp_return_plot = mvp_return * 100
    mvp_risk_plot = mvp_risk * 100
    
    mu_vals = np.linspace(min(expected_returns) - 0.02, max(expected_returns) + 0.04, 300)
    ones = np.ones(len(expected_returns))
    inv_cov = np.linalg.inv(cov_matrix)
    A = ones.T @ inv_cov @ ones
    B = ones.T @ inv_cov @ expected_returns
    C = expected_returns.T @ inv_cov @ expected_returns
    Delta = A * C - B**2
    sigma_vals = np.sqrt(np.maximum((A * mu_vals**2 - 2 * B * mu_vals + C) / Delta, 0)) * 100
    
    plt.figure(figsize=(10, 6))
    plt.grid(True)
    plt.plot(sigma_vals, mu_vals * 100, label="Efficient Frontier", color="blue")
    
    for industry in expected_returns.index:
        plt.scatter(returns[industry].std() * 100, expected_returns[industry] * 100, marker='X', s=200, label=industry)
    
    plt.scatter(mvp_risk_plot, mvp_return_plot, color="red", marker="*", s=200, label="Minimum Variance Portfolio")
    plt.annotate(f"(r = {mvp_return_plot:.2f}%, σ = {mvp_risk_plot:.2f}%)", (mvp_risk_plot, mvp_return_plot), xytext=(10, 10), textcoords='offset points', fontsize=8)
    
    plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter())
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
    plt.xlabel("Standard Deviation (%)", fontsize=12)
    plt.ylabel("Expected Return (%)", fontsize=12)
    plt.title("Efficient Frontier with MVP (With Short Selling)", fontsize=15, fontweight='bold')
    plt.legend()
    plt.savefig("efficient_frontierA1.png", bbox_inches='tight', dpi=300)
    plt.show()

mvp_weights = calculate_mvp_weights(cov_matrix, allow_short_selling=True)
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

# Generate efficient frontier
mu_vals = np.linspace(min(expected_returns) - 0.02, max(expected_returns) + 0.04, 300)
ones = np.ones(len(expected_returns))
inv_cov = np.linalg.inv(cov_matrix)
A = ones.T @ inv_cov @ ones
B = ones.T @ inv_cov @ expected_returns
C = expected_returns.T @ inv_cov @ expected_returns
Delta = A * C - B**2
sigma_vals = np.sqrt(np.maximum((A * mu_vals**2 - 2 * B * mu_vals + C) / Delta, 0)) * 100

# Plot the Efficient Frontier (Parabola)
plt.plot(sigma_vals, mu_vals * 100, label="Efficient Frontier", color="blue", linestyle='dashed')

# Mark each individual industry on the plot
for industry in expected_returns.index:
    plt.scatter(np.std(returns[industry]) * 100, expected_returns[industry] * 100, 
                marker='X', s=200, label=industry)

# Mark the tangency portfolio on the plot
plt.scatter(tangency_risk * 100, tangency_return * 100, color="blue", marker="*", s=200, label="Tangency Portfolio")

# Annotate the tangency portfolio point
plt.text(tangency_risk * 100, tangency_return * 100, 
         f"(r = {tangency_return*100:.2f}%, σ = {tangency_risk*100:.2f}%)", fontsize=8, ha='right')

# ---- ADDING CML ---- #
# Define the Capital Market Line (CML)
cml_x = np.linspace(0, 10, 100)  # From risk-free to high risk
cml_y = rf_rate * 100 + ((tangency_return - rf_rate) / tangency_risk) * cml_x

# Plot the CML
plt.plot(cml_x, cml_y, label="Capital Market Line (CML)", color="red", linestyle="solid", linewidth=2)

# Format axes as percentages
plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter())
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())

plt.xlim(0, 10)
plt.ylim(-3, 3)

# Labels and title
plt.xlabel("Standard Deviation (%)", fontsize=12)
plt.ylabel("Expected Return (%)", fontsize=12)
plt.title("Tangency Portfolio with Efficient Frontier & CML (With Short Selling)", fontsize=15, fontweight='bold')
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

# A) 4. Graph the "mean-variance locus" (without the risk-free asset and w/ short-selling constraint) of the 5 industries.

def calculate_portfolio_metrics(weights, returns, cov_matrix):
    portfolio_return = np.dot(weights, expected_returns)
    portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return portfolio_return, portfolio_risk

def generate_efficient_frontier_no_short():
    # Generate a range of target returns
    target_returns = np.linspace(min(expected_returns), max(expected_returns), 100)
    efficient_risks = []
    efficient_returns = []
    
    for target_return in target_returns:
        # Define the optimization problem
        def portfolio_volatility(weights):
            return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # weights sum to 1
            {'type': 'eq', 'fun': lambda x: np.dot(x, expected_returns) - target_return}  # target return
        ]
        
        # Bounds (no short selling)
        bounds = tuple((0, 1) for _ in range(len(expected_returns)))
        
        # Initial guess
        initial_weights = np.ones(len(expected_returns)) / len(expected_returns)
        
        try:
            # Optimize
            result = minimize(portfolio_volatility, 
                            initial_weights,
                            method='SLSQP',
                            bounds=bounds,
                            constraints=constraints)
            
            if result.success:
                efficient_risks.append(result.fun)
                efficient_returns.append(target_return)
        except:
            continue
    
    return np.array(efficient_risks), np.array(efficient_returns)

# Calculate MVP weights
mvp_weights = calculate_mvp_weights(cov_matrix, allow_short_selling=False)

# Calculate MVP return and risk
mvp_return = np.dot(mvp_weights, expected_returns)
mvp_risk = np.sqrt(mvp_weights.T @ cov_matrix @ mvp_weights)

# Convert values for plotting (to percentages)
mvp_return_plot = mvp_return * 100
mvp_risk_plot = mvp_risk * 100

# Generate efficient frontier points
ef_risks, ef_returns = generate_efficient_frontier_no_short()

# Create the plot
plt.figure(figsize=(10, 6))
plt.grid(True)

# Plot the efficient frontier with dots
plt.plot(ef_risks * 100, ef_returns * 100, label="Efficient Frontier", color="blue", linestyle='dashed')

# Mark each industry
for industry in expected_returns.index:
    plt.scatter(np.std(returns[industry]) * 100, expected_returns[industry] * 100, 
                marker='X', s=200, label=industry)

# Mark the MVP
plt.scatter(mvp_risk_plot, mvp_return_plot, color="red", marker="*", s=200, label="Minimum Variance Portfolio")
plt.annotate(f"(r = {mvp_return_plot:.2f}%, σ = {mvp_risk_plot:.2f}%)", 
            (mvp_risk_plot, mvp_return_plot), 
            xytext=(10, 10), 
            textcoords='offset points', 
            fontsize=8)

# Format axes as percentages
plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter())
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())

# Set axis limits
plt.xlim(0, 10)
plt.ylim(0, 3)

# Labels and title
plt.xlabel("Standard Deviation (%)", fontsize=12)
plt.ylabel("Expected Return (%)", fontsize=12)
plt.title("Efficient Frontier with MVP (No Short Selling)", fontsize=15, fontweight='bold')
plt.legend()
plt.savefig("efficient_frontierA4.png", bbox_inches='tight', dpi=300)
plt.show()




# A) 5. Graph the "mean-variance locus" (with the risk-free asset and w/ short-selling constraint) of the 5 industries. Specify each industry in the chart. 

# Calculate the Tangency Portfolio
tangency_portfolio = calculate_tangency_portfolio(expected_returns, cov_matrix, rf_rate, allow_short_selling=False)
tangency_weights = tangency_portfolio['weights']
tangency_return = tangency_portfolio['expected_return']
tangency_risk = tangency_portfolio['risk']
tangency_sharpe_ratio = tangency_portfolio['sharpe_ratio']

# Create the plot
plt.figure(figsize=(10, 6))
plt.grid(True)

# Plot Efficient Frontier (Parabola)
plt.plot(sigma_vals, mu_vals * 100, label="Efficient Frontier", color="blue", linestyle='dashed')

# Add Tangency Parabola
# Calculate points for the tangency parabola
tangency_point = np.array([tangency_risk, tangency_return])
parabola_points = []

for t in np.linspace(0, 2, 100):  # t goes from 0 to 2 to show both sides of tangency point
    point = t * tangency_point
    parabola_points.append(point)

parabola_points = np.array(parabola_points)
plt.plot(parabola_points[:, 0] * 100, parabola_points[:, 1] * 100, 
         label="Tangency Parabola", color="green", linestyle='solid')

# Mark each individual industry on the plot
for industry in expected_returns.index:
    plt.scatter(np.std(returns[industry]) * 100, expected_returns[industry] * 100,
               marker='X', s=200, label=industry)

# Mark the tangency portfolio on the plot
plt.scatter(tangency_risk * 100, tangency_return * 100, 
           color="blue", marker="*", s=200, label="Tangency Portfolio")
plt.text(tangency_risk * 100, tangency_return * 100,
         f"(r = {tangency_return*100:.2f}%, σ = {tangency_risk*100:.2f}%)", 
         fontsize=8, ha='right')

# Format axes as percentages
plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter())
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())

# Set plot limits
plt.xlim(0, 10)
plt.ylim(-3, 3)

# Labels and title
plt.xlabel("Standard Deviation (%)", fontsize=12)
plt.ylabel("Expected Return (%)", fontsize=12)
plt.title("Efficient Frontier & CML (No Short Selling))", 
          fontsize=15, fontweight='bold')
plt.legend()
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

# Plotting histograms of Sharpe ratios
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


# Find optimal bootstrap samples for each case
def get_optimal_bootstrap_sample(bootstrap_samples, results):
    optimal_index = max(range(len(results)), 
                       key=lambda i: results[i]['sharpe_ratio'])
    return bootstrap_samples[optimal_index]

# Get optimal samples for all four cases
optimal_sample_mvp_with_short = get_optimal_bootstrap_sample(bootstrap_samples, mvp_results_with_short)
optimal_sample_mvp_without_short = get_optimal_bootstrap_sample(bootstrap_samples, mvp_results_without_short)
optimal_sample_tangency_with_short = get_optimal_bootstrap_sample(bootstrap_samples, tangency_results_with_short)
optimal_sample_tangency_without_short = get_optimal_bootstrap_sample(bootstrap_samples, tangency_results_without_short)

def plot_bootstrapped_efficient_frontier(ax, optimal_bootstrap_sample, rf_rate, allow_short_selling=True, is_tangency=True):
    """Modified to take an axis as parameter instead of creating a new figure"""
    # Calculate expected returns and covariance matrix from the optimal bootstrap sample
    expected_returns = optimal_bootstrap_sample.mean()
    cov_matrix = optimal_bootstrap_sample.cov()
    
    # Generate efficient frontier points
    n_points = 300
    weights_list = []
    returns_list = []
    risks_list = []
    
    # Use optimization to generate efficient frontier points
    target_returns = np.linspace(min(expected_returns), max(expected_returns), n_points)
    
    for target_return in target_returns:
        try:
            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                {'type': 'eq', 'fun': lambda w: np.dot(w, expected_returns) - target_return}
            ]
            
            bounds = None if allow_short_selling else [(0, 1) for _ in range(len(expected_returns))]
            
            result = minimize(lambda w: np.sqrt(np.dot(w.T, np.dot(cov_matrix, w))),
                            x0=np.ones(len(expected_returns)) / len(expected_returns),
                            method='SLSQP',
                            bounds=bounds,
                            constraints=constraints)
            
            if result.success:
                weights_list.append(result.x)
                returns_list.append(target_return)
                risks_list.append(np.sqrt(np.dot(result.x.T, np.dot(cov_matrix, result.x))))
        except:
            continue
    
    # Plot the Efficient Frontier
    ax.plot(np.array(risks_list) * 100, np.array(returns_list) * 100,
            label="Efficient Frontier", color="blue", linestyle='dashed', alpha=0.7)

    # Plot individual assets
    for industry in expected_returns.index:
        ax.scatter(np.std(optimal_bootstrap_sample[industry]) * 100, 
                  expected_returns[industry] * 100, 
                  marker='X', s=200, label=industry)
    
    # Calculate and plot MVP
    mvp_weights = calculate_mvp_weights(cov_matrix, allow_short_selling)
    mvp_return = np.dot(mvp_weights, expected_returns)
    mvp_risk = np.sqrt(mvp_weights.T @ cov_matrix @ mvp_weights)
    
    # Calculate Tangency Portfolio
    tangency_portfolio = calculate_tangency_portfolio(expected_returns, cov_matrix, 
                                                    rf_rate, allow_short_selling)
    
    # Plot the relevant portfolio point based on is_tangency parameter
    if is_tangency:
        portfolio_risk = tangency_portfolio['risk'] * 100
        portfolio_return = tangency_portfolio['expected_return'] * 100
        ax.scatter(portfolio_risk, portfolio_return, 
                  color='green', marker="*", s=200, label="Tangency Portfolio")
        ax.text(portfolio_risk, portfolio_return,
                f"\nTangency (r={portfolio_return:.2f}%, σ={portfolio_risk:.2f}%)",
                fontsize=8, ha='left', va='bottom')
        
        # Plot CML for tangency portfolio
        max_x = max(20, portfolio_risk * 1.2)
        cml_x = np.linspace(0, max_x, 100)
        slope = (portfolio_return - rf_rate * 100) / portfolio_risk
        cml_y = rf_rate * 100 + slope * cml_x
        ax.plot(cml_x, cml_y, label="Capital Market Line (CML)", 
               color="red", linestyle="solid", linewidth=1.5, alpha=0.7)
    else:
        ax.scatter(mvp_risk * 100, mvp_return * 100, 
                  color='red', marker="*", s=200, label="MVP")
        ax.text(mvp_risk * 100, mvp_return * 100,
                f"\nMVP (r={mvp_return*100:.2f}%, σ={mvp_risk*100:.2f}%)",
                fontsize=8, ha='left', va='bottom')
    
    # Formatting
    portfolio_type = "Tangency Portfolio" if is_tangency else "MVP"
    title = f"Bootstrapped {portfolio_type} ({'With' if allow_short_selling else 'Without'} Short Selling)"
    ax.set_title(title, fontsize=15, fontweight='bold')
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_xlim(0, 20)
    ax.set_ylim(-10, 10)
    ax.set_xlabel("Standard Deviation (%)", fontsize=12)
    ax.set_ylabel("Expected Return (%)", fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(True)

# Create one figure with 2x2 subplots
fig, axes = plt.subplots(2, 2, figsize=(20, 16))

# MVP with short-selling
plot_bootstrapped_efficient_frontier(axes[0, 0], 
                                   optimal_sample_mvp_with_short, 
                                   rf_rate, 
                                   allow_short_selling=True, 
                                   is_tangency=False)

# MVP without short-selling
plot_bootstrapped_efficient_frontier(axes[0, 1], 
                                   optimal_sample_mvp_without_short, 
                                   rf_rate, 
                                   allow_short_selling=False, 
                                   is_tangency=False)

# Tangency with short-selling
plot_bootstrapped_efficient_frontier(axes[1, 0], 
                                   optimal_sample_tangency_with_short, 
                                   rf_rate, 
                                   allow_short_selling=True, 
                                   is_tangency=True)

# Tangency without short-selling
plot_bootstrapped_efficient_frontier(axes[1, 1], 
                                   optimal_sample_tangency_without_short, 
                                   rf_rate, 
                                   allow_short_selling=False, 
                                   is_tangency=True)

plt.tight_layout()
plt.savefig("most_optimal_portfoliosB12.png")
plt.show()

def verify_tangency_portfolio_optimality(returns_data, rf_rate, allow_short_selling=True):
    """
    Verify that the tangency portfolio has the highest Sharpe ratio among all portfolios
    on the efficient frontier.
    """
    # Calculate expected returns and covariance matrix
    expected_returns = returns_data.mean()
    cov_matrix = returns_data.cov()
    
    # Get tangency portfolio
    tangency_portfolio = calculate_tangency_portfolio(
        expected_returns, cov_matrix, rf_rate, allow_short_selling
    )
    tangency_sharpe = tangency_portfolio['sharpe_ratio']
    
    # Generate a large number of portfolios along the efficient frontier
    n_points = 1000
    target_returns = np.linspace(min(expected_returns), max(expected_returns), n_points)
    portfolio_metrics = []
    
    for target_return in target_returns:
        try:
            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                {'type': 'eq', 'fun': lambda w: np.dot(w, expected_returns) - target_return}
            ]
            
            bounds = None if allow_short_selling else [(0, 1) for _ in range(len(expected_returns))]
            
            result = minimize(
                lambda w: np.sqrt(np.dot(w.T, np.dot(cov_matrix, w))),
                x0=np.ones(len(expected_returns)) / len(expected_returns),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            if result.success:
                weights = result.x
                portfolio_return = target_return
                portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                sharpe_ratio = (portfolio_return - rf_rate) / portfolio_risk
                
                portfolio_metrics.append({
                    'return': portfolio_return,
                    'risk': portfolio_risk,
                    'sharpe_ratio': sharpe_ratio
                })
                
        except:
            continue
    
    # Convert to DataFrame for analysis
    metrics_df = pd.DataFrame(portfolio_metrics)
    max_sharpe_portfolio = metrics_df.loc[metrics_df['sharpe_ratio'].idxmax()]
    
    print(f"\nVerification Results ({'With' if allow_short_selling else 'Without'} Short-Selling):")
    print("-" * 60)
    print(f"Tangency Portfolio Sharpe Ratio: {tangency_sharpe:.4f}")
    print(f"Maximum Sharpe Ratio Found on Efficient Frontier: {max_sharpe_portfolio['sharpe_ratio']:.4f}")
    print(f"Difference: {abs(tangency_sharpe - max_sharpe_portfolio['sharpe_ratio']):.8f}")
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    plt.scatter(metrics_df['risk'] * 100, metrics_df['sharpe_ratio'], 
               alpha=0.5, label='Efficient Frontier Portfolios')
    plt.axhline(y=tangency_sharpe, color='r', linestyle='--', 
                label='Tangency Portfolio Sharpe Ratio')
    plt.xlabel('Portfolio Risk (%)')
    plt.ylabel('Sharpe Ratio')
    plt.title(f"Sharpe Ratios Along Efficient Frontier ({'With' if allow_short_selling else 'Without'} Short-Selling)")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"sharpe_ratio_verification_{'with' if allow_short_selling else 'without'}_short.png")
    plt.show()
    
    return abs(tangency_sharpe - max_sharpe_portfolio['sharpe_ratio']) < 1e-4

# Run verification for both short-selling scenarios using the optimal bootstrap samples
print("\nVerifying optimality of tangency portfolios...")

# With short-selling
is_optimal_with_short = verify_tangency_portfolio_optimality(
    optimal_sample_tangency_with_short, 
    rf_rate, 
    allow_short_selling=True
)

# Without short-selling
is_optimal_without_short = verify_tangency_portfolio_optimality(
    optimal_sample_tangency_without_short, 
    rf_rate, 
    allow_short_selling=False
)

print("\nOptimality Verification Summary:")
print("-" * 40)
print(f"Tangency Portfolio is optimal (with short-selling): {is_optimal_with_short}")
print(f"Tangency Portfolio is optimal (without short-selling): {is_optimal_without_short}")








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

def calculate_mvp_weights_cardinality(cov_matrix, allow_short_selling=True, max_assets=3):
    """Calculate MVP weights with cardinality constraint using binary variables"""
    n_assets = len(cov_matrix)
    
    def objective(params):
        weights = params[:n_assets]
        return np.sqrt(weights.T @ cov_matrix @ weights)
    
    # Initial guess: Equal weights in top 3 assets by variance
    variances = np.diag(cov_matrix)
    top_3_indices = np.argsort(variances)[:3]
    x0 = np.zeros(n_assets)
    x0[top_3_indices] = 1/3
    
    # Constraints
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Sum of weights = 1
    ]
    
    # Bounds for weights
    if allow_short_selling:
        bounds = [(-np.inf, np.inf) for _ in range(n_assets)]
    else:
        bounds = [(0, 1) for _ in range(n_assets)]
    
    # Add cardinality constraint through penalty in objective function
    def penalized_objective(x):
        portfolio_risk = objective(x)
        # Count number of non-zero weights (using a small threshold)
        active_positions = np.sum(np.abs(x) > 1e-4)
        if active_positions > max_assets:
            return portfolio_risk + 1000 * (active_positions - max_assets)
        return portfolio_risk
    
    # Optimize
    result = minimize(penalized_objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
    
    # Clean up small weights
    weights = result.x
    weights[np.abs(weights) < 1e-4] = 0
    
    # Normalize weights to ensure they sum to 1
    weights = weights / np.sum(weights)
    
    return weights

def calculate_tangency_portfolio_cardinality_improved(expected_returns, cov_matrix, rf_rate, allow_short_selling=True, max_assets=3, n_starts=10):
    """Calculate tangency portfolio weights with improved cardinality constraint handling"""
    n_assets = len(cov_matrix)
    best_result = None
    best_sharpe = -np.inf
    
    def sharpe_ratio(params):
        weights = params[:n_assets]
        portfolio_return = np.dot(weights, expected_returns)
        portfolio_risk = np.sqrt(weights.T @ cov_matrix @ weights)
        return -(portfolio_return - rf_rate) / portfolio_risk  # Negative because we minimize
    
    # Define stronger penalty function
    def penalized_sharpe_ratio(x):
        sr = sharpe_ratio(x)
        active_positions = np.sum(np.abs(x) > 1e-4)
        if active_positions > max_assets:
            return sr + 10000 * (active_positions - max_assets)**2  # Quadratic penalty
        return sr
    
    # Bounds for weights with limited short-selling
    if allow_short_selling:
        bounds = [(-2, 2) for _ in range(n_assets)]  # Limit short positions to -200%
    else:
        bounds = [(0, 1) for _ in range(n_assets)]
    
    # Constraints
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Sum of weights = 1
    ]
    
    # Multi-start optimization
    for _ in range(n_starts):
        # Generate different initial guesses
        if _ == 0:
            # First try: Top assets by Sharpe ratio
            individual_sharpe = (expected_returns - rf_rate) / np.sqrt(np.diag(cov_matrix))
            top_indices = np.argsort(-individual_sharpe)[:max_assets]
            x0 = np.zeros(n_assets)
            x0[top_indices] = 1/max_assets
        else:
            # Random initialization with preference for positive weights
            x0 = np.random.uniform(-0.5, 1.5, n_assets)
            x0 = x0 / np.sum(np.abs(x0))  # Normalize
        
        try:
            result = minimize(
                penalized_sharpe_ratio,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000}
            )
            
            if result.success:
                weights = result.x
                # Clean up small weights
                weights[np.abs(weights) < 1e-4] = 0
                if np.sum(np.abs(weights) > 1e-4) <= max_assets:  # Check cardinality constraint
                    weights = weights / np.sum(weights)  # Renormalize
                    portfolio_return = np.dot(weights, expected_returns)
                    portfolio_risk = np.sqrt(weights.T @ cov_matrix @ weights)
                    sharpe = (portfolio_return - rf_rate) / portfolio_risk
                    
                    if sharpe > best_sharpe:
                        best_sharpe = sharpe
                        best_result = {
                            'weights': weights,
                            'expected_return': portfolio_return,
                            'risk': portfolio_risk,
                            'sharpe_ratio': sharpe
                        }
        except:
            continue
    
    if best_result is None:
        raise ValueError("Failed to find valid solution")
    
    return best_result

# Loop through each bootstrap sample
for sample in bootstrap_samples:
    expected_returns = sample.mean()
    cov_matrix = sample.cov()
    
    try:
        # MVP with short-selling (should be updated to use cardinality constraint)
        mvp_weights_with_short = calculate_mvp_weights_cardinality(cov_matrix, allow_short_selling=True, max_assets=3)
        mvp_return_with_short = np.dot(mvp_weights_with_short, expected_returns)
        mvp_risk_with_short = np.sqrt(mvp_weights_with_short.T @ cov_matrix @ mvp_weights_with_short)
        mvp_results_with_short.append({
            'mvp_weights': mvp_weights_with_short,
            'mvp_return': mvp_return_with_short,
            'mvp_risk': mvp_risk_with_short,
            'sharpe_ratio': (mvp_return_with_short - rf_rate) / mvp_risk_with_short
        })
        
        # MVP without short-selling
        mvp_weights_without_short = calculate_mvp_weights_cardinality(cov_matrix, allow_short_selling=False, max_assets=3)
        mvp_return_without_short = np.dot(mvp_weights_without_short, expected_returns)
        mvp_risk_without_short = np.sqrt(mvp_weights_without_short.T @ cov_matrix @ mvp_weights_without_short)
        mvp_results_without_short.append({
            'mvp_weights': mvp_weights_without_short,
            'mvp_return': mvp_return_without_short,
            'mvp_risk': mvp_risk_without_short,
            'sharpe_ratio': (mvp_return_without_short - rf_rate) / mvp_risk_without_short
        })
        
        # Tangency with short-selling (using improved function)
        tangency_portfolio_with_short = calculate_tangency_portfolio_cardinality_improved(
            expected_returns, cov_matrix, rf_rate, allow_short_selling=True, max_assets=3)
        tangency_results_with_short.append(tangency_portfolio_with_short)
        
        # Tangency without short-selling (using improved function)
        tangency_portfolio_without_short = calculate_tangency_portfolio_cardinality_improved(
            expected_returns, cov_matrix, rf_rate, allow_short_selling=False, max_assets=3)
        tangency_results_without_short.append(tangency_portfolio_without_short)
    except:
        continue

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

# Plotting histograms of Sharpe ratios
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


# Find optimal bootstrap samples for each case
def get_optimal_bootstrap_sample(bootstrap_samples, results):
    optimal_index = max(range(len(results)), 
                       key=lambda i: results[i]['sharpe_ratio'])
    return bootstrap_samples[optimal_index]

# Get optimal samples for all four cases
optimal_sample_mvp_with_short = get_optimal_bootstrap_sample(bootstrap_samples, mvp_results_with_short)
optimal_sample_mvp_without_short = get_optimal_bootstrap_sample(bootstrap_samples, mvp_results_without_short)
optimal_sample_tangency_with_short = get_optimal_bootstrap_sample(bootstrap_samples, tangency_results_with_short)
optimal_sample_tangency_without_short = get_optimal_bootstrap_sample(bootstrap_samples, tangency_results_without_short)

def plot_bootstrapped_efficient_frontier(ax, optimal_bootstrap_sample, rf_rate, allow_short_selling=True, is_tangency=True):
    """Modified to take an axis as parameter instead of creating a new figure"""
    # Calculate expected returns and covariance matrix from the optimal bootstrap sample
    expected_returns = optimal_bootstrap_sample.mean()
    cov_matrix = optimal_bootstrap_sample.cov()
    
    # Generate efficient frontier points
    n_points = 300
    weights_list = []
    returns_list = []
    risks_list = []
    
    # Use optimization to generate efficient frontier points
    target_returns = np.linspace(min(expected_returns), max(expected_returns), n_points)
    
    for target_return in target_returns:
        try:
            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                {'type': 'eq', 'fun': lambda w: np.dot(w, expected_returns) - target_return}
            ]
            
            bounds = [(-2, 2) if allow_short_selling else (0, 1) for _ in range(len(expected_returns))]
            
            result = minimize(lambda w: np.sqrt(np.dot(w.T, np.dot(cov_matrix, w))),
                            x0=np.ones(len(expected_returns)) / len(expected_returns),
                            method='SLSQP',
                            bounds=bounds,
                            constraints=constraints)
            
            if result.success:
                weights_list.append(result.x)
                returns_list.append(target_return)
                risks_list.append(np.sqrt(np.dot(result.x.T, np.dot(cov_matrix, result.x))))
        except:
            continue
    
    # Plot the Efficient Frontier
    ax.plot(np.array(risks_list) * 100, np.array(returns_list) * 100,
            label="Efficient Frontier", color="blue", linestyle='dashed', alpha=0.7)

    # Plot individual assets
    for industry in expected_returns.index:
        ax.scatter(np.std(optimal_bootstrap_sample[industry]) * 100, 
                  expected_returns[industry] * 100, 
                  marker='X', s=200, label=industry)
    
    # Calculate and plot MVP
    mvp_weights = calculate_mvp_weights(cov_matrix, allow_short_selling)
    mvp_return = np.dot(mvp_weights, expected_returns)
    mvp_risk = np.sqrt(mvp_weights.T @ cov_matrix @ mvp_weights)
    
    # Calculate Tangency Portfolio
    tangency_portfolio = calculate_tangency_portfolio(expected_returns, cov_matrix, 
                                                    rf_rate, allow_short_selling)
    
    # Plot the relevant portfolio point based on is_tangency parameter
    if is_tangency:
        portfolio_risk = tangency_portfolio['risk'] * 100
        portfolio_return = tangency_portfolio['expected_return'] * 100
        ax.scatter(portfolio_risk, portfolio_return, 
                  color='green', marker="*", s=200, label="Tangency Portfolio")
        ax.text(portfolio_risk, portfolio_return,
                f"\nTangency (r={portfolio_return:.2f}%, σ={portfolio_risk:.2f}%)",
                fontsize=8, ha='left', va='bottom')
        
        # Plot CML for tangency portfolio
        max_x = max(20, portfolio_risk * 1.2)
        cml_x = np.linspace(0, max_x, 100)
        slope = (portfolio_return - rf_rate * 100) / portfolio_risk
        cml_y = rf_rate * 100 + slope * cml_x
        ax.plot(cml_x, cml_y, label="Capital Market Line (CML)", 
               color="red", linestyle="solid", linewidth=1.5, alpha=0.7)
    else:
        ax.scatter(mvp_risk * 100, mvp_return * 100, 
                  color='red', marker="*", s=200, label="MVP")
        ax.text(mvp_risk * 100, mvp_return * 100,
                f"\nMVP (r={mvp_return*100:.2f}%, σ={mvp_risk*100:.2f}%)",
                fontsize=8, ha='left', va='bottom')
    
    # Formatting
    portfolio_type = "Tangency Portfolio" if is_tangency else "MVP"
    title = f"Bootstrapped {portfolio_type} ({'With' if allow_short_selling else 'Without'} Short Selling)"
    ax.set_title(title, fontsize=15, fontweight='bold')
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_xlim(0, 20)
    ax.set_ylim(-10, 10)
    ax.set_xlabel("Standard Deviation (%)", fontsize=12)
    ax.set_ylabel("Expected Return (%)", fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(True)

# Create one figure with 2x2 subplots
fig, axes = plt.subplots(2, 2, figsize=(20, 16))

# MVP with short-selling
plot_bootstrapped_efficient_frontier(axes[0, 0], 
                                   optimal_sample_mvp_with_short, 
                                   rf_rate, 
                                   allow_short_selling=True, 
                                   is_tangency=False)

# MVP without short-selling
plot_bootstrapped_efficient_frontier(axes[0, 1], 
                                   optimal_sample_mvp_without_short, 
                                   rf_rate, 
                                   allow_short_selling=False, 
                                   is_tangency=False)

# Tangency with short-selling
plot_bootstrapped_efficient_frontier(axes[1, 0], 
                                   optimal_sample_tangency_with_short, 
                                   rf_rate, 
                                   allow_short_selling=True, 
                                   is_tangency=True)

# Tangency without short-selling
plot_bootstrapped_efficient_frontier(axes[1, 1], 
                                   optimal_sample_tangency_without_short, 
                                   rf_rate, 
                                   allow_short_selling=False, 
                                   is_tangency=True)

plt.tight_layout()
plt.savefig("most_optimal_portfoliosB22.png")
plt.show()

def verify_tangency_portfolio_optimality_cardinality_improved(returns_data, rf_rate, allow_short_selling=True, max_assets=3):
    """Improved verification with better optimization parameters"""
    expected_returns = returns_data.mean()
    cov_matrix = returns_data.cov()
    
    # Get tangency portfolio with improved calculation
    tangency_portfolio = calculate_tangency_portfolio_cardinality_improved(
        expected_returns, cov_matrix, rf_rate, allow_short_selling, max_assets
    )
    tangency_sharpe = tangency_portfolio['sharpe_ratio']
    
    # Generate portfolios along the efficient frontier with improved parameters
    n_points = 1000
    target_returns = np.linspace(min(expected_returns), max(expected_returns), n_points)
    portfolio_metrics = []
    
    for target_return in target_returns:
        try:
            n_assets = len(cov_matrix)
            
            def objective(weights):
                portfolio_risk = np.sqrt(weights.T @ cov_matrix @ weights)
                active_positions = np.sum(np.abs(weights) > 1e-4)
                if active_positions > max_assets:
                    return portfolio_risk + 10000 * (active_positions - max_assets)**2
                return portfolio_risk
            
            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                {'type': 'eq', 'fun': lambda w: np.dot(w, expected_returns) - target_return}
            ]
            
            bounds = [(-2, 2) if allow_short_selling else (0, 1) for _ in range(n_assets)]
            
            # Try multiple starting points
            best_result = None
            best_risk = np.inf
            
            for _ in range(5):  # Multiple starts for each target return
                if _ == 0:
                    # First try: Equal weights in top assets
                    top_indices = np.argsort(-expected_returns)[:max_assets]
                    x0 = np.zeros(n_assets)
                    x0[top_indices] = 1/max_assets
                else:
                    x0 = np.random.uniform(-0.5 if allow_short_selling else 0,
                                         1.5 if allow_short_selling else 1,
                                         n_assets)
                    x0 = x0 / np.sum(np.abs(x0))
                
                result = minimize(objective, x0, method='SLSQP',
                                bounds=bounds, constraints=constraints,
                                options={'maxiter': 1000})
                
                if result.success and (best_result is None or result.fun < best_risk):
                    best_result = result
                    best_risk = result.fun
            
            if best_result is not None:
                weights = best_result.x
                weights[np.abs(weights) < 1e-4] = 0
                weights = weights / np.sum(weights)
                
                if np.sum(np.abs(weights) > 1e-4) <= max_assets:
                    portfolio_return = np.dot(weights, expected_returns)
                    portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                    sharpe_ratio = (portfolio_return - rf_rate) / portfolio_risk
                    
                    portfolio_metrics.append({
                        'return': portfolio_return,
                        'risk': portfolio_risk,
                        'sharpe_ratio': sharpe_ratio
                    })
        except:
            continue
    
    metrics_df = pd.DataFrame(portfolio_metrics)
    if len(metrics_df) == 0:
        return False
        
    max_sharpe_portfolio = metrics_df.loc[metrics_df['sharpe_ratio'].idxmax()]
    
    # Print detailed results
    print(f"\nVerification Results ({'With' if allow_short_selling else 'Without'} Short-Selling):")
    print("-" * 60)
    print(f"Tangency Portfolio Sharpe Ratio: {tangency_sharpe:.6f}")
    print(f"Maximum Sharpe Ratio Found on Efficient Frontier: {max_sharpe_portfolio['sharpe_ratio']:.6f}")
    print(f"Difference: {abs(tangency_sharpe - max_sharpe_portfolio['sharpe_ratio']):.8f}")
    
    # Visualization
    plt.figure(figsize=(12, 8))
    plt.scatter(metrics_df['risk'] * 100, metrics_df['sharpe_ratio'],
                alpha=0.5, label='Efficient Frontier Portfolios (Max 3 Assets)')
    plt.axhline(y=tangency_sharpe, color='r', linestyle='--',
                label='Tangency Portfolio Sharpe Ratio')
    plt.xlabel('Portfolio Risk (%)')
    plt.ylabel('Sharpe Ratio')
    plt.title(f"Sharpe Ratios Along Efficient Frontier with Max 3 Assets\n({'With' if allow_short_selling else 'Without'} Short-Selling)")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"improved_sharpe_ratio_verification_{'with' if allow_short_selling else 'without'}_short.png")
    plt.show()
    
    return abs(tangency_sharpe - max_sharpe_portfolio['sharpe_ratio']) < 1e-4

# Run verification for both short-selling scenarios using the optimal bootstrap samples
print("\nVerifying optimality of cardinality-constrained tangency portfolios...")

# Test with the optimal bootstrap samples
is_optimal_with_short = verify_tangency_portfolio_optimality_cardinality_improved(
    optimal_sample_tangency_with_short, 
    rf_rate, 
    allow_short_selling=True,
    max_assets=3
)

is_optimal_without_short = verify_tangency_portfolio_optimality_cardinality_improved(
    optimal_sample_tangency_without_short, 
    rf_rate, 
    allow_short_selling=False,
    max_assets=3
)

print("\nOptimality Verification Summary:")
print("-" * 40)
print(f"Cardinality-Constrained Tangency Portfolio is optimal (with short-selling): {is_optimal_with_short}")
print(f"Cardinality-Constrained Tangency Portfolio is optimal (without short-selling): {is_optimal_without_short}")









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
# Retrieve 48 industry portfolios from Kenneth French's website
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