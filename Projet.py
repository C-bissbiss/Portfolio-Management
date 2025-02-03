# Import all packages for the project
import pandas as pd
import numpy as np
import pandas_datareader as web
import warnings
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

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
    Calculate Minimum Variance Portfolio (MVP) weights
    
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
    if allow_short_selling == True:
        # MVP with short-selling (quadratic programming)
        def portfolio_variance(weights):
            return weights.T @ cov_matrix @ weights
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # weights sum to 1
        ]
        
        # Bounds (with short-selling)
        bounds = [(-np.inf, np.inf) for _ in range(len(cov_matrix))]
        
        # Initial guess: equal weights
        initial_weights = np.ones(len(cov_matrix)) / len(cov_matrix)
        
        # Optimization
        result = minimize(
            portfolio_variance, 
            initial_weights, 
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        mvp_weights = result.x
    else:
        # MVP with no short-selling (quadratic programming)
        def portfolio_variance(weights):
            return weights.T @ cov_matrix @ weights
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # weights sum to 1
        ]
        
        # Bounds (no short-selling)
        bounds = [(0, np.inf) for _ in range(len(cov_matrix))]
        
        # Initial guess: equal weights
        initial_weights = np.ones(len(cov_matrix)) / len(cov_matrix)
        
        # Optimization
        result = minimize(
            portfolio_variance, 
            initial_weights, 
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        mvp_weights = result.x
    
    return mvp_weights

# Calculate MVP weights
mvp_weights = calculate_mvp_weights(cov_matrix, allow_short_selling=True)

# Calculate MVP return and risk
mvp_return = np.dot(mvp_weights, expected_returns)
mvp_risk = np.sqrt(mvp_weights.T @ cov_matrix @ mvp_weights)

sharpe_ratios = mvp_return / mvp_risk  # No risk-free asset

# Convert values for graphing purposes
mvp_return_plot = mvp_return * 100
mvp_risk_plot = mvp_risk  * 100
expected_returns_plot = expected_returns * 100
mvp_return_plot = mvp_return * 100
mvp_risk_plot = mvp_risk * 100

# Plot Efficient Frontier
plt.figure(figsize=(10, 6))
plt.grid(True)

# Mark each industry
for industry in expected_returns.index:
    plt.scatter(np.std(returns[industry]) * 100, expected_returns[industry] * 100, 
                marker='X', s=200, label=industry)

# Mark MVP
plt.scatter(mvp_risk_plot, mvp_return_plot, color="red", marker="*", s=200, label="Minimum Variance Portfolio")

# Add x and y value to the MVP point (coordinates)
plt.text(mvp_risk_plot, mvp_return_plot, f"(r = {mvp_return_plot:.2f}%), σ = {mvp_risk_plot:.2f}%)", fontsize=8, ha='right')

# Format axes as percentages
plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter())
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())

plt.xlim(0, 10)
plt.ylim(-3, 3)

# Labels and title
plt.xlabel("Standard Deviation (%)", fontsize=12)
plt.ylabel("Expected Return (%)", fontsize=12)
plt.title("Efficient Frontier with MVP (With Short Selling)", fontsize=15, fontweight='bold')
plt.legend()
plt.savefig("efficient_frontierA1.png")
plt.show()

# Print the MVP weights
print("Minimum Variance Portfolio Weights:", mvp_weights * 100, "%")
print("Minimum Variance Portfolio Expected Return:", mvp_return * 100, "%")
print("Minimum Variance Portfolio S:", mvp_risk * 100, "%")






# A) 2. Calculate tangency portfolio and CML (w/ risk-free rate and w/ short-selling)

# Retrieve risk-free rate from Kenneth French website
def retrieve_risk_free_rate():
    rf_data = web.famafrench.FamaFrenchReader('F-F_Research_Data_Factors', start='2019-12', end='2024-12')
    rf = rf_data.read()[0]['RF'] / 100  # Convert to decimal
    return rf.mean()

rf_rate = retrieve_risk_free_rate()

inv_cov_matrix = np.linalg.inv(cov_matrix)

# Calculate the Tangency Portfolio (Optimal Risky Portfolio)
# Compute excess returns vector (expected returns over the risk-free rate)
excess_returns = expected_returns - rf_rate

# Calculate tangency portfolio weights
tangency_weights = inv_cov_matrix @ excess_returns / (ones.T @ inv_cov_matrix @ excess_returns)

# Compute the tangency portfolio's expected return and risk
tangency_return = np.dot(tangency_weights, expected_returns)
tangency_risk = np.sqrt(tangency_weights.T @ cov_matrix @ tangency_weights)

# Compute the tangency portfolio's Sharpe ratio
tangency_sharpe_ratio = (tangency_return - rf_rate) / tangency_risk

# Display results for the tangency portfolio
print("Tangency Portfolio Weights:", tangency_weights * 100, "%")
print("Tangency Portfolio Expected Return:", tangency_return * 100, "%")
print("Tangency Portfolio Risk (Std Dev):", tangency_risk * 100, "%")
print("Tangency Portfolio Sharpe Ratio:", tangency_sharpe_ratio)

# Plotting the Efficient Frontier along with the Capital Market Line (CML)

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






# A) 3.  Describe the tangent portfolio and its characteristics

print("\n------ Tangency Portfolio Characteristics ------")
for asset, weight in zip(expected_returns.index, tangency_weights):
    print(f"{asset}: {weight*100:.2f}%")
    
print(f"\nTangency Portfolio Expected Return: {tangency_return*100:.2f}%")
print(f"Tangency Portfolio Standard Deviation: {tangency_risk*100:.2f}%")
print(f"Tangency Portfolio Sharpe Ratio: {tangency_sharpe_ratio:.4f}")

# Ensure weights sum to 1
print(f"\nSum of Tangency Portfolio Weights: {np.sum(tangency_weights):.4f} (should be close to 1)")

# Ensure no negative weights if needed
if np.any(tangency_weights < 0):
    print("Warning: Some assets have negative weights (short selling is occurring).")

# Verification: Compare the Tangency Portfolio Sharpe Ratio with other portfolios
simulated_sharpe_ratios = (expected_returns - rf_rate) / np.sqrt(np.diag(cov_matrix))  # Simulated Sharpe Ratios
max_simulated_sharpe = np.max(simulated_sharpe_ratios)

print(f"\nMaximum Simulated Portfolio Sharpe Ratio: {max_simulated_sharpe:.4f}")
print(f"Tangency Portfolio Sharpe Ratio: {tangency_sharpe_ratio:.4f}")

if tangency_sharpe_ratio >= max_simulated_sharpe:
    print("Verification Successful: The Tangency Portfolio has the highest Sharpe ratio.")
else:
    print("Warning: Some individual assets may have a higher Sharpe ratio. Review constraints.")





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

# Print the MVP weights 2 decimal places
print("Minimum Variance Portfolio Weights:", mvp_weights * 100, "%")
print("Minimum Variance Portfolio Expected Return:", mvp_return * 100, "%")
print("Minimum Variance Portfolio Standard Deviation:", mvp_risk * 100, "%")


