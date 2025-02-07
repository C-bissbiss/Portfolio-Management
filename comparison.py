import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import pandas_datareader as web

def retrieve_industry_data():
    """
    Retrieve 48 industry portfolios from Kenneth French website
    """
    # Instantiate the reader for 48 Industry Portfolios to capture the monthly data
    industry_reader = web.famafrench.FamaFrenchReader('48_Industry_Portfolios', start='2019-12', end='2024-12')
    
    # Read the data
    portfolios_data = industry_reader.read()[0]  # Assuming we need the first table [0]
    
    # Clean missing data
    portfolios_data = portfolios_data.replace([-99.99, -999], np.nan)
    
    return portfolios_data

# Retrieve the data
portfolios_data = retrieve_industry_data()

# Drop all industries except for Rtail, Whlsl, BusSv, Comps, and Cnstr
selected_industries = ['Rtail', 'Whlsl', 'BusSv', 'Comps', 'Cnstr']
portfolios_data = portfolios_data[selected_industries].dropna()

# Compute mean returns and covariance matrix
mean_returns = portfolios_data.mean()
cov_matrix = portfolios_data.cov()

# Compute MVP parameters
inv_cov = np.linalg.inv(cov_matrix)
ones = np.ones(len(mean_returns))
A = ones.T @ inv_cov @ ones
B = ones.T @ inv_cov @ mean_returns
C = mean_returns.T @ inv_cov @ mean_returns
Delta = A * C - B**2

# Generate a range of expected returns (mu) to capture the full locus
mu_vals = np.linspace(min(mean_returns) - 2, max(mean_returns) + 4, 300)

# Compute standard deviation for each mu using the minimum-variance equation
sigma_vals = np.sqrt(np.maximum((A * mu_vals**2 - 2 * B * mu_vals + C) / Delta, 0))

# Plot the Full Mean-Variance Locus (Full Parabola)
plt.figure(figsize=(10, 6))
plt.plot(sigma_vals, mu_vals, label="Mean-Variance Locus", color="blue")

# Define different colors for each industry
industry_colors = ['blue', 'orange', 'purple', 'yellow', 'pink']  

# Plot each industry's mean and variance 
for i, industry in enumerate(selected_industries):
    std_dev = np.sqrt(cov_matrix.loc[industry, industry])  # Standard deviation (risk)
    mean_return = mean_returns[industry]  # Mean return
    color = industry_colors[i % len(industry_colors)]  
    plt.scatter(std_dev, mean_return, marker='o', s=50, label=industry, color=color)

# Plot the Global Minimum Variance Portfolio (GMVP)
gmvp_std = np.sqrt(1 / A)
gmvp_return = B / A
plt.scatter(gmvp_std, gmvp_return, color='red', marker='o', s=120, label="Global Minimum Variance Portfolio")
plt.text(gmvp_std - 0.3, gmvp_return, "GMVP", fontsize=12, verticalalignment='bottom', horizontalalignment='right', color='red')

print(f"GMVP Return: {gmvp_return}, GMVP Std Dev: {gmvp_std}")

# Add reference lines at y = 0 and x = 0
plt.axhline(y=0, color='orange', linestyle='-', linewidth=1) 
plt.axvline(x=0, color='orange', linestyle='-', linewidth=1)  

# Labels and title
plt.xlabel("Standard Deviation (Risk)")
plt.ylabel("Expected Return")
plt.title("Full Mean-Variance Locus without the risk-free asset")
plt.legend()
plt.grid()
plt.show()