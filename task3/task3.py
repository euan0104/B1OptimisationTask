import numpy as np
from scipy.optimize import minimize

import matplotlib.pyplot as plt

# Define Rosenbrock's function
def rosenbrock(x):
    return sum(100.0 * (x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)

# Perform optimization using Nelder-Mead Simplex algorithm
def optimize_rosenbrock(initial_guess):
    result = minimize(rosenbrock, initial_guess, method='Nelder-Mead')
    return result

# Generate three distinct random initializations
initial_guesses = [np.random.rand(2) * 10 - 5 for _ in range(3)]

# Store results for visualization
results = []

# Optimize from each initial guess
for initial_guess in initial_guesses:
    result = optimize_rosenbrock(initial_guess)
    results.append(result)

# Plot convergence
plt.figure(figsize=(10, 6))
for i, result in enumerate(results):
    plt.plot(result.nit, result.fun, label=f'Run {i+1}')
plt.xlabel('Iterations')
plt.ylabel('Function Value')
plt.title('Convergence of Rosenbrock Function Minimization')
plt.legend()
plt.grid(True)
plt.show()