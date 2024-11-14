import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Define the Rosenbrock function
def rosenbrock(x):
    return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2

# Create a grid of points for contour plot
x = np.linspace(-2, 2, 400)
y = np.linspace(-2, 2, 400)
X, Y = np.meshgrid(x, y)
Z = rosenbrock([X, Y])

lv = np.geomspace(1,36000,40)

# Generate random initializations
num_initializations = 3
initializations = np.random.uniform(-2, 2, size=(num_initializations, 2))

# Execute and plot for each initialization
for i, initial in enumerate(initializations):
    result = minimize(rosenbrock, initial, method='Nelder-Mead', options={'return_all': True})
    path = result.allvecs
    path = np.array(path)
    
    plt.figure()
    cp = plt.contour(X, Y, Z, levels=lv, cmap='viridis')
    plt.colorbar(cp, label='Gradient')
    plt.plot(path[:, 0], path[:, 1], marker='o', label='Path')
    plt.plot(path[0, 0], path[0, 1], 'ro', label='Start Point')
    plt.plot(path[-1, 0], path[-1, 1], 'go', label='End Point')
    plt.title(f'Nelder-Mead Convergence from Initialization {i+1}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc='best')
    plt.show()