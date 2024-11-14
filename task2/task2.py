import math
import numpy as np
import matplotlib.pyplot as plt

# Define the Rosenbrock function
def rosenbrock(x, y):
    return 100 * (y - x**2)**2 + (1 - x)**2

# Define the gradient of the Rosenbrock function
def rosenbrock_gradient(x, y):
    df_dx = -400 * x * (y - x**2) - 2 * (1 - x)
    df_dy = 200 * (y - x**2)
    return np.array([df_dx, df_dy])

# Define the Hessian of the Rosenbrock function
def rosenbrock_hessian(x, y):
    d2f_dx2 = 1200 * x**2 - 400 * y + 2
    d2f_dxdy = -400 * x
    d2f_dy2 = 200
    return np.array([[d2f_dx2, d2f_dxdy], [d2f_dxdy, d2f_dy2]])

# Gradient descent algorithm
def gradient_descent(x0, y0, learning_rate=0.001, max_iter=10000, tol=1e-6):
    x, y = x0, y0
    path = [(x, y)]
    for _ in range(max_iter):
        grad = rosenbrock_gradient(x, y)
        x_new = x - learning_rate * grad[0]
        y_new = y - learning_rate * grad[1]
        path.append((x_new, y_new))
        if np.linalg.norm([x_new - x, y_new - y]) < tol:
            break
        x, y = x_new, y_new
    return np.array(path)

# Newton's method algorithm
def newton_method(x0, y0, max_iter=10000, tol=1e-6):
    x, y = x0, y0
    path = [(x, y)]
    for _ in range(max_iter):
        grad = rosenbrock_gradient(x, y)
        hess = rosenbrock_hessian(x, y)
        delta = np.linalg.solve(hess, grad)
        x_new = x - delta[0]
        y_new = y - delta[1]
        path.append((x_new, y_new))
        if np.linalg.norm([x_new - x, y_new - y]) < tol:
            break
        x, y = x_new, y_new
    return np.array(path)

# Gauss-Newton method algorithm (example, assuming it's similar to Newton's method)
def gauss_newton_method(x0, y0, max_iter=10000, tol=1e-6):
    x, y = x0, y0
    path = [(x, y)]
    for _ in range(max_iter):
        grad = rosenbrock_gradient(x, y)
        hess = rosenbrock_hessian(x, y)
        delta = np.linalg.solve(hess, grad)
        x_new = x - delta[0]
        y_new = y - delta[1]
        path.append((x_new, y_new))
        if np.linalg.norm([x_new - x, y_new - y]) < tol:
            break
        x, y = x_new, y_new
    return np.array(path)

# Create a grid of points for contour plot
x = np.linspace(-2, 2, 400)
y = np.linspace(-2, 2, 400)
X, Y = np.meshgrid(x, y)
Z = rosenbrock(X, Y)

# Random initializations
num_initializations = 3
initializations = np.random.uniform(-2, 2, size=(num_initializations, 2))

# Methods dictionary
methods = {
    "Gradient Descent": gradient_descent,
    "Newton's Method": newton_method,
    "Gauss-Newton Method": gauss_newton_method
}


#levels generator
##lv = np.zeros(800)
#for k in range(1, 801):
#    lv[k-1] = math.exp(0.2*k)


lv = np.geomspace(1,36000,40)


# Execute and plot for each method and each initialization
for method_name, method in methods.items():
    for i, (initial_x, initial_y) in enumerate(initializations):
        path = method(initial_x, initial_y)
        plt.figure()
        cp = plt.contour(X, Y, Z, levels=lv, cmap='viridis')
        plt.colorbar(cp, label='Gradient')
        plt.plot(path[:, 0], path[:, 1], marker='o', label='Path')
        plt.plot(path[0, 0], path[0, 1], 'ro', label='Start Point')
        plt.plot(path[-1, 0], path[-1, 1], 'go', label='End Point')
        plt.title(f'{method_name} Convergence from Initialization {i+1}')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend(loc='best')
        plt.show()