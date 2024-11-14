import numpy as np

import matplotlib.pyplot as plt

def rosenbrock(x):
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

def gradient(x):
    grad = np.zeros_like(x)
    grad[0] = -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0]**2)
    grad[1] = 200 * (x[1] - x[0]**2)
    return grad

def hessian(x):
    hess = np.zeros((2, 2))
    hess[0, 0] = 2 - 400 * x[1] + 1200 * x[0]**2
    hess[0, 1] = -400 * x[0]
    hess[1, 0] = -400 * x[0]
    hess[1, 1] = 200
    return hess

def gradient_descent(x0, lr=0.001, tol=1e-6, max_iter=10000):
    x = x0
    path = [x]
    for _ in range(max_iter):
        grad = gradient(x)
        x = x - lr * grad
        path.append(x)
        if np.linalg.norm(grad) < tol:
            break
    return x, path

def newton_method(x0, tol=1e-6, max_iter=100):
    x = x0
    path = [x]
    for _ in range(max_iter):
        grad = gradient(x)
        hess = hessian(x)
        x = x - np.linalg.inv(hess).dot(grad)
        path.append(x)
        if np.linalg.norm(grad) < tol:
            break
    return x, path

def gauss_newton_method(x0, lr=0.001, tol=1e-6, max_iter=10000):
    x = x0
    path = [x]
    for _ in range(max_iter):
        grad = gradient(x)
        hess = hessian(x)
        x = x - lr * np.linalg.inv(hess).dot(grad)
        path.append(x)
        if np.linalg.norm(grad) < tol:
            break
    return x, path

def plot_convergence(paths, title):
    plt.figure(figsize=(10, 6))
    for path in paths:
        path = np.array(path)
        plt.plot(path[:, 0], path[:, 1], marker='o')
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()
    plt.show()

# Initializations
initializations = [np.random.randn(2) for _ in range(3)]

# Gradient Descent
gd_paths = [gradient_descent(x0)[1] for x0 in initializations]
plot_convergence(gd_paths, 'Gradient Descent Convergence')

# Newton's Method
newton_paths = [newton_method(x0)[1] for x0 in initializations]
plot_convergence(newton_paths, 'Newton\'s Method Convergence')

# Gauss-Newton Method
gn_paths = [gauss_newton_method(x0)[1] for x0 in initializations]
plot_convergence(gn_paths, 'Gauss-Newton Method Convergence')