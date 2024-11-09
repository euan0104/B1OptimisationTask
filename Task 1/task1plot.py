import numpy as np
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt

# Define the function
def f(x, y):
    return 100 * (y - x**2)**2 + (1 - x)**2

# Create a grid of points
x = np.linspace(-2, 2, 400)
y = np.linspace(-2, 2, 400)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

# Plot 2D contour
plt.figure()
cp = plt.contourf(X, Y, Z, levels=50, cmap='viridis')
plt.colorbar(cp)
plt.plot(1, 1, 'ro')  # Plot the minimum point
plt.title('2D Contour Plot')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# Plot 3D surface
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
ax.scatter(1, 1, f(1, 1), color='r')  # Plot the minimum point
ax.set_title('3D Surface Plot')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x, y)')
plt.show()