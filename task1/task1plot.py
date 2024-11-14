import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class rosenblock:

    # Define the function
    @staticmethod
    def f(x, y):
        return 100 * (y - x**2)**2 + (1 - x)**2

    def plot(self):
        
        self.Z = f(self.X, self.Y)
        plt.figure(self)
        cp = plt.contourf(self.X, self.Y, self.Z, levels=50, cmap='viridis')
        plt.colorbar(cp)
        plt.plot(1, 1, 'ro')  # Plot the minimum point
        plt.title('2D Contour Plot')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()

        # Plot 3D surface
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(self.X, self.Y, self.Z, cmap='viridis', edgecolor='none')
        ax.scatter(1, 1, self.f(1, 1), color='r')  # Plot the minimum point
        ax.set_title('3D Surface Plot')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('f(x, y)')
        plt.show()