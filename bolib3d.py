import numpy as np

import matplotlib.pyplot as plt
from matplotlib import pyplot, cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D

import bolib

plt.style.use('ggplot')


def plot_3d(objective_function, length=100):
    """
    Plot 3D functions
    :param objective_function:
    :type objective_function:
    :param length:
    :type length:
    :return:
    :rtype:
    """
    bounds = objective_function.get_bounds()

    if len(bounds) != 2:
        return

    x_grid = np.linspace(bounds[0][0], bounds[0][1], length)
    y_grid = np.linspace(bounds[1][0], bounds[1][1], length)
    x_grid, y_grid = np.meshgrid(x_grid, y_grid)
    grid = np.vstack((x_grid.flatten(), y_grid.flatten())).T
    z_points = objective_function.evaluate(grid)
    z_points = z_points.reshape(length, length)

    fig = pyplot.figure()
    axis = fig.gca(projection='3d')

    surf = axis.plot_surface(x_grid, y_grid,
                             z_points, rstride=1, cstride=1,
                             cmap=cm.jet, linewidth=0, antialiased=False,
                             alpha=0.3)
    axis.contour(x_grid.tolist(), y_grid.tolist(), z_points.tolist(),
                 zdir='z', offset=z_points.min(), cmap=cm.jet)

    axis.set_xlim(bounds[0][0], bounds[0][1])
    axis.set_ylim(bounds[1][0], bounds[1][1])
    pyplot.title(objective_function.__class__.__name__)
    axis.zaxis.set_major_locator(LinearLocator(10))
    axis.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    fig.colorbar(surf, shrink=0.5, aspect=5)

    pyplot.show()


if __name__ == '__main__':
    OFS = [
        bolib.ofs.Branin,
        bolib.ofs.Camelback,
        bolib.ofs.Hartmann,
        bolib.ofs.Rastrigin,
        bolib.ofs.Rosenbrock,
        bolib.ofs.Schwefel,
        bolib.ofs.Sphere
    ]

    for objective_function_class in OFS:
        objective_function = objective_function_class()
        plot_3d(objective_function)