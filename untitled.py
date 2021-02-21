# galaxy collision simulation by Omar Hassan, Devon Joseph, and Kyle Castrojeres

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as sp
from mpl_toolkits.mplot3d import Axes3D
#plt.rcParams['axes.facecolor'] = 'black'

steps = 10000  # time steps
dt = 1e-2  # dt value for Euler's method

## Constants
G = 1  # gravitational constant

## Sphere generation
n_0 = 1  # number of black holes
black = np.ones((n_0, 7)) * [300000, 0, 0, 0, 1, 1, 1]  # black hole array

n = 100  # number of spheres
sphere = np.ones((n, 7))  # array to store sphere objects with a mass and position attribute respectively
sphere[:, 1:4] = sphere[:, 1:4] * np.random.randint(-100, 100,
                                                    size=(n, 3))  # giving each sphere a random momentum from -100, 100
sphere[:, 4:] = sphere[:, 4:] * np.random.randint(-50, 50,
                                                  size=(n, 3))  # giving each sphere random positions from -50, 50


def force():
    r_sphere = np.linalg.norm(sphere[:, 4:], axis=1)
    r_black = np.linalg.norm(black[:, 4:], axis=1)
    f_vec = np.zeros((n, 3))
    for i in range(n_0):
        f_mag = G * (sphere[:, 0] * black[i, 0]) / (r_sphere - r_black[i]) ** 2
        f_vec = f_vec + f_mag[:, None] * ((sphere[:, 4:] - black[0, 4:])
                                          / (np.abs(r_sphere - r_black[i])[:, None]))
    return -f_vec


## Simulation

fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(1, 1, 1, projection='3d', xlim=(-100, 100), ylim=(-100, 100), zlim=(-100, 100))
# ax.set_xticks([])
# ax.set_yticks([])
# ax.set_zticks([])
# ax.grid(False)
# ax.axis('off')

for i in range(steps):
    # ax.set_xlim3d(black[:, 4] - 100, black[:, 4] + 100)
    # ax.set_ylim3d(black[:, 5] - 100, black[:, 6] + 100)
    # ax.set_zlim3d(black[:, 5] - 100, black[:, 6] + 100)
    sphere[:, 1:4] = sphere[:, 1:4] + dt * force()
    sphere[:, 4:] = sphere[:, 4:] + dt * sphere[:, 1:4] / (sphere[:, 0][:, None])
    #black[:, 4:] = black[:, 4:] + dt * black[:, 1:4] / (black[:, 0][:, None])
    ax.scatter3D(black[:, 4], black[:, 5], black[:, 6], color='black')
    ax.scatter3D(sphere[:, 4], sphere[:, 5], sphere[:, 6], color="b", alpha=0.5, s=5)
    plt.show(block=False)
    plt.pause(dt)
    #pts.remove()
    del ax.collections[:]
    # plt.clf()

# Do correlation
