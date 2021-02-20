# galaxy collision simulation by Omar Hassan, Devon Joseph, and Kyle Castrojeres

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

steps = 10000  # time steps
dt = 1e-1  # dt value for Euler's method

## Constants
G = 1  # gravitational constant

## Sphere generation
n_0 = 1  # number of black holes
black = np.ones((n_0, 7)) * [300000, 0, 0, 0, 0, 0, 0]  # black hole array

n = 10  # number of spheres
sphere = np.ones((n, 7))  # array to store sphere objects with a mass and position attribute respectively
sphere[:, 1:4] = sphere[:, 1:4] * np.random.randint(-100, 100,
                                                    size=(n, 3))  # giving each sphere a random momentum from 0-10
sphere[:, 4:] = sphere[:, 4:] * np.random.randint(-50, 50,
                                                  size=(n, 3))  # giving each sphere random positions from 0-100


def force():
    r_sphere = np.linalg.norm(sphere[:, 4:], axis=1)
    r_black = np.linalg.norm(black[:, 4:], axis=1)
    f_mag = np.zeros(n)
    f_vec = np.zeros((n, 3))
    for i in range(n_0):
        f_mag += G * (sphere[:, 0] * black[0, 0]) / (r_sphere - r_black[0]) ** 2
        f_vec[:, 0] = f_vec[:, 0] + f_mag * ((sphere[:, 4] - black[0, 4]) / (np.abs(r_sphere - r_black[i])))
        f_vec[:, 1] = f_vec[:, 1] + f_mag * ((sphere[:, 5] - black[0, 5]) / (np.abs(r_sphere - r_black[i])))
        f_vec[:, 2] = f_vec[:, 2] + f_mag * ((sphere[:, 6] - black[0, 6]) / (np.abs(r_sphere - r_black[i])))
    return -f_vec


## Simulation

fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(1, 1, 1, projection='3d')

for i in range(steps):
    if i < 20:
        plt.clf()
        ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.set_xlim3d(-100, 100)
    ax.set_ylim3d(-100, 100)
    ax.set_zlim3d(-100, 100)
    sphere[:, 1:4] = sphere[:, 1:4] + dt * force()
    sphere[:, 4:] = sphere[:, 4:] + dt * sphere[:, 1:4]
    ax.scatter3D(black[:, 4], black[:, 5], black[:, 6], color='black')
    ax.scatter3D(sphere[:, 4], sphere[:, 5], sphere[:, 6], color="b", alpha=0.5, s=5)
    plt.show(block=False)
    plt.pause(dt)
    # plt.clf()

# Do correlation
