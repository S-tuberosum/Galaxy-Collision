# galaxy collision simulation by Omar Hassan, Devon Joseph, and Kyle Castrojeres

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
#plt.rcParams['axes.facecolor'] = 'black'

steps = 10000  # time steps
dt = 0.05  # dt value for Euler's method

## Constants
G = 1  # gravitational constant

## Sphere generation
n_0 = 1  # number of black holes
black = np.ones((n_0, 7)) * [300000, 0, 0, 0, 0, 0, 0]  # black hole array

n = 2000  # number of spheres
sphere = np.ones((n, 7))  # array to store sphere objects with a mass and position attribute respectively
sphere[:, 4:] = sphere[:, 4:] * np.random.randint(30, 60, size=(n, 3))
sphere[:, 1:4] = sphere[:, 1:4] * np.sqrt(G * black[0, 0]/(np.linalg.norm(sphere[:, 4:], axis=1)))[:, None]/2
sphere[:, 2:3] = sphere[:, 2:3] * -1
sphere[:, 3:4] = 0


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
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
# ax.grid(False)
# ax.axis('off')

ax.scatter3D(black[:, 4], black[:, 5], black[:, 6], color='black')
graph = ax.scatter3D(sphere[:, 4], sphere[:, 5], sphere[:, 6], color="b", alpha=0.5, s=1)


def update(i):
    sphere[:, 1:4] = sphere[:, 1:4] + dt * force()
    sphere[:, 4:] = sphere[:, 4:] + dt * sphere[:, 1:4] / (sphere[:, 0][:, None])
    graph._offsets3d = (sphere[:, 4], sphere[:, 5], sphere[:, 6])


ani = animation.FuncAnimation(fig, update, steps, interval=1)
plt.show()

