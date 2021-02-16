import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

steps = 10000  # time steps
dt = 1e-1  # dt value for Euler's method

## Constants
G = 1  # gravitational constant ##N*m^2 / kg^2

## Sphere generation
n_0 = 1  # number of black holes
black = np.ones((n_0, 7)) * [1000, 0, 0, 0, 25, 25, 25]  # black hole array

n = 100  # number of spheres
sphere = np.ones((n, 7))  # array to store sphere objects with a mass and position attribute respectively
sphere[:, 1:4] = sphere[:, 1:4] * np.random.rand(n, 3) * 0  # giving each sphere a random momentum from 0-10
sphere[:, 4:7] = sphere[:, 4:7] * np.random.rand(n, 3) * 50  # giving each sphere random positions from 0-100

def force():
    r_sphere = np.linalg.norm(sphere[:, 4:7], axis=1)
    r_black = np.linalg.norm(black[:, 4:7], axis=1)
    f_mag = np.zeros(n)
    f_vec = np.zeros((n, 3))
    for i in range(n_0):
        f_mag += G*sphere[:, 0]*black[i, 0] / (r_sphere-r_black[i])**2
        for k in range(3):
            f_vec[:, k] = f_vec[:, k] + f_mag * (sphere[:, k+4] - black[i, k+4])/(r_sphere - r_black[i])
    return -1*f_vec


## Simulation

fig = plt.figure(figsize=(7,7))
for i in range(steps):
    ax = fig.add_subplot(1,1,1, projection='3d')
    ax.set_xlim3d(0,50)
    ax.set_ylim3d(0,50)
    ax.set_zlim3d(0,50)
    p = sphere[:, 1:4] + dt*force()
    sphere[:, 4:7] = sphere[:, 4:7] + dt*p
    ax.scatter3D(black[:, 4], black[:, 5], black[:, 6], color = 'black')
    ax.scatter3D(sphere[:, 4], sphere[:, 5], sphere[:, 6], color = 'blue', alpha=0.5)
    plt.show(block=False)
    plt.pause(dt)
    plt.clf()


# Do correlation
