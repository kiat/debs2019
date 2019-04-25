import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.gca( projection='3d')

# Scatter graph
N = 100
X = np.random.uniform(-90, 90, N)
Y = np.random.uniform(-90, 90, N)
Z = np.random.uniform(-3, 29, N)
ax.scatter(X, Y, Z)

# Cylinder
x=np.linspace(-120, 120, 100)
z=np.linspace(-3, 29, 100)
Xc, Zc=np.meshgrid(x, z)
Yc = np.sqrt(14400-Xc**2)

# Draw parameters
rstride = 20
cstride = 10
ax.plot_surface(Xc, Yc, Zc, alpha=0.2, rstride=rstride, cstride=cstride)
ax.plot_surface(Xc, -Yc, Zc, alpha=0.2, rstride=rstride, cstride=cstride)

ax.view_init(elev=20, azim=60)

ax.set_xlabel("Z")
ax.set_ylabel("X")
ax.set_zlabel("Y")
plt.show()

fig.savefig("/home/saeed/data.pdf", bbox_inches='tight')