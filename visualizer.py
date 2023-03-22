import mpl_toolkits
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
x = np.arange(-1, 1, .025)
y = np.arange(-1, 1, .025)
X, Y = np.meshgrid(x, y)


z = ((6.983 * X**2) + (12.415 * Y**2) - X)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, z,cmap='hsv', edgecolor='black')
ax.set_title('Suface Plot')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.grid(True)
plt.show()



