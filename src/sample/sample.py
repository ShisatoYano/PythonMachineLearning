import math
import numpy as np
import matplotlib.pyplot as plt

pi = math.pi

x = np.linspace(0, 2 * pi, 100)

y = np.sin(x)

plt.plot(x, y)

plt.show()
