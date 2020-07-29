# Copyright (c) 2019 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""

"""

__author__ = "Kyle Vitatus Lopin"
# installed libraries
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint



class ModelParameters():
    def __init__(self):
        self.k_hydro = 0.04  # h**-1


class CompostModel():
    def __init__(self, params: ModelParameters):
        pass


def model(y, t):
    k_hydro = 0.04
    dydt = -k_hydro * y
    return dydt


y0 = 1
t = np.linspace(0, 100)

y = odeint(model, y0, t)

plt.plot(t, y)
plt.show()
