# Copyright (c) 2019 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""

"""

__author__ = "Kyle Vitatus Lopin"
# installed libraries
import matplotlib.pyplot as plt
import numpy as np


class ModelParameters():
    def __init__(self):
        k_hydro_environ = 0.04  # h**-1
        B_sat = 10**-4  # g g**-1
        B_growth = 1


class CompostModel():
    def __init__(self):
        pass


S_bact = 62

time = np.linspace(1, 1000, 1000)
bact = [1]
for t in time:
    b_1 = bact[-1]
    bact_growth = 0.2 * b_1 * (1 - b_1/S_bact)
    bact_death = .04 * b_1
    bact.append(bact[-1] + bact_growth - bact_death)


print(time)
plt.plot(time, bact[1:])
plt.show()


