# Copyright (c) 2019 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""

"""

__author__ = "Kyle Vitatus Lopin"
# standard libraries

# installed libraries
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

NUM_SAMPLES = 3

time = np.linspace(1, 200, 200)
time_samples = [50, 100, 150]


def k1(temp, a, b, c):
    # print('====')
    # print(temp, a, b, c)
    return a * np.exp(b * temp) + c


def k2(temp, a, b, c):
    return -a * np.exp(-b * temp) + c


def temperatures(time):
    return 8 * np.sin(time/10+5.5) + 1 * np.sin(time/20) + \
           33 + 4 * np.random.rand(len(time))

temperature_list = temperatures(time)
plt.plot(time, temperature_list)
plt.show()

k1_params = [1, 0.05, 0]
k2_params = [50, 0.1, 20]
# k_params = k1_params.copy()
# k_params.extend(k2_params)

def diff_steps(times, a1, b1, c1):
    k2_params = [50, 0.1, 20]
    _bulk = [1]
    _degrade = [0]
    _CO2 = [0]
    time2 = [0]
    # k1_params = [a1, b1, c1]
    for i, temp in enumerate(temperature_list):
        _k1 = k1(temp, a1, b1, c1)
        # print('k1 = ', _k1, temp)
        # print('ll = ', a1, b1, c1)
        _k2 = k2(temp, *k2_params)
        degrade_step = _k1 * _bulk[-1] / 400
        # print(degrade_step)
        CO2_step = _k2 * _degrade[-1] / 400
        _bulk.append(_bulk[-1]-degrade_step)
        # print(_degrade)
        _degrade.append(_degrade[-1]+degrade_step-CO2_step)
        _CO2.append(_CO2[-1]+CO2_step)
        time2.append(time2[-1]+1)

    bulk = []
    degrade = []
    CO2 = []
    for t in times:
        # print(_bulk)
        # print(t)
        #
        # print(times)
        # print(_bulk[int(t)])
        bulk.append(_bulk[int(t)])
        degrade.append(_degrade[int(t)])
        CO2.append(_CO2[int(t)])
    return bulk, degrade, CO2, time2


def diff_steps_cf(times, a1, b1, c1):
    k2_params = [50, 0.1, 20]
    _bulk = [1]
    _degrade = [0]
    _CO2 = [0]
    time2 = [0]
    # k1_params = [a1, b1, c1]
    for i, temp in enumerate(temperature_list):
        _k1 = k1(temp, a1, b1, c1)
        # print('k1 = ', _k1, temp)
        # print('ll = ', a1, b1, c1)
        _k2 = k2(temp, *k2_params)
        degrade_step = _k1 * _bulk[-1] / 400
        # print(degrade_step)
        CO2_step = _k2 * _degrade[-1] / 400
        _bulk.append(_bulk[-1]-degrade_step)
        # print(_degrade)
        _degrade.append(_degrade[-1]+degrade_step-CO2_step)
        _CO2.append(_CO2[-1]+CO2_step)
        time2.append(time2[-1]+1)

    bulk = []
    degrade = []
    CO2 = []
    for t in times:

        bulk.append(_bulk[int(t)])
        degrade.append(_degrade[int(t)])
        CO2.append(_CO2[int(t)])
    return bulk


# k1_s = k1(temperature_list, *k1_params)
# k2_s = k2(temperature_list, *k2_params)
print('=====')
# print(k1_s)

bulk, degrade, CO2, time2 = diff_steps(time, *k1_params)
plt.plot(time, bulk, label="bulk plastic")
plt.plot(time, CO2, label="CO2 produced")

bulk_samples = []
for time_sample in time_samples:
    bulk_samples.append(bulk[time_sample])
print(len(time2), len(bulk))
print(time_samples)
print(bulk_samples)
plt.scatter(time_samples, bulk_samples, marker='x', c='red', s=50)
print('=====>>>>>>>>>>>>>>>>>>')
print(time_samples)
print(bulk_samples)
fit_values, _ = curve_fit(diff_steps_cf, time_samples, bulk_samples, p0=(1.0, 0.1, 0))

print(fit_values)
print('*******')
print(diff_steps_cf(time_samples, *fit_values))

bulk_fit, degrade_fit, CO2_fit, time2_fit = diff_steps(time, *fit_values)

plt.plot(time, bulk_fit, label="bulk plastic fit", linestyle='--')
plt.plot(time, CO2_fit, label="CO2 produced fit", linestyle='--')

plt.show()

temp = np.linspace(20, 50, 50)
k1_actual = k1(temp, *k1_params)
k1_fit = k1(temp, *fit_values)

plt.plot(temp, k1_actual, label='k1')
plt.plot(temp, k1_fit, label='k1_fit')

plt.legend()
plt.show()

