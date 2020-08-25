# Copyright (c) 2019 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""

"""

__author__ = "Kyle Vitatus Lopin"

from neuron import h, gui

soma = h.Section(name='soma')
soma.insert('pas')
print("type(soma) = {}".format(type(soma)))
print("type(soma(0.5)) ={}".format(type(soma(0.5))))
mech = soma(0.5).pas
print(dir(mech))
print(mech.g)
print(soma(0.5).pas.g)
print(soma)
asyn = h.AlphaSynapse(soma(0.5))

dir(asyn)
print("asyn.e = {}".format(asyn.e))
print("asyn.gmax = {}".format(asyn.gmax))
print("asyn.onset = {}".format(asyn.onset))
print("asyn.tau = {}".format(asyn.tau))
# Let’s assign the onset of this synapse to occur at 20 ms and the maximal conductance to 1.

asyn.onset = 20
asyn.gmax = 1
# Let’s look at the state of our cell using neuron’s psection().

h.psection()
