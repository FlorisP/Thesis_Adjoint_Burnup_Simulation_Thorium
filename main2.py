import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

class Neuclide:
    def __init__(self,name, initDensity, decay, nccs):
        self.name = name
        self.initDensity = initDensity
        self.decay = decay
        self.nccs = nccs

Th231 = Neuclide('Th231', initDensity = 0, decay = 1/25, nccs = 0)
Th232 = Neuclide('Th232', initDensity = 1000, decay = 0, nccs = 0.1)
Th233 = Neuclide('Th233', initDensity = 0, decay = 60/22, nccs = 5)

Pa231 = Neuclide('Pa231', initDensity = 0, decay = 0, nccs = 0)
Pa233 = Neuclide('Pa233', initDensity = 0, decay = 0, nccs = 0)
Pa234 = Neuclide('Pa234', initDensity = 0, decay = 0, nccs = 0)
Pa235 = Neuclide('Pa235', initDensity = 0, decay = 0, nccs = 0)

U233 = Neuclide('U233', initDensity = 0, decay = 0, nccs = 0)
U234 = Neuclide('U234', initDensity = 0, decay = 0, nccs = 0)
U235 = Neuclide('U235', initDensity = 0, decay = 0, nccs = 0)

Neuclides = [Th231,Th232,Th233,Pa231,Pa233,Pa234,Pa235,U233,U234,U235]

def odefunc(y, t, flux):
    NTh232, NTh231, NTh233, NPa231 = y
    
    funTh232 = - NTh232*(Th232.decay + flux * Th232.nccs)
    funTh231 = - NTh231*(Th231.decay) + 0.1 * NTh232 * flux * Th232.nccs
    funTh233 = - NTh233*(Th231.decay) + 0.9 * NTh232 * flux * Th232.nccs
    
    funPa231 = - NPa231*(Pa231.decay) + NTh231*(Th231.decay)
    
    dydt = [funTh232, funTh231,funTh233, funPa231]
    return dydt


initial = [Th232.initDensity, Th231.initDensity, Th233.initDensity,  Pa231.initDensity]
t = np.linspace(0, 100, 101)
flux = 1

sol = odeint(odefunc, initial, t, args=(flux,))


plt.plot(t, sol[:, 0], label=Th232.name)
plt.plot(t, sol[:, 1], label=Th231.name)
plt.plot(t, sol[:, 2], label=Th233.name)
plt.plot(t, sol[:, 3], label=Pa231.name)
plt.legend(loc='best')
plt.xlabel('t (hours)')
plt.grid()
plt.show()