#%% INITIALIZE

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy


class Nuclide:
    def __init__(self, name, protons, mass, bmDecay = 0, captureCS = 0, fissionCS = 0, ntwonCS = 0, initDensity = 0, extraction = 0):
        self.name = name
        self.protons = protons
        self.mass = mass
        self.initDensity = initDensity
        self.bmDecay = bmDecay
        self.extraction = extraction
        self.captureCS = captureCS
        self.fissionCS = fissionCS
        self.ntwonCS = ntwonCS

# TIME & FLUX
days = 1000
stepspersec = 0.001

_secinday = 86400
timesteps = int(_secinday*days*stepspersec)+1

dtime = days*_secinday/timesteps
_time = np.linspace(0, _secinday*days, timesteps)
_timeinhours = _time/3600
_timeindays = _timeinhours/24

#     per cm^2 per sec     cm^2 to barns
flux =  10**14              * 10**-24 *50

#%% NUCLIDES
# bmDecay: lambda = ln(2)/half-life in 1/(days*secinday) 
# captureCS: Barns 1 b = 10^{-24} cm^2
# initDensity: Atoms per (barn cm)
_Th231 = Nuclide('Th231', 90, 231, bmDecay = np.log(2)/(1.063*_secinday))
_Th232 = Nuclide('Th232', 90, 232,  captureCS = 0.396, fissionCS = 0.011, ntwonCS = 2.234*10**(-3), initDensity = 0.0056)
_Th233 = Nuclide('Th233', 90, 233, bmDecay = np.log(2)/(0.016*_secinday))

_Pa231 = Nuclide('Pa231', 91, 231,  captureCS = 2.980, fissionCS = 0.258, ntwonCS=1.479*10**(-3))
_Pa232 = Nuclide('Pa232', 91, 232, bmDecay = np.log(2)/(1.312*_secinday))
_Pa233 = Nuclide('Pa233', 91, 233, bmDecay = np.log(2)/(26.975*_secinday),  captureCS = 1.056, fissionCS = 0, ntwonCS=0.612*10**(-3))
_Pa234 = Nuclide('Pa234', 91, 234, bmDecay = np.log(2)/(0.279*_secinday))
_Pa235 = Nuclide('Pa235', 91, 235, bmDecay = np.log(2)/(0.017*_secinday))

_U232  = Nuclide('U232', 92, 232,  captureCS = 0.662, fissionCS = 2.264, ntwonCS=1.117*10**(-3))
_U233  = Nuclide('U233', 92, 233,  captureCS = 0.261, fissionCS = 2.774, ntwonCS=1.137*10**(-3), initDensity = 0.0007)
_U234  = Nuclide('U234', 92, 234,  captureCS = 0.602, fissionCS = 0.349, ntwonCS=0.222*10**(-3))
_U235  = Nuclide('U235', 92, 235,  captureCS = 0.535, fissionCS = 1.906, ntwonCS=1.490*10**(-3))

# Williams
_U237  = Nuclide('U237',  92, 237, captureCS = 35.225, fissionCS = 0.609, ntwonCS=7.902*10**(-3), bmDecay = np.log(2)/(6.751*_secinday))
_U238  = Nuclide('U238',  92, 238, captureCS = 5.254, fissionCS = 0.101, ntwonCS=4.058*10**(-3))
_U239  = Nuclide('U239x',  92, 239, bmDecay = np.log(2)/(0.016*_secinday))
_U240  = Nuclide('U240x',  92, 240, bmDecay = np.log(2)/(0.588*_secinday))

_Ne237 = Nuclide('Ne237', 93, 237, captureCS = 34.396, fissionCS = 0.503, ntwonCS=0.816*10**(-3))
_Ne238 = Nuclide('Ne238x', 93, 238, captureCS = 14.282, fissionCS = 140.580, ntwonCS=4.987*10**(-3),  bmDecay = np.log(2)/(2.1*_secinday))
_Ne239 = Nuclide('Ne239', 93, 239, captureCS = 13.971, fissionCS = 0.593, ntwonCS=1.230*10**(-3),  bmDecay = np.log(2)/(2.356*_secinday))
_Ne240 = Nuclide('Ne240', 93, 240,  bmDecay = np.log(2)/(0.04*_secinday))

_Pu238 = Nuclide('Pu238', 94, 238, captureCS = 29.029, fissionCS = 2.383, ntwonCS=0.271*10**(-3))
_Pu239 = Nuclide('Pu239', 94, 239, captureCS = 61.704, fissionCS = 107.540, ntwonCS=1.094*10**(-3))
_Pu240 = Nuclide('Pu240', 94, 240, captureCS = 220.980, fissionCS = 0.630, ntwonCS=1.326*10**(-3))

# Poisons
_Xe135 = Nuclide('Xe135', 54, 135, bmDecay = np.log(2)/(0.380*_secinday), captureCS = 10**6)
_Xe136 = Nuclide('Xe136', 54, 136)
_Te135 = Nuclide('Te135', 52, 135, bmDecay = np.log(2)/(0.01*_secinday))
_I135 = Nuclide('I135', 53, 135, bmDecay = np.log(2)/(0.274*_secinday))
_Cs135 = Nuclide('Cs135', 55, 135)

FissionData = pd.DataFrame({'Th232': [1.39, 6.53, 24.5, 22.8, 0],
                    'Pa231': [6.57, 21.6, 19.3, 40.2, 0],
                    'U232': [17.3, 45.6, 6.79, 28.3, 0],
                    'U233': [18.2, 28.5, 5.20, 26.5, 1.63],
                    'U234': [18.4, 34.4, 5.84, 29.0, 1.45],
                    'U235': [14.6, 21.8, 10.5, 31.5, 0]
                    }, index=['Xe135','Xe136','Te135', 'I135', 'Cs135'])

Nuclides = [_Th231,_Th232,_Th233,_Pa231,_Pa232,_Pa233,_Pa234,_Pa235,_U232,_U233,_U234,_U235]
#Nuclides = [_Th231,_Th232,_Th233,_Pa231,_Pa232,_Pa233,_Pa234,_Pa235,_U232,_U233,_U234,_U235,_Xe135,_Xe136,_Te135,_I135,_Cs135]
StartNuclides = [_Th232, _U233]
NostartNuclides = [_Th231,_Th233,_Pa231,_Pa232,_Pa233,_Pa234,_Pa235,_U232,_U234,_U235]
ThNuclides = [_Th231,_Th232,_Th233]
PaNuclides = [_Pa231,_Pa232,_Pa233,_Pa234,_Pa235]
UNuclides = [_U232,_U233,_U234,_U235]
PoisNuclides = [ _Xe135,_Xe136,_Te135,_I135,_Cs135]
NopoisNuclides = [_Th231,_Th232,_Th233,_Pa231,_Pa232,_Pa233,_Pa234,_Pa235,_U232,_U233,_U234,_U235,]

WillNuclides = [_U237, _U238, _U239, _U240, _Ne237, _Ne238, _Ne239,_Ne240, _Pu238, _Pu239, _Pu240]
WillplotNuclides = [_U237, _U238, _U239, _U240]
#Nuclides = WillNuclides

N0 = np.zeros(len(Nuclides))  # Initial density
_Identity = np.identity(len(Nuclides))
for _NeuNum in range(len(Nuclides)):
    N0[_NeuNum] = Nuclides[_NeuNum].initDensity
  
    
#%% MAKE MATRIX  
def Matrix(Nuclides, fissionbool = True, productbool = True):
    Arr = np.zeros((len(Nuclides),len(Nuclides)))
    FissionArr = np.zeros((len(Nuclides),len(Nuclides)))
    for _NeuNum in range(len(Nuclides)):  
        
        #Beta Minus Decay
        if(Nuclides[_NeuNum].bmDecay!=0):
            Arr[_NeuNum,_NeuNum] += -Nuclides[_NeuNum].bmDecay  # Subtract from own Neucleus
            
            for _neunum in range(len(Nuclides)): # Add to other Neucleus
                if(Nuclides[_neunum].mass == Nuclides[_NeuNum].mass and Nuclides[_neunum].protons ==  Nuclides[_NeuNum].protons + 1):                
                    Arr[_neunum,_NeuNum] += Nuclides[_NeuNum].bmDecay
            
        # Neutron Capture
        if(Nuclides[_NeuNum].captureCS!=0):
            Arr[_NeuNum,_NeuNum] += -Nuclides[_NeuNum].captureCS * flux  # Subtract from own Neucleus
            
            for _neunum in range(len(Nuclides)): # Add to other Neucleus
                if(Nuclides[_neunum].mass == Nuclides[_NeuNum].mass +1 and Nuclides[_neunum].protons ==  Nuclides[_NeuNum].protons):     
                    Arr[_neunum,_NeuNum] += Nuclides[_NeuNum].captureCS * flux
        
        # (n,2n) Reaction
        if(Nuclides[_NeuNum].ntwonCS!=0):
            Arr[_NeuNum,_NeuNum] += -Nuclides[_NeuNum].ntwonCS * flux  # Subtract from own Neucleus
            
            for _neunum in range(len(Nuclides)): # Add to other Neucleus
                if(Nuclides[_neunum].mass == Nuclides[_NeuNum].mass - 1 and Nuclides[_neunum].protons ==  Nuclides[_NeuNum].protons):     
                    Arr[_neunum,_NeuNum] += Nuclides[_NeuNum].ntwonCS * flux
                    
        # Fission
        if(Nuclides[_NeuNum].fissionCS!=0):        
            Arr[_NeuNum,_NeuNum] += -Nuclides[_NeuNum].fissionCS * flux  # Subtract from own Neucleus
            
            if(fissionbool):    
                FissionArr[_NeuNum,_NeuNum] += Nuclides[_NeuNum].fissionCS * flux  # Add to Fission Counter matrix
            if(productbool):
                _FissionVec = FissionData[Nuclides[_NeuNum].name]                
                for _neunum in range(len(Nuclides)):
                    if(Nuclides[_neunum].name in _FissionVec.index):
                        Arr[_neunum,_NeuNum] += _FissionVec[Nuclides[_neunum].name] * 10**-3 * Nuclides[_NeuNum].fissionCS * flux 
             
        # Extraction
        if(Nuclides[_NeuNum].extraction!=0):
            Arr[_NeuNum,_NeuNum] += -Nuclides[_NeuNum].extraction  # Subtract from own Neucleus
    
    if(fissionbool):
        return Arr, FissionArr
    else: return Arr

Arr,FissionArr = Matrix(Nuclides)
#Arr,FissionArr = Matrix(Nuclides, productbool = False)

#%% FORWARD CALCULATIONS UNPERTURBED

# Calculates Density from initial density (and matrix, nuclides, times)
def DensityCalc(initDensity, Array):    

    Ntemp = np.zeros((len(Nuclides),timesteps+1))
    Ntemp[:,0] = initDensity # Initial density
    dArrayTot = _Identity+Array*dtime
    
    for _t in range(len(_time)):     
        Ntemp[:,_t+1] = dArrayTot.dot(Ntemp[:,_t])
        
    return Ntemp # N array of densities over time

N = DensityCalc(N0, Arr)    


# Calculates power per time step
def PowerCalc(FissionsPerBarnCM): # volume in /barn centimeter    
    jouleperfission = 200 * 10**6 * 1.602 * 10**(-19) # 200 MeV to J
    volume = 9 * 10**28 *100 # from BarnCm to 9 m^3
    return FissionsPerBarnCM * jouleperfission * volume *10**(-9) /dtime

FissionsPerBarnCMPerDTime = np.zeros(timesteps)
Power = np.zeros(timesteps)
for _t in range(len(_time)):     
    FissionsPerBarnCMPerDTime[_t] = sum(FissionArr.dot(N[:,_t])) * dtime
    Power[_t] = PowerCalc(FissionsPerBarnCMPerDTime[_t])


#%% ADJOINT FUNCTION CALCULATIONS
# N0, A already defined

# vector function h
h = np.zeros(len(Nuclides))
#h[1] = 1

# WILLIAMS Pu239, Pu240
#h[9] = h[10]= 1

# U233, U235
# h[9] = h[11] =1

# U232, U233, U235
# h[8] = h[9] = h[11] =1

# Response R
#R = np.dot(N[:,-1],h) # Voorbeeld Response with deltafunction at t_f

# RADIOTOXICITY
h[1] = 1.08 * 10**(7)       # Th-232
h[3] = 6.28 * 10**(12)      # Pa-231
h[5] = 6.28 * 10**(10)      # Pa-233
h[8] = 1.21 * 10**(15)      # U-232
h[9] = 6.28 * 10**(10)      # U-233
h[10] = 3.95 * 10**(10)     # U-234
h[11] = 1.31 * 10**(7)      # U-235
    
unpR = sum(h*N[:,-1])

# Calculates Adjoint Importance Function
def NStarCalc(htemp):    
    NStartemp = np.zeros((len(Nuclides),timesteps+1))    
    NStartemp[:,-1] = htemp
    ArrStarTot = _Identity + np.transpose(Arr) * dtime
    
    for _t in reversed(range(len(_time))):  
        NStartemp[:,_t] = ArrStarTot.dot(NStartemp[:,_t+1])
        
    return NStartemp


NStar = NStarCalc(h)

#%%  RESPONSE INITIAL DENSITY PERTURBATION

#Linspace perturbations
PertSize = np.linspace(0,2,10)
dRvec = np.zeros(len(PertSize))
realdRvec = np.zeros(len(PertSize))

N0pert = deepcopy(N0)

for pert in range(len(PertSize)):      

    # Introduce Perturbations
    N0pert[1] = N0[1]*PertSize[pert] 
    # N0pert[2] = PertSize[pert]
    # N0pert[5] = PertSize[pert]
    
    # Perturbation response calculations
    dRvec[pert] = NStar[:,0].dot(N0pert-N0) 
    
    # Real response calculations
    NPert = DensityCalc(N0pert, Arr)
    realdRvec[pert] = sum((NPert[:,-1] - N[:,-1])*h)

plt.plot(PertSize*0.0056, realdRvec/unpR, label = "Forward Solution", linewidth = 3)
plt.plot(PertSize*0.0056, dRvec/unpR, label = "Perturbation Approximation", linestyle =  ':',linewidth = 4)

plt.legend() # loc = 'lower center'
plt.xlabel("Initial Density Th232 Atoms  1/(barn cm)")
plt.ylabel("Relative Response Change  (\u0394 R/R\u2080)")
plt.xlim(left = 0) # , right = 1000
#plt.ylim(bottom = 0) #, top = 0.05780

#%%  RESPONSE MATRIX PERTURBATION

# PertSize = np.linspace(0,3,30) # FLUX, 
PertSize = np.linspace(1,2000,25) # EXTRACTION
# PertSize = np.linspace(0.8,1.2,30) # CROSS SECTION, 

# save original values
ogflux = flux 
ogcapture = np.zeros(len(Nuclides))
ogfission = np.zeros(len(Nuclides))
for _NeuNum in range(len(Nuclides)):
    ogcapture[_NeuNum] = Nuclides[_NeuNum].captureCS
    ogfission[_NeuNum] = Nuclides[_NeuNum].fissionCS

dRvec = np.zeros(len(PertSize))
realdRvec = np.zeros(len(PertSize))
for pert in range(len(PertSize)):       
    
    # INTRODUCE PERTURBATIONS
    # FLUX
    # flux = ogflux * PertSize[pert] 
    
    # EXTRACTION
    # _Th232.extraction = np.log(2)/( PertSize[pert]*_secinday)# PertSize[pert] 
    for _NeuNum in range(len(Nuclides)):
        Nuclides[_NeuNum].extraction = np.log(2)/( PertSize[pert]*_secinday)
    # _Pa231.extraction = np.log(2)/( PertSize[pert]*_secinday)# PertSize[pert] 
    
    # CROSS SECTION
    # for _NeuNum in range(len(Nuclides)):
    #     Nuclides[_NeuNum].captureCS = ogcapture[_NeuNum] * PertSize[pert] 
    #     Nuclides[_NeuNum].fissionCS = ogfission[_NeuNum] * PertSize[pert] 
    
    # Create perturbed array     
    ArrPert = Matrix(Nuclides, False, False)
    
    # Perturbation response calculations
    dArr = ArrPert - Arr    
    dAN = dArr.dot(N)    
    dRvec[pert] = sum(sum(NStar*dAN))*dtime
    
    # Real response calculations
    NPert = DensityCalc(N0, ArrPert)
    realdRvec[pert] = sum((NPert[:,-1] - N[:,-1])*h)

# Reset original values
flux = ogflux
for _NeuNum in range(len(Nuclides)):
    Nuclides[_NeuNum].extraction = 0
    Nuclides[_NeuNum].captureCS = ogcapture[_NeuNum]
    Nuclides[_NeuNum].fissionCS = ogfission[_NeuNum]
 
fig, ax = plt.subplots()
plt.plot(PertSize, realdRvec/unpR, label = "Forward Solution")
plt.plot(PertSize, dRvec/unpR, label = "Perturbation Approximation")
plt.legend() # loc = 'lower center'
plt.ylabel("Relative Response Change  (\u0394 R/R\u2080)")

# FLUX
# plt.xlabel("Flux Relative to Unperturbed Flux")
# plt.xlim(left = 0, right = 3) # , right = 1000

# EXTRACTION
plt.xlabel("Extraction 'half-life' of all Actinides  (days)")
plt.ylim(bottom = -10, top = 1) # , top = 0.05780
plt.xlim(left = 0,  right = 2000) # , right = 1000
ax.yaxis.set_ticks_position('both')

# CAPTURE
# plt.xlabel("Cross Section values relative to Unperturbed Cross Sections")
# plt.xlim(left = 0.8, right = 1.2) # , right = 1000

#%% ################################### PLOTTING

# PLOT PERTURBATIONS

plt.plot(PertSize, dRvec, label = "Perturbation")
plt.plot(PertSize, realdRvec, label = "Forward")

plt.legend() # loc = 'lower center'
plt.xlabel("x label")
plt.ylabel("y label")


#%%  PLOT DENSITY
fig, ax = plt.subplots()
ax.set_prop_cycle(color=["#8b0000", "#008000","#000080","#ff0000","#ffd700","#ff1493","#911eb4","#000000","#696969","#87cefa","#1e90ff", "#7fff00"])

for _nucli in PoisNuclides:
    plt.plot(_timeindays,N[Nuclides.index(_nucli),:-1], label = _nucli.name)

#plt.title("Densities of Actinides in Thorium Cycle")
plt.legend(bbox_to_anchor=(1.0, 0.95)) # loc = 'lower center'
plt.xlabel("Time (days)")
plt.ylabel("Atoms 1/(barn cm)")
plt.xlim(left = 0, right = 1000)
#plt.ylim(bottom = 10**-10, top = 10**-1)
plt.yscale("log")



#%% PLOT Power
plt.plot(_timeindays,Power)
plt.xlabel("Time (days)")
plt.ylabel("Power (GW)")
plt.xlim(left = 0, right = 1000)
#plt.ylim(bottom = 0.05770, top = 0.05780)

#%% PLOT IMPORTANCE
fig, ax = plt.subplots()
ax.set_prop_cycle(color=["#8b0000", "#008000","#000080","#ff0000","#ff1493","#911eb4","#87cefa","#ffd700","#696969","#000000","#1e90ff", "#7fff00"])

for _nucli in Nuclides:
    plt.plot(_timeindays,NStar[Nuclides.index(_nucli),:-1], label = _nucli.name) 

#for _nucli in WillplotNuclides:
#    plt.plot(_timeindays,NStar[Nuclides.index(_nucli),:-1], label = _nucli.name) 

# for _nucli in ThNuclides:
#     plt.plot(_timeindays,NStar[Nuclides.index(_nucli),:-1], label = _nucli.name) 
# for _nucli in PaNuclides:
#     plt.plot(_timeindays,NStar[Nuclides.index(_nucli),:-1], label = _nucli.name)
    
#plt.plot(_timeindays,NStar[Nuclides.index(_Pa235),:-1], label = "Pa235")
# plt.plot(_timeindays,NStar[Nuclides.index(_Pa232),:-1], label = "Pa232,\n Pa235")
# plt.plot(_timeindays,NStar[Nuclides.index(_Th233),:-1], label = "Th233,\n Pa233")
# plt.plot(_timeindays,NStar[Nuclides.index(_Th231),:-1], label = "Th231,\n Pa231")
# plt.plot(_timeindays,NStar[Nuclides.index(_Pa234),:-1], label = "Pa234")
# plt.plot(_timeindays,NStar[Nuclides.index(_Th232),:-1], label = "Th232")

plt.legend(bbox_to_anchor=(1.0, 0.8)) # loc = 'lower center'
plt.yscale("log")
plt.xlabel("Time (days)")
plt.ylabel("N*(t)")
#plt.ylim(top = 1.5)# bottom = 10**(-3), 
plt.xlim(left = 0)


