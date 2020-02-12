
# coding: utf-8

# # Fachkurs 2020, Thurley Gruppe

# In[1]:

#!/usr/bin/env python3

"""
Created on Tue Feb 12 17:30:07 2019

@author: philipp burt
script to simulate diffusion inside an immunological synapse
modified for fachkurs WS 2020 by Patrick Brunner
"""
#==============================================================================
# import modules
#==============================================================================
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import j0, j1, jn_zeros
sns.set(context = "talk", style = "ticks")

#import import_ipynb
import functions as functions


#==============================================================================
# parameters
#==============================================================================

# system parameters

# synapse contact area radius (micrometer)
a = 2.0
# synaptic distance (micrometer)
l = 0.02
# IL2/ IL2-R binding rate 1 / (nM h) ???
k_on = 111.6 * (10./6) * (1./3600)
# receptor expression levels (molecules / cell)
R_resp = 0
R = 100 / (np.pi * a**2)
# IL2 secretion rate (molecules / sec)
q = 10
# IL2 diffusion constant (micrometer**2 / sec)
D = 10
# iterations, increase to get more precise results but dont blow up your computer
n_max = 20


# plotting parameters

# density of grid (increase for nicer resolution, but will slow down computation)
res = 50

# define a saturation level (greater il2 concentrations than this will all have the same colour)
saturation_level = 2000

# synapse geometry (radius = 0 return division by zero)
radius_start = -a
radius_stop = a
zylinder_start = 0
zylinder_stop = l
r_arr = np.linspace(radius_start, radius_stop, res)
z_arr = np.linspace(zylinder_start, l, res)

# design grid
r_mesh, z_mesh = np.meshgrid(r_arr, z_arr)
intensity = np.zeros_like(r_mesh)


#==============================================================================
# computation
#==============================================================================

# for each element in grid, compute the IL2 concentration
for i, r in enumerate(r_arr):
    for j, z in enumerate(z_arr):
        il2_conc = functions.bessel_sum(r, z, n_max, a, l, R, R_resp, k_on, q, D)

        # convert back to il2 concentration from number of molecules per cubic micrometer
        # also convert to picomolar concentration
        
        il2_conc = 0.6 * 1e3 * il2_conc 
        if il2_conc > saturation_level:
            il2_conc = saturation_level
        intensity[j,i] = il2_conc

# remove 1 dimension because intensity should be bounded by the grid
intensity = intensity[:-1, :-1]
# solution is symmetrical
intensity = np.flip(intensity, 1) + intensity

#==============================================================================
# plotting
#==============================================================================

# type of colormap, if you want to, you can choose your own fancy colormap, just google matplotlib colormaps
cmap = "jet"

fig, ax = plt.subplots(figsize = (6,4))
heatmap = ax.pcolormesh(r_mesh, z_mesh, intensity, cmap = cmap, edgecolors = "none")
ax.set_xlabel(r"synapse radius ($\mu m$)")
ax.set_ylabel(r"cell-cell distance ($\mu m$)")

fig.colorbar(heatmap, label = 'IL2 (pM)')
plt.tight_layout()
plt.show()
