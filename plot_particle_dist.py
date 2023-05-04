#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plot_particle_dist.py

Script for plotting various distributions
of outgoing particles.

"""

import plotters.particle_distribution as distribution
import scipy.constants as const
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

# Plotting Params
plt.rcParams["figure.figsize"] = (10,8)
plt.rcParams["figure.autolayout"] = True
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15
plt.rcParams['legend.fontsize'] = 16

# Useful prefix's
pico = 1e-12
femto = 1e-15
micron = 1e-6

# conversion from keV to kelvin
keV_to_K = (const.e*1e3)/const.k

################################################################################
#  File read setup
################################################################################

# Location of data files
path = "../half_omega"

# Path for picture outputs to go
output_path = '../Plots'

# String before the number in the SDF files for accumulated field files
sdf_prefix = "probes"

# Flag for which probe to use
probe_flag = "y_upper_probe"

#outgoing_e_probe_ & conv_e_probe_ (for run0)

# Extract all filenames with given sdf prefix
fnames = f'{path}/{sdf_prefix}*.sdf'
# Sort in ascending order
files = np.array(sorted(glob.glob(fnames)))

################################################################################
# Define simulation setup
################################################################################

# Laser Wavelength (units : m)
lambda_0 = 1.314e-6

# Electron Temperature (units : keV)
T_e_keV = 4.3
# Eectron Temperature (units : K)
T_e_K = T_e_keV * keV_to_K

################################################################################
#  Plotting setup
################################################################################

# Make sure directory exists
try:
    os.mkdir(output_path)
    print(f'Created {output_path} directory')
except:
    print(f'{output_path} directory already exists')

# Initiate plotting class
plots = distribution.plots(files, probe_flag, output_path, lambda_0, T_e_K)

########################################################################################################################
# Momentum  
########################################################################################################################

p_distribution = False

if p_distribution:

    # Minimum time/times to plot (units : s)
    t_min = 0.0 * pico
    # Maximum time/times to plot (units : s)
    t_max = 9 * pico 
    # Momentum componant/s to use,  options 'x', 'y', 'z', 'xy', 'xz', 'yz' 'xyz'
    componants = 'xyz'
    # Minimum momentum to plot (units : p_th = m_e*v_th)
    p_min = 4.0 
    # Maximum momentum to plot (units : p_th = m_e*v_th)
    p_max = 25
    # Number of discrete bins to plot
    nbins = 1000
    # Plot Maxwellian Fits
    maxwell_plot = True

    print('---------------------------------------------------------------')
    print(f'Plotting outgoing momentum distribution for {t_min / pico} - {t_max /pico} ps')
    plots.plot_p_dist(t_min, t_max, componants, p_min, p_max, nbins, maxwell_plot)


########################################################################################################################
# Energy
########################################################################################################################

E_distribution = True

if E_distribution:

    # Minimum time/times to plot (units : s)
    t_min = 0.0 * pico
    # Maximum time/times to plot (units : s)
    t_max = 15.0 * pico 
    # Weight distribution function by E 
    weighted = True
    # Minimum energy to plot (units : keV)
    E_min = 30
    # Maximum momentum to plot (units : keV)
    E_max = 2500
    # Number of discrete bins to plot
    nbins = 1000
    # Plot Maxwellian Fits
    maxwell_plot = True

    print('---------------------------------------------------------------')
    print(f'Plotting outgoing energy distribution for {t_min / pico} - {t_max /pico} ps')
    plots.plot_E_dist(t_min, t_max, E_min, E_max, nbins, weighted, maxwell_plot)