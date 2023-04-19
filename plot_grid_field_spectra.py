#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plot_grid_field_spectra.py

Script for plotting various figures that require
grid field data.

i.e you can plot:

kx_vs_ky
x_vs_ky
x_vs_kx

for a chosen field

"""


import plotters.grid_field_spectra as field_spectra
import scipy.constants as const
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

# Plotting Params
plt.rcParams["figure.figsize"] = (10,8)
plt.rcParams["figure.autolayout"] = True
plt.rcParams['lines.linewidth'] = 3
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
path = "../half_omega_long_run"

# Path for picture outputs to go
output_path = '../Plots'

# String before the number in the SDF files for accumulated field files
sdf_prefix = "regular"

# Extract all filenames with given sdf prefix
fnames = f'{path}/{sdf_prefix}*.sdf'
# Sort in ascending order
files = np.array(sorted(glob.glob(fnames)))

# Particular field to take fft of. Read from sdf file directory with naming style
# "Electric_Field_E{x,y or z}", and similar for magnetic field.
field_name = "Electric_Field_Ex"

################################################################################
# Define simulation setup
################################################################################

# Laser Wavelength (units : m)
lambda_0 = 1.314e-6

# Electron Temperature (units : keV)
T_e_keV = 4.3
# Eectron Temperature (units : K)
T_e_K = T_e_keV * keV_to_K

# Density profile can be 'exponential' or 'linear'
density_profile = 'exponential'
# Density at x = 0 (units : n_cr)
n_0 = 0.03
# Density scale length (units : m)
L_n = 101 * lambda_0

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
plots = field_spectra.plots(files, field_name, output_path, \
                            lambda_0, T_e_K, density_profile, n_0, L_n)


# ------------------------------------------------------------------------------
kx_vs_ky = True

if kx_vs_ky:
    # Time of grid snapshot (units : s)
    snap_time = 2.0 * pico
    # Minimum density point to start taking FFT from (units : n_cr)
    n_min = 0.1
    # Maximum density point to start taking FFT from (units : n_cr)
    n_max =  0.2
    # Wavenumber (kx) range to plot
    kx_range = [-2, 2]
    # Wavenumber (ky) range to plot
    ky_range = [-2, 2]
    # Plot SRS curves
    plot_srs = True
    # Densities to plot SRS curves for
    n_srs=[0.1, 0.18]
    # Plot TPD curves
    plot_tpd=True
    # Densities to plot TPD curves for
    n_tpd=[0.2,0.23]

  
    # Plot for given value/values
    if np.isscalar(snap_time):
        print('---------------------------------------------------------------')
        print(f'Plotting kx_vs_ky for {snap_time / pico} ps')
        plots.plot_kx_vs_ky(snap_time, n_min, n_max, kx_range, ky_range,\
                            plot_srs, n_srs, plot_tpd, n_tpd)
    else:
        for time in snap_time:
            print('---------------------------------------------------------------')
            print(f'Plotting kx_vs_ky for {time/ pico} ps')
            plots.plot_kx_vs_ky(time, n_min, n_max, kx_range, ky_range,\
                                plot_srs, n_srs, plot_tpd, n_tpd)


# ------------------------------------------------------------------------------
x_vs_ky = False

if x_vs_ky:
    # Time of grid snapshot (units : s)
    snap_time = 2.0 * pico
    # Minimum density point to start taking FFT from (units : n_cr)
    n_min = 0.1
    # Maximum density point to start taking FFT from (units : n_cr)
    n_max =  0.2
    # Wavenumber (ky) range to plot
    k_range = [-2, 2]

  
    # Plot for given value/values
    if np.isscalar(snap_time):
        print('---------------------------------------------------------------')
        print(f'Plotting x_vs_ky for {snap_time / pico} ps')
        plots.plot_x_vs_ky(snap_time, n_min, n_max, k_range)
    else:
        for time in snap_time:
            print('---------------------------------------------------------------')
            print(f'Plotting x_vs_ky for {time/ pico} ps')
            plots.plot_x_vs_ky(time, n_min, n_max, k_range)


# ------------------------------------------------------------------------------
x_vs_kx = False

if x_vs_kx:
    # Time of grid snapshot (units : s)
    snap_time = 2.0 * pico
    # Minimum density point to start taking FFT from (units : n_cr)
    n_min = 0.1
    # Maximum density point to start taking FFT from (units : n_cr)
    n_max =  0.25
    # Window size for STFT
    x_window = 1000
    # Number of discrete spatial bins for STFT
    x_bins = 500
    # Wavenumber (ky) range to plot
    k_range = [-2, 2]


  
    # Plot for given value/values
    if np.isscalar(snap_time):
        print('---------------------------------------------------------------')
        print(f'Plotting x_vs_kx for {snap_time / pico} ps')
        plots.plot_x_vs_kx(snap_time, n_min, n_max, x_window, x_bins, k_range)
    else:
        for time in snap_time:
            print('---------------------------------------------------------------')
            print(f'Plotting x_vs_kx for {time/ pico} ps')
            plots.plot_x_vs_kx(time, n_min, n_max, x_window, x_bins, k_range)