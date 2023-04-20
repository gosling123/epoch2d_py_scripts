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
import calculations.plasma_calculator as plasma
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
path = "../../shared/run0"

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
lambda_0 = 0.351e-6

# Electron Temperature (units : keV)
T_e_keV = 4.5
# Eectron Temperature (units : K)
T_e_K = T_e_keV * keV_to_K

# Density profile can be 'exponential' or 'linear'
density_profile = 'exponential'
# Density at x = 0 (units : n_cr)
n_0 = 0.1
# Density scale length (units : m)
L_n = 600 * micron

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
    snap_time = 2.5 * pico
    # Minimum density point to start taking FFT from (units : n_cr)
    n_min = plasma.density_exponential(n_0, L_n, x=350*micron)
    # Maximum density point to start taking FFT from (units : n_cr)
    n_max =  plasma.density_exponential(n_0, L_n, x=400*micron)
    # Wavenumber (kx) range to plot
    kx_range = [-2, 2]
    # Wavenumber (ky) range to plot
    ky_range = [-2, 2]
    # Plot SRS curves
    plot_srs = False
    # Densities to plot SRS curves for
    n_srs=[n_min, n_max]
    # Angle range to plot SRS polar curve for
    srs_angles = [0, 30]
    # Plot TPD curves
    plot_tpd = True
    # Densities to plot TPD curves for
    n_tpd = [0.2,0.24]
    # Angle range to plot TPD polar curve for
    tpd_angles = [0, 360]

  
    # Plot for given value/values
    if np.isscalar(snap_time):
        print('---------------------------------------------------------------')
        print(f'Plotting kx_vs_ky for {snap_time / pico} ps')
        plots.plot_kx_vs_ky(snap_time, n_min, n_max, kx_range, ky_range,\
                            plot_srs, n_srs, srs_angles, plot_tpd, n_tpd, tpd_angles)
    else:
        for time in snap_time:
            print('---------------------------------------------------------------')
            print(f'Plotting kx_vs_ky for {time/ pico} ps')
            plots.plot_kx_vs_ky(time, n_min, n_max, kx_range, ky_range,\
                                plot_srs, n_srs, srs_angles, plot_tpd, n_tpd, tpd_angles)


# ------------------------------------------------------------------------------
x_vs_ky = False

if x_vs_ky:
    # Time of grid snapshot (units : s)
    snap_time = 2.5 * pico
    # Minimum density point to start taking FFT from (units : n_cr)
    n_min = 0.1
    # Maximum density point to start taking FFT from (units : n_cr)
    n_max =  0.25
    # Wavenumber (ky) range to plot
    k_range = [-2, 2]
    # Plot SRS curve
    plot_srs = True 
    # Angle to plot srs curve (angle of sacttred EM wave) (units : degrees)
    srs_angle = 160
    # Plot TPD curve
    plot_tpd = True
    # Angle to plot TPD curve (centred angle of two LW) (units : degrees)
    # For angle at maximum linear growth set to 'max_lin_growth'
    tpd_angle = 45

  
    # Plot for given value/values
    if np.isscalar(snap_time):
        print('---------------------------------------------------------------')
        print(f'Plotting x_vs_ky for {snap_time / pico} ps')
        plots.plot_x_vs_ky(snap_time, n_min, n_max, k_range,\
                           plot_srs, srs_angle, plot_tpd, tpd_angle)
    else:
        for time in snap_time:
            print('---------------------------------------------------------------')
            print(f'Plotting x_vs_ky for {time/ pico} ps')
            plots.plot_x_vs_ky(time, n_min, n_max, k_range,\
                               plot_srs, srs_angle, plot_tpd, tpd_angle)


# ------------------------------------------------------------------------------
x_vs_kx = True

if x_vs_kx:
    # Time of grid snapshot (units : s)
    snap_time = 2.5 * pico
    # Minimum density point to start taking FFT from (units : n_cr)
    n_min = 0.1
    # Maximum density point to start taking FFT from (units : n_cr)
    n_max =  0.25
    # Window size for STFT
    x_window = 1000
    # Number of discrete spatial bins for STFT
    x_bins = 100
    # Wavenumber (ky) range to plot
    k_range = [-2, 2]
    # Plot SRS curve
    plot_srs = True 
    # Angle to plot srs curve (angle of sacttred EM wave) (units : degrees)
    srs_angle = 160
    # Plot TPD curve
    plot_tpd = True
    # Angle to plot TPD curve (centred angle of two LW) (units : degrees)
    # For angle at maximum linear growth set to 'max_lin_growth'
    tpd_angle='max_lin_growth'

  
    # Plot for given value/values
    if np.isscalar(snap_time):
        print('---------------------------------------------------------------')
        print(f'Plotting x_vs_kx for {snap_time / pico} ps')
        plots.plot_x_vs_kx(snap_time, n_min, n_max, x_window, x_bins, k_range,\
                           plot_srs, srs_angle, plot_tpd, tpd_angle)
    else:
        for time in snap_time:
            print('---------------------------------------------------------------')
            print(f'Plotting x_vs_kx for {time/ pico} ps')
            plots.plot_x_vs_kx(time, n_min, n_max, x_window, x_bins, k_range,\
                               plot_srs, srs_angle, plot_tpd, tpd_angle)