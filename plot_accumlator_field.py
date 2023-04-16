#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plot_accumulator_field.py

Script for plotting various figures that require accumulated field
data.

i.e you can plot:

kx_vs_omega
ky_vs_omega
omega_vs_y
x_vs_omega
omega_vs_time

for a chosen accumulator and field

"""


import Calculations.plasma_calculator as plasma
from Plotters.accumulator_field_fft import accumulator_field_fft_plots
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
path = "../half_omega_long_run"

# Path for picture outputs to go
output_path = '../Plots'

# String before the number in the SDF files for accumulated field files
sdf_prefix = "acc_field"

# Extract all filenames with given sdf prefix
fnames = f'{path}/{sdf_prefix}*.sdf'
# Sort in ascending order
files = np.array(sorted(glob.glob(fnames)))

# Flag for which accumulator strip to use
acc_flag = "y_upper"

# Particular field to take fft of. Read from sdf file directory with naming style
# "Electric_Field_E{x,y or z}", and similar for magnetic field.
field_name = "Electric_Field_Ex"

################################################################################
# Define simulation setup
################################################################################

# Laser Wavelength
lambda_0 = 1.314e-6

# Electron Temperature (keV)
T_e_keV = 4.3
T_e_K = T_e_keV * keV_to_K

# Density profile can be 'exponential' or 'linear'
density_profile = 'exponential'
n_0 = 0.03
L_n = 101 * lambda_0

################################################################################
#  Plotting setup
################################################################################

# Make sure directory exists
try:
    os.mkdir(output_path)
except:
    print('', end='\n')

# Initiate plotting class
plots = accumulator_field_fft_plots(files, acc_flag, field_name, output_path, \
                                    lambda_0, T_e_K, density_profile, n_0, L_n)


# ------------------------------------------------------------------------------
kx_vs_omega = True

if acc_flag[0] == 'x':
    print("ERROR: Plot is for kx-omega spectrum so y diagnostic strip required, seeting plot boolean to False")
    kx_vs_omega = False

if kx_vs_omega:

    # Minimum time/times to plot
    t_min = 8.0 * pico
    # Maximum time/times to plot
    t_max = 12.0 * pico
    # Minimum density point to start taking FFT from
    n_min = 0.15
    # Maximum density point to start taking FFT from
    n_max =  0.25
    # Wavenumber range to plot
    k_range = [-2, 2]
    # Frequency range to plot
    omega_range = [0.0, 1.2]

    # Plot for given value/values
    if np.isscalar(t_min) and np.isscalar(t_max):
        print('---------------------------------------------------------------')
        print(f'Plotting kx_vs_omega for {t_min / pico} - {t_max /pico} ps')
        plots.plot_kx_vs_omega(t_min, t_max, n_min, n_max, k_range, omega_range)
    else:
        for min_, max_ in zip(t_min, t_max):
            print('---------------------------------------------------------------')
            print(f'Plotting kx_vs_omega for {min_ / pico} - {max_ /pico} ps')
            plots.plot_kx_vs_omega(min_, max_, n_min, n_max, k_range, omega_range)


# ------------------------------------------------------------------------------
# ky_vs_omega = True

# if acc_flag[0] == 'y' or acc_flag == 'strip_y0':
#     print("ERROR: Plot is for ky-omega spectrum so x diagnostic strip required, seeting plot boolean to False")
#     ky_vs_omega = False

# if ky_vs_omega:


