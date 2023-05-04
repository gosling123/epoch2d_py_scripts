#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plot_accumulator_field_spectra.py

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


import plotters.accumulator_field_spectra as field_spectra
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
path = "../half_omega"

# Path for picture outputs to go
output_path = '../Plots_half_omega'

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
field_name = "Magnetic_Field_Bz"

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
plots = field_spectra.plots(files, acc_flag, field_name, output_path, \
                                    lambda_0, T_e_K, density_profile, n_0, L_n)


# ------------------------------------------------------------------------------
kx_vs_omega = False

if kx_vs_omega:
    # Minimum time/times to plot (units : s)
    t_min = 1.5 * pico
    # Maximum time/times to plot (units : s)
    t_max = 2.5 * pico
    # Minimum density point to start taking FFT from (units : n_cr)
    n_min = plasma.density_exponential(n_0, L_n, x=300*micron)
    # Maximum density point to start taking FFT from (units : n_cr)
    n_max = plasma.density_exponential(n_0, L_n, x=350*micron)
    # Wavenumber range to plot
    k_range = [0.25, 1.55]
    # Frequency range to plot (units : omega_0 (laser))
    omega_range = [0.4, 0.5]
    # Plot EPW dispersion Curve
    plot_disp_epw = True
    # Plot EPW dispersion Curve
    plot_disp_em = False
    # Plot SRS curve
    plot_srs = False
    # Angle to plot srs curve (angle of sacttred EM wave) (units : degrees)
    srs_angle = 140
    # Plot TPD curve
    plot_tpd = True
    # Angle to plot TPD curve (centred angle of two LW) (units : degrees)
    # For angle at maximum linear growth set to 'max_lin_growth'
    tpd_angle='max_lin_growth'
    

    # Plot for given value/values
    if np.isscalar(t_min) and np.isscalar(t_max):
        print('---------------------------------------------------------------')
        print(f'Plotting kx_vs_omega for {t_min / pico} - {t_max /pico} ps')
        plots.plot_kx_vs_omega(t_min, t_max, n_min, n_max, k_range, omega_range, \
                               plot_disp_epw, plot_disp_em, plot_srs, srs_angle, plot_tpd, tpd_angle)
    else:
        for min_, max_ in zip(t_min, t_max):
            print('---------------------------------------------------------------')
            print(f'Plotting kx_vs_omega for {min_ / pico} - {max_ /pico} ps')
            plots.plot_kx_vs_omega(min_, max_, n_min, n_max, k_range, omega_range, \
                                   plot_disp_epw, plot_disp_em, plot_srs, srs_angle, plot_tpd, tpd_angle)


# ------------------------------------------------------------------------------
x_vs_omega = False

if x_vs_omega:

    # Minimum time/times to plot (units : s)
    t_min = np.arange(0, 34, 1) * pico
    # Maximum time/times to plot (units : s)
    t_max =  t_min + 1.0 * pico
    # Minimum density point to start taking FFT from (units : n_cr)
    n_min = 0.06
    # Maximum density point to start taking FFT from (units : n_cr)
    n_max =  1.0
    # Frequency range to plot (units : omega_0 (laser))
    omega_range = [0.0, 1.6]
    # Plot SRS curve
    plot_srs = False
    # Angle to plot srs curve (angle of sacttred EM wave) (units : degrees)
    srs_angle = 160
    # Plot TPD curve
    plot_tpd = True
    # Angle to plot TPD curve (centred angle of two LW) (units : degrees)
    # For angle at maximum linear growth set to 'max_lin_growth'
    tpd_angle='max_lin_growth'

    # Plot for given value/values
    if np.isscalar(t_min) and np.isscalar(t_max):
        print('---------------------------------------------------------------')
        print(f'Plotting x_vs_omega for {t_min / pico} - {t_max /pico} ps')
        plots.plot_x_vs_omega(t_min, t_max, n_min, n_max, omega_range, \
                               plot_srs, srs_angle, plot_tpd, tpd_angle)
    else:
        for min_, max_ in zip(t_min, t_max):
            print('---------------------------------------------------------------')
            print(f'Plotting x_vs_omega for {min_ / pico} - {max_ /pico} ps')
            plots.plot_x_vs_omega(min_, max_, n_min, n_max, omega_range, \
                                  plot_srs, srs_angle, plot_tpd, tpd_angle)


# ------------------------------------------------------------------------------
omega_vs_y = False

if omega_vs_y:

    # Minimum time/times to plot (units : s)
    t_min = 30.0 * pico
    # Maximum time/times to plot (units : s)
    t_max = 34.3 * pico
    # Minimum y point to take FFT in (units : m)
    y_min = -150 * micron
    # Maximum y point to take FFT in (units : m)
    y_max =  1300 * micron
    # Frequency range to plot (units : omega_0 (laser))
    omega_range = [0.4, 0.6]
    # Plot SRS frequency bounds
    plot_srs = True
    # Plot TPD frequency bounds
    plot_tpd = True
    # Minimum density for bound (units : n_cr)
    n_min=0.2
    # Maximum density for bound (units : n_cr)
    n_max=0.249

    # Plot for given value/values
    if np.isscalar(t_min) and np.isscalar(t_max):
        print('---------------------------------------------------------------')
        print(f'Plotting omega_vs_y for {t_min / pico} - {t_max /pico} ps')
        plots.plot_omega_vs_y(t_min, t_max, y_min, y_max, omega_range,\
                              n_min, n_max, plot_srs, plot_tpd)
    else:
        for min_, max_ in zip(t_min, t_max):
            print('---------------------------------------------------------------')
            print(f'Plotting omega_vs_y for {min_ / pico} - {max_ /pico} ps')
            plots.plot_omega_vs_y(min_, max_, y_min, y_max, omega_range,\
                              n_min, n_max, plot_srs, plot_tpd)


# ------------------------------------------------------------------------------
omega_vs_time = True

if omega_vs_time:
    # Minimum time to start taking FFT from
    t_min = 20 * pico
    # Maximum time to start taking FFT from
    t_max = 35 * pico
    # Number of x or y slices to average over
    space_slices = 3
    # Number of time bins
    t_bins = 100
    # Size of moving FFT window
    t_window = 1000
    # Frequenecy range to plot (units of laser frequency)
    omega_range = [1.4, 1.6]
    # Plot SRS frequency bounds
    plot_srs = False
    # Plot TPD frequency bounds
    plot_tpd = False
    # Minimum density for bound (units : n_cr)
    n_srs=[0.21,0.25]
    # Maximum density for bound (units : n_cr)
    n_tpd=[0.21,0.249]

    # Plot for given value/values
    if np.isscalar(t_min) and np.isscalar(t_max):
        print('---------------------------------------------------------------')
        print(f'Plotting omega_vs_time for {t_min / pico} - {t_max /pico} ps')
        plots.plot_omega_vs_time(t_min, t_max, space_slices, t_bins, t_window, omega_range, \
                            n_srs, n_tpd, plot_srs, plot_tpd)
    else:
        for min_, max_ in zip(t_min, t_max):
            print('---------------------------------------------------------------')
            print(f'Plotting omega_vs_time for {min_ / pico} - {max_ /pico} ps')
            plots.plot_omega_vs_time(min_, max_, space_slices, t_bins, t_window, omega_range, \
                            n_srs, n_tpd, plot_srs, plot_tpd)


















