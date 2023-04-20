#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

srs_calculator.py

Script housing all the functions used to calculate quantities relating
to the SRS laser-plasma interaction.

"""


# import libraries
import numpy as np
import warnings
import scipy.constants as const
from calculations import plasma_calculator as plasma
from calculations import laser_calculator as laser

################################################################################
# Wavenumber matching conditions - EM daughter
################################################################################

def srs_wns_EM(n_e, v_th, lambda_0, angle):

    """
    Returns the allowd SRS wavenumbers from the matching conditions,
    for a given density and temperature and angle, (EM daughter wave). 
    Output wavenumbers are normalised by k_0 (vacuum wavenumber).

    n_e = Electron number density (units : n_cr)
    v_th = Electron thermal speed (units : ms^-1)
    lambda_0 = Vacuum laser wavelength (units : m)
    angle = Angle between k_0 and k_s (units : degrees)
    
    """

    if n_e > 0.25:
        return float('NaN'),float('NaN')

    # convert from degrees to radians
    theta = angle * np.pi/180.0
    
    # Vacuum wavenumber m^-1
    k_0 = laser.wavenumber(lambda_0)
    # Frequency rads^-1
    omega_0 = laser.omega(lambda_0)
    # Laser wavenumber 
    k_L = laser.wavenumber_plasma(lambda_0, n_e)
    
    # Polynomial coefficients for |k_s|
    coeffs = np.zeros(5)
    # Zeroth power coefficient
    coeffs[4] = (omega_0**4/const.c**4)*(1.0-4.0*n_e) + (3.0*v_th**2/const.c**4)*k_L**2*(3.0*v_th**2 * k_L**2 - 2.0*omega_0**2)
    # First power coefficient
    coeffs[3] = (12.0*v_th**2/const.c**4)*k_L*np.cos(theta) * (omega_0**2 - 3.0*v_th**2*k_L**2)
    # Second power coefficient
    coeffs[2] = (6.0*v_th**2/const.c**2)*k_L**2 * ((6.0*v_th**2/const.c**2)*(np.cos(theta))**2 - (1.0 - 3.0*v_th**2/const.c**2)) - (2.0*omega_0**2/const.c**2)*(1.0 + 3.0*v_th**2/const.c**2)
    # Third power coefficient
    coeffs[1] = 12.0*v_th**2/const.c**2 * (1.0 - 3.0*v_th**2/const.c**2)*k_L*np.cos(theta)
    # Fourth power coefficient
    coeffs[0] = (1.0 - 3.0*v_th**2/const.c**2)**2

    # Solve quartic
    ks_mag = np.roots(coeffs) / k_0

    # Only want soloutions that are physical (i.e >0)
    # and scattered wave cannot have greater frequenecy than laser (i.e <1)
    
    ks_mag = ks_mag[np.where(np.logical_and(ks_mag >= 0.0, ks_mag <= k_L/k_0))]
  
    if len(ks_mag) == 1:
        k_x = ks_mag * np.cos(theta)
        k_y = ks_mag * np.sin(theta)
        return k_x, k_y
    elif len(ks_mag) > 1:
        # Should only have one unique soloution
        warnings.warn("Multi-valued solution")
        return float('NaN'),float('NaN')
    else:
        warnings.warn("No solution")
        return float('NaN'),float('NaN')

def srs_EM_k_y(v_th, n_e, angle, lambda_0):

    """
    Returns the allowd SRS wavenumbers (k_y only) from the matching conditions,
    for a given density and temperature and angle, (EM daughter wave). 
    Output wavenumbers are normalised by k_0 (vacuum wavenumber).

    v_th = Electron thermal speed (units : ms^-1)
    n_e = Electron number density (units : n_cr)
    angle = Angle between k_0 and k_s (units : degrees)
    lambda_0 = Vacuum laser wavelength (units : m)
    """
    k_y_vals = np.array([])

    for i in range(len(n_e)):
        # Extract componants from Quartic root finding
        k_x,k_y = srs_wns_EM(n_e[i], v_th, lambda_0, angle)
        k_y_vals = np.append(k_y_vals, k_y)

    # Output chosen componant
    return k_y_vals


def srs_EM_k_x(v_th, n_e, angle, lambda_0):

    """
    Returns the allowd SRS wavenumbers (k_y only) from the matching conditions,
    for a given density and temperature and angle, (EM daughter wave). 
    Output wavenumbers are normalised by k_0 (vacuum wavenumber).

    v_th = Electron thermal speed (units : ms^-1)
    n_e = Electron number density (units : n_cr)
    angle = Angle between k_0 and k_s (units : degrees)
    lambda_0 = Vacuum laser wavelength (units : m)
    """

    k_x_vals = np.array([])

    for i in range(len(n_e)): 
        # Extract componants from Quartic root finding
        k_x,k_y = srs_wns_EM(n_e[i], v_th, lambda_0, angle)
        k_x_vals = np.append(k_x_vals, k_x)

    # Output chosen componant
    return k_x_vals


def srs_wns_EM_polar(n_e, v_th, lambda_0, angle_min = 0, angle_max = 360):

    """
    Returns the allowd SRS wavenumbers from the matching conditions, for a
    range of angles, for a given density and temperature, (EM daughter wave). 
    Output wavenumbers are normalised by k_0 (vacuum wavenumber).

    n_e = Electron number density (units : n_cr)
    v_th = Electron thermal speed (units : ms^-1)
    lambda_0 = Vacuum laser wavelength (units : m)
    angle_min = Minimum angle between k_0 and k_s (units : degrees)
    angle_min = Maximum angle between k_0 and k_s (units : degrees) 
    """

    theta = np.linspace(angle_min, angle_max, 1000)

    # Store componant values for all angles in range
    k_x = np.zeros(len(theta))
    k_y = np.zeros(len(theta))

    for i in range(len(theta)):
        k_x[i], k_y[i] = srs_wns_EM(n_e, v_th, lambda_0, angle = theta[i])

    # Output componants for polar plot
    return k_x, k_y


################################################################################
# Wavenumber matching conditions - EPW daughter
################################################################################

def srs_wns_EPW(n_e, v_th, lambda_0, angle):

    """
    Returns the allowd SRS wavenumbers from the matching conditions,
    for a given density and temperature and angle, (EPW daughter wave). 
    Output wavenumbers are normalised by k_0 (vacuum wavenumber).

    n_e = Electron number density (units : n_cr)
    v_th = Electron thermal speed (units : ms^-1)
    lambda_0 = Vacuum laser wavelength (units : m)
    angle = Angle between k_0 and k_s (units : degrees)
    """
    
    # Vacuum wavenumber
    k_0 = laser.wavenumber(lambda_0)

    # Find the scattered EM wave wavenumbers
    ks_x, ks_y = srs_wns_EM(n_e, v_th, lambda_0, angle)

    # Employ the use of the SRS matching conditions

    # Laser wavenumber (normalised by k_0)
    k_L = laser.wavenumber_plasma(lambda_0, n_e) / k_0

    # For x : k_L = ks_x + kepw_x 
    k_x = k_L - ks_x
    # For x : 0 = ks_y + kepw_y, so magnitudes are equal
    k_y = -ks_y

    return k_x, k_y


def srs_EPW_k_y(v_th, n_e, angle, lambda_0):

    """
    Returns the allowd SRS wavenumbers (k_y only) from the matching conditions,
    for a given density and temperature and angle, (EPW daughter wave). 
    Output wavenumbers are normalised by k_0 (vacuum wavenumber).

    v_th = Electron thermal speed (units : ms^-1)
    n_e = Electron number density (units : n_cr)
    angle = Angle between k_0 and k_s (units : degrees)
    lambda_0 = Vacuum laser wavelength (units : m)
    """

    k_y_vals = np.array([])

    for i in range(len(n_e)):
        # Extract componants from Quartic root finding
        k_x,k_y = srs_wns_EPW(n_e[i], v_th, lambda_0, angle)
        k_y_vals = np.append(k_y_vals, k_y)

    # Output chosen componant
    return k_y_vals


def srs_EPW_k_x(v_th, n_e, angle, lambda_0):

    """
    Returns the allowd SRS wavenumbers (k_y only) from the matching conditions,
    for a given density and temperature and angle, (EPW daughter wave). 
    Output wavenumbers are normalised by k_0 (vacuum wavenumber).

    v_th = Electron thermal speed (units : ms^-1)
    n_e = Electron number density (units : n_cr)
    angle = Angle between k_0 and k_s (units : degrees)
    lambda_0 = Vacuum laser wavelength (units : m)
    """

    k_x_vals = np.array([])

    for i in range(len(n_e)): 
        # Extract componants from Quartic root finding
        k_x,k_y = srs_wns_EPW(n_e[i], v_th, lambda_0, angle)
        k_x_vals = np.append(k_x_vals, k_x)

    # Output chosen componant
    return k_x_vals

def srs_wns_EPW_polar(n_e, v_th, lambda_0, angle_min = 0, angle_max = 360):

    """
    Returns the allowd SRS wavenumbers from the matching conditions, for a
    range of angles, for a given density and temperature, (EPW daughter wave).
    Output wavenumbers are normalised by k_0 (vacuum wavenumber).

    n_e = Electron number density (units : n_cr)
    v_th = Electron thermal speed (units : ms^-1)
    lambda_0 = Vacuum laser wavelength (units : m)
    angle_min = Minimum angle between k_0 and k_s (units : degrees)
    angle_min = Maximum angle between k_0 and k_s (units : degrees) 
    """

    theta = np.linspace(angle_min, angle_max, 1000)

    # Store componant values for all angles in range
    k_x = np.zeros(len(theta))
    k_y = np.zeros(len(theta))

    for i in range(len(theta)):
        k_x[i], k_y[i] = srs_wns_EPW(n_e, v_th, lambda_0, angle = theta[i])

    # Output componants for polar plot
    return k_x, k_y


################################################################################
# Frequencies
################################################################################

def srs_omega_EM(n_e, T_e, angle, lambda_0):

    """
    Returns the frequencies of daughter EM waves that are valid 
    due to SRS matching conditions. Use's the EM dispersion
    relation.

    n_e = Electron number density array (units : n_cr)
    T_e = Electron temperature (units : K)
    angle = Angle between k_0 and k_s (units : degrees)
    lambda_0 = Vacuum laser wavelength (units : m)
    normed = Logical flag to set output to be normalised by the laser frequency
    """

    v_th = plasma.electron_thermal_speed(T_e)
    # Find magnitude of SRS at given angle, density, temperature
    k_x = srs_EM_k_x(v_th, n_e, angle, lambda_0)
    k_y = srs_EM_k_y(v_th, n_e, angle, lambda_0)
    k_srs_EM = np.sqrt(k_x**2 + k_y**2)
    # Require k's to be normlaised by k_0
    omega_srs_EM = plasma.dispersion_EM(n_e, k_srs_EM)

    return omega_srs_EM

def srs_omega_EPW(n_e, T_e, angle, lambda_0):

    """
    Returns the frequencies of daughter EPW waves that are valid 
    due to SRS matching conditions. Use's the warm EPW dispersion
    relation.

    n_e = Electron number density array (units : n_cr)
    T_e = Electron temperature (units : K)
    angle = Angle between k_0 and k_s (units : degrees)
    lambda_0 = Vacuum laser wavelength (units : m)
    normed = Logical flag to set output to be normalised by the laser frequency
    """

    v_th = plasma.electron_thermal_speed(T_e)
    # Find magnitude of SRS at given angle, density, temperature
    k_x = srs_EPW_k_x(v_th, n_e, angle, lambda_0)
    k_y = srs_EPW_k_y(v_th, n_e, angle, lambda_0)
    k_srs_EPW = np.sqrt(k_x**2 + k_y**2)
    # Require k's to be normlaised by k_0
    omega_srs_EPW = plasma.dispersion_EPW(n_e, T_e, k_srs_EPW)

    return omega_srs_EPW


################################################################################
# Linear growth
################################################################################

def landau_cutoff_index(T_e, n_e, lambda_0, angle, cutoff = 0.3):

    """
    Returns the estimated index location for the Landau cutoff,
    required for k_y vs x plot of Electrostatic field. The number
    density paramter is used as an array of values in this case.

    T_e = Electron temperature (units : K)
    n_e = Electron number density array (units : n_cr)
    lambda_0 = Vacuum laser wavelength (units : m)
    angle = Angle between k_0 and k_s (units : degrees)
    cutoff = Strength of Landau damping i.e k * Debeye length = cutoff (units : dimensionless) 
    """

    v_th = plasma.electron_thermal_speed(T_e)
    # Set vacuum value as required to get correct dimensions
    k_0 = laser.wavenumber(lambda_0)
    # Find magnitude of SRS EPW daughter at given angle, density, temperature
    k_x = srs_EPW_k_x(v_th, n_e, angle, lambda_0)
    k_y = srs_EPW_k_y(v_th, n_e, angle, lambda_0)
    k_srs_EPW = np.sqrt(k_x**2 + k_y**2) * k_0
    # Find the Debeye length 
    lambda_D = plasma.Debeye_length(T_e, n_e, lambda_0)
    # Estimate Landau damping strength
    landau_damp = k_srs_EPW * lambda_D
    # Find index where Langmuir waves are not damped
    index_cut = np.where(landau_damp < cutoff)[0]
    if len(index_cut) == 0:
        print(f'Landau damping coefficient k lambda_D is never lower than {cutoff} in this density range')
        return None
    else:
        return index_cut[0]


