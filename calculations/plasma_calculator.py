#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

plasma_calculator.py

Script housing functions used to calculate common plasma 
quantities, dispersion relations and denisty profiles.

"""

# import libraries
import numpy as np
import scipy.constants as const
from calculations import laser_calculator as laser

# Whether to use relativistic correction for omega_pe
relativistic = False

################################################################################
# Useful plasma quantities
################################################################################

def critical_density(lambda_0):

    """
    Calculates critcal density of plasma from laser wavelength
    in vacuum. (units : m^-3)

    lambda_0 = Vacuum laser wavelength (units : m)
    """

    omega_0 = laser.omega(lambda_0)

    return omega_0**2 * const.m_e * const.epsilon_0 / const.e**2

def electron_thermal_speed(T_e):

    """
    Calculates electron thermal speed from given electron
    temperature. (units : ms^-1).

    T_e = Electron temperature (units : K)
    """

    return np.sqrt(const.k * T_e  / const.m_e)

def electron_plasma_freq(n_e, lambda_0, v_th, relativistic=relativistic):

    """
    Calculates electron plasma frequencey from given electron
    number density. (units : rads^-1).

    n_e = Electron temperature (units : n_cr)
    lambda_0 = Vacuum laser wavelength (units : m)
    """

    omega_0 = laser.omega(lambda_0)
    omega_pe = np.sqrt(n_e) * omega_0
    if relativistic:
        # First order relativistic corection [Pegoraro and Porcelli, 1984]
        # brought about by the relativistic mass increase of high-energy thermal electrons.
        factor = np.sqrt(1.0 - 2.5*v_th**2/const.c**2)
        return factor * omega_pe
    else:
        return omega_pe

def Debeye_length(T_e, n_e, lambda_0, relativistic=relativistic):

    """
    Calculates Debeye length from electron thermal speed
    and electron plasma frequencey. (units : m).

    T_e = Electron temperature (units : K)
    n_e = Electron temperature (units : n_cr)
    lambda_0 = Vacuum laser wavelength (units : m)
    """

    v_th = electron_thermal_speed(T_e) # ms^-1
    omega_pe = electron_plasma_freq(n_e, lambda_0, v_th, relativistic) # rads^-1

    return v_th / omega_pe

################################################################################
# Typical density profiles
################################################################################

def density_exponential(n_0, L_n, x):

    """
    Calculates exponential density profile
    for a given scale length and initial
    density.

    n_0 = Number density at x = 0 (units : n_cr)
    L_n = Density scale length (units : m)
    x = x-space locations (units : m)
    """
    return n_0 * np.exp(x / L_n)

def density_linear(n_0, L_n, x):

    """
    Calculates linear density profile
    for a given scale length and intial
    density.

    n_0 = Number density at x = 0 (or x_min) (units : n_cr)
    L_n = Density scale length (units : m)
    x = x-space locations (units : m)
    """
    return n_0 * (1.0 + x / L_n)



def x_locs_exponential(n_0, L_n, x, n_min, n_max):

    """
    Calculates x locations for given density range
    for an exponential density profile.

    n_0 = Number density at x = 0 (units : n_cr)
    L_n = Density scale length (units : m)
    x = x-space locations (units : m)
    n_min = Minimum number density (units : n_cr)
    n_max = Maximum number density (units : n_cr)
    """

    x_min = np.log(n_min/n_0) * L_n
    x_max = np.log(n_max/n_0) * L_n

    idx_min = np.where(x - x_min >= 0)[0]
    idx_max = np.where(x - x_max >= 0)[0]

    if len(idx_max) == 0:
        idx_max = -1
        print(f'{n_max}' + ' n_cr is not within range of the average number density, plotting up to max value ')
        return x[idx_min[0]:idx_max]
    else:
        return x[idx_min[0]:idx_max[0]]

    
def x_locs_linear(n_0, L_n, x, n_min, n_max):

    """
    Calculates x locations for given density range
    for a linear density profile.

    n_min = Minimum number density (units : n_cr)
    n_max = Maximum number density (units : n_cr)
    x = x-space locations (units : m)
    """

    x_min = (n_min/n_0 - 1.0) * L_n
    x_max = (n_max/n_0 - 1.0) * L_n

    idx_min = np.where(x - x_min >= 0)[0]
    idx_max = np.where(x - x_max >= 0)[0]

    if len(idx_max) == 0:
        idx_max = -1
        print(f'{n_max}' + ' n_cr is not within range of the average number density, plotting up to max value ')
        return x[idx_min[0]:idx_max]
    else:
        return x[idx_min[0]:idx_max[0]]

    

################################################################################
# Dispersion relations - Omega
################################################################################

# Dispersion relations for common wave modes. Returns frequenecies
# for given wavenumbers

# Electron Plasma Wave (Warm) 
def dispersion_EPW(n_e, T_e, lambda_0, k, relativistic=relativistic):

    """
    Calculates electron plasma wave frequencies for a 
    range of given wavenumbers k. These wavenumbers are
    assumed to be normalised by k_0, (laser wavenumber
    in vacuum).

    
    n_e = Electron temperature (units : n_cr)
    T_e = Electron temperature (units : K)
    k = Wavenumber magnitude (units : k_0)
    """

    # So that function can work if n_e, k are given as np arrays,
    # lists or scalars
    if np.isscalar(n_e) and np.isscalar(k):
            n_e = n_e
    elif np.isscalar(n_e):
        n_e = np.ones(len(k)) * n_e
    else:
        n_e = n_e

    v_th = electron_thermal_speed(T_e)
    omega_0 = laser.omega(lambda_0)
    omega_pe = electron_plasma_freq(n_e, lambda_0, v_th, relativistic)
    omega = np.sqrt(omega_pe**2/omega_0**2 + 3.0 * v_th**2 * k **2 / const.c**2)

    return omega


# Electromagnetic Wave
def dispersion_EM(n_e, v_th, lambda_0, k, relativistic=relativistic):

    """
    Calculates EM wave frequencies for a 
    range of given wavenumbers k. These wavenumbers are
    assumed to be normalised by k_0, (laser wavenumber
    in vacuum). The outputted frequency has units of 
    omega_0 (i.e laser frequency).

    n_e = Electron temperature (units : n_cr)
    k = Wavenumber magnitude (units : k_0)

    """

    # So that function can work if n_e, k are given as np arrays,
    # lists or scalars
    if np.isscalar(n_e) and np.isscalar(k):
            n_e = n_e
    elif np.isscalar(n_e):
        n_e = np.ones(len(k)) * n_e
    else:
        n_e = n_e

    omega_0 = laser.omega(lambda_0)
    omega_pe = electron_plasma_freq(n_e, lambda_0, v_th, relativistic)

    omega = np.sqrt(omega_pe**2/omega_0**2 + k**2)
    return omega


# Stokes Branch
def Stokes_branch(n_e, k):

    """
    Calculates the Stoke branch wave frequencies for a 
    range of given wavenumbers k, and desnity point.
    These wavenumbers are assumed to be normalised by k_0, 
    (laser wavenumber in vacuum), and the desnity in units
    of n_cr. The outputted frequency has units of 
    omega_0 (i.e laser frequency).

    n_e = Electron temperature (units : n_cr)
    k = Wavenumber magnitude (units : k_0)
    """


    omega_stoke = 1.0 - np.sqrt(n_e + (k - 1.0)**2)
    

    return omega_stoke



