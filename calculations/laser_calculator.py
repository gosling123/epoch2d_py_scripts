#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

laser_calculator.py

Script housing functions used to calculate common laser 
quantities.

"""

# import libraries
import scipy.constants as const
import numpy as np

def omega(lambda_0):
    """
    Calculates laser angular frequency for 
    given wavelength.

    lambda_0 = Wavelength (units : m)
    """
    return 2.0 * const.c * const.pi / lambda_0

def wavenumber(lambda_0):
    """
    Calculates laser wavenumber for 
    given wavelength.

    lambda_0 = Wavelength (units : m)
    """
    return 2.0 * const.pi / lambda_0

def wavenumber_plasma(lambda_0, n_e):
    """
    Calculates laser wavenumber for 
    given wavelength in a plasma with
    electron density n_e.

    lambda_0 = Wavelength (units : m)
    n_e = Electron number density (units : n_cr)
    """
    k_0 = wavenumber(lambda_0)
    return k_0 * np.sqrt(1.0 - n_e)

def E_normalisation(lambda_0):
    """
    Calculates E field normlaisation 
    constant for given wavelength.

    lambda_0 = Wavelength (units : m)
    """
    omega_0 = omega(lambda_0)
    return const.e / (const.m_e * omega_0 * const.c)

def B_normalisation(lambda_0):
    """
    Calculates B field normlaisation 
    constant for given wavelength.

    lambda_0 = Wavelength (units : m)
    """
    omega_0 = omega(lambda_0)
    return const.e / (const.m_e * omega_0)