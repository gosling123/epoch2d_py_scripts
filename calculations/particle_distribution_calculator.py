#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

particle_distribution_calculator.py

Script housing functions used to calculate common plasma 
particle distributions.

"""

# import libraries
import numpy as np
import scipy.constants as const


def max_boltz_momentum(p, T_e, deg=3):

    """
    Returns the normlaised Maxwell-Boltzmann
    distribution, for given momentum range and
    temperature.

    p = Momentum domain (units : ms^-1)
    T_e = Electron temperature (units : K)

    """
    counts = (2.0*np.pi*const.m_e*const.k*T_e)**(-0.5*deg) * np.exp(-p**2 / (2.0 * const.m_e*const.k*T_e))
    density = counts / (np.sum(counts)*np.diff(p)[0])
    return density




def max_boltz_energy(E, T_e, weighted=False):

    """
    Returns the normlaised Maxwell-Boltzmann
    distribution, for given energies and
    temperature.

    p = Momentum domain (units : ms^-1)
    T_e = Electron temperature (units : K)

    """
    # Get exponential factor counts
    counts = 2.0*np.sqrt(E/(np.pi*(const.k*T_e)**3))*np.exp(-E/(const.k*T_e))
    density = counts / (np.sum(counts)*np.diff(E)[0])
    if weighted:
        return density * E
    else:
        return density
