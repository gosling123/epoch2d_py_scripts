#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

plasma_calculator.py

Script housing clases used to calculate common plasma 
quantities, dispersion relations and denisty profiles.

"""

# import libraries
import numpy as np
import scipy.constants as const


################################################################################
# Useful plasma quantities
################################################################################

class plasma_params:

    def __init__(self, lambda_0, T_e):

        """
        Class constructor function.

        lambda_0 = Vacuum laser wavelength (units : m)
        T_e = Electron temperature (units : K)
        """

        # Laser Varibales
        self.lambda_0 = lambda_0
        self.omega_0 = 2.0 * const.pi * const.c / self.lambda_0
        self.k_0 = 2.0 * const.pi / self.lambda_0

        self.T_e = T_e

    
    def critical_density(self):

        """
        Calculates critcal density of plasma from laser wavelength
        in vacuum. (units : m^-3)

        """
        return self.omega_0**2 * const.m_e * const.epsilon_0 / const.e**2

    def electron_thermal_speed(self):

        """
        Calculates electron thermal speed from given electron
        temperature. (units : ms^-1).
        
        """
        return np.sqrt(const.k * self.T_e  / const.m_e)

    
    def electron_plasma_freq(self, n_e):

        """
        Calculates electron plasma frequencey from given electron
        number density. (units : rads^-1).

        n_e = Electron temperature (units : n_cr)
        """

        # Using that omega_pe^2 = (n_e * e^2) / (eps_0 * m_e) 
        # and omega_0^2 = (n_cr * e^2) / (eps_0 * m_e)

        return np.sqrt(n_e) * self.omega_0

    def Debeye_length(self, n_e):

        """
        Calculates Debeye length from electron thermal speed
        and electron plasma frequencey. (units : m).

        n_e = Electron temperature (units : n_cr)
        """

        omega_pe = self.electron_plasma_freq(n_e) # rads^-1
        v_th = self.electron_thermal_speed() # ms^-1

        return v_th / omega_pe


################################################################################
# Typical density profiles
################################################################################

class density:

    def __init__(self, n_0, L_n):

        """
        Class constructor function.

        n_0 = Number density at x = 0 (or x_min) (units : n_cr)
        L_n = Density scale length (units : m)
        """

        self.n_0 = n_0
        self.L_n = L_n

    def exponential(self, x):

        """
        Calculates exponential density profile
        for given domain.

        x = x-space locations (units : m)
        """
        return self.n_0 * np.exp(x / self.L_n)

    def linear(self, x):

        """
        Calculates linear density profile
        for given domain.

        x = x-space locations (units : m)
        """
        return self.n_0 * (1.0 + x / self.L_n)

    def x_locs_exponential(self, n_min, n_max, x):

        """
        Calculates x locations for given density range
        for an exponential density profile.

        n_min = Minimum number density (units : n_cr)
        n_max = Maximum number density (units : n_cr)
        x = x-space locations (units : m)
        """

        x_min = np.log(n_min/self.n_0) * self.L_n
        x_max = np.log(n_max/self.n_0) * self.L_n

        idx_min = np.where(x - x_min >= 0)[0]
        idx_max = np.where(x - x_max >= 0)[0]

        if len(idx_max) == 0:
            idx_max = -1
            print(f'{n_max}' + ' n_cr is not within range of the average number density, plotting up to max value ')
            return x[idx_min[0]:idx_max]
        else:
            return x[idx_min[0]:idx_max[0]]

    def x_locs_linear(self, n_min, n_max, x):

        """
        Calculates x locations for given density range
        for an exponential density profile.

        n_min = Minimum number density (units : n_cr)
        n_max = Maximum number density (units : n_cr)
        x = x-space locations (units : m)
        """

        x_min = (n_min/self.n_0 - 1.0) * self.L_n
        x_max = (n_max/self.n_0 - 1.0) * self.L_n

        idx_min = np.where(x - x_min >= 0)[0]
        idx_max = np.where(x - x_max >= 0)[0]

        if len(idx_max) == 0:
            idx_max = -1
            print(f'{n_max}' + ' n_cr is not within range of the average number density, plotting up to max value ')
            return x[idx_min[0]:idx_max]
        else:
            return x[idx_min[0]:idx_max[0]]

    




################################################################################
# Dispersion Relations
################################################################################

class dispersion_relations():

    def __init__(self, n_e, T_e, lambda_0):

        """
        Class constructor function.

        n_e = Electron number density (units : n_cr)
        T_e = Electron temperature (units : K)
        lambda_0 = Vacuum laser wavelength (units : m)
        """

        self.n_e = n_e
        self.T_e = T_e
        self.lambda_0 = lambda_0

        self.plasma_params = plasma_params(lambda_0 = self.lambda_0, T_e = self.T_e)
        self.v_th = self.plasma_params.electron_thermal_speed()


    # Electron Plasma Wave (Warm) 
    def dispersion_EPW_omega(self, k):

        """
        Calculates electron plasma wave frequencies for a 
        range of given wavenumbers k. These wavenumbers are
        assumed to be normalised by k_0 (laser wavenumber
        in vacuum). Output frequencies are thus normalised
        by the laser frequency.

        k = Wavenumber magnitude (units : k_0)
        """

        if np.isscalar(self.n_e) and np.isscalar(k):
            n_e = self.n_e
        elif np.isscalar(self.n_e):
            n_e = np.ones(len(k)) * self.n_e
        else:
            n_e = self.n_e

        omega = np.sqrt(self.n_e + 3.0 * self.v_th**2 * k **2 / const.c**2)
        return omega


    # Electromagnetic Wave
    def dispersion_EM_omega(self, k):

        """
        Calculates EM wave frequencies for a 
        range of given wavenumbers k. These wavenumbers are
        assumed to be normalised by k_0, (laser wavenumber
        in vacuum). The outputted frequency has units of 
        omega_0 (i.e laser frequency).

        k = Wavenumber magnitude (units : k_0)

        """

        if np.isscalar(self.n_e) and np.isscalar(k):
            n_e = self.n_e
        elif np.isscalar(self.n_e):
            n_e = np.ones(len(k)) * self.n_e
        else:
            n_e = self.n_e

        omega = np.sqrt(n_e + np.power(k,2))
        return omega


