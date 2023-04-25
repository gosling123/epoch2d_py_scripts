#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

tpd_calculator.py

Script housing all the functions used to calculate quantities relating
to the TPD laser-plasma interaction.

"""


# import libraries
import numpy as np
import scipy.constants as const
import warnings
from calculations import plasma_calculator as plasma
from calculations import laser_calculator as laser


# Whether to use relativistic correction for omega_pe
relativistic = True

################################################################################
# Wavenumber matching conditions
################################################################################


def tpd_k_y(v_th, n_e, angle, lambda_0):

    """
    Returns the allowd TPD wavenumbers (k_y only) from the matching conditions,
    for a given density and temperature and angle. 
    Output wavenumbers are normalised by k_0 (vacuum wavenumber).

    v_th = Electron thermal speed (units : ms^-1)
    n_e = Electron number density (units : n_cr)
    angle = Angle between k - k_0/2 and k_0 (units : degrees)
    lambda_0 = Vacuum laser wavelength (units : m)
    """

    if angle == 'max_lin_growth':
        # Get angles for which matching conditions intersect max growth curve
        theta = np.zeros(len(n_e))
        for i in range(len(n_e)):
            theta[i] = tpd_angle_max_growth(n_e[i], v_th, lambda_0) * np.pi/180.0
    else:
        # convert from degrees to radians
        theta = angle * np.pi/180.0

    # Laser 
    k_0 = laser.wavenumber(lambda_0)
    k_L = laser.wavenumber_plasma(lambda_0, n_e)
    omega_0 = laser.omega(lambda_0)
    # Plasmon Frequency
    omega_pe = plasma.electron_plasma_freq(n_e, lambda_0, v_th, relativistic = relativistic)
    # Magnitude of wavevector for given angle (normalised by k_0)
    mag_square =  const.c**2 / (3.0 * v_th**2) * (0.25 - omega_pe**2/omega_0**2) - 0.25 * k_L**2 * const.c**2 / omega_0**2
    mag_square /= 1.0 - 3.0 * v_th**2 / omega_0**2 * k_L**2 * np.cos(theta)**2

    # x componant 
    k_y = np.sqrt(mag_square) * np.sin(theta)

    return k_y

def tpd_k_x(v_th, n_e, angle, lambda_0):

    """
    Returns the allowd TPD wavenumbers (k_x only) from the matching conditions,
    for a given density and temperature and angle. 
    Output wavenumbers are normalised by k_0 (vacuum wavenumber).

    v_th = Electron thermal speed (units : ms^-1)
    n_e = Electron number density (units : n_cr)
    angle = Angle between k - k_0/2 and k_0 (units : degrees)
    lambda_0 = Vacuum laser wavelength (units : m)
    """
    
    if angle == 'max_lin_growth':
        # Get angles for which matching conditions intersect max growth curve
        theta = np.zeros(len(n_e))
        for i in range(len(n_e)):
            theta[i] = tpd_angle_max_growth(n_e[i], v_th, lambda_0) * np.pi/180.0
    else:
        # convert from degrees to radians
        theta = angle * np.pi/180.0
    
    # Laser 
    k_0 = laser.wavenumber(lambda_0)
    k_L = k_0 * np.sqrt(1.0 - n_e)
    omega_0 = laser.omega(lambda_0)
    # Plasmon Frequency
    omega_pe = plasma.electron_plasma_freq(n_e, lambda_0, v_th, relativistic = relativistic)
    # Magnitude of wavevector for given angle (normalised by k_0)
    mag_square =  const.c**2 / (3.0 * v_th**2) * (0.25 - omega_pe**2/omega_0**2) - 0.25 * k_L**2 * const.c**2 / omega_0**2
    mag_square /= 1.0 - 3.0 * v_th**2 / omega_0**2 * k_L**2 * np.cos(theta)**2

    # x componant 
    k_x = np.sqrt(mag_square) * np.cos(theta) + 0.5 * k_L / k_0

    return k_x



def tpd_wns_polar(n_e, v_th, lambda_0, angle_min = 0, angle_max = 360):

    """
    Returns the allowd TPD wavenumbers from the matching conditions, at all
    angles, for a given density and temperature. Output wavenumbers are 
    normalised by k_0 (vacuum wavenumber).

    n_e = Electron number density (units : n_cr)
    v_th = Electron thermal speed (units : ms^-1)
    lambda_0 = Vacuum laser wavelength (units : m)
    angle_min = Minimum angle between k - k0/2 and k0 (units : degrees)
    angle_min = Maximum angle between k - k0/2 and k0 (units : degrees) 
    """

    # Convert degrees tp radians
    theta_min = const.pi * angle_min / 180.0
    theta_max = const.pi * angle_max / 180.0
    # Laser 
    k_0 = laser.wavenumber(lambda_0)
    k_L =  laser.wavenumber_plasma(lambda_0, n_e) / k_0
    omega_0 = laser.omega(lambda_0)

    # Angle between k - k0/2 and k0
    theta = np.linspace(theta_min, theta_max, 100)
    # print(v_th)
    # Magnitude of wavevector for given angle (normalised by k_0)
    # Plasmon Frequency
    omega_pe = plasma.electron_plasma_freq(n_e, lambda_0, v_th, relativistic = relativistic)
    k_tpd_rad = np.sqrt(const.c**2 / (3.0 * v_th**2) * (0.25 - omega_pe**2/omega_0**2) - 0.25*k_L**2)
    k_tpd_rad /= np.sqrt( 1.0 - 3.0 * v_th**2 / omega_0**2 * (k_L*k_0)**2 * np.cos(theta)**2) 
    # Find componants from (k_x - 0.5 k_0)^2 + k_y^2 = R^2(theta)
    k_x = k_tpd_rad * np.cos(theta) + k_L/2.0
    k_y = k_tpd_rad * np.sin(theta)

    return k_x, k_y


def tpd_wns_pairs(v_th, n_e, angle, lambda_0, componants = 'both'):

    """
    Returns the componants of the pair of Langmuir waves that satisfy
    the TPD matching conditions, for given densies, angle and temperature.
    Output wavenumbers are normalised by k_0 (vacuum wavenumber).

    v_th = Electron thermal speed (units : ms^-1)
    n_e = Electron number density (units : n_cr)
    angle = Angle between k - k_0/2 and k_0 (units : degrees)
    lambda_0 = Vacuum laser wavelength (units : m)
    """

    # Vacuum wavenumber
    k_0 = laser.wavenumber(lambda_0)
    # Laser wavenumber at n_e
    k_L = laser.wavenumber_plasma(lambda_0, n_e) / k_0
    # Allowed Langmuir wave numbers from TPD matching
    k1_x = tpd_k_x(v_th, n_e, angle, lambda_0)
    k1_y = tpd_k_y(v_th, n_e, angle, lambda_0)
    # Second LW defined via matching conditions
    k2_x = k_L -k1_x
    k2_y = - k1_y

    if componants == 'both':
        # Return both componants for the two LW
        return k1_x, k1_y, k2_x, k2_y
    elif componants == 'x':
        # Return only the x-componant of the two
        return k1_x, k2_x
    elif componants == 'y':
        # Return only the y-componant of the two
        return k1_y, k2_y
    else:
        warnings.warn("componants argument must be taken as both, x or y")
        return None


################################################################################
# Frequencey
################################################################################

def tpd_omegas(n_e, T_e, angle, lambda_0):

    """
    Returns the frequencies of EPW's that are valid due to 
    TPD matching conditions. Use's the warm EPW dispersion
    relation.

    n_e = Electron number density array (units : n_cr)
    T_e = Electron temperature (units : K)
    angle = Angle between k - k_0/2 and k_0 (units : degrees)
    lambda_0 = Vacuum laser wavelength (units : m)
    normed = Logical flag to set output to be normalised by the laser frequency
    """
    v_th = plasma.electron_thermal_speed(T_e)

    if angle == 'max_lin_growth':
        # Get angles for which matching conditions intersect max growth curve
        theta = np.zeros(len(n_e))
        for i in range(len(n_e)):
            theta[i] = tpd_angle_max_growth(n_e[i], v_th, lambda_0) * np.pi/180.0
    else:
        # convert from degrees to radians
        theta = angle * np.pi/180.0

    # Find magnitude of TPD at given angle, density, temperature
    k1_x, k1_y, k2_x, k2_y = tpd_wns_pairs(v_th, n_e, angle, lambda_0, componants = 'both')
    k_1 = np.sqrt(k1_x**2 + k1_y**2)
    k_2 = np.sqrt(k2_x**2 + k2_y**2) 
    # Require k's to be normlaised by k_0
    omega_1 = plasma.dispersion_EPW(n_e, T_e, lambda_0, k_1, relativistic)
    omega_2 = plasma.dispersion_EPW(n_e, T_e, lambda_0, k_2, relativistic)

    return omega_1, omega_2


################################################################################
# Linear growth
################################################################################

def tpd_max_lin_growth(n_e, lambda_0):

    """
    Returns the TPD wavenumbers that correspond to maximal linear growth.
    Output wavenumbers are normalised by k_0 (vacuum wavenumber).

    n_e = Electron number density (units : n_cr)
    """
    # Vacuum wavenumber
    k_0 = laser.wavenumber(lambda_0)
    # Laser wavenumber in plasma
    k_L = laser.wavenumber_plasma(lambda_0, n_e) / k_0
    # k_x range to plot over
    k_x = np.linspace(-3.0, 3.0, 5000)
    x_part = (k_x - 0.5 * k_L)**2
    # Avoid computing imaginary region
    indx = np.where(x_part > 0.25*k_L**2)[0]
    k_x = k_x[indx[0]:indx[-1]]
    x_part = (k_x-0.5*k_L)**2
    # Find k_y values over k_x range
    k_y = np.sqrt(x_part - 0.25*k_L**2)

    return k_x, k_y

def landau_cutoff_index(T_e, n_e, lambda_0, angle, cutoff = 0.3):

    """
    Returns the estimated index location for the Landau cutoff,
    required for k_y vs x plot of Electrostatic field. The number
    density paramter is used as an array of values in this case.

    T_e = Electron temperature (units : K)
    n_e = Electron number density array (units : n_cr)
    lambda_0 = Vacuum laser wavelength (units : m)
    angle = Angle between k - k_0/2 and k_0 (units : degrees)
    cutoff = Strength of Landau damping i.e k * Debeye length = cutoff (units : dimensionless) 
    """

    
    v_th = plasma.electron_thermal_speed(T_e)
    
    if angle == 'max_lin_growth':
        # Get angles for which matching conditions intersect max growth curve
        theta = np.zeros(len(n_e))
        for i in range(len(n_e)):
            theta[i] = tpd_angle_max_growth(n_e[i], v_th, lambda_0) * np.pi/180.0
    else:
        # convert from degrees to radians
        theta = angle * np.pi/180.0

    
    # Set vacuum value as required to get correct dimensions
    k_0 = laser.wavenumber(lambda_0)
    # Find magnitude of TPD at given angle, density, temperature
    k1_x, k1_y, k2_x, k2_y = tpd_wns_pairs(v_th, n_e, angle, lambda_0, componants = 'both')
    k_1 = np.sqrt(k1_x**2 + k1_y**2) * k_0
    k_2 = np.sqrt(k2_x**2 + k2_y**2) * k_0
    # Find the Debeye length (relativistic flag -> uses first order correction to omega_pe)
    lambda_D = plasma.Debeye_length(T_e, n_e, lambda_0, relativistic = relativistic)
    # Estimate Landau damping strength
    landau_coeff_1  = k_1 * lambda_D
    landau_coeff_2  = k_2 * lambda_D
    # Find index where Langmuir waves are not damped
    landau_cut_1 = np.where(landau_coeff_1 < cutoff)[0]
    landau_cut_2 = np.where(landau_coeff_2 < cutoff)[0]

    if len(landau_cut_1) == 0 and len(landau_cut_2) == 0:
        print(f'Landau damping coefficient k lambda_D is never lower than {cutoff} in this density range')
        return None
    elif len(landau_cut_1) == 0:
        return landau_cut_2[0]
    elif len(landau_cut_2) == 0:
        return landau_cut_1[0]
    else:
        return max(landau_cut_1[0], landau_cut_2[0])
        

def tpd_angle_max_growth(n_e, v_th, lambda_0):

    """
    Returns the estimated angle for which maximum linear growth is observed.
    Angle is given in degrees

    n_e = Electron number density array (units : n_cr)
    v_th = Electron thermal speed (units : ms^-1)
    lambda_0 = Vacuum laser wavelength (units : m)
    """

    # Laser 
    k_0 = laser.wavenumber(lambda_0)
    k_L = laser.wavenumber_plasma(lambda_0, n_e)
    omega_0 = laser.omega(lambda_0)
    # Check if circle approximation is viable 
    thermal_factor = 1.0 / (1.0 - 3.0 * v_th**2 / omega_0**2 * k_L**2) 
    if thermal_factor <= 1.10:
        # Find componants from analytic formulae
        x_comp = np.sqrt(const.c**2 / (6.0 * v_th**2) * (0.25 - n_e))
        y_comp = np.sqrt(const.c**2 / (6.0 * v_th**2) * (0.25 - n_e) - 0.25 * k_L**2 * const.c**2 / omega_0**2)
        theta = np.arctan(y_comp / x_comp)
        # Want it in degrees
        theta *= 180.0/np.pi
        return theta
    
    else:
        theta = np.linspace(0, 90, 100000)
        # Start with large number to not set of the sort
        diff = 1000
        for i in range(len(theta)):
            # From matching conditions
            k_x_mc = tpd_k_x(v_th, n_e, theta[i], lambda_0)
            k_y_mc = tpd_k_y(v_th, n_e, theta[i], lambda_0)
            # Estimate deviation
            diff_new = np.abs((k_x_mc - 0.5 * k_L/k_0)**2 - k_y_mc**2 - 0.25 * k_L**2/k_0**2)
            # Find angle which has minimum deviation
            if diff_new > diff:
                continue
            else:
                diff = diff_new
                theta_max_growth = theta[i]
        return theta_max_growth