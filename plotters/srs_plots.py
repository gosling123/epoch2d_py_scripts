#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
srs_plots.py 

File that houses class for defining SRS plot curves
"""

# Import libraries
import sys
sys.path.append("..")
import matplotlib.pyplot as plt
import numpy as np
import calculations.srs_calculator as srs
import calculations.plasma_calculator as plasma

# Set colours for plot
em_colour = 'blue'
epw_colour = 'white'


class plots:
    """ Class housing functions to call for plotting SRS curves"""

    def __init__(self, T_e, lambda_0):

        """
        Class constructor function

        T_e = Electron temperature (units : K)
        lambda_0 = Laser wavelength in vacuum (units : m)
        """

        # Base laser plasma variables required
        self.T_e = T_e
        self.lambda_0 = lambda_0

        # Electron thermal speed (key varibale for most plots)
        self.v_th = plasma.electron_thermal_speed(self.T_e)

    ########################################################################################################################
    # kx vs omega
    ########################################################################################################################

    def kx_vs_omega_EPW(self, n_e, theta, ax):

        """
        Plots the curve for expcted wavenumbers (kx) and
        frequencies at a given desnities of scattered EPW
        generated by SRS.

        n_e = Densities to calculate wavenumbers and frequencies at (units : n_cr)
        theta = Angle of scatter that defines SRS process. Angle is defined
                as the angle between the EM wavevector and the laser wavevector. (units : degrees) 
        ax = Name of plot to plot onto (best to use plt.gca())
        """

        # Get wavenumbers and frequencies for scattered EPW
        k_epw = srs.srs_EPW_k_x(self.v_th, n_e, theta, self.lambda_0)
        omega_epw = srs.srs_omega_EPW(n_e, self.T_e, theta, self.lambda_0)
        # Plot
        ax.plot(k_epw, omega_epw, c=epw_colour, label = r'SRS EPW ($\theta$ =' + f'{np.round(theta, 1)}\N{DEGREE SIGN})')    
        

    def kx_vs_omega_EM(self, n_e, theta,  ax):  

        """
        Plots the curve for expcted wavenumbers (kx) and
        frequencies at a given desnities of scattered EM
        generated by SRS.

        n_e = Densities to calculate wavenumbers and frequencies at (units : n_cr)
        theta = Angle of scatter that defines SRS process. Angle is defined
                as the angle between the EM wavevector and the laser wavevector. (units : degrees) 
        ax = Name of plot to plot onto (best to use plt.gca())
        """
        
        # Get wavenumbers and frequencies for scattered EMW
        k_em = srs.srs_EM_k_x(self.v_th, n_e, theta, self.lambda_0) 
        omega_em = srs.srs_omega_EM(n_e, self.T_e, theta, self.lambda_0)
        # Plot
        ax.plot(k_em, omega_em, c=em_colour, label = r'SRS EM ($\theta$ =' + f'{np.round(theta, 1)}\N{DEGREE SIGN})')

    ########################################################################################################################
    # x vs omega
    ########################################################################################################################

    def x_vs_omega_EPW(self, n_e, theta, x, ax):

        """
        Plots the curve for expcted frequencies at a 
        spatial/desnity location. This plots the frequencies
        of the scattered EPW.

        n_e = Densities to calculate wavenumbers and frequencies at (units : n_cr)
        theta = Angle of scatter that defines SRS process. Angle is defined
                as the angle between the EM wavevector and the laser wavevector. (units : degrees) 
        x = Spatial scale to plot over (units : m)
        ax = Name of plot to plot onto (best to use plt.gca())
        """
        
        # Get frequencies
        omega_epw = srs.srs_omega_EPW(n_e, self.T_e, theta, self.lambda_0)
        # Plot
        ax.plot(x, omega_epw, c=epw_colour, label = r'SRS EPW ($\theta$ =' + f'{np.round(theta, 1)}\N{DEGREE SIGN})')

    def x_vs_omega_EM(self, n_e, theta, x, ax):   

        """
        Plots the curve for expcted frequencies at a 
        spatial/desnity location. This plots the frequencies
        of the scattered EM wave.

        n_e = Densities to calculate wavenumbers and frequencies at (units : n_cr)
        theta = Angle of scatter that defines SRS process. Angle is defined
                as the angle between the EM wavevector and the laser wavevector. (units : degrees) 
        x = Spatial scale to plot over (units : m)
        ax = Name of plot to plot onto (best to use plt.gca())
        """
        
        # Get frequencies
        omega_em = srs.srs_omega_EM(n_e, self.T_e, theta, self.lambda_0)
        # Plot
        ax.plot(x, omega_em, c=em_colour, label = r'SRS EM ($\theta$ =' + f'{np.round(theta, 1)}\N{DEGREE SIGN})')

    
    ########################################################################################################################
    # omega range 
    ########################################################################################################################

    def omega_EM(self, axis, n_min, n_max, ax):

        """
        Finds the range of frequncies for scattred EM waves
        from SRS for a given density range.

        axis = Sets oreintation of plot. Either = 'x' or 'y'.
        n_min = Minimum density (units : n_cr)
        n_max = Maximum density (units : n_cr)
        ax = Name of plot to plot onto (best to use plt.gca())
        """

        # Take number density points
        num_dens = np.linspace(n_min, n_max, 50)
        # Look for all angles
        angles = np.linspace(0, 360, 50)

        # Store max and min values
        omega_min = np.zeros(len(angles))
        omega_max = np.zeros(len(angles))
        

        for i in range(len(angles)):
            omega_em = srs.srs_omega_EM(num_dens, self.T_e, angles[i], self.lambda_0)
            # Extract min/max frequncies for given angle
            idx = np.where(np.isnan(omega_em) == False)[0]
            omega_min[i] = omega_em[idx].min()
            omega_max[i] = omega_em[idx].max()
        
        # Plots bounds to get idea of range of frequcies relating to scattered EM waves due to SRS
        if axis == 'x':
            ax.axhline(omega_min.min(), c=em_colour)
            ax.axhline(omega_max.max(), c=em_colour, label = f'SRS EM ({np.round(n_min,2)}-{np.round(n_max,2)}' + r' $n_{cr}$)')
        elif axis == 'y':
            ax.axvline(omega_min.min(), c=em_colour)
            ax.axvline(omega_max.max(), c=em_colour, label = f'SRS EM ({np.round(n_min,2)}-{np.round(n_max,2)}' + r' $n_{cr}$)')


    def omega_EPW(self, axis, n_min, n_max, ax):

        """
        Finds the range of frequncies for scattred EPW waves
        from SRS for a given density range.

        axis = Sets oreintation of plot. Either = 'x' or 'y'.
        n_min = Minimum density (units : n_cr)
        n_max = Maximum density (units : n_cr)
        ax = Name of plot to plot onto (best to use plt.gca())
        """

        # Take number density points
        num_dens = np.linspace(n_min, n_max, 50)
        # Look for all angles
        angles = np.linspace(0, 360, 50)

        # Store max and min values
        omega_min = np.zeros(len(angles))
        omega_max = np.zeros(len(angles))
        
        for i in range(len(angles)):
            omega_epw = srs.srs_omega_EPW(num_dens, self.T_e, angles[i], self.lambda_0)
            # Extract min/max frequncies for given angle
            idx = np.where(np.isnan(omega_epw) == False)[0]
            omega_min[i] = omega_epw[idx].min()
            omega_max[i] = omega_epw[idx].max()
        
        # Plots bounds to get idea of range of frequcies relating to scattered EPW waves due to SRS
        if axis == 'x':
            ax.axhline(omega_min.min(), c=epw_colour)
            ax.axhline(omega_max.max(), c=epw_colour, label = f'SRS ({np.round(n_min,2)}-{np.round(n_max,2)}' + r' $n_{cr}$)')
        elif axis == 'y':
            ax.axvline(omega_min.min(), c=epw_colour)
            ax.axvline(omega_max.max(), c=epw_colour, label = f'SRS ({np.round(n_min,2)}-{np.round(n_max,2)}' + r' $n_{cr}$)')


    ########################################################################################################################
    # kx vs ky
    ########################################################################################################################


    def kx_vs_ky_EM(self, n_vals, angle_range, ax):

        """
        Finds the wavevector componants for scattred EM waves
        from SRS for a given densities.

        n_vals = List of density values to plot for (units : n_cr)
        angle_range = Range of scattering angles to plot for [theta_min, theta_max] (units : degrees)
        ax = Name of plot to plot onto (best to use plt.gca())
        """
        
        # For if n_vals is given of the form n_vals = {num}
        if np.isscalar(n_vals):
            # Extract k componants
            k_x, k_y = srs.srs_wns_EM_polar(n_vals, self.v_th, self.lambda_0,\
                                            angle_min = angle_range[0], angle_max = angle_range[-1])
            # Plot
            ax.plot(k_x, k_y, c=em_colour, label = f'SRS EM ($n_e$ = ' + f'{np.round(n_vals,2)}' + r' $n_{cr}$)')
        else:
            for i in range(len(n_vals)):
                # Extract k componants at each density
                k_x, k_y = srs.srs_wns_EM_polar(n_vals[i], self.v_th, self.lambda_0,\
                                                angle_min = angle_range[0], angle_max = angle_range[-1])
                # Plot
                if i == 0:
                    # Just so only one label is present in legend
                    ax.plot(k_x, k_y, c=em_colour, label = r'SRS EM ($n_e$ = '+ f'{np.round(np.array(n_vals).min(),2)} - {np.round(np.array(n_vals).max(),2)}' + r' $n_{cr}$)')
                else:
                    ax.plot(k_x, k_y, c=em_colour)


    def kx_vs_ky_EPW(self, n_vals, angle_range, ax):

        """
        Finds the wavevector componants for scattred EPW waves
        from SRS for a given densities.

        n_vals = List of density values to plot for (units : n_cr)
        angle_range = Range of scattering angles to plot for [theta_min, theta_max] (units : degrees)
        ax = Name of plot to plot onto (best to use plt.gca())
        """
        
        # For if n_vals is given of the form n_vals = {num}
        if np.isscalar(n_vals):
            # Extract k componants
            k_x, k_y = srs.srs_wns_EPW_polar(n_vals, self.v_th, self.lambda_0,\
                                            angle_min = angle_range[0], angle_max = angle_range[-1])
            # Plot
            ax.plot(k_x, k_y, c=epw_colour, label = f'SRS EPW ($n_e$ = ' + f'{np.round(n_vals,2)}' + r' $n_{cr}$)')
        else:
            for i in range(len(n_vals)):
                # Extract k componants at each density
                k_x, k_y = srs.srs_wns_EPW_polar(n_vals[i], self.v_th, self.lambda_0,\
                                                 angle_min = angle_range[0], angle_max = angle_range[-1])
                # Plot
                if i == 0:
                    # Just so only one label is present in legend
                    ax.plot(k_x, k_y, c=epw_colour, label = r'SRS EPW ($n_e$ = '+ f'{np.round(np.array(n_vals).min(),2)} - {np.round(np.array(n_vals).max(),2)}' + r' $n_{cr}$)')
                else:
                    ax.plot(k_x, k_y, c=epw_colour)

    
    ########################################################################################################################
    # x vs ky
    ########################################################################################################################

    def x_vs_ky_EM(self, n_e, theta, x, ax):

        """
        Plots the curve for expcted wavenumbers (ky) at a 
        spatial/desnity location. This plots the wavenumbers
        of the scattered EM wave.

        n_e = Densities to calculate wavenumbers and frequencies at (units : n_cr)
        theta = Angle of scatter that defines SRS process. Angle is defined
                as the angle between the EM wavevector and the laser wavevector. (units : degrees) 
        x = Spatial scale to plot over (units : m)
        ax = Name of plot to plot onto (best to use plt.gca())
        """

        # Extract componant and plot
        k_y = srs.srs_EM_k_y(self.v_th, n_e, theta, self.lambda_0)
        ax.plot(x, k_y, c=em_colour, label = r'SRS ($\theta$ =' + f'{np.round(theta, 1)}\N{DEGREE SIGN})')
        # Plot for conjugate of FFT
        ax.plot(x, -k_y, c=em_colour)

    def x_vs_ky_EPW(self, n_e, theta, x, ax):

        """
        Plots the curve for expcted wavenumbers (ky) at a 
        spatial/desnity location. This plots the wavenumbers
        of the scattered EPW.

        n_e = Densities to calculate wavenumbers and frequencies at (units : n_cr)
        theta = Angle of scatter that defines SRS process. Angle is defined
                as the angle between the EM wavevector and the laser wavevector. (units : degrees) 
        x = Spatial scale to plot over (units : m)
        ax = Name of plot to plot onto (best to use plt.gca())
        """

        # Extract componant and plot
        k_y = srs.srs_EPW_k_y(self.v_th, n_e, theta, self.lambda_0)
        ax.plot(x, k_y, c=epw_colour, label = r'SRS ($\theta$ =' + f'{np.round(theta, 1)}\N{DEGREE SIGN})')
        # Plot for conjugate of FFT
        ax.plot(x, -k_y, c=epw_colour)

        # Plot landau cutoff k_epw lambda_D ~ 0.3
        landau_cutoff_srs = srs.landau_cutoff_index(self.T_e, n_e, self.lambda_0, theta, cutoff = 0.3)
        if landau_cutoff_srs is not None:
            ax.axvline(x[landau_cutoff_srs], ls = '--', c=epw_colour, label = r'SRS Landau Cutoff ($\theta$ =' + f'{np.round(theta, 1)}\N{DEGREE SIGN})')


    ########################################################################################################################
    # x vs kx
    ########################################################################################################################


    def x_vs_kx_EM(self, n_e, theta, x, ax):

        """
        Plots the curve for expcted wavenumbers (kx) at a 
        spatial/desnity location. This plots the wavenumbers
        of the scattered EM wave.

        n_e = Densities to calculate wavenumbers and frequencies at (units : n_cr)
        theta = Angle of scatter that defines SRS process. Angle is defined
                as the angle between the EM wavevector and the laser wavevector. (units : degrees) 
        x = Spatial scale to plot over (units : m)
        ax = Name of plot to plot onto (best to use plt.gca())
        """

        # Extract componant and plot
        k_x = srs.srs_EM_k_x(self.v_th, n_e, theta, self.lambda_0)
        ax.plot(x, k_x, c=em_colour, label = r'SRS EM ($\theta$ =' + f'{np.round(theta, 1)}\N{DEGREE SIGN})')
        # Plot for conjugate of FFT
        ax.plot(x, -k_x, c=em_colour)

    def x_vs_kx_EPW(self, n_e, theta, x, ax):

        """
        Plots the curve for expcted wavenumbers (kx) at a 
        spatial/desnity location. This plots the wavenumbers
        of the scattered EPW.

        n_e = Densities to calculate wavenumbers and frequencies at (units : n_cr)
        theta = Angle of scatter that defines SRS process. Angle is defined
                as the angle between the EM wavevector and the laser wavevector. (units : degrees) 
        x = Spatial scale to plot over (units : m)
        ax = Name of plot to plot onto (best to use plt.gca())
        """

        # Extract componant and plot
        k_x = srs.srs_EPW_k_x(self.v_th, n_e, theta, self.lambda_0)
        ax.plot(x, k_x, c=epw_colour, label = r'SRS EPW ($\theta$ =' + f'{np.round(theta, 1)}\N{DEGREE SIGN})')
        # Plot for conjugate of FFT
        ax.plot(x, -k_x, c=epw_colour)

        # # Plot landau cutoff k_epw lambda_D ~ 0.3
        # landau_cutoff_srs = srs.landau_cutoff_index(self.T_e, n_e, self.lambda_0, theta, cutoff = 0.3)
        # if landau_cutoff_srs is not None:
        #     ax.axvline(x[landau_cutoff_srs], ls = '--', c=epw_colour, label = r'SRS Landau Cutoff ($\theta$ =' + f'{np.round(theta, 1)}\N{DEGREE SIGN})')

