#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

accumulator_field_fft.py

File which houses class that defines plotting routines
for fourier transformed field strips.

"""



# Import libraries
import sys

sys.path.append("..")

import sdf
import matplotlib.pyplot as plt
import numpy as np
import glob
import scipy.constants as const
from matplotlib.colors import LogNorm
from matplotlib import cm
import os
from datetime import datetime

import calculations.plasma_calculator as plasma
import calculations.laser_calculator as laser
import plotters.srs_plots as srs
import plotters.tpd_plots as tpd
import get_data.accumulator_field_spectra as field_spectra



# Colour map style
cmap = cm.jet

# Useful prefix's
pico = 1e-12
femto = 1e-15
micron = 1e-6

# conversion from keV to kelvin
keV_to_K = (const.e*1e3)/const.k

class plots():

    """ Class that houses fft plotting routines for accumulated field data"""

    def __init__(self, files, acc_flag, field_name, output_path, \
                lambda_0, T_e, density_profile, n_0, L_n):

                """
                Class constructor function

                files = Organised list of required files to read
                acc_flag = Flag for which accumulator strip to use
                field_name = Particular field to take fft of. Read from sdf file directory with naming style
                             "Electric_Field_E{x,y or z}", and similar for magnetic field.
                output_path = Directory to store figures in
                lambda_0 = Vacuum laser wavelength (units : m)
                T_e = Electron temperature (units : K)
                density_profile = String which sets type of density profile.
                                  Either 'exponential' or 'linear'.
                n_0 = Number density at x = 0 (or x_min) (units : n_cr)
                L_n = Density scale length (units : m)
                """

                # File reading setup
                self.files = files
                self.nfile = len(files)
                self.field_name = field_name
                self.acc_flag = acc_flag
                self.output_path = output_path

                # Base plasma/laser parameters required
                self.lambda_0 = lambda_0
                self.T_e = T_e

                # Thermal speed for LPI curves
                self.v_th = plasma.electron_thermal_speed(self.T_e)

                # Class to extract accumulator data
                self.field_data = field_spectra.data(files=self.files, acc_flag=self.acc_flag, field_name=self.field_name, lambda_0=self.lambda_0, T_e = self.T_e)
                
                # Variables to define density
                self.density_profile = density_profile
                self.n_0 = n_0
                self.L_n = L_n
                
                
    ########################################################################################################################
    # kx vs omega plot
    ########################################################################################################################

    def plot_kx_vs_omega(self, t_min, t_max, n_min, n_max, k_range, omega_range,\
                         plot_srs=False, srs_angle=180, plot_tpd=False, tpd_angle='max_lin_growth'):
                                                            

        """
        Function to plot kx_vs_omega for chosen time, density. 
        The plot is also cut to a defined wavenumber and frequency
        range.

        
        t_min = Minimum time to plot around (units : s)
        t_max = Maximum time to plot around (units : s)
        n_min = Minimum electron density to plot around (units : n_cr)
        n_max = Maximum electron density to plot around (units : n_cr)
        k_range = Range of wavenumbers to plot given as a list of the
                  form [k_min, k_max]. (units : k_0)
        omega_range = Range of frequencies to plot given as a list of the
                  form [omega_min, omega_max]. (units : omega_0)
        plot_srs = Logical flag to plot wavenumber/omega values due to 
                   SRS in given density region.
        srs_angle = Scattering angle of EM wave to plot SRS curves for (units : degrees)
        plot_tpd = Logical flag to plot wavenumber/omega values due to 
                   TPD in given density region.
        tpd_angle = Centred scatter angle to plot TPD curves for. (units : degrees) 
                    For angle corresponding to max lin growth set to 'max_lin_growth'
        """
        

        # Create sub-directory to store results in
        try:
            os.mkdir(f'{self.output_path}/kx_vs_omega/')
        except:
            print('', end='\n')
        
        # Required to get correct X range
        self.field_data.setup_variables()
        X = self.field_data.X_centres

        # Find X locations for given density range
        if self.density_profile == 'exponential':
            X = plasma.x_locs_exponential(n_0=self.n_0, L_n=self.L_n, x = X, n_min=n_min, n_max=n_max)
            n_e = plasma.density_exponential(self.n_0, self.L_n, X) # For LPI curves
        elif self.density_profile == 'linear':
            X = plasma.x_locs_linear(n_0=self.n_0, L_n=self.L_n, x = X, n_min=n_min, n_max=n_max)
            n_e = plasma.density_linear(self.n_0, self.L_n, X) # For LPI curves
        
        # Extract required data and perform required FFT process
        self.field_data.kx_vs_omega_fft(t_min=t_min, t_max=t_max, x_min=X.min(), x_max=X.max())

        # Required data
        field_fourier = self.field_data.field_fourier
        k_space = self.field_data.k_space
        omega_space = self.field_data.omega_space 

        # Plot for given omega region
        idx_omega_min = np.where(omega_space - omega_range[0] >= 0)[0][0]
        idx_omega_max = np.where(omega_space - omega_range[1] >= 0)[0]

        if len(idx_omega_max) == 0:
            print('Max value of omega range set to high, default to maximum value')
            idx_omega_max = -1
        else:
            idx_omega_max = idx_omega_max[0]

        # Plot for given k region
        idx_k_min = np.where(k_space - k_range[0] >= 0)[0][0]
        idx_k_max = np.where(k_space - k_range[1] >= 0)[0]

        if len(idx_k_max) == 0:
            print('Max value of k range set to high, default to maximum value')
            idx_k_max = -1
        else:
            idx_k_max = idx_k_max[0]

        # Slice data into given range
        field_fourier = field_fourier[idx_omega_min:idx_omega_max, idx_k_min:idx_k_max]
        omega_space = omega_space[idx_omega_min:idx_omega_max]
        k_space = k_space[idx_k_min:idx_k_max]

        # Begin plotting
        print('Plotting kx_vs_omega')
        fig = plt.figure() 
    
        # Log norm cbar scaling
        vmax = field_fourier.max()
        vmin = vmax*1e-5

        # Plot image
        fft_plot = plt.imshow(field_fourier, cmap=cmap, norm = LogNorm(vmin=vmin, vmax=vmax), interpolation='gaussian', \
                              aspect='auto', extent=[k_space.min(), k_space.max(),omega_space.min(),omega_space.max()], origin="lower")

        # Set plot limits using inputs
        plt.ylim(omega_space.min(),omega_space.max())
        plt.xlim(k_space.min(), k_space.max())

        # Add colour bar
        cbar = plt.colorbar(fft_plot, ax = plt.gca())
        plt.ylabel(r"$\omega / \omega_0$")
        # Set the label automatically
        plt.xlabel(r'$c k_x / \omega_0$')
        cbar.set_label(r'|' +  str(self.field_name[-2:]) + r'$(k_x, \omega)$ |$^2$', rotation=270, labelpad=25)

        # Plot LPI curves:

        if plot_srs:
            print('Plotting SRS curves')
            # SRS plotting class
            plots = srs.plots(self.T_e, n_e, srs_angle, self.lambda_0)
            if self.field_name[-2:] == 'Bz':
                # Don't plot EPW for pure EM componant
                plots.kx_vs_omega_EM(ax=plt.gca())
            else:
                plots.kx_vs_omega_EM(ax=plt.gca())
                plots.kx_vs_omega_EPW(ax=plt.gca())
       
        if self.field_name[-2:] == 'Bz':
            # Again, don't plot EPW for pure EM componant
            plot_tpd = False
        
        if plot_tpd:
            print('Plotting TPD curves')
            # SRS plotting class
            plots = tpd.plots(self.T_e, n_e, tpd_angle, self.lambda_0)
    
            plots.kx_vs_omega(ax=plt.gca())
        
        plt.legend()
        
        # Save figure
        time_min = np.round(self.field_data.times.min() / pico, 2)
        time_max = np.round(self.field_data.times.max() / pico, 2)

        plot_name = f'{self.field_name[-2:]}_{self.acc_flag}_{time_min}-{time_max}_ps_{n_min}-{n_max}_n_cr.png'
        print(f'Saving Figure to {self.output_path}/kx_vs_omega/{plot_name}')
        plt.tight_layout()
        fig.savefig(f'{self.output_path}/kx_vs_omega/{plot_name}')

        # Append to output_fig file to keep track
        output_file = open(f'{self.output_path}/output_figs.txt',"a")
        now = datetime.now()
        # dd/mm/YY H:M:S
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        output_file.write(f'\nSaved kx_vs_omega/{plot_name} at {dt_string}')
        output_file.close()

      
    ########################################################################################################################
    # x vs omega plot
    ########################################################################################################################

    def plot_x_vs_omega(self, t_min, t_max, n_min, n_max, omega_range):
                        # plot_srs=False, srs_angle=180, plot_tpd=False, tpd_angle='max_lin_growth'):

        """
        Function to plot x_vs_omega for chosen time, density. 
        The plot is also cut to a defined frequency
        range.

        
        t_min = Minimum time to plot around (units : s)
        t_max = Maximum time to plot around (units : s)
        n_min = Minimum electron density to plot around (units : n_cr)
        n_max = Maximum electron density to plot around (units : n_cr)
        omega_range = Range of frequencies to plot given as a list of the
                  form [omega_min, omega_max]. (units : omega_0)
        plot_srs = Logical flag to plot wavenumber/omega values due to 
                   SRS in given density region.
        srs_angle = Scattering angle of EM wave to plot SRS curves for (units : degrees)
        plot_tpd = Logical flag to plot wavenumber/omega values due to 
                   TPD in given density region.
        tpd_angle = Centred scatter angle to plot TPD curves for. (units : degrees) 
                    For angle corresponding to max lin growth set to 'max_lin_growth'
        """

         # Create sub-directory to store results in
        try:
            os.mkdir(f'{self.output_path}/x_vs_omega/')
        except:
            print('', end='\n')
        
        # Required to get correct X range
        self.field_data.setup_variables()
        X = self.field_data.X_centres

        # Find X locations for given density range
        if self.density_profile == 'exponential':
            X = plasma.x_locs_exponential(n_0=self.n_0, L_n=self.L_n, x = X, n_min=n_min, n_max=n_max)
            n_e = plasma.density_exponential(self.n_0, self.L_n, X) # For LPI curves
        elif self.density_profile == 'linear':
            X = plasma.x_locs_linear(n_0=self.n_0, L_n=self.L_n, x = X, n_min=n_min, n_max=n_max)
            n_e = plasma.density_linear(self.n_0, self.L_n, X) # For LPI curves
        
        # Extract required data and perform required FFT process
        self.field_data.x_vs_omega_fft(t_min=t_min, t_max=t_max, x_min=X.min(), x_max=X.max())

        # Required data cut by x_min x_max
        field_fourier = self.field_data.field_fourier
        omega_space = self.field_data.omega_space
        X = self.field_data.X_centres

        # Plot for given omega region
        idx_omega_min = np.where(omega_space - omega_range[0] >= 0)[0][0]
        idx_omega_max = np.where(omega_space - omega_range[1] >= 0)[0]

        if len(idx_omega_max) == 0:
            print('Max value of omega range set to high, default to maximum value')
            idx_omega_max = -1
        else:
            idx_omega_max = idx_omega_max[0]

        # Slice data into given range
        field_fourier = field_fourier[:,idx_omega_min:idx_omega_max]
        omega_space = omega_space[idx_omega_min:idx_omega_max]

        # Begin plotting
        print('Plotting x_vs_omega')
        fig, ax = plt.subplots()
    
        # Log norm cbar scaling
        vmax = field_fourier.max()
        vmin = vmax*1e-5

        # Plot image
        fft_plot = ax.imshow(field_fourier.T, cmap=cmap, norm = LogNorm(vmin=vmin, vmax=vmax), interpolation='gaussian', \
                              aspect='auto', extent=[X.min()/micron, X.max()/micron,omega_space.min(),omega_space.max()], origin="lower")

        # Set plot limits using inputs
        ax.set_ylim(omega_space.min(),omega_space.max())
        ax.set_xlim(X.min()/micron, X.max()/micron)

        # Add colour bar
        cbar = plt.colorbar(fft_plot, ax = plt.gca())
        ax.set_ylabel(r"$\omega / \omega_0$")
        # Set the label automatically
        ax.set_xlabel(r'$ X \, (\mu \, m)$')
        cbar.set_label(r'|' +  str(self.field_name[-2:]) + r'$(x, \omega)$ |$^2$', rotation=270, labelpad=25)


        # Add density scale on top x axis
        new_tick_locations = np.linspace(X.min()/micron, X.max()/micron, 4)

        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(new_tick_locations)

        if self.density_profile == 'exponential':
            dens_ticks = plasma.density_exponential(self.n_0, self.L_n, new_tick_locations * micron)
        elif self.density_profile == 'linear':
            dens_ticks = plasma.density_linear(self.n_0, self.L_n, new_tick_locations * micron)

        ax2.set_xticklabels(np.round(dens_ticks,2))
        ax2.set_xlabel(r"$n_e / n_{cr}$")

        # Save figure
        time_min = np.round(self.field_data.times.min() / pico, 2)
        time_max = np.round(self.field_data.times.max() / pico, 2)

        plot_name = f'{self.field_name[-2:]}_{self.acc_flag}_{time_min}-{time_max}_ps_{n_min}-{n_max}_n_cr.png'
        print(f'Saving Figure to {self.output_path}/x_vs_omega/{plot_name}')
        plt.tight_layout()
        fig.savefig(f'{self.output_path}/x_vs_omega/{plot_name}')

        # Append to output_fig file to keep track
        output_file = open(f'{self.output_path}/output_figs.txt',"a")
        now = datetime.now()
        # dd/mm/YY H:M:S
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        output_file.write(f'\nSaved x_vs_omega/{plot_name} at {dt_string}')
        output_file.close()


    ########################################################################################################################
    # omega vs y plot
    ########################################################################################################################

    def plot_omega_vs_y(self, t_min, t_max, y_min, y_max, omega_range):
                        
        """
        Function to plot omega_vs_y for chosen time, density. 
        The plot is also cut to a defined frequency
        range.

        
        t_min = Minimum time to plot around (units : s)
        t_max = Maximum time to plot around (units : s)
        y_min = Minimum y-posistion to plot around (units : m)
        y_max = Maximum y-posistion to plot around (units : m)
        omega_range = Range of frequencies to plot given as a list of the
                  form [omega_min, omega_max]. (units : omega_0)
        """

         # Create sub-directory to store results in
        try:
            os.mkdir(f'{self.output_path}/omega_vs_y/')
        except:
            print('', end='\n')
        
        # Extract required data and perform required FFT process
        self.field_data.omega_vs_y_fft(t_min=t_min, t_max=t_max, y_min=y_min, y_max=y_max)

        # Required data
        field_fourier = self.field_data.field_fourier
        omega_space = self.field_data.omega_space
        Y = self.field_data.Y_centres

        # Plot for given omega region
        idx_omega_min = np.where(omega_space - omega_range[0] >= 0)[0][0]
        idx_omega_max = np.where(omega_space - omega_range[1] >= 0)[0]

        if len(idx_omega_max) == 0:
            print('Max value of omega range set to high, default to maximum value')
            idx_omega_max = -1
        else:
            idx_omega_max = idx_omega_max[0]

        # Slice data into given range
        field_fourier = field_fourier[:,idx_omega_min:idx_omega_max]
        omega_space = omega_space[idx_omega_min:idx_omega_max]

        # Begin plotting
        print('Plotting omega_vs_y')
        fig, ax = plt.subplots()
    
        # Log norm cbar scaling
        vmax = field_fourier.max()
        vmin = vmax*1e-5

        # Plot image
        fft_plot = ax.imshow(field_fourier, cmap=cmap, norm = LogNorm(vmin=vmin, vmax=vmax), interpolation='gaussian', \
                              aspect='auto', extent=[omega_space.min(),omega_space.max(),Y.min()/micron, Y.max()/micron], origin="lower")

        # Set plot limits using inputs
        ax.set_xlim(omega_space.min(),omega_space.max())
        ax.set_ylim(Y.min()/micron, Y.max()/micron)

        # Add colour bar
        cbar = plt.colorbar(fft_plot, ax = plt.gca())
        ax.set_xlabel(r"$\omega / \omega_0$")
        # Set the label automatically
        ax.set_ylabel(r'$ Y \, (\mu \, m)$')
        cbar.set_label(r'|' +  str(self.field_name[-2:]) + r'$(\omega, y)$ |$^2$', rotation=270, labelpad=25)

        # Save figure
        time_min = np.round(self.field_data.times.min() / pico, 2)
        time_max = np.round(self.field_data.times.max() / pico, 2)

        plot_name = f'{self.field_name[-2:]}_{self.acc_flag}_{time_min}-{time_max}_ps.png'
        print(f'Saving Figure to {self.output_path}/omega_vs_y/{plot_name}')
        plt.tight_layout()
        fig.savefig(f'{self.output_path}/omega_vs_y/{plot_name}')

        # Append to output_fig file to keep track
        output_file = open(f'{self.output_path}/output_figs.txt',"a")
        now = datetime.now()
        # dd/mm/YY H:M:S
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        output_file.write(f'\nSaved omega_vs_y/{plot_name} at {dt_string}')
        output_file.close()

    ########################################################################################################################
    # omega vs time plot
    ########################################################################################################################

    # def plot_omega_vs_time(self, t_min, time_max):




