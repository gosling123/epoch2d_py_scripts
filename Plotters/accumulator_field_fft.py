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

import Calculations.plasma_calculator as plasma
import Utils.get_accumulator_data as acc_data

# Useful prefix's
pico = 1e-12
femto = 1e-15
micron = 1e-6

# conversion from keV to kelvin
keV_to_K = (const.e*1e3)/const.k

class accumulator_field_fft_plots():

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

                # Basic plasma calculator class
                self.plasma_params = plasma.plasma_params(lambda_0=self.lambda_0, T_e=self.T_e)
                # Class to extract accumulator data
                self.field_data = acc_data.field_data(files=self.files, acc_flag=self.acc_flag, field_name=self.field_name, lambda_0=self.lambda_0, T_e = self.T_e)
                
                # Variables to define density
                self.density_profile = density_profile
                self.n_0 = n_0
                self.L_n = L_n
                # Class to extract density
                self.density = plasma.density(n_0=self.n_0, L_n=self.L_n)


    def plot_kx_vs_omega(self, t_min, t_max, n_min, n_max, k_range, omega_range):

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
        """
        

        # Create sub-directory to store results in
        dir_store = f'{self.output_path}/kx_vs_omega/{self.acc_flag}'
        try:
            os.mkdir(f'{self.output_path}/kx_vs_omega/')
        except:
            print('', end='\n')
        try:
            os.mkdir(dir_store)
        except:
            print('', end='\n')

        
        # Required to get correct X range
        self.field_data.setup_variables()
        self.X = self.field_data.X_centres

        # Find X locations for given density range
        if self.density_profile == 'exponential':
            self.X = self.density.x_locs_exponential(n_min=n_min, n_max=n_max, x = self.X)
        elif self.density_profile == 'linear':
            self.X = self.density.x_locs_linear(n_min=n_min, n_max=n_max, x = self.X)

        # Extract required data and perform required FFT process
        self.field_data.kx_vs_omega_fft(t_min=t_min, t_max=t_max, x_min=self.X.min(), x_max=self.X.max())

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
        cmap = cm.jet

        # Log norm cbar scaling
        vmax = field_fourier.max()
        vmin = vmax*1e-5

        # Plot image
        fft_plot = plt.imshow(field_fourier, cmap=cm.jet, norm = LogNorm(vmin=vmin, vmax=vmax), interpolation='gaussian', \
                              aspect='auto', extent=[k_space.min(), k_space.max(),omega_space.min(),omega_space.max()], origin="lower")

        # Set plot limits using inputs
        plt.ylim(omega_range[0], omega_range[1])
        plt.xlim(k_range[0], k_range[1])

        # Add colour bar
        cbar = plt.colorbar(fft_plot, ax = plt.gca())
        plt.ylabel(r"$\omega / \omega_0$")
        # Set the label automatically
        plt.xlabel(r'$c k_x / \omega_0$')
        cbar.set_label(r'|' +  str(self.field_name[-2:]) + r'$(k_x, \omega)$ |$^2$', rotation=270, labelpad=25)

        plt.tight_layout()


        time_min = np.round(self.field_data.times.min() / pico, 2)
        time_max = np.round(self.field_data.times.max() / pico, 2)

        plot_name = f'{self.field_name[-2:]}_{time_min}-{time_max}_ps_{n_min}-{n_max}.png'
        print(f'Saving Figure to {dir_store}/{plot_name}')
        fig.savefig(f'{dir_store}/{plot_name}')


    # def plot_ky_vs_omega(self, t_min, time_max, y_min, y_max):

        

    # def plot_x_vs_omega(self, t_min, time_max, ne_min, ne_max):

    
    # def plot_omega_vs_y(self, t_min, time_max, y_min, y_max):

        

    # def plot_omega_vs_time(self, t_min, time_max):




