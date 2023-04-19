#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

grid_field_spectra.py

File which houses class that defines plotting routines
for fourier transformed field strips.

"""



# Import libraries
import sys

sys.path.append("..")

import sdf
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as const
from matplotlib.colors import LogNorm
from matplotlib import cm
import os
from datetime import datetime

import calculations.plasma_calculator as plasma
import calculations.laser_calculator as laser
from plotters import srs_plots as srs
from plotters import tpd_plots as tpd
import get_data.grid_field_spectra as field_spectra

# Colour map style
cmap = cm.jet

# Useful prefix's
pico = 1e-12
femto = 1e-15
micron = 1e-6

# conversion from keV to kelvin
keV_to_K = (const.e*1e3)/const.k

class plots():

    """ Class that houses fft plotting routines for grid field spectra data"""

    def __init__(self, files, field_name, output_path, \
                lambda_0, T_e, density_profile, n_0, L_n):

        """
        Class constructor function

        files = Organised list of required files to read
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
        self.output_path = output_path

        # Base plasma/laser parameters required
        self.lambda_0 = lambda_0
        self.T_e = T_e

        # Thermal speed for LPI curves
        self.v_th = plasma.electron_thermal_speed(self.T_e)

        # Class to extract accumulator data
        self.field_data = field_spectra.data(files=self.files, field_name=self.field_name, lambda_0=self.lambda_0, T_e = self.T_e)
                
        # Variables to define density
        self.density_profile = density_profile
        self.n_0 = n_0
        self.L_n = L_n

    ########################################################################################################################
    # kx vs ky plot
    ########################################################################################################################

    def plot_kx_vs_ky(self, snap_time, n_min, n_max, kx_range, ky_range,\
                      plot_srs=False, n_srs=[0.1, 0.18], plot_tpd=False, n_tpd=[0.2,0.23]):

        
        # Create sub-directory to store results in
        try:
            os.mkdir(f'{self.output_path}/kx_vs_ky/')
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

        self.field_data.kx_vs_ky_fft(snap_time=snap_time, x_min=X.min(), x_max=X.max())

    
        field_fourier = self.field_data.field_fourier
        k_x = self.field_data.k_x
        k_y = self.field_data.k_y

        # Plot for given k region
        kx_indicies = np.where((k_x >= kx_range[0]) & (k_x <= kx_range[-1]))[0]
        if len(kx_indicies) == 0:
            sys.exit('ERROR (kx_vs_ky): Inputted wavenumber (kx) scale is incorrect. Please ensure it is units of k_0 (laser) and in range.')
        # Plot for given k region
        ky_indicies = np.where((k_y >= ky_range[0]) & (k_y <= ky_range[-1]))[0]
        if len(kx_indicies) == 0:
            sys.exit('ERROR (kx_vs_ky): Inputted wavenumber (ky) scale is incorrect. Please ensure it is units of k_0 (laser) and in range.')

        k_x = k_x[kx_indicies]
        k_y = k_y[ky_indicies]
        field_fourier = field_fourier[kx_indicies[0]:kx_indicies[-1], ky_indicies[0]:ky_indicies[-1]]

        print(f'Plotting kx_vs_ky')

       # Colour plot scale (lognorm)
        vmax = field_fourier.max()
        vmin = vmax*1e-4
    
        fig, ax = plt.subplots()

        fft_plot = ax.imshow(field_fourier.T, cmap=cmap, norm = LogNorm(vmin=vmin, vmax=vmax), interpolation='gaussian', \
                             aspect='auto', extent=[k_x.min(), k_x.max(),k_y.min(),k_y.max()], origin="lower")

        # Colour bar to get current axis
        cbar = plt.colorbar(fft_plot, ax = plt.gca())

        cbar.set_label(r'|' +  str(self.field_name[-2:]) + r'$(k_x, k_y)$ |$^2$', rotation=270, labelpad=25)
        plt.xlabel(r'$c k_x / \omega_0$')
        plt.ylabel(r'$c k_y / \omega_0$')
        plt.ylim(k_y.min(), k_y.max())
        plt.xlim(k_x.min(), k_x.max())

        # Plot LPI curves
        if plot_srs:
            print('Plotting SRS curves')
            # SRS plotting class
            plots = srs.plots(self.T_e, self.lambda_0)
            if self.field_name[-2:] == 'Bz':
                # Don't plot EPW for pure EM componant
                plots.kx_vs_ky_EM(n_vals=n_srs, ax=plt.gca())
            else:
                plots.kx_vs_ky_EM(n_vals=n_srs, ax=plt.gca())
                plots.kx_vs_ky_EPW(n_vals=n_srs, ax=plt.gca())

        if self.field_name[-2:] == 'Bz':
            # Again, don't plot EPW for pure EM componant
            plot_tpd = False
        
        if plot_tpd:
            print('Plotting TPD curves')
            # TPD plotting class
            plots = tpd.plots(self.T_e, self.lambda_0)
    
            plots.kx_vs_ky(n_vals=n_tpd, ax=plt.gca())
        
        plt.legend()
       
        # Save figure
        time = np.round(self.field_data.time / pico, 3)
        plot_name = f'{self.field_name[-2:]}_{time}_ps_{n_min}-{n_max}_n_cr.png'
        print(f'Saving Figure to {self.output_path}/kx_vs_ky/{plot_name}')
        plt.tight_layout()
        fig.savefig(f'{self.output_path}/kx_vs_ky/{plot_name}')

        # Append to output_fig file to keep track
        output_file = open(f'{self.output_path}/output_figs.txt',"a")
        now = datetime.now()
        # dd/mm/YY H:M:S
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        output_file.write(f'\nSaved kx_vs_ky/{plot_name} at {dt_string}')
        output_file.close()

    ########################################################################################################################
    # x vs ky plot
    ########################################################################################################################

    def plot_x_vs_ky(self, snap_time, n_min, n_max, k_range):

        
        # Create sub-directory to store results in
        try:
            os.mkdir(f'{self.output_path}/x_vs_ky/')
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

        self.field_data.x_vs_ky_fft(snap_time=snap_time, x_min=X.min(), x_max=X.max())

    
        field_fourier = self.field_data.field_fourier
        k_y = self.field_data.k_y
        X = self.field_data.X_centres

        # Plot for given k region
        k_indicies = np.where((k_y >= k_range[0]) & (k_y <= k_range[-1]))[0]
        if len(k_indicies) == 0:
            sys.exit('ERROR (x_vs_ky): Inputted wavenumber (ky) scale is incorrect. Please ensure it is units of k_0 (laser) and in range.')

        field_fourier = field_fourier[:, k_indicies]
        k_y = k_y[k_indicies]

        print(f'Plotting x_vs_ky')

        # Colour plot scale (lognorm)
        vmax = field_fourier.max()
        vmin = vmax*1e-3

        fig, ax = plt.subplots()

        fft_plot = ax.imshow(field_fourier.T, cmap=cmap, norm = LogNorm(vmin=vmin, vmax=vmax), interpolation='gaussian', \
                             aspect='auto', extent=[X.min()/micron, X.max()/micron, k_y.min(), k_y.max()], origin="lower")
        # Colour bar to get current axis
        cbar = plt.colorbar(fft_plot, ax = plt.gca())
        # Label plot
        cbar.set_label(r'|' +  str(self.field_name[-2:]) + r'$(x, k_y)$ |$^2$', rotation=270, labelpad=25)
        ax.set_xlabel(r'$X (\mu m)$')
        ax.set_ylabel(r'$c k_y / \omega_0$')

        ax.set_xlim(X.min()/micron, X.max()/micron)
        ax.set_ylim(k_y.min(), k_y.max())

        # Add density scale on top x axis
        new_tick_locations = np.linspace(X.min(), X.max(), 4)

        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(new_tick_locations/micron)

        if self.density_profile == 'exponential':
            dens_ticks = plasma.density_exponential(self.n_0, self.L_n, new_tick_locations)
        elif self.density_profile == 'linear':
            dens_ticks = plasma.density_linear(self.n_0, self.L_n, new_tick_locations)

        ax2.set_xticklabels(np.round(dens_ticks,2))
        ax2.set_xlabel(r"$n_e / n_{cr}$")

        # Save figure
        time = np.round(self.field_data.time / pico, 3)
        plot_name = f'{self.field_name[-2:]}_{time}_ps_{n_min}-{n_max}_n_cr.png'
        print(f'Saving Figure to {self.output_path}/x_vs_ky/{plot_name}')
        plt.tight_layout()
        fig.savefig(f'{self.output_path}/x_vs_ky/{plot_name}')

        # Append to output_fig file to keep track
        output_file = open(f'{self.output_path}/output_figs.txt',"a")
        now = datetime.now()
        # dd/mm/YY H:M:S
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        output_file.write(f'\nSaved x_vs_ky/{plot_name} at {dt_string}')
        output_file.close()

            
    ########################################################################################################################
    # x vs kx plot
    ########################################################################################################################

    def plot_x_vs_kx(self, snap_time, n_min, n_max, x_window, x_bins, k_range):


        # Create sub-directory to store results in
        try:
            os.mkdir(f'{self.output_path}/x_vs_kx/')
        except:
            print('', end='\n')

        self.field_data.x_vs_kx_stft(snap_time, x_window, x_bins, k_range)

        X = self.field_data.x_data
        field_fourier = self.field_data.field_fourier
        k_x = self.field_data.k_x

        # Find X locations for given density range
        if self.density_profile == 'exponential':
            n_e = plasma.density_exponential(self.n_0, self.L_n, X)
        elif self.density_profile == 'linear':
            n_e = plasma.density_linear(self.n_0, self.L_n, X)

        # Plot for given density region
        x_indicies = np.where((n_e >= n_min) & (n_e <= n_max))[0]
        if len(x_indicies) == 0:
            sys.exit('ERROR (x_vs_kx): Inputted density scale is incorrect. Please ensure it is units of n_cr and in range.')

        X = X[x_indicies]
        field_fourier = field_fourier[x_indicies,:]

        print(f'Plotting x_vs_kx')

        # Colour plot scale (lognorm)
        vmax = field_fourier.max()
        vmin = vmax*1e-3

        fig, ax = plt.subplots()


        fft_plot = ax.imshow(field_fourier.T, cmap=cmap, norm = LogNorm(vmin=vmin, vmax=vmax), interpolation='gaussian', \
                             aspect='auto', extent=[X.min()/micron, X.max()/micron,k_x.min(),k_x.max()], origin="lower")
        # Colour bar to get current axis
        cbar = plt.colorbar(fft_plot, ax = plt.gca())

        cbar.set_label(r'|' +  str(self.field_name[-2:]) + r'$(x, k_x)$ |$^2$', rotation=270, labelpad=25)
        ax.set_xlabel(r'$X (\mu m)$')
        ax.set_ylabel(r'$c k_x / \omega_0$')
        ax.set_ylim(k_x.min(),k_x.max())
        ax.set_xlim(X.min()/micron, X.max()/micron)

        # Add density scale on top x axis
        new_tick_locations = np.linspace(X.min(), X.max(), 4)

        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(new_tick_locations/micron)

        if self.density_profile == 'exponential':
            dens_ticks = plasma.density_exponential(self.n_0, self.L_n, new_tick_locations)
        elif self.density_profile == 'linear':
            dens_ticks = plasma.density_linear(self.n_0, self.L_n, new_tick_locations)

        ax2.set_xticklabels(np.round(dens_ticks,2))
        ax2.set_xlabel(r"$n_e / n_{cr}$")

        # Save figure
        time = np.round(self.field_data.time / pico, 3)
        plot_name = f'{self.field_name[-2:]}_{time}_ps_{n_min}-{n_max}_n_cr.png'
        print(f'Saving Figure to {self.output_path}/x_vs_kx/{plot_name}')
        plt.tight_layout()
        fig.savefig(f'{self.output_path}/x_vs_kx/{plot_name}')

        # Append to output_fig file to keep track
        output_file = open(f'{self.output_path}/output_figs.txt',"a")
        now = datetime.now()
        # dd/mm/YY H:M:S
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        output_file.write(f'\nSaved x_vs_kx/{plot_name} at {dt_string}')
        output_file.close()
    
        