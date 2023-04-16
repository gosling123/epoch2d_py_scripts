#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Import libraries

"""
get_accumulator_data.py

File which houses the classes which extracts required
accumulator data for plotting. Mainly field plots.

"""

import sys

sys.path.append("..")

import sdf
import matplotlib.pyplot as plt
import numpy as np
import glob
import scipy.constants as const
from matplotlib.colors import LogNorm
from matplotlib import cm

import Calculations.plasma_calculator as plasma


def print_progress_bar(index, total, label):
    """
    prints progress bar for loops.

    :param index : current index of loop
    :param total : total number of indicies to loop over
    :param label : print statement next to progress bar

    """
    n_bar = 80 # Progress bar width
    progress = index / total
    sys.stdout.write('\r')
    sys.stdout.write(f"[{'=' * int(n_bar * progress):{n_bar}s}] {int(100 * progress)}%  {label}")
    sys.stdout.flush()

class field_data:


    def __init__(self, files, acc_flag, field_name, lambda_0, T_e):


        """
        Class constructor function

        files = Organised list of required files to read
        acc_flag = Flag for which accumulator strip to use
        field_name = Particular field to take fft of. Read from sdf file directory with naming style
                             "Electric_Field_E{x,y or z}", and similar for magnetic field.

        lambda_0 = Vacuum laser wavelength (units : m)
        T_e = Electron temperature (units : K)
             
        """
        
        # File reading setup
        self.files = files
        self.nfiles = len(files)
        self.field_name = field_name
        self.acc_flag = acc_flag

        # Field variable name in sdf file dictionary
        self.var_name = f"{self.field_name}_Acc_{self.acc_flag}_acc_field"

        # Base plasma/laser parameters required
        self.lambda_0 = lambda_0
        self.T_e = T_e
        # Basic plasma calculator class
        self.plasma_params = plasma.plasma_params(lambda_0=self.lambda_0, T_e=self.T_e)
        # Required laser normalisations
        self.k_0 = self.plasma_params.k_0
        self.omega_0 = self.plasma_params.omega_0
        # Field normalisation constant
        self.field_norm = const.e / (const.m_e * self.omega_0 * const.c)


    def setup_variables(self):

        """
        Function that performs the necessary setup 
        and extraction of key grid variables to be 
        able to store required field data.
        
        """

        # Read last file to get end time
        d = sdf.read(self.files[-1])
        # End time of simulation
        self.t_end = d.__dict__['Header']['time']
        
        # Define grid varibales
        var = d.__dict__[self.var_name]

        # Number of grid cells
        self.N_x = var.dims[0]
        self.N_y = var.dims[1]

        # X-space
        self.X_edges = var.grid.data[0]
        self.X_centres = 0.5 * (self.X_edges[1:] + self.X_edges[:-1])
        self.dx = self.X_edges[1] - self.X_edges[0]
        # Y-space
        self.Y_edges = var.grid.data[1]
        self.Y_centres = 0.5 * (self.Y_edges[1:] + self.Y_edges[:-1])
        self.dy = self.Y_edges[1] - self.Y_edges[0]

        del d


    
    # def load_x_strip_field_data(self, t_min, t_max, y_min, y_max):

    #     self.setup_variables()

    # if self.N_y == 1:
    #         sys.exit(f'Cannot Perform loading of x-strip data for {self.acc_flag} accumulator')

    
    def load_y_strip_field_data(self, t_min, t_max, x_min, x_max):

        """
        Function that loads the required field data for given 
        y-strip.

        t_min = Minimum time to plot around (units : s)
        t_max = Maximum time to plot around (units : s)
        x_min = Minimum x posistion to plot around (units : m)
        x_max = Maximum x-posistion to plot around (units : m)
        """
        
        # Get required data to store field data
        self.setup_variables()

        # Prevention from using the wrong accumulator strip
        if self.N_x == 1:
            sys.exit(f'Cannot Perform loading of y-strip data for {self.acc_flag} accumulator')

        # Get data for X range
        x_idx_min = np.where(self.X_centres - x_min >= 0)[0][0]
        x_idx_max = np.where(self.X_centres - x_max >= 0)[0]
        if len(x_idx_max) == 0:
            x_idx_max = -1
        else:
            x_idx_max = x_idx_max[0]
        self.X_centres = self.X_centres[x_idx_min:x_idx_max]
        self.N_x = len(self.X_centres)

        # Get data for time range
        file_index_min = int(t_min/self.t_end * self.nfiles)
        file_index_max = int(t_max/self.t_end * self.nfiles)
        # Set file names to be in range
        self.files_cut = self.files[file_index_min:file_index_max+1]
        self.nfiles_cut = len(self.files_cut)

        # Set empty arrays to store times and field data
        self.times = np.empty(shape = 0)
        self.field_data = np.empty([0, self.N_x])

        print('Extracting Required Data')
        # Read through sdf files and store each result
        for i in range(self.nfiles_cut):
            #print_progress_bar(i+1, self.nfiles_cut, f"Reading {self.files_cut[i]} ({i+1}/{self.nfiles_cut})")
        
            d = sdf.read(self.files_cut[i])
            field_acc = d.__dict__[self.var_name]
            time_acc = field_acc.grid.data[-1]
	    
	        # Store by concatenation to get correct format
            self.times = np.concatenate((self.times, time_acc))
            self.field_data = np.concatenate((self.field_data, field_acc.data.T[:,0,x_idx_min:x_idx_max]*self.field_norm))    

        print('', end='\n')
        del d

        # Temporal resolution
        self.N_t = len(self.times)
        self.dt = self.times[1] - self.times[0]



    def kx_vs_omega_fft(self, t_min, t_max, x_min, x_max):

        """
        Function extracts the required fourier transform to
        plot kx_vs_omega.

        t_min = Minimum time to plot around (units : s)
        t_max = Maximum time to plot around (units : s)
        x_min = Minimum x posistion to plot around (units : m)
        x_max = Maximum x-posistion to plot around (units : m)
        """
        
        # Load required data
        self.load_y_strip_field_data(t_min, t_max, x_min, x_max)
        
        # 2D window
        # Apply windowing to FFT (hanning usually the best)
        w_t = np.hanning(self.N_t)
        w_x = np.hanning(self.N_x)
        window_func = np.outer(w_t, w_x)
        # Coefficient to normalise amplitudes
        amp_coeff = 2.0 / (self.N_t * np.mean(w_t) * self.N_x * np.mean(w_x))

        # Fourier space
        self.omega_space = np.fft.fftshift(np.fft.fftfreq(self.N_t, self.dt / 2.0 / np.pi)) / self.omega_0
        self.k_space = np.fft.fftshift(np.fft.fftfreq(self.N_x, self.dx / 2.0 / np.pi)) / self.k_0

        print('Extracting kx-omega fft')
        # use fftshift to get of the form -k => k
        field_fft_2d = np.fft.fftshift(np.fft.fft2(window_func * self.field_data))
        self.field_fourier =  (amp_coeff * np.abs(field_fft_2d))**2



        