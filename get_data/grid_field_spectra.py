#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Import libraries

"""
accumulator_field_spectra.py

File which houses the classes which extracts required
spatial grid data for plotting

"""

import sys
sys.path.append("..")
import sdf
import numpy as np
import scipy.constants as const
from math import floor, ceil
import calculations.laser_calculator as laser


class data:
    """
    Class that contains required functions to extract data for 
    grid field spectra plots.
    """

    def __init__(self, files, field_name, lambda_0, T_e):


        """
        Class constructor function

        files = Organised list of required files to read
        field_name = Particular field to take fft of. Read from sdf file directory with naming style
                             "Electric_Field_E{x,y or z}", and similar for magnetic field.

        lambda_0 = Vacuum laser wavelength (units : m)
        T_e = Electron temperature (units : K)
             
        """
        
        # File reading setup
        self.files = files
        self.nfiles = len(files)
        self.field_name = field_name
        
        # Base plasma/laser parameters required
        self.lambda_0 = lambda_0
        self.T_e = T_e
        
        # Required laser normalisations
        self.k_0 = laser.wavenumber(self.lambda_0)
        self.omega_0 = laser.omega(self.lambda_0)
        # Field normalisation constant
        if self.field_name[-2] == 'E':
            self.field_norm = laser.E_normalisation(self.lambda_0)
        elif self.field_name[-2] == 'B':
            self.field_norm = laser.B_normalisation(self.lambda_0)
        else:
            sys.exit('ERROR: Naming of field_name is incorrect. Plese use the form "Electric_Field_E{x,y or z}", or similar for magnetic field.')


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
        var = d.__dict__[self.field_name]
        

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

    def load_grid_field_data(self, snap_time):

        """
        Loads the required field and grid data for chosen
        time snapshot.

        snap_time = Simuation time of the field grid to FFT (units : s)
       
        """

        # Setup key variables
        self.setup_variables()

        # Find file index closest to given snapshot time
        file_index = int(snap_time/self.t_end * self.nfiles)
        # Set file name
        fname = self.files[file_index]
        
        # Read data file for snap_time
        d = sdf.read(fname)
        # Extract chosen field data
        self.field_data = np.array(d.__dict__[self.field_name].data)
        self.field_data *= self.field_norm
        # Extract time for plot
        self.time = d.Header["time"]


    def kx_vs_ky_fft(self, snap_time, x_min, x_max):

        """
        Extracts the 2D spatial FFT of chosen field.
        FFT is done for domain defined by the given
        minimum and maximum x values.

        snap_time = Simuation time of the field grid to FFT (units : s)
        x_min = Minimum x-posistion to take FFT around (units : m)
        x_max = Maximum x-posistion to take FFT around (units : m)
       
        """


        # Load required grid data
        self.load_grid_field_data(snap_time)

        x_indicies = np.where((self.X_centres >= x_min) & (self.X_centres <= x_max))[0]
        if len(x_indicies) == 0:
            sys.exit('ERROR (kx_vs_ky): Inputted density scale is incorrect. Please ensure it is in units n_cr and in range.')

        # Define quantities for given density range
        self.X_centres = self.X_centres[x_indicies]
        self.field_data = self.field_data[x_indicies,:]
        self.N_x = len(self.X_centres)

        # Find possible wavenumbers from grid (normalised by k_0)
        self.k_x = 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(n = self.N_x, d = self.dx)) / self.k_0
        self.k_y = 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(n = self.N_y, d = self.dy)) / self.k_0

        # 2D hanning window
        # Apply windowing to FFT (hanning usually the best)
        w_x = np.hanning(self.N_x)
        w_y = np.hanning(self.N_y)
        window_func = np.outer(w_x, w_y)
        # Coefficient to normalise amplitudes
        amp_coeff = 2.0 / (self.N_x * np.mean(w_x) * self.N_y * np.mean(w_y))

        print('Extracting kx_vs_ky FFT')
        # use fftshift to get of the form -k => k
        field_fft_2d = np.fft.fftshift(np.fft.fft2(window_func * self.field_data))
        self.field_fourier =  (amp_coeff * np.abs(field_fft_2d))**2

    def x_vs_ky_fft(self, snap_time, x_min, x_max):
        
        """
        Extracts the 1D spatial FFT (y) of chosen field.
        FFT is done for domain defined by the given
        minimum and maximum x values.

        snap_time = Simuation time of the field grid to FFT (units : s)
        x_min = Minimum x-posistion to take FFT around (units : m)
        x_max = Maximum x-posistion to take FFT around (units : m)
       
        """

        # Load required grid data
        self.load_grid_field_data(snap_time)

        x_indicies = np.where((self.X_centres >= x_min) & (self.X_centres <= x_max))[0]
        if len(x_indicies) == 0:
            sys.exit('ERROR (kx_vs_ky): Inputted density scale is incorrect. Please ensure it is in units n_cr and in range.')

        # Define quantities for given density range
        self.X_centres = self.X_centres[x_indicies]
        self.field_data = self.field_data[x_indicies,:]
        self.N_x = len(self.X_centres)

        # Find possible wavenumbers from grid (normalised by k_0)
        self.k_y = 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(n = self.N_y, d = self.dy)) / self.k_0

        # Create list to store y-FFT of each grid strip
        field_fourier = []
        # Apply windowing to FFT (hanning usually the best)
        window_func = np.hanning(self.N_y)
        # Coefficient to normalise amplitudes
        ampCoeff = 2.0 / (self.N_y * np.mean(window_func))

        print('Extracting x_vs_ky FFT')

        for i in range(self.N_x):
            # use fftshift to get of the form -k => k
            fft_y = np.fft.fftshift(np.fft.fft(window_func * self.field_data[i,:]))  
            field_fourier.append((ampCoeff * (np.abs(fft_y)))**2)

        # Store as numpy array as it's easy to plot
        self.field_fourier = np.array(field_fourier)

    def x_vs_kx_stft(self, snap_time, x_window, x_bins, k_range):

        """
        Extracts the Short Time Fourier Transform (STFT)
        of chosen field. The STFT process is defined by the
        size of the window, (i.e frequency resolution) and number of
        spatial bins to perform this FFT window over (i.e spatial resolution)

        snap_time = Simuation time of the field grid to FFT (units : s)
        x_window = Length of FFT window for STFT process 
        x_bins = Number of dicrete dections to perform the process over. 
        k_range = Wavenumber range (kx) to store output for (units : k_0)
       
        """

        # Load required grid data
        self.load_grid_field_data(snap_time)

        # Hop size for STFT process
        x_hop = int((self.N_x - x_window) / x_bins)
        # Apply windowing to FFT (hanning usually the best)
        window_func = np.hanning(x_window)
        # Coefficient to normalise amplitudes
        ampCoeff = 2.0 / (x_window * np.mean(window_func))

        # Find possible wavenumbers from grid (normalised by k_0)
        self.k_x = 2.0 * np.pi *np.fft.fftshift(np.fft.fftfreq(n = x_window, d = self.dx)) / self.k_0

        # Plot for given k region
        k_indicies = np.where((self.k_x >= k_range[0]) & (self.k_x <= k_range[-1]))[0]
        if len(k_indicies) == 0:
            sys.exit('ERROR (x_vs_kx): Inputted wavenumber (kx) scale is incorrect. Please ensure it is units of k_0 (laser) and in range.')

        # Get wavenumbers in range
        self.k_x = self.k_x[k_indicies]

        # Store required plot data
        field_fourier = []
        x_data = []


        print('Extracting x_vs_kx STFT')

        for i in range(x_bins):
            fourier_space_av = 0
            for j in range(self.N_y):
                # STFT steps
                F = self.field_data[i*x_hop:x_window+i*x_hop ,j]
                # Absolute value to have a real number
                fft_field = np.fft.fftshift(np.fft.fft(window_func * F))
                fourier_space_av = np.add(fourier_space_av, (ampCoeff * (np.abs(fft_field)))**2)
            # Average over y-space
            fourier_space_av /= self.N_y
            # FFT in each time window
            field_fourier.append(fourier_space_av[k_indicies])
            # x_data.append(self.X_centres[i*x_hop])
            x_data.append(self.X_centres[i*x_hop])

        # Use numpy arrays for plots
        self.field_fourier = np.array(field_fourier)
        self.x_data = np.array(x_data)

    



    