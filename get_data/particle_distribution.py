#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Import libraries

"""
particle_distribution.py

File which houses the class which extracts required
data for plotting distributions of outgoing
particles.

"""

import sys
sys.path.append("..")
import sdf
import numpy as np
import scipy.constants as const
from math import floor, ceil
import calculations.plasma_calculator as plasma

def print_progress_bar(index, total, label):
    """
    prints progress bar for loops.

    :param index : current index of loop
    :param total : total number of indicies to loop over
    :param label : print statement next to progress bar

    """
    n_bar = 80  # Progress bar width
    progress = index / total
    sys.stdout.write('\r')
    sys.stdout.write(f"[{'=' * int(n_bar * progress):{n_bar}s}] {int(100 * progress)}%  {label}")
    sys.stdout.flush()

# Conversion from keV to Joules
keV_to_J = const.e*1e3

class data:
    """
    Class that contains required functions to extract data for 
    outgoing particle distributions.
    """

    def __init__(self, files, probe_flag, lambda_0, T_e):

        """
        Class constructor function

        files = Organised list of required files to read
        probe_flag = Flag for which probe to use
        lambda_0 = Vacuum laser wavelength (units : m)
        T_e = Electron temperature (units : K)
        """

        self.files = files
        self.nfiles = len(self.files)
        self.probe_flag = probe_flag
        
        # Required plasma parameters
        self.lambda_0 = lambda_0
        self.T_e = T_e

        # Electron thermal speed and momentum
        self.v_th = plasma.electron_thermal_speed(self.T_e)
        self.p_th = const.m_e * self.v_th

    def get_file_range(self, t_min, t_max):

        # Read last file to get end time
        d = sdf.read(self.files[-1])
        # Extract end sim-time
        self.t_end = d.__dict__['Header']['time']
        
        # Get data close to time range (save's reading all files)
        file_index_min = floor(t_min/self.t_end * self.nfiles)
        file_index_max = ceil(t_max/self.t_end * self.nfiles)

        # Set file name
        self.files = self.files[file_index_min:file_index_max]
        #self.files = self.files[file_index_min:file_index_max+1]
        self.nfiles = len(self.files)

        del d
    
    def load_momenta_histogram(self, t_min, t_max, componants, nbins):
        
        # Get required files to read
        self.get_file_range(t_min, t_max)
        # Set momentum bins for histogram
        p_bins_edges = np.linspace(0.0 * self.p_th, 100 * self.p_th, nbins)
        self.p_bins_centre = 0.5 * (p_bins_edges[1:] + p_bins_edges[:-1])

        # Find distribution from each time segment and sum to get full distribution
        self.p_distribution = np.zeros(len(self.p_bins_centre))
        self.time = np.array([])

        for i in range(self.nfiles):
            print_progress_bar(i+1, self.nfiles, f"Computing momentum distribution {self.files[i]}")
            # Extract data from files
            d = sdf.read(self.files[i])
            # Data output time
            self.time = np.append(self.time, d.__dict__['Header']['time'])
            
            if componants == 'x' or componants == 'y' or componants == 'z':
                # Momentum magnitude
                p = d.__dict__[f'{self.probe_flag}_P{componants}'].data 
            elif componants == 'xy' or componants == 'xz' or componants == 'yz':
                # Momentum componants
                p1 = d.__dict__[f'{self.probe_flag}_P{componants[0]}'].data
                p2 = d.__dict__[f'{self.probe_flag}_P{componants[1]}'].data
                # Momentum magnitude
                p = np.sqrt(p1**2 + p2**2)
            elif componants == 'xyz':
                # Momentum componants
                p1 = d.__dict__[f'{self.probe_flag}_P{componants[0]}'].data
                p2 = d.__dict__[f'{self.probe_flag}_P{componants[1]}'].data
                p3 = d.__dict__[f'{self.probe_flag}_P{componants[2]}'].data
                # Momentum magnitude
                p = np.sqrt(p1**2 + p2**2 + p3**2)
            else:
                sys.exit('ERROR (p_distribution): componants argument inputted incorrectly ,\
                          please set it to either x, y, z, xy, xz, yz, xyz')
            
            # Weights of number of particles passing
            particle_weights = d.__dict__[f'{self.probe_flag}_weight'].data
            # Histogram for given data
            histogram, _ =  np.histogram(p, bins = p_bins_edges, weights = particle_weights, density = True)
            # Sum to get data for total time period
            self.p_distribution = np.add(self.p_distribution, histogram)

        print('',end='\n')


    def load_energy_histogram(self, t_min, t_max, nbins, weighted = True):
        
        # Get required files to read
        self.get_file_range(t_min, t_max)
        # Set momentum bins for histogram
        E_bins_edges = np.linspace(0.0 * keV_to_J, 2500 * keV_to_J, nbins)
        self.E_bins_centre = 0.5 * (E_bins_edges[1:] + E_bins_edges[:-1])

        # Find distribution from each time segment and sum to get full distribution
        self.E_distribution = np.zeros(len(self.E_bins_centre))
        self.time = np.array([])

        for i in range(self.nfiles):
            print_progress_bar(i+1, self.nfiles, f"Computing energy distribution {self.files[i]}")
            # Extract data from files
            d = sdf.read(self.files[i])
            # Data output time
            self.time = np.append(self.time, d.__dict__['Header']['time'])
            # Extract momentum componants if outgoing particles
            px = d.__dict__[f'{self.probe_flag}_Px'].data 
            py = d.__dict__[f'{self.probe_flag}_Py'].data 
            pz = d.__dict__[f'{self.probe_flag}_Pz'].data
            # Magnitude
            p = np.sqrt(px**2 + py**2 + pz**2)
            # Lorentz factor
            gamma = np.sqrt(1.0 + (p/(const.m_e*const.c))**2)
            # Convert to Kinetic energy (relativistic)
            E = (gamma - 1.0) * const.m_e * const.c**2
            # Weights of number of particles passing
            particle_weights = d.__dict__[f'{self.probe_flag}_weight'].data
            # Histogram for given data
            histogram, _ =  np.histogram(E, bins = E_bins_edges, weights = particle_weights, density = True)
            # Sum to get data for total time period
            if weighted:
                self.E_distribution = np.add(self.E_distribution, histogram*self.E_bins_centre)
            else:
                self.E_distribution = np.add(self.E_distribution, histogram)

        print('',end='\n')
        







        


