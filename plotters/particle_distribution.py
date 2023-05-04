#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Import libraries

"""
particle_distribution.py

File which houses the class which
holds plotting functions for distributions 
of outgoing particles.
"""


# Import libraries
import sys
sys.path.append("..")
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as const
from matplotlib.colors import LogNorm
from matplotlib import cm
import os
from datetime import datetime
import get_data.particle_distribution as distribution
import calculations.plasma_calculator as plasma
import calculations.particle_distribution_calculator as dist_funcs
from scipy.ndimage import uniform_filter1d

# Useful prefix's
pico = 1e-12
femto = 1e-15
micron = 1e-6

# conversion from keV to Kelvin
keV_to_K = (const.e*1e3)/const.k
# Conversion from keV to Joules
keV_to_J = const.e*1e3

# Line colour for plot
line_colour = 'black'


def loss_func(fit, data):
    N = len(data)
    sum_ = 0
    for i in range(N):
        l = fit[i] - data[i]
        sum_ += l*l
    
    loss = sum_/N

    return loss

class plots:
    
    """ Class that houses plotting routines for outgoing particle distributions"""

    def __init__(self, files, probe_flag, output_path, lambda_0, T_e):

        """
        Class constructor function

        files = Organised list of required files to read
        probe_flag = Flag for which probe to use
        output_path = Directory to store figures in
        lambda_0 = Vacuum laser wavelength (units : m)
        T_e = Electron temperature (units : K)
        """

        self.files = files
        self.nfiles = len(self.files)
        self.probe_flag = probe_flag
        self.output_path = output_path
        
        # Required plasma parameters
        self.lambda_0 = lambda_0
        self.T_e = T_e

        # Electron thermal speed and momentum
        self.v_th = plasma.electron_thermal_speed(self.T_e)
        self.p_th = const.m_e * self.v_th

        # Class to extract distribution data
        self.dist_data = distribution.data(files=self.files, probe_flag=self.probe_flag, lambda_0=self.lambda_0, T_e=self.T_e)



    ########################################################################################################################
    # Momentum
    ########################################################################################################################

    def plot_p_dist(self, t_min, t_max, componants, p_min, p_max, nbins, maxwell_plot = False):

        """
        Function ot plot momentum distribution of outgoing 
        particles.

        t_min = Minimum time to extract probe data from (units : s)
        t_min = Maximum time to extract probe data from (units : s)
        componants = Which componants of momentum to use. 
                     (options 'x', 'y', 'z', 'xy', 'xz', 'yz' 'xyz')
        p_min = Minimum momentum to plot for (units : p_th = m_e * v_th)
        p_max = Maximum momentum to plot for (units : p_th = m_e * v_th)
        nbins = Number of discrete bins to plot for
        """

        # Create sub-directory to store results in
        try:
            os.mkdir(f'{self.output_path}/p_dist/')
        except:
            print('', end='\n')

        # Call required function to extract  wanted data
        self.dist_data.load_momenta_histogram(t_min, t_max, componants, nbins)

        # Extract data

        p_bins = self.dist_data.p_bins_centre
        p_distribution = self.dist_data.p_distribution
        # Normalise by it's maximum value
        p_distribution /= p_distribution.max()
        time = self.dist_data.time / pico

        # Required to find time at which data is taken from
        dt = time[1] - time[0]

        print('Plotting Exiting Particle Momentum Distribution ')

        fig, ax = plt.subplots()

        # momentum distribution
        ax.plot(p_bins/ self.p_th, p_distribution, c = 'black', lw = 4,label = f'Probe Data ({np.round(np.abs(time[0] - dt), 2)} - {np.round(time[-1], 2)}) ps')

        # Axes limits
        ax.set_xlim(p_min, p_max)
        ax.set_yscale('log')
        ax.set_ylim(1e-6 * p_distribution.max(), 1.05 * p_distribution.max())

        # Label plot
        if componants == 'x':
            ax.set_xlabel(r'$p_x \,\, (m_e v_{th})$')
            ax.set_ylabel(r'$f(p_{x})$')
        elif componants == 'y':
            ax.set_xlabel(r'$p_y \,\, (m_e v_{th})$')
            ax.set_ylabel(r'$f(p_{y})$')
        elif componants == 'z':
            ax.set_xlabel(r'$p_z \,\, (m_e v_{th})$')
            ax.set_ylabel(r'$f(p_{z})$')
        elif componants == 'xy':
            ax.set_xlabel(r'$p_{xy} \,\, (m_e v_{th})$')
            ax.set_ylabel(r'$f(p_{xy})$')
        elif componants == 'xz':
            ax.set_xlabel(r'$p_{xz} \,\, (m_e v_{th})$')
            ax.set_ylabel(r'$f(p_{xz})$')           
        elif componants == 'xyz':
            ax.set_xlabel(r'$p \,\, (m_e v_{th})$')
            ax.set_ylabel(r'$f(p)$')  

        if maxwell_plot:
            T_vals = [40, 75, 100]
            deg = len(componants)
            for T in T_vals:
                p_maxwell = np.linspace(p_min * self.p_th, p_max * self.p_th, 1000)
                maxwell = dist_funcs.max_boltz_momentum(p_maxwell, T*keV_to_K, deg)
                ax.plot(p_maxwell/self.p_th, maxwell/maxwell.max(), label = '$T_e$ = ' + str(np.round(T,2)) + ' keV')


        plt.grid(color = 'black', linestyle = '--', linewidth = 1)

        ax.legend()

        # Save figure
        plot_name = f'{self.probe_flag}_{np.round(np.abs(time[0] - dt), 2)}-{np.round(time[-1], 2)}_ps.png'
        print(f'Saving Figure to {self.output_path}/p_dist/{plot_name}')
        plt.tight_layout()
        fig.savefig(f'{self.output_path}/p_dist/{plot_name}')

        # Append to output_fig file to keep track
        output_file = open(f'{self.output_path}/output_figs.txt',"a")
        now = datetime.now()
        # dd/mm/YY H:M:S
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        output_file.write(f'\nSaved p_dist/{plot_name} at {dt_string}')
        output_file.close()


    ########################################################################################################################
    # Energy
    ########################################################################################################################

    def plot_E_dist(self, t_min, t_max, E_min, E_max, nbins, weighted = False, maxwell_plot = False):

        """
        Function ot plot energy distribution of outgoing 
        particles.

        t_min = Minimum time to extract probe data from (units : s)
        t_min = Maximum time to extract probe data from (units : s)
        E_min = Minimum energy to plot for (units : keV)
        E_max = Maximum energy to plot for (units : keV)
        nbins = Number of discrete bins to plot for
        weighted = To return weighted distribution (E * f(E)) or not
        """

        # Create sub-directory to store results in
        try:
            os.mkdir(f'{self.output_path}/E_dist/')
        except:
            print('', end='\n')

        # Call required function to extract  wanted data
        self.dist_data.load_energy_histogram(t_min, t_max, nbins, weighted)

        # Extract data
        E_bins = self.dist_data.E_bins_centre
        E_distribution = self.dist_data.E_distribution
        # Normalise by it's maximum value
        # norm = E_distribution.max()
        # E_distribution /= norm
        time = self.dist_data.time / pico

        # Required to find time at which data is taken from
        #dt = time[1] - time[0]

        print('Plotting Exiting Particle Energy Distribution ')

        fig, ax = plt.subplots()

        # Energy distribution
        # ax.plot(E_bins/keV_to_J, E_distribution, c='black', lw=4, label = f'Probe Data ({np.round(np.abs(time[0] - dt), 2)} - {np.round(time[-1], 2)}) ps')
        ax.plot(E_bins/keV_to_J, E_distribution, c='black', lw=4, label = f'Probe Data')
        # Axes limits
        ax.set_xlim(E_min, E_max)
        ax.set_yscale('log')
        ax.set_ylim(1e-5 * E_distribution.max(), 1.05 * E_distribution.max())
            
        if weighted:
            ax.set_xlabel(r'$E \,\, (keV)$')
            ax.set_ylabel(r'$E \cdot f(E)$')
        else:
            ax.set_xlabel(r'$E \,\, (keV)$')
            ax.set_ylabel(r'$f(E)$')

        if maxwell_plot:
            T_vals = np.array([10, 40, 65, 100, 150])
            for T in T_vals:
                E_maxwell = np.linspace(E_min * keV_to_J, E_max*keV_to_J , 1000)
                maxwell = dist_funcs.max_boltz_energy(E_maxwell, T*keV_to_K, weighted)
                ax.plot(E_maxwell/keV_to_J, maxwell, label = '$T_e$ = ' + str(np.round(T,2)) + ' keV')


        
        plt.grid(color = 'black', linestyle = '--', linewidth = 1)

        ax.legend()

        # Save figure
        # plot_name = f'{self.probe_flag}_{np.round(np.abs(time[0] - dt), 2)}-{np.round(time[-1], 2)}_ps.png'
        plot_name = f'{self.probe_flag}.png'
        print(f'Saving Figure to {self.output_path}/E_dist/{plot_name}')
        plt.tight_layout()
        fig.savefig(f'{self.output_path}/E_dist/{plot_name}')

        # Append to output_fig file to keep track
        output_file = open(f'{self.output_path}/output_figs.txt',"a")
        now = datetime.now()
        # dd/mm/YY H:M:S
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        output_file.write(f'\nSaved E_dist/{plot_name} at {dt_string}')
        output_file.close()