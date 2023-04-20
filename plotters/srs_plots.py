#!/usr/bin/env python3
# -*- coding: utf-8 -*-


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

import calculations.srs_calculator as srs
import calculations.plasma_calculator as plasma


em_colour = 'red'
epw_colour = 'green'


class plots:

    def __init__(self, T_e, lambda_0):

        self.T_e = T_e
        self.lambda_0 = lambda_0

        self.v_th = plasma.electron_thermal_speed(self.T_e)

    def kx_vs_omega_EPW(self, n_e, theta, ax):

        # Get wavenumbers and frequencies
        k_epw = srs.srs_EPW_k_x(self.v_th, n_e, theta, self.lambda_0)
        omega_epw = srs.srs_omega_EPW(n_e, self.T_e, theta, self.lambda_0)
        # Plot
        ax.plot(k_epw, omega_epw, c=epw_colour, label = r'SRS EPW ($\theta$ =' + f'{np.round(theta, 1)}\N{DEGREE SIGN})')    

    def kx_vs_omega_EM(self, n_e, theta,  ax):   
        
        # Get wavenumbers and frequencies
        k_em = srs.srs_EM_k_x(self.v_th, n_e, theta, self.lambda_0) 
        omega_em = srs.srs_omega_EM(n_e, self.T_e, theta, self.lambda_0)
        # Plot
        ax.plot(k_em, omega_em, c=em_colour, label = r'SRS EM ($\theta$ =' + f'{np.round(theta, 1)}\N{DEGREE SIGN})')

    def x_vs_omega_EPW(self, n_e, theta, x, ax):   
        
        # Getfrequencies
        omega_epw = srs.srs_omega_EPW(n_e, self.T_e, theta, self.lambda_0)
        # Plot
        ax.plot(x, omega_epw, c=epw_colour, label = r'SRS EPW ($\theta$ =' + f'{np.round(theta, 1)}\N{DEGREE SIGN})')

    def x_vs_omega_EM(self, n_e, theta, x, ax):   
        
        # Getfrequencies
        omega_em = srs.srs_omega_EM(n_e, self.T_e, theta, self.lambda_0)
        # Plot
        ax.plot(x, omega_em, c=em_colour, label = r'SRS EM ($\theta$ =' + f'{np.round(theta, 1)}\N{DEGREE SIGN})')

    def omega_EM(self, axis, n_min, n_max, ax):

        num_dens = np.linspace(n_min, n_max, 50)
        angles = np.linspace(0, 360, 50)

        omega_min = np.zeros(len(angles))
        omega_max = np.zeros(len(angles))
        

        for i in range(len(angles)):
            omega_em = srs.srs_omega_EM(num_dens, self.T_e, angles[i], self.lambda_0)
        
            idx = np.where(np.isnan(omega_em) == False)[0]
    
            omega_min[i] = omega_em[idx].min()
            omega_max[i] = omega_em[idx].max()
        
        if axis == 'x':
            ax.axhline(omega_min.min(), c=em_colour)
            ax.axhline(omega_max.max(), c=em_colour, label = f'SRS EM ({np.round(n_min,2)}-{np.round(n_max,2)}' + r' $n_{cr}$)')
        elif axis == 'y':
            ax.axvline(omega_min.min(), c=em_colour)
            ax.axvline(omega_max.max(), c=em_colour, label = f'SRS EM ({np.round(n_min,2)}-{np.round(n_max,2)}' + r' $n_{cr}$)')


    def omega_EPW(self, axis, n_min, n_max, ax):

        num_dens = np.linspace(n_min, n_max, 50)
        angles = np.linspace(0, 360, 50)

        omega_min = np.zeros(len(angles))
        omega_max = np.zeros(len(angles))
        
        
        for i in range(len(angles)):
            omega_epw = srs.srs_omega_EPW(num_dens, self.T_e, angles[i], self.lambda_0)

            idx = np.where(np.isnan(omega_epw) == False)[0]
            omega_min[i] = omega_epw[idx].min()
            omega_max[i] = omega_epw[idx].max()
        
        if axis == 'x':
            ax.axhline(omega_min.min(), c=epw_colour)
            ax.axhline(omega_max.max(), c=epw_colour, label = f'SRS EPW ({np.round(n_min,2)}-{np.round(n_max,2)}' + r' $n_{cr}$)')
        elif axis == 'y':
            ax.axvline(omega_min.min(), c=epw_colour)
            ax.axvline(omega_max.max(), c=epw_colour, label = f'SRS EPW ({np.round(n_min,2)}-{np.round(n_max,2)}' + r' $n_{cr}$)')


    def kx_vs_ky_EM(self, n_vals, angle_range, ax):
        
        if np.isscalar(n_vals):
            k_x, k_y = srs.srs_wns_EM_polar(n_vals, self.v_th, self.lambda_0,\
                                            angle_min = angle_range[0], angle_max = angle_range[-1])
            ax.plot(k_x, k_y, c=em_colour, label = f'SRS EM ($n_e$ = ' + f'{np.round(n_vals,2)}' + r' $n_{cr}$)')
        else:
            for i in range(len(n_vals)):
                k_x, k_y = srs.srs_wns_EM_polar(n_vals[i], self.v_th, self.lambda_0,\
                                                angle_min = angle_range[0], angle_max = angle_range[-1])
                if i == 0:
                    ax.plot(k_x, k_y, c=em_colour, label = r'SRS EM ($n_e$ = '+ f'{np.round(np.array(n_vals).min(),2)} - {np.round(np.array(n_vals).max(),2)}' + r' $n_{cr}$)')
                else:
                    ax.plot(k_x, k_y, c=em_colour)


    def kx_vs_ky_EPW(self, n_vals, angle_range, ax):
        
        if np.isscalar(n_vals):
            k_x, k_y = srs.srs_wns_EPW_polar(n_vals, self.v_th, self.lambda_0,\
                                            angle_min = angle_range[0], angle_max = angle_range[-1])
            ax.plot(k_x, k_y, c=epw_colour, label = f'SRS EPW ($n_e$ = ' + f'{np.round(n_vals,2)}' + r' $n_{cr}$)')
        else:
            for i in range(len(n_vals)):
                k_x, k_y = srs.srs_wns_EPW_polar(n_vals[i], self.v_th, self.lambda_0,\
                                                 angle_min = angle_range[0], angle_max = angle_range[-1])
                if i == 0:
                    ax.plot(k_x, k_y, c=epw_colour, label = r'SRS EPW ($n_e$ = '+ f'{np.round(np.array(n_vals).min(),2)} - {np.round(np.array(n_vals).max(),2)}' + r' $n_{cr}$)')
                else:
                    ax.plot(k_x, k_y, c=epw_colour)

    def x_vs_ky_EM(self, n_e, theta, x, ax):

        k_y = srs.srs_EM_k_y(self.v_th, n_e, theta, self.lambda_0)
        ax.plot(x, k_y, c=em_colour, label = r'SRS ($\theta$ =' + f'{np.round(theta, 1)}\N{DEGREE SIGN})')
        ax.plot(x, -k_y, c=em_colour)

    def x_vs_ky_EPW(self, n_e, theta, x, ax):

        k_y = srs.srs_EPW_k_y(self.v_th, n_e, theta, self.lambda_0)
        ax.plot(x, k_y, c=epw_colour, label = r'SRS ($\theta$ =' + f'{np.round(theta, 1)}\N{DEGREE SIGN})')
        ax.plot(x, -k_y, c=epw_colour)

        landau_cutoff_srs = srs.landau_cutoff_index(self.T_e, n_e, self.lambda_0, theta, cutoff = 0.3)
        if landau_cutoff_srs is not None:
            ax.axvline(x[landau_cutoff_srs], ls = '--', c=epw_colour, label = r'SRS Landau Cutoff ($\theta$ =' + f'{np.round(theta, 1)}\N{DEGREE SIGN})')

    def x_vs_kx_EM(self, n_e, theta, x, ax):

        k_x = srs.srs_EM_k_x(self.v_th, n_e, theta, self.lambda_0)
        ax.plot(x, k_x, c=em_colour, label = r'SRS EM ($\theta$ =' + f'{np.round(theta, 1)}\N{DEGREE SIGN})')
        ax.plot(x, -k_x, c=em_colour)

    def x_vs_kx_EPW(self, n_e, theta, x, ax):

        k_x = srs.srs_EPW_k_x(self.v_th, n_e, theta, self.lambda_0)
        ax.plot(x, k_x, c=epw_colour, label = r'SRS EPW ($\theta$ =' + f'{np.round(theta, 1)}\N{DEGREE SIGN})')
        ax.plot(x, -k_x, c=epw_colour)

        landau_cutoff_srs = srs.landau_cutoff_index(self.T_e, n_e, self.lambda_0, theta, cutoff = 0.3)
        if landau_cutoff_srs is not None:
            ax.axvline(x[landau_cutoff_srs], ls = '--', c=epw_colour, label = r'SRS Landau Cutoff ($\theta$ =' + f'{np.round(theta, 1)}\N{DEGREE SIGN})')

