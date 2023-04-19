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


em_colour = 'white'
epw_colour = 'red'


class plots:

    def __init__(self, T_e, n_e, theta, lambda_0):

        self.T_e = T_e
        self.n_e = n_e
        self.theta = theta
        self.lambda_0 = lambda_0

        self.v_th = plasma.electron_thermal_speed(self.T_e)

    def kx_vs_omega_EPW(self, ax):

        # Get wavenumbers and frequencies
        k_epw = srs.srs_EPW_k_x(self.v_th, self.n_e, self.theta, self.lambda_0)
        omega_epw = srs.srs_omega_EPW(self.n_e, self.T_e, self.theta, self.lambda_0)
        # Plot
        ax.plot(k_epw, omega_epw, c=epw_colour, label = 'SRS EPW')    

    def kx_vs_omega_EM(self, ax):   
        
        # Get wavenumbers and frequencies
        k_em = srs.srs_EM_k_x(self.v_th, self.n_e, self.theta, self.lambda_0) 
        omega_em = srs.srs_omega_EM(self.n_e, self.T_e, self.theta, self.lambda_0)
        # Plot
        ax.plot(k_em, omega_em, c=em_colour, label = 'SRS EM')

    def x_vs_omega_EPW(self, x, ax):   
        
        # Getfrequencies
        omega_epw = srs.srs_omega_EPW(self.n_e, self.T_e, self.theta, self.lambda_0)
        # Plot
        ax.plot(x, omega_epw, c=epw_colour, label = 'SRS EPW')

    def x_vs_omega_EM(self, x, ax):   
        
        # Getfrequencies
        omega_em = srs.srs_omega_EM(self.n_e, self.T_e, self.theta, self.lambda_0)
        # Plot
        ax.plot(x, omega_em, c=em_colour, label = 'SRS EM')

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
            ax.axhline(omega_max.max(), c=em_colour, label = f'SRS EM {n_min}-{n_max}' + r' $n_{cr}$')
        elif axis == 'y':
            ax.axvline(omega_min.min(), c=em_colour)
            ax.axvline(omega_max.max(), c=em_colour, label = f'SRS EM {n_min}-{n_max}' + r' $n_{cr}$')


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
            ax.axhline(omega_max.max(), c=epw_colour, label = f'SRS EPW {n_min}-{n_max}' + r' $n_{cr}$')
        elif axis == 'y':
            ax.axvline(omega_min.min(), c=epw_colour)
            ax.axvline(omega_max.max(), c=epw_colour, label = f'SRS EPW {n_min}-{n_max}' + r' $n_{cr}$')

