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

import calculations.tpd_calculator as tpd
import calculations.plasma_calculator as plasma


plot_colour = 'white'


class plots:

    def __init__(self, T_e, lambda_0):

        self.T_e = T_e
        self.lambda_0 = lambda_0

        self.v_th = plasma.electron_thermal_speed(self.T_e)

    def kx_vs_omega(self, n_e, theta, ax):

        k1, k2 = tpd.tpd_wns_pairs(self.v_th, n_e, theta, self.lambda_0, componants = 'x')
        omega1, omega2 = tpd.tpd_omegas(n_e, self.T_e, theta, self.lambda_0)

        print(k2)

        ax.plot(k1, omega1, c=plot_colour)
        ax.plot(k2, omega2, c=plot_colour)

    def x_vs_omega(self, n_e, theta, x, ax):

        omega1, omega2 = tpd.tpd_omegas(n_e, self.T_e, theta, self.lambda_0)

        ax.plot(x, omega1, c=plot_colour)
        ax.plot(x, omega2, c=plot_colour, label = r'TPD EPW ($\theta$ =' + f'{np.round(theta, 1)}\N{DEGREE SIGN})')

    def omega(self, axis, n_min, n_max, ax):

        num_dens = np.linspace(n_min, n_max, 50)
        angles = np.linspace(0, 360, 50)

        omega_min = np.zeros(len(angles))
        omega_max = np.zeros(len(angles))
        
        for i in range(len(angles)):
            omega1, omega2 = tpd.tpd_omegas(num_dens, self.T_e, angles[i], self.lambda_0)
             
            idx1 = np.where(np.isnan(omega1) == False)[0]
            idx2 = np.where(np.isnan(omega2) == False)[0]
            omega_min[i] = min(omega1[idx1].min(),omega2[idx2].min())
            omega_max[i] = min(omega1[idx1].max(),omega2[idx2].max())
        

        if axis == 'x':
            ax.axvline(omega_min.min(), c=plot_colour)
            ax.axvline(omega_max.max(), c=plot_colour, label = f'TPD EPW ({n_min}-{n_max}' + r' $n_{cr}$)')
        elif axis == 'y':
            ax.axhline(omega_min.min(), c=plot_colour)
            ax.axhline(omega_max.max(), c=plot_colour, label = f'TPD EPW ({n_min}-{n_max}' + r' $n_{cr}$)')
            
    def kx_vs_ky(self, n_vals, angle_range, ax):
        
        if np.isscalar(n_vals):
            k_x, k_y = tpd.tpd_wns_polar(n_vals, self.v_th, self.lambda_0, angle_range[0], angle_range[-1])
            ax.plot(k_x, k_y, c=plot_colour, label = f'TPD EPW ($n_e$ = ' + f'{np.round(n_vals,2)}' + r' $n_{cr}$)')
        else:
            for i in range(len(n_vals)):
                k_x, k_y = tpd.tpd_wns_polar(n_vals[i], self.v_th, self.lambda_0, angle_range[0], angle_range[-1])
                if i == 0:
                    ax.plot(k_x, k_y, c=plot_colour, label = r'TPD EPW ($n_e$ = '+ f'{np.round(np.array(n_vals).min(),2)} - {np.round(np.array(n_vals).max(),2)}' + r' $n_{cr}$)')
                else:
                    ax.plot(k_x, k_y, c=plot_colour)

        # Plot linear growth rate curve for max density asked to plot
        # Curve is very similar for all k_L values (i.e for all n_e values)
        k_x_growth, k_y_growth = tpd.tpd_max_lin_growth(np.array(n_vals).max(), self.lambda_0)
        # Have postive and negative solutions
        ax.plot(k_x_growth, k_y_growth, c=plot_colour, ls = '--')
        ax.plot(k_x_growth, -k_y_growth, c=plot_colour, ls = '--', label = 'TPD Growth Curve')

    def x_vs_ky(self, n_e, theta, x, ax):


        k1_y, k2_y = tpd.tpd_wns_pairs(self.v_th, n_e, theta, self.lambda_0, componants = 'y')
        ax.plot(x, k1_y, c=plot_colour)
        if theta == 'max_lin_growth':
            ax.plot(x, k2_y, c=plot_colour, label = r'TPD EPW ($\theta$ = $\theta_{MLG}$)')
        else:
            ax.plot(x, k2_y, c=plot_colour, label = r'TPD EPW ($\theta$ =' + f'{np.round(theta, 1)}\N{DEGREE SIGN})')

        # Find location where LW are Landau damped
        landau_cutoff_tpd = tpd.landau_cutoff_index(self.T_e, n_e, self.lambda_0, theta, cutoff = 0.3)
        if landau_cutoff_tpd is not None:
            if theta == 'max_lin_growth':
                ax.axvline(x[landau_cutoff_tpd], ls = '--', c=plot_colour, label = r'TPD Landau Cutoff ($\theta$ = $\theta_{MLG}$)')
            else:
                ax.axvline(x[landau_cutoff_tpd], ls = '--', c=plot_colour, label = r'TPD Landau Cutoff ($\theta$ =' + f'{np.round(theta, 1)}\N{DEGREE SIGN})')

    def x_vs_kx(self, n_e, theta, x, ax):


        k1_x, k2_x = tpd.tpd_wns_pairs(self.v_th, n_e, theta, self.lambda_0, componants = 'x')
        ax.plot(x, k1_x, c=plot_colour)
        if theta == 'max_lin_growth':
            ax.plot(x, k2_x, c=plot_colour, label = r'TPD EPW ($\theta$ = $\theta_{MLG}$)')
        else:
            ax.plot(x, k2_x, c=plot_colour, label = r'TPD EPW ($\theta$ =' + f'{np.round(theta, 1)}\N{DEGREE SIGN})')

        ax.plot(x, -k1_x, c=plot_colour)
        ax.plot(x, -k2_x, c=plot_colour)

        # Find location where LW are Landau damped
        landau_cutoff_tpd = tpd.landau_cutoff_index(self.T_e, n_e, self.lambda_0, theta, cutoff = 0.3)
        if landau_cutoff_tpd is not None:
            if theta == 'max_lin_growth':
                ax.axvline(x[landau_cutoff_tpd], ls = '--', c=plot_colour, label = r'TPD Landau Cutoff ($\theta$ = $\theta_{MLG}$)')
            else:
                ax.axvline(x[landau_cutoff_tpd], ls = '--', c=plot_colour, label = r'TPD Landau Cutoff ($\theta$ =' + f'{np.round(theta, 1)}\N{DEGREE SIGN})')
