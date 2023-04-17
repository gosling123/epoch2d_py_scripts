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


em_color = 'green'
epw_color = 'magenta'


class plots:

    def __init__(self, T_e, n_e, theta, lambda_0):

        self.T_e = T_e
        self.n_e = n_e
        self.theta = theta
        self.lambda_0 = lambda_0

        self.v_th = plasma.electron_thermal_speed(self.T_e)

    def kx_vs_omega_EPW(self, ax):

        k_epw = srs.srs_EPW_k_x(self.v_th, self.n_e, self.theta, self.lambda_0)
        omega_epw = srs.srs_omega_EPW(self.n_e, self.T_e, self.theta, self.lambda_0)

        ax.plot(k_epw, omega_epw, c=epw_color, label = 'SRS EPW')    

    def kx_vs_omega_EM(self, ax):   
            
            
        k_em = srs.srs_EM_k_x(self.v_th, self.n_e, self.theta, self.lambda_0) 
        omega_em = srs.srs_omega_EM(self.n_e, self.T_e, self.theta, self.lambda_0)

        ax.plot(k_em, omega_em, c=em_color, label = 'SRS EM')