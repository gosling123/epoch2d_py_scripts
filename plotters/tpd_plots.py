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


plot_colour = 'black'


class plots:

    def __init__(self, T_e, n_e, theta, lambda_0):

        self.T_e = T_e
        self.n_e = n_e
        self.theta = theta
        self.lambda_0 = lambda_0

        self.v_th = plasma.electron_thermal_speed(self.T_e)

    def kx_vs_omega(self, ax):

        k1, k2 = tpd.tpd_wns_pairs(self.v_th, self.n_e, self.theta, self.lambda_0, componants = 'x')
        omega1, omega2 = tpd.tpd_omegas(self.n_e, self.T_e, self.theta, self.lambda_0)

        ax.plot(k1, omega1, c=plot_colour)
        ax.plot(k2, omega2, c=plot_colour, label = 'TPD EPW')
