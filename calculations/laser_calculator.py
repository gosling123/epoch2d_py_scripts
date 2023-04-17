#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

plasma_calculator.py

Script housing functions used to calculate common laser 
quantities.

"""

# import libraries
import numpy as np
import scipy.constants as const

def omega(lambda_0):
    return 2.0 * const.c * const.pi / lambda_0

def wavenumber(lambda_0):
    return 2.0 * const.pi / lambda_0

def E_normalisation(lambda_0):
    omega_0 = omega(lambda_0)
    return const.e / (const.m_e * omega_0 * const.c)

def B_normalisation(lambda_0):
    omega_0 = omega(lambda_0)
    return const.e / (const.m_e * omega_0)