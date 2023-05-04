import numpy as np
import calculations.plasma_calculator as plasma
import calculations.srs_calculator as srs
import calculations.tpd_calculator as tpd
import scipy.constants as const
import matplotlib.pyplot as plt
import sys

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

# conversion from keV to kelvin
keV_to_K = (const.e*1e3)/const.k

T_e = 4.3*keV_to_K 
lambda_0 = 1.314e-6 

v_th = plasma.electron_thermal_speed(T_e)

n_e = np.linspace(0.06, 0.25, 100)

angles = np.linspace(0, 360, 100)

for i, angle in enumerate(angles):
    print_progress_bar(i+1, len(angles), label=f'TPD theta = {np.round(angle,2)}')

    omega1, omega2 = tpd.tpd_omegas(n_e, T_e, angle, lambda_0)

    plt.plot(n_e, omega1-0.5, c='black', alpha = 0.2)
    plt.plot(n_e, omega2-0.5, ls='--', c='black', alpha=0.2)

print('',end='\n')


for i, angle in enumerate(angles):
    print_progress_bar(i+1, len(angles), label=f'SRS theta = {np.round(angle,2)}')

    omega = srs.srs_omega_EPW(n_e, T_e, angle, lambda_0)

    plt.plot(n_e, omega-0.5, c='blue', alpha = 0.2)

print('',end='\n')

plt.axhline(y=-0.025, c='r', ls='--', lw=3)
plt.axhline(y=0.025, c='r', ls='--', lw=3)
plt.xlim(0.18, 0.25)
plt.ylim(-0.1, 0.1)

plt.savefig('wns.png')

