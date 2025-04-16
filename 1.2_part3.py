# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 22:08:35 2025

@author: alexa
"""



#Import Python files
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import solve_ivp

show_plots = False


#Assignment Data

mu_sun = 1.3271244*10**11
mu_earth = 3.986*10**5
radius_earth = 6378.14
J_2_earth = 1082.63*10**-6
mu_moon = 4902.8
radius_moon = 1737.4
r_mean = 384400


def time_orbit(a, μ):
    T = 2 * np.pi * np.sqrt(a**3 / μ)
    time = T 
    return time

parking_altitude = 220
r_p = parking_altitude + radius_earth
parking_velocity = np.sqrt(mu_earth/r_p)


r_a_array = radius_moon*np.arange(1.1, 1.4, 0.01)

a = 0.5 * (r_p + r_a_array)
e = (r_a_array - r_p) / (r_a_array + r_p)


cos_theta2_array = np.zeros(r_a_array.shape[0])


for i in range(r_a_array.shape[0]):
    cos_theta2_array[i] = (a[i] * (1 - e[i]**2) / r_a_array[i] - 1) / e[i]

# semi_major_axis = 0.5*(parking_radius + radius_apogee)
# #print(semi_major_axis)

# transfer_e = (radius_apogee - parking_radius) / (radius_apogee + parking_radius)


# cos_theta_A = (semi_major_axis * ((1 - transfer_e**2) / radius_apogee) - 1) / transfer_e


#cos_theta_A = np.clip(cos_theta_A, -1.0, 1.0)


theta_2 = np.arccos(cos_theta_A)
print(cos_theta_A)




















theta_eccentric = 2*np.arctan(np.tan(theta_2/2) * ((1-transfer_e)/(1+transfer_e))**(0.5))



theta_mean = theta_eccentric - transfer_e*np.sin(theta_eccentric)


time_total = time_orbit(semi_major_axis, mu_earth)

delta_t = (time_total / (2 * np.pi)) * theta_mean

#print(delta_t)

    










