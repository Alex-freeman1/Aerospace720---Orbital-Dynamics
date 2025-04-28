# -*- coding: utf-8 -*-
"""
Created on Sat Apr 26 18:01:19 2025

@author: alexa
"""

#Import Python files
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import solve_ivp
from scipy import signal

show_plots = True



#Assignment Data

mu_sun = 1.3271244*10**11
mu_earth = 3.986*10**5
radius_earth = 6378.14
J_2_earth = 1082.63*10**-6
mu_moon = 4902.8
radius_moon = 1737.4
r_mean = 384400


'''
Functions
'''

def norm(vec):
    return np.linalg.norm(vec)

pi = np.pi
def radians(deg):
    rads = deg * (pi/180)
    return rads


def get_mean_anamoly(del_t, M_old, a):
    nu_t = (mu_sun / (a**3))**0.5
    mean_anamoly_t = M_old + nu_t * (del_t)
    
    return mean_anamoly_t

def time_orbit(a, μ):
    T = 2 * np.pi * np.sqrt(a**3 / μ)
    time = T 
    return time

def rotation_matrix(i, Omega, omega):
    cos_O, sin_O = np.cos(Omega), np.sin(Omega)
    cos_i, sin_i = np.cos(i), np.sin(i)
    cos_w, sin_w = np.cos(omega), np.sin(omega)
    
    R = np.array([
        [cos_O * cos_w - sin_O * sin_w * cos_i, -cos_O * sin_w - sin_O * cos_w * cos_i, sin_O * sin_i],
        [sin_O * cos_w + cos_O * sin_w * cos_i, -sin_O * sin_w + cos_O * cos_w * cos_i, -cos_O * sin_i],
        [sin_w * sin_i, cos_w * sin_i, cos_i]
    ])
    
    return R

'''
Newton-Rasphon Method to return the eccentric anamoly to a certain tolerance
'''

#1.1.1 ----------------------------------------------------------------------


def Kepler(e, M, tol = 1e-12, max_i = 1000):
    E = M
    
    
    for i in range(max_i):
        
        f_E = E - e * np.sin(E) - M
        f_prime = 1 - e * np.cos(E)
        
        del_E = f_E / f_prime
        
        
        E_new = E - del_E
        if np.abs(del_E) < tol:
            theta = 2*np.arctan(np.tan(E_new/2) * ((1+e)/(1-e))**(0.5))
            return theta
        E = E_new
 
        



    
E_ae0 = [2460705.5, 1.495988209443421E+08, 1.669829008180246E-02, radians(3.248050135173038E-03), 
          radians(1.744712892867145E+02), radians(2.884490093009512E+02), radians(2.621190445180298E+01)]


A_ae0 = [2460705.5, 3.764419202360106E+08, 6.616071771587672E-01, radians(3.408286057191753),
          radians(2.713674649188756E+02), radians(1.343644678984687E+02), radians(1.693237490356061E+01)]



#1.1.2 ----------------------------------------------------------------------


trueAnamoly_asteroidt_0 = Kepler(A_ae0[2], A_ae0[6])
meanAnamolyt_100 = get_mean_anamoly(100*(3600*24), A_ae0[6], A_ae0[1])
trueAnamoly_asteroidt_100 = Kepler(A_ae0[2], meanAnamolyt_100)

# print(trueAnamoly_asteroidt_0)
# print(trueAnamoly_asteroidt_100)



Obj2_t0 = A_ae0.copy()
Obj2_t0[6] = trueAnamoly_asteroidt_0
Obj2_t0 = Obj2_t0[1:]


Obj2_t100 = A_ae0.copy()
Obj2_t100[6] = trueAnamoly_asteroidt_100
Obj2_t100 = Obj2_t100[1:]

#1.1.3 ----------------------------------------------------------------------

t_0 = 2460705.5*(3600*24)

def COE2RV(arr, mu):
    a, e, i, Omega, omega, theta_var = arr[0:6]
    h = np.sqrt(mu * a * (1 - e**2))
    r = a*(1-(e**2))/(1 + e*np.cos(theta_var))
    
    arr_r = np.array([r*np.cos(theta_var), r*np.sin(theta_var), 0])
    arr_v = (mu/h)* np.array([-np.sin(theta_var), e + np.cos(theta_var), 0])

    #Rotate position and velocity from perifocal to inertial frame using the transfomration matrix
    R_matrix = rotation_matrix(i, Omega, omega)
   
    r_ijk = R_matrix @ arr_r
    v_ijk = R_matrix @ arr_v
    return r_ijk, v_ijk




state_vector_0 = np.array(COE2RV(Obj2_t0, mu_sun))
#print(state_vector_0) # t = t0



state_vector_100 = np.array(COE2RV(Obj2_t100, mu_sun))
#print(state_vector_100) # t = t0 + 100

# print('\n')
# print(state_vector_0 - state_vector_100)

t_0_days = 2460705.5
days_convert = 3600*24
#1.1.4 ----------------------------------------------------------------------

def Ephemeris(t, OBJdata, mu):

    time, a, e, i, Omega, omega, mean_anamoly = OBJdata[0:7]
    nu_t = (mu / (a**3))**0.5
    
    t = t - t_0_days*days_convert
    mean_anamoly_t = mean_anamoly + nu_t * (t)

    h = np.sqrt(mu * a * (1 - e**2))
    
    theta_var = Kepler(e, mean_anamoly_t)
    r = a*(1-(e**2))/(1 + e*np.cos(theta_var))
    
    arr_r = np.array([r*np.cos(theta_var), r*np.sin(theta_var), 0])
    arr_v = (mu/h)* np.array([-np.sin(theta_var), e + np.cos(theta_var), 0])
    
    R_matrix = rotation_matrix(i, Omega, omega)
    r_ijk = R_matrix @ arr_r
    v_ijk = R_matrix @ arr_v
    
    return r_ijk, v_ijk




years_shown_i = 1
t_array = days_convert*np.arange(t_0_days,t_0_days+years_shown_i*365, 1) 

x_earth = np.zeros((6,len(t_array)))
x_asteroid = np.zeros((6,len(t_array)))

 
for r in range(len(t_array)):
    x_earth[0:6, r] = np.hstack(Ephemeris(t_array[r], E_ae0, mu_sun))
    x_asteroid[0:6, r] = np.hstack(Ephemeris(t_array[r], A_ae0, mu_sun))
    

t_day_array = np.arange(0, years_shown_i*365, 1)
time_Earth = time_orbit(E_ae0[1], mu_sun)/days_convert # Convert from seconds to days
orbital_percent_E = (t_day_array / time_Earth) 


plt.plot(orbital_percent_E,[norm(x_earth[0:3, t]) for t in range(len(t_array))], label="Earth position", color='b')
plt.plot(orbital_percent_E, [norm(x_asteroid[0:3, t]) for t in range(len(t_array))], label="Asteroid position", color='r')
plt.title("Magnitude of the Position of Earth and 2024 YR4 Asteroid")
plt.xlabel("Percent of Earth's Orbit (%)")
plt.ylabel("Position Vector (km)")
if show_plots:
    plt.show()
else:
    plt.close()

plt.figure()
plt.plot(orbital_percent_E,[norm(x_earth[3:, t]) for t in range(len(t_array))], label="Earth velocity", color='b')
plt.plot(orbital_percent_E, [norm(x_asteroid[3:, t]) for t in range(len(t_array))], label="Asteroid velocity", color='r')
plt.legend()
plt.xlabel("Percent of Earth's Orbit (%)")
plt.ylabel("Velocity Vector (km/s)")
plt.title("Magnitude of the Position of Earth and 2024 YR4 Asteroid")

if show_plots:
    plt.show()
else:
    plt.close()
    
years_shown = 10
t_total = days_convert*np.arange(t_0_days, t_0_days + years_shown*365, 1)



normed_diff = []
for t in t_total:
    
    normed_diff.append(norm(Ephemeris(t,E_ae0, mu_sun)[0] - Ephemeris(t,A_ae0, mu_sun)[0]))
    
    
plt.figure()
#plt.plot(t_total*(10/t_total[-1]), normed_diff)
plt.plot(t_total/(days_convert*365), normed_diff)
plt.title("Absolute distance seperation of Earth and 2024 YR4")
plt.xlabel("Time in J2000 (years)")
plt.ylabel("Distance Seperation (km)")
if show_plots:
    plt.show()
else:
    plt.close()
    
    



