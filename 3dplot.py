# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 13:23:42 2025

@author: alexa
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 22:08:35 2025

@author: alexa
"""



#Import Python files
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D




#Assignment Data

mu_sun = 1.3271244*10**11
mu_earth = 3.986*10**5
radius_earth = 6378.14
J_2_earth = 1082.63*10**-6
mu_moon = 4902.8
radius_moon = 1737.4
r_mean = 384400


def norm(vec):
    return np.linalg.norm(vec)



def newton_interation(y, dydx, init_guess, tol, i=1000):
    x = init_guess

    
        




pi = np.pi
def radians(deg):
    rads = deg * (pi/180)
    return rads

E_ae0 = [2460705.5, 1.495988209443421E+08, 1.669829008180246E-02, radians(3.248050135173038E-03), 
         radians(1.744712892867145E+02), radians(2.884490093009512E+02), radians(2.621190445180298E+01)]


A_ae0 = [2460705.5, 3.764419202360106E+08, 6.616071771587672E-01, radians(3.408286057191753),
         radians(2.713674649188756E+02), radians(1.343644678984687E+02), radians(1.693237490356061E+01)]
ecc = E_ae0[2]


'''
Newton-Rasphon Method to return the eccentric anamoly to a certain tolerance
'''
def Kepler(e, M, tol = 1e-12, max_i = 1000):
    E = M
    
    
    for i in range(max_i):
        
        f_E = E - e * np.sin(E) - M
        f_prime = 1 - e * np.cos(E)
        
        del_E = f_E / f_prime
        
        
        E_new = E - del_E
        if np.abs(del_E) < tol:
            return E_new
        E = E_new
 
        

'''
Function to calculate the true anamoly from the eccentric anamoly
'''
def theta(E_anomaly):
    theta = 2*np.arctan(np.tan(E_anomaly/2) * ((1+ecc)/(1-ecc))**(0.5))
    return theta

'''
Function to return some mean anamoly at some time t
'''

def mean_anamoly(del_t, M_old, a):
    nu_t = (mu_sun / (a**3))**0.5
    M_new = M_old + nu_t * (del_t)
    
    return M_new
 

# Question 1: Calculate the true anamoly of Earth (held in a variable called 'trueTheta_earth')   
E_earth = Kepler(E_ae0[2], E_ae0[6]) # Eccentric anamoly
#print(E_earth) 
trueTheta_earth = theta(E_earth)     # True anamoly
#print(trueTheta_earth)
        

# Question 2 part a: Calculate the true anamoly of the Asteroid at t0 
E_asteroid = Kepler(A_ae0[2], A_ae0[6])
#print(E_asteroid)
trueTheta_asteroid = theta(E_asteroid)
#print(trueTheta_asteroid)


#Question 2 part b: Calculate the true anamoly of the Asteroid at t0 + 100 days
mean_100 = mean_anamoly(100, A_ae0[6], A_ae0[1])
E_100 = Kepler(A_ae0[2], mean_100)
trueTheta_100 = theta(E_100)
#print(trueTheta_100)



#--------------------------------------------------------------


def COE2RV(arr, mu):
    state_x = arr
    time = state_x[0]
    a = state_x[1]
    e = state_x[2]
    i = state_x[3]
    Omega = state_x[4]
    omega = state_x[5]
    mean_a = state_x[6]
    
    h = np.sqrt(mu * a * (1 - e**2))
    
    eccentric_a = Kepler(e, mean_a)
    theta_var = theta(eccentric_a)
    r = a*(1-(e**2))/(1 + e*np.cos(theta_var))
    
    arr_r = np.array([r*np.cos(theta_var), r*np.sin(theta_var), 0])
    arr_v = (mu/h)* np.array([-np.sin(theta_var), e + np.cos(theta_var), 0])
    
    
    
    

    #Rotate position and velocity from perifocal to inertial frame using the transfomration matrix
  
    cos_O, sin_O = np.cos(Omega), np.sin(Omega)
    cos_i, sin_i = np.cos(i), np.sin(i)
    cos_w, sin_w = np.cos(omega), np.sin(omega)
    
    R = np.array([
        [cos_O * cos_w - sin_O * sin_w * cos_i, -cos_O * sin_w - sin_O * cos_w * cos_i, sin_O * sin_i],
        [sin_O * cos_w + cos_O * sin_w * cos_i, -sin_O * sin_w + cos_O * cos_w * cos_i, -cos_O * sin_i],
        [sin_w * sin_i, cos_w * sin_i, cos_i]
    ])
    
    r_ijk = R @ arr_r
    v_ijk = R @ arr_v
    return r_ijk, v_ijk




# state_vector_0 = np.array(COE2RV(E_ae0, mu_sun))
# print(state_vector_0) # t = t0


# E_ae0[6] = mean_anamoly(100, E_ae0[6], E_ae0[1])
# state_vector_100 = np.array(COE2RV(E_ae0, mu_sun))
# print(state_vector_100) # t = t0 + 100

# print('\n')
# print(state_vector_0 - state_vector_100)



#--------------------------------------------------------------

def Ephemeris(t, OBJdata, mu):
    
    time = OBJdata[0]
    a = OBJdata[1]
    e = OBJdata[2]
    i = OBJdata[3]
    Omega = OBJdata[4]
    omega = OBJdata[5]
    mean_a = OBJdata[6]
    
    
    
    nu_t = (mu / (a**3))**0.5
    # del_t = t - time
    M_new = mean_a + nu_t * (t)
    
    
    h = np.sqrt(mu * a * (1 - e**2))
    
    eccentric_anamoly = Kepler(e, M_new)
    theta_var = theta(eccentric_anamoly)
    r = a*(1-(e**2))/(1 + e*np.cos(theta_var))
    
    arr_r = np.array([r*np.cos(theta_var), r*np.sin(theta_var), 0])
    arr_v = (mu/h)* np.array([-np.sin(theta_var), e + np.cos(theta_var), 0])
    
    
    
    

    #Rotate position and velocity from perifocal to inertial frame using the transfomration matrix
  
    cos_O, sin_O = np.cos(Omega), np.sin(Omega)
    cos_i, sin_i = np.cos(i), np.sin(i)
    cos_w, sin_w = np.cos(omega), np.sin(omega)
    
    R = np.array([
        [cos_O * cos_w - sin_O * sin_w * cos_i, -cos_O * sin_w - sin_O * cos_w * cos_i, sin_O * sin_i],
        [sin_O * cos_w + cos_O * sin_w * cos_i, -sin_O * sin_w + cos_O * cos_w * cos_i, -cos_O * sin_i],
        [sin_w * sin_i, cos_w * sin_i, cos_i]
    ])
    
    r_ijk = R @ arr_r
    v_ijk = R @ arr_v
    
    x = r_ijk[0]
    y = r_ijk[1]
    z = r_ijk[2]
    
    vx = v_ijk[0]
    vy = v_ijk[1]
    vz = v_ijk[2]
    return np.array([x, y, z]), np.array([vx, vy, vz]) 







t_array = (3600*24)*np.arange(0,365, 1)
pos_array_E = []
for t in t_array:
    
    normed_vec = norm(Ephemeris(t,E_ae0, mu_sun)[0])
    #print(normed_vec)
    pos_array_E.append(normed_vec)

position_magnitude_E = np.array(pos_array_E)


#velocities_E = np.array([Ephemeris(t,E_ae0, mu_sun)[1] for t in t_array])  # Extract velocity

pos_array_A = []
for t in t_array:
    
    normed_vec = norm(Ephemeris(t,A_ae0, mu_sun)[0])
    #print(normed_vec)
    pos_array_A.append(normed_vec)

position_magnitude_A = np.array(pos_array_A)
    
    
#velocities_A = np.array([Ephemeris(t,A_ae0, mu_sun)[1] for t in t_array])




def time_orbit(a, μ):
    T = 2 * np.pi * np.sqrt(a**3 / μ)
    time = T 
    return time

time_Earth = time_orbit(E_ae0[1], mu_sun)


orbital_percent_E = (t_array / time_Earth) 

time_Asteroid = time_orbit(A_ae0[1], mu_sun)
orbital_percent_A = (t_array / time_Asteroid) 

val = False
if val:
    
    plt.plot(orbital_percent_E, position_magnitude_E, label="|r| - Earth", color='b')
    plt.plot(orbital_percent_A, position_magnitude_A, label="|r| - Asteroid ", color='r')
    plt.legend()
    plt.show()

#----------------------------------------------------

years = 10
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

t_values = (3600*24)*np.arange(0,(years*365), 1)


r_earth = np.array([Ephemeris(t, E_ae0, mu_sun)[0] for t in t_values])
r_asteroid = np.array([Ephemeris(t, A_ae0, mu_sun)[0] for t in t_values])

# Extract x, y, z components
x_earth, y_earth, z_earth = r_earth[:, 0], r_earth[:, 1], r_earth[:, 2]
x_ast, y_ast, z_ast = r_asteroid[:, 0], r_asteroid[:, 1], r_asteroid[:, 2]



ax.plot(x_earth, y_earth, z_earth, label="Earth Orbit", color="b")

# Plot Asteroid orbit
ax.plot(x_ast, y_ast, z_ast, label="Asteroid Orbit", color="r")

# Mark the Sun at (0,0,0)
ax.scatter(0, 0, 0, color='yellow', s=100, label="Sun")

# Labels and title
ax.set_xlabel("X (AU)")
ax.set_ylabel("Y (AU)")
ax.set_zlabel("Z (AU)")
ax.set_title("3D Orbits of Earth and Asteroid")
ax.legend()
ax.grid()

plt.show()














