# -*- coding: utf-8 -*-
"""
Created on Fri Apr  4 13:51:59 2025

@author: alexa
"""



import numpy as np
def norm(vec):
    return np.linalg.norm(vec)



x_test = np.array([1,2,3])
y_test = np.array([2,3,4])
z_test = np.array([5,6,7])

v_total = np.sqrt(x_test**2 + y_test**2 + z_test**2)

