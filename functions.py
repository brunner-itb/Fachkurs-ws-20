
# coding: utf-8

# In[ ]:

import numpy as np
from scipy.special import j0, j1, jn_zeros

#==============================================================================
# functions
#==============================================================================
def get_alpha_n(n_max,a):
    bessel_roots = jn_zeros(0, n_max)
    alpha_n = bessel_roots / a
    
    assert alpha_n.all() > 0
    
    return alpha_n

def get_coeff(a, l, alpha_n, R, R_resp, k_on, q, D):
    """
    returns coefficients A_n and B_n as list    
    """
    p = (k_on * R_resp) / (np.pi * D * a**2)
    
    denom_a = a * alpha_n * j1(alpha_n*a)
    denom_b = alpha_n * k_on * (R+R_resp) * np.cosh(alpha_n*l)
    denom_c = a**2 * alpha_n**2 * D * np.pi + k_on * R * p
    denom_d = np.sinh(alpha_n*l)
    denom = denom_a * (denom_b + denom_c * denom_d)
             
    #print R, denom_a, denom_b, denom_c
    assert denom != 0
    
    a_n = 2 * q * (alpha_n * np.cosh(alpha_n*l) + p * np.sinh(alpha_n*l)) / denom
    b_n = -2 * q * (p * np.cosh(alpha_n*l) + alpha_n * np.sinh(alpha_n*l)) / denom
    
    return [a_n, b_n]

def bessel_sum(r, z, n_max, a, l, R, R_resp, k_on, q, D):
    
    # for each n in n_max, get the corresponding alpha and coefficients
    alphas = get_alpha_n(n_max, a)
    coeff = [get_coeff(a, l, alpha_n, R, R_resp, k_on, q, D) for alpha_n in alphas]
    intensity = 0
    
    # calculate the sum of bessel functions times coefficients from n=1 up to n=n_max 
    for i in range(n_max):
        # get coefficients and alphas (previously calculated)
        # note the indexing, since n starts with 1
        a_n = coeff[i][0]
        b_n = coeff[i][1]
        alpha_n = alphas[i]

              
        intensity = intensity + (j0(alpha_n*r)*(a_n*np.cosh(alpha_n*z) + b_n*np.sinh(alpha_n * z)))
    
    if intensity < 0:
        intensity = 0
        
    assert intensity >= 0
    
    return intensity

