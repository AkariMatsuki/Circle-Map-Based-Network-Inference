import numpy as np
from sklearn.linear_model import LinearRegression
import sys


import scipy


def growth_period(x, K):
    return x[K:] - x[:-K]

# Maximum likelihood estimation based on the circle map with fixed alpha
def MLE_cm(x, dt, alpha): 
    N = np.size(x, axis=0) # number of oscillators
    L = np.size(x, axis=1)  # number of data points 
    
    omega, b, sigma = np.array([]), np.array([]), np.array([])
    
    ##### Estimation of the basic period #####
    omega_eff_j = (x[:, -1] - x[:, 0]) / ((L-1)*dt)
    T_mean = np.mean(2*np.pi / omega_eff_j)
    K_j = round(T_mean / dt) # number of data points per cycle

    for j in range(N):
        
        x_dec = x[:, ::K_j]
        L_dec = np.size(x_dec, axis=1)
        sin_dX = np.sin(x_dec.T - (x_dec[j]).reshape(L_dec, 1) + alpha ) 
        sin_dX = np.delete(sin_dX, j, axis=1) # delete self-coupling, which is not inferred
        

        Dphi = growth_period(x_dec[j], 1)
        X = sin_dX[:-1]*K_j*dt  

        
        reg = LinearRegression().fit(X, Dphi )
        b_j_reg = reg.coef_.reshape(1, N-1)
        b_j = np.insert(b_j_reg, j, 0, axis=1) # insert zero self-couplings
        
        b = np.append(b.reshape(j, N), b_j, axis=0 ) # size of b is (j, N)
        omega = np.append(omega, reg.intercept_/(K_j*dt)) 

        couple_term = (b_j_reg * X/ (K_j*dt)).T
        sum_res_j = np.sum(( Dphi - ( reg.intercept_ / (K_j*dt) + np.sum(couple_term, axis=0) ) *K_j*dt )**2 )
        v =  sum_res_j / ((L_dec-1)*K_j*dt)
        sigma = np.append(sigma, np.sqrt(v))
        
    return omega, b, sigma

# Function for optimization of alpha
def opt_func(alpha, x, dt): 
    N = np.size(x, axis=0) # number of oscillators
    L = np.size(x, axis=1)  # number of data points 
    omega, b, sigma = np.array([]), np.array([]), np.array([])
    ll = 0 ###### log-likelihood 

    ##### Estimation of the basic period #####
    omega_eff_j = (x[:, -1] - x[:, 0]) / ((L-1)*dt)
    T_mean = np.mean(2*np.pi / omega_eff_j)
    K_j = round(T_mean / dt) # number of data points per cycle

    for j in range(N):
    
        x_dec = x[:, ::K_j]
        L_dec = np.size(x_dec, axis=1)

        sin_dX = np.sin(x_dec.T - (x_dec[j]).reshape(L_dec, 1) + alpha  ) ### shape (L_dec, N)
        sin_dX = np.delete(sin_dX, j, axis=1) # delete self-coupling, which is not inferred
        

        Dphi = growth_period(x_dec[j], 1)
        X = sin_dX[:-1]*K_j*dt  

        
        reg = LinearRegression().fit(X, Dphi )
        b_j_reg = reg.coef_.reshape(1, N-1)
        b_j = np.insert(b_j_reg, j, 0, axis=1) # insert zero self-couplings
        
        b = np.append(b.reshape(j, N), b_j, axis=0 ) # size of b is (j, N)
        omega = np.append(omega, reg.intercept_/(K_j*dt)) 

        couple_term = (b_j_reg * X/ (K_j*dt)).T
        sum_res_j = np.sum(( Dphi - ( reg.intercept_ / (K_j*dt) + np.sum(couple_term, axis=0) ) *K_j*dt )**2 )
        v =  sum_res_j / ((L_dec-1)*K_j*dt)
        #print(np.sqrt(v))
        sigma_j = np.sqrt(v)
        sigma = np.append(sigma, sigma_j)
        ll += - (L_dec-1) * np.log(sigma_j) - sum_res_j / (2*sigma_j**2*K_j*dt)
    print(alpha, ll) 
    return -ll

# Infer parameters including natural frequencies, coupling strengths, noise strengths, and alpha.
def couple_inf_cm(x, dt):
    opt_alpha, opt_l_minus, ierr, numfunc =scipy.optimize.fminbound(func=opt_func, x1=-np.pi/2, x2 =np.pi/2, args=(x, dt), full_output=True)
    print(opt_l_minus, numfunc)
    omega, b, sigma = MLE_cm(x, dt, opt_alpha)
    return omega, b, sigma, opt_alpha





######### Load phase data ########
file_path_phase = [file_path_to_phase_data]
phase = np.load(file_path_phase)
##################################

########## Coupling inference ###############
tau = 0.01 #sampling_interval_of_phase_data
omega, b, sigma, alpha = couple_inf_cm(phase, tau)
#################################


######## save inference results #######
file_path_b = [file_path_to_inferred_coupling_stregnths]
np.save(file_path_b, b)

file_path_omega = [file_path_to_inferred_natural_freqs]
np.save(file_path_omega, omega)

file_path_sigma = [file_path_to_inferred_noise_stregnths]
np.save(file_path_sigma, sigma)

file_path_alpha = [file_path_to_inferred_alpha]
np.save(file_path_alpha, alpha)
########################################


