import numpy as np
from sklearn.linear_model import LinearRegression
import sys
import scipy



def growth_period(x, K):
    return x[K:] - x[:-K]

# Maximum likelihood estimation in the naive method
def MLE_naive(x, dt, alpha): 
    N = np.size(x, axis=0) # number of oscillators
    L = np.size(x, axis=1)  # number of data points 
    
    omega, b, sigma = np.array([]), np.array([]), np.array([])
    
    for j in range(N):
        print(j)
        sin_dX = np.sin(x.T - (x[j]).reshape(L, 1) + alpha)
        sin_dX = np.delete(sin_dX, j, axis=1) ## self-couplingをなくす
       

        X = sin_dX[:-1]*dt
        Y = growth_period(x[j], 1)
        
        reg =  LinearRegression().fit(X, Y)
        
        b_j_reg = reg.coef_.reshape(1, N-1)
        b_j = np.insert(b_j_reg, j, 0, axis=1) ## sinのself-coupling =0を挿入
        
        b = np.append(b.reshape(j, N), b_j, axis=0 ) ####### bのサイズは(j, N)
        omega = np.append(omega, reg.intercept_/dt) 
        
        couple_term = (b_j_reg * X/ dt).T
        sum_res_j = np.sum(( Y - ( reg.intercept_ / (dt) + np.sum(couple_term, axis=0) ) *dt )**2 )
        v =  sum_res_j / ((L-1)*dt)
        sigma_j = np.sqrt(v)
        sigma = np.append(sigma, sigma_j)
        
    return omega, b, sigma

# Function for optimization of alpha
def opt_func_naive(alpha, x, dt):
    N = np.size(x, axis=0) # number of oscillators
    L = np.size(x, axis=1)  # number of data points 
    omega, b, sigma = np.array([]), np.array([]), np.array([])
    ll = 0 ###### log-likelihood

    for j in range(N):
        #print(j)
        sin_dX = np.sin(x.T - (x[j]).reshape(L, 1) + alpha)
        sin_dX = np.delete(sin_dX, j, axis=1) # delete self-coupling, which is not inferred
       

        X = sin_dX[:-1]*dt
        Y = growth_period(x[j], 1)
        
        reg =  LinearRegression().fit(X, Y)
        
        b_j_reg = reg.coef_.reshape(1, N-1)
        b_j = np.insert(b_j_reg, j, 0, axis=1) # sinのself-coupling =0を挿入
        
        b = np.append(b.reshape(j, N), b_j, axis=0 ) # size of b is (j, N)
        omega = np.append(omega, reg.intercept_/dt) 
        
        couple_term = (b_j_reg * X/ dt).T
        sum_res_j = np.sum(( Y - ( reg.intercept_ / (dt) + np.sum(couple_term, axis=0) ) *dt )**2 )
        v =  sum_res_j / ((L-1)*dt)
        sigma_j = np.sqrt(v)
        sigma = np.append(sigma, sigma_j)
        ll += - (L-1) * np.log(sigma_j) - sum_res_j / (2*sigma_j**2*dt)
        
    print(alpha, ll) 
    return -ll


def couple_inf_naive(x, dt):
    opt_alpha, opt_l_minus, ierr, numfunc =scipy.optimize.fminbound(func=opt_func_naive, x1=-np.pi/2, x2 =np.pi/2, args=(x, dt), full_output=True)
    omega, b, sigma = MLE_naive(x, dt, opt_alpha)
    return omega, b, sigma, opt_alpha



######### Load phase data ########
file_path_phase = [file_path_to_phase_data]
phase = np.load(file_path_phase)
##################################



########## 結合推定 ###########
tau = 0.1 #sampling_interval_of_phase_data
omega, b, sigma, alpha = couple_inf_naive(phase, tau)
#############################


######## save inference results #######
file_path_b = [file_path_to_inferred_coupling_stregnths]
np.save(file_path_b, b)

file_path_omega = [file_path_to_inferred_natural_freqs]
np.save(file_path_omega, omega)

file_path_sigma = [file_path_to_inferred_noise_stregnths]
np.save(file_path_sigma, sigma)

file_path_alpha = [file_path_to_inferred_alpha]
np.save(file_path_alpha, alpha)


