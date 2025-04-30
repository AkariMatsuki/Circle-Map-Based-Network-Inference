import numpy as np
import sdeint

def brusselator(t, X, A, B, d, conmat): # function for Brusselator model
    '''
    X: x1,x2,..,xN, y1, y2,..., yN
    A: array with (N,)
    B: array with (N,)
    conmat: array with (N,N)
    '''
    N = round(len(X)/2)
    x,y = X[:N], X[N:]
    
    dis_x = x.reshape(N,1) - x.reshape(1,N)
    dis_y = y.reshape(N,1) - y.reshape(1,N)
    
    coupling_x = np.diag(np.dot(conmat, dis_x)).reshape(N)
    coupling_y = d*np.diag(np.dot(conmat, dis_y)).reshape(N)
    
    dx = A + x**2*y - B*x - x + coupling_x
    dy = B*x - x**2 *y + coupling_y
    
    dX = np.append(dx, dy)
    return dX

def G_whitenoise(x,t,sigma): # function for white noise
    N = round(len(x))
    return np.identity(N)*sigma

def simulation_brusselator(y_init, tau, sim_time, A, B, d, conmat, sigma): # function for simulation
    tspan = np.linspace(0.0, sim_time-tau, round(sim_time/tau))
    T = len(tspan)

    def det_func(y,t):
        return brusselator(t, y, A, B, d, conmat)
    def stc_func(y,t):
        return G_whitenoise(y,t,sigma)
    
    result = sdeint.itoEuler(f=det_func, G=stc_func, y0=y_init, tspan=tspan).T
    return result


# Main simulation
path_to_output_file = "../output_Brusselator/simulation_brusselator.npy" # Path to save the result

######## Parmeter setting ###########
K=0.001
A0 = np.load("../data/props_oscillators.npy") # Value of A_i in Brusseltor model.
M = len(A0) # number of oscillators
Bc = 1 + A0**2 
mu0= 0.04 
B0 = Bc*(1+mu0)
sigma0 = 0.002
d0 = 5/4
print("mu, d, K, sigma", mu0, d0, K, sigma0)
print("A", A0)
######################################


########## Coupling matrix ###########
con_mat0 = np.load("../output_Brusselator/network_brusselator.npy").T * K 
######################################



############# Simulation #############
y_init0 = np.append(A0 , B0) # Initial condition
tau0 = 0.01 # Time step
sim_time0 = 300000 # Simulation time
result0 = simulation_brusselator(y_init0, tau0, sim_time0, A0, B0, d0, con_mat0, sigma0) # Simulation
######################################


########## Save result #############
file_path_result = [path_to_output_file]
np.save(file_path_result, result0)