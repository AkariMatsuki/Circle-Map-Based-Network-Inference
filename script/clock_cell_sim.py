import numpy as np
import sdeint



def clock_cell(t, X, conmat, T=1.0, n=5.6645, V1=6.8355, K1=2.7266, V2=8.4297, K2=0.2910, 
                      Vc=6.7924, Kc=4.8283, L=0.0, k3=0.1177, 
                      V4=1.0841, K4=8.1343, k5=0.3352, 
                      V6=4.6645, K6=9.9849, k7=0.2282, V8=3.5216, K8=7.4519):
    '''
    X: x1,x2,..,xN, y1, y2,..., yN, z1, z2,..., zN, r1, r2,..., rN
    conmat: array with (N,N). conmat[i,i] = 1
    '''
    
    N = round(len(X)/4)
    
    x, y, z, r = X[:N], X[N:2*N], X[2*N:3*N], X[3*N:4*N]
    
    f = np.dot(conmat, r.reshape(N,1)).reshape(N,) 
    
    dx =T*( V1 * (K1**n)/(K1 **n + z **n) - V2 *x/(K2 + x) + Vc * (f) / (Kc+f) + L ) 
    dy = T*( k3 * x - V4*y / (K4 + y) ) 
    dz = T*( k5 * y - V6*z / (K6 + z) ) 
    dr = T*( k7 * x - V8*r / (K8 + r) ) 

    
    dX = np.block([dx, dy, dz, dr])
    return dX


def G_whitenoise(x, t, sigma):
    N = round(len(x))
    return np.identity(N)*sigma

path_to_output_file = "../output_clock_cell/simulation_clock_cell.npy" # Path to save the result

K0, Kself0=0.01, 0.9 # coupling strength and self-coupling strength.
n0 = 5.0
file_path_T = "../data/props_oscillators.npy" # path to data file of scaling factors tau_1, ..., tau_10 in the clock cell model 
T0 =  np.load(file_path_T) 
N0 = len(T0) # number of oscillators

sigma0 = 0.002


########## Coupling matrix ###########
con_mat0 = np.load("../data/mat_unidir.npy")*K0 # path to data file coupling matrix. 
    # mat_inter.npy: no inter-group interaction 
    # # mat_unidir.npy: unidirectional interaction 
    # mat_bidir.npy: two-way or bidirectional interaction
con_mat0 += np.identity(N0)*Kself0
######################################


############# Simulation #############
y_init0 = np.block([np.ones(N0)*2.1, np.ones(N0)*2.0, np.ones(N0)*1.6, np.ones(N0)*1.2]) # initial condition
tau0 = 0.04 # step size of simulation
sim_time0 = 10000 # total simulation time (the observation duration)

args0 = {'T':T0, 'n':n0}

def det_func(y,t):
    return clock_cell(t, y, conmat=con_mat0, **args0 )
def stc_func(y,t):
    return G_whitenoise(y, t, sigma=sigma0)

tspan0 = np.linspace(0.0, sim_time0-tau0, round(sim_time0/tau0))
result0 = sdeint.itoEuler(f=det_func, G=stc_func, y0=y_init0, tspan=tspan0).T
######################################


########## Save result #############
file_path_result = [path_to_output_file]
np.save(file_path_result, result0)
