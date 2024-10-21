# Network inference based on circle map from oscillatory signals
Network inference from oscillatory signals based on circle map 

## script
Script used in "Network inference applicable to both synchronous and desynchronous systems from oscillatory"
- brusselator_sim.py: generate the synthetic data based on the Brusselator model.
- clock_cell_sim.py: generate synthetic data based on clock cell model (Locke et al BMC systems biolo, 2008)
- couopl_inf_cm.py: circle-map-based coupling inference
- coupl_inf_naive.py: naive coupling inference 

## data
Data used in "Network inference applicable to both synchronous and desynchronous systems from oscillatory". 
All data are synthetic data.
- mat_nointer.npy: coupling strength matrix of clock cell network with no inter-group interaction 
- mat_unidir.npy: coupling strength matrix of clock cell network with unidirectional interaction
- mat_bidir.npy: coupling strength matrix of clock cell network with bidirectional interaction
- network_brusselator.npy: coupling strength matrix of Brusselator oscillators network.
- props_oscillators.npy: Values of A_i in Brusselators model and tau_i in 



