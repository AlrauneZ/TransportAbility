#!/usr/bin/env python3
"""
Python script to run upscaling simulation for transport ability as random 
function of porosity (statistics) based on observed porosity characteristics
of catalytical particles, analysis of sub-REV behaviour

- generation of ensemble of networks 
- statistical analysis of networks
- calculation of effective values of networks
"""

# import numpy as np
from TA_Simulation import TA_Ensemble_Simulation

### Ensemble settings 
settings = dict(
    mat_type = 'FCC_2_1',
    dim = 2,        # dimensionality of networks: 2D or 3D
    res = 2,        # measurement resolution
    n_network = 4,  # number of nodes per direction within each network
    pmean = 0.3,    # porosity statistics of nodes
    pstd = 0.15,    
    n_ens=5,        # number or networks
    ens_seed = 123456*7*8,
)

ENS_TA1 = TA_Ensemble_Simulation(
    task_root = '../results/FCC_2-1_d{}_r{}-{}_N{}'.format(settings['dim'],settings['res'],settings['res']*settings['n_network'],settings['n_ens']),
    **settings
    )

### setup ta statistics (of individual network nodes) from input data
ENS_TA1.set_ta(file_data = '../data/FCC_2-1_por_ta_data_d2_r2.csv')

### sample network node values of porosity, connectivity and ta 
ENS_TA1.sample()

### calculate and save statistics of networks (from individual nodes)
ENS_TA1.statistics() 
ENS_TA1.save_ensemble_samples()

### run numerical transport simulation on each network to gain ta_eff of networks
ENS_TA1.ta_network_flow()
ENS_TA1.save_ensemble_taeff()

###-------------------------------------------------------------------------###
