import os
import nest
import numpy as np
import pylab as plt
from os.path import join


dt = 0.01
t_simulation = 200.0
state = "gamma"
nthreads = 1
common_params = {    # Parameters common to all neurons.
    'C_m': 100.,
    'E_L': -60.,
    'Delta_T': 2.,
    'V_reset': -65.,
    'V_th': -50.,
    'E_in': -80.0,  # Reversal potential
    'E_ex': 0.0,
    'tau_syn_ex': 4.,  # Alpha function tau
    'tau_syn_in': 3.,
    # "V_m": -70.0
}
common_params['g_L'] = common_params['C_m'] / 10.0

E_cell_params = dict(common_params, **{'Ncells': 200,
                                       'IXmean': 30.0,
                                       'IXsd': 20.0,
                                       'p_rate': 1.0,
                                       'Nrec': 50})

I_cell_params = dict(common_params, **{'Ncells': 50,
                                       'IXmean': 30.,
                                       'IXsd': 80.,
                                       'p_rate': 400.0,
                                       'Nrec': 50})
IIp = {  # I - cell to I - cell
    # 'Tau': 3.,  # Alpha function tau
    'weight': -1.5,  # Peak conductance
    'delay': dt,
    'p': 0.05}

EIp = {  # I - cell to E - cell
    # 'Tau': 3.5,  # Alpha function tau
    'weight': -0.6,  # Peak conductance
    'delay': dt,
    'p': 0.1}

IEp = {  # E - cell to I - cell
    # 'Tau': 4.0,  # lpha function tau
    'weight': 0.2,  # Peak conductance
    'delay': dt,
    'p': 0.05}

EEp = {  # E - cell to E - cell
    # 'Tau': 4.0,  # lpha function tau
    'weight': 0.1,  # Peak conductance
    'delay': dt,
    'p': 0.05}

IXp = {  # External to I - cell
    # 'Tau': 4*msecond,  # Alpha function tau
    'weight': 1.1,  # Peak conductance  (0.8 in paper)
    'delay': dt}

EXp = {  # External to E - cell
    # 'Tau': 4*msecond,  # Alpha function tau
    'weight': 1.4,  # Peak conductance  (1 in paper)
    'delay': dt}

# par = {"N": 1,
#        "t_simulation": 2000.0}
