'''
Class for simulating input networks from Akam et al Neuron 2010.  Network can be set to either asychronous, gamma oscillating, or beta oscillating state when it is initialised. The input networks were simulated in Nest simulator for the original paper and when I built them in Brian they showed substanitally lower firing rates for reasons that I do not fully understand but which may be due to the connection function working somewhat differently. I have adjusted a couple of synaptic weights (where indicated) to bring the firing rates back  to aproximately where they were in the Nest implementation. The resulting networks behave very similarly to those in the paper but have slightly lower oscillation frequencies.

# Copyright (c) Thomas Akam 2012.  Licenced under the GNU General Public License v3.
'''

import os
import time
import pylab as plt
import numpy as np
from numpy.random import randn
import brian2 as b2
from brian2.equations.equations import Equations
import logging
logger = logging.getLogger('ftpuploader')

b2.seed(1)
np.random.seed(1)


def simulate():

    common_params = {    # Parameters common to all neurons.
        'C': 100*b2.pfarad,
        'tau_m': 10*b2.ms,
        'EL': -60*b2.mV,
        'DeltaT': 2*b2.mV,
        'Vreset': -65,  # *b2.mV
        'VTmean': -50*b2.mV,
        'VTsd': 2*b2.mV
    }

    common_params['gL'] = common_params['C'] / common_params['tau_m']

    E_cell_params = dict(common_params, **{'Ncells': num_E_cells,
                                           'IXmean': 30*b2.pA,
                                           'IXsd': 20*b2.pA})


    eqs = Equations(
        """
        Im = IX + 
            gL * (EL - vm) + 
            gL * DeltaT * exp((vm - VT) / DeltaT) - 
            ge * (vm - Erev_e) - 
            gi * (vm - Erev_i) - 
            gx * (vm - Erev_x) : amp
        dgi/dt = (1*nsiemens-gi)/Tau_i - gi/Tau_i : siemens
        dgx/dt = (1*nsiemens-gx)/Tau_x - gx/Tau_x : siemens
        dge/dt = (1*nsiemens-ge)/Tau_e - ge/Tau_e : siemens
        VT : volt
        IX : amp
        dvm/dt = Im / C : volt
        """
    )

    param_E_syn = {"Erev_i": 0.0*b2.mV,
                   "Erev_x": 0.0*b2.mV,
                   "Erev_e": -80.0*b2.mV,
                   "Tau_i": 3.0*b2.ms,
                   "Tau_e": 4.0*b2.ms,
                   "Tau_x": 4.0*b2.ms,
                   "w_i": 0.6,  # *b2.nsiemens,  # Peak conductance
                   # *b2.nsiemens,  # Peak conductance  (1 in paper)
                   "w_x": 1.4,
                   "w_e": 0.1,  # *b2.nsiemens,  # Peak conductance
                   "p_i": 0.1,  # ./I_cell_params['Ncells'],  # ! 200
                   "p_e": 0.05,  # /E_cell_params['Ncells'],  # ! 400
                   }

    if state == "beta":
        param_E_syn['w_x'] = 0.55 * b2.nS
        param_E_syn['Tau_x'] = 12 * b2.ms
        param_E_syn['w_e'] = 0.05 * b2.nS
        param_E_syn['Tau_e'] = 12 * b2.ms
        param_E_syn['w_i'] = 0.1 * b2.nS
        param_E_syn['Tau_i'] = 15 * b2.ms


    E_cells = b2.NeuronGroup(E_cell_params['Ncells'],
                             model=eqs,
                             method=integration_method,
                             threshold="vm > 0.*mV",
                             reset="vm={}*mV".format(E_cell_params['Vreset']),
                             refractory="vm > 0.*mV",
                             namespace={**common_params,
                                        **param_E_syn,
                                        })

    # Poisson_to_E = b2.PoissonGroup(
    #     E_cell_params['Ncells'], rates=input_rates())  # ! input_rates

    cEE = b2.Synapses(E_cells,
                      E_cells,
                      on_pre='ge+={}*nsiemens'.format(param_E_syn["w_e"]))
    # cEX = b2.Synapses(Poisson_to_E,
    #                   E_cells,
    #                   method=integration_method,
    #                   on_pre="gx += {}*nsiemens".format(param_E_syn["w_x"]))
    # cEX.connect(j='i')


    # Initialise random parameters.

    E_cells.VT = (randn(len(E_cells)) *
                  E_cell_params['VTsd'] + E_cell_params['VTmean'])
    E_cells.IX = (randn(len(E_cells)) *
                  E_cell_params['IXsd'] + E_cell_params['IXmean'])

    spike_monitor_E = b2.SpikeMonitor(E_cells)

    rate_monitor_E = b2.PopulationRateMonitor(E_cells)

    state_monitor_E = state_monitor_I = None
    if record_volrages:
        state_monitor_E = b2.StateMonitor(E_cells, "vm", record=True)
        state_monitor_I = b2.StateMonitor(I_cells, "vm", record=True)

    net = b2.Network(E_cells)
    if record_volrages:
        net.add(state_monitor_E)
    net.add(spike_monitor_E)
    net.add(rate_monitor_E)
    # net.add(cEX)
    # Randomise initial membrane potentials.
    E_cells.vm = randn(len(E_cells)) * 10 * b2.mV - 60 * b2.mV

    print('Simulation running...')

    start_time = time.time()
    b2.run(sim_duration*b2.ms)

    duration = time.time() - start_time
    print('Simulation time:', duration, 'seconds')


def plot(spike_monitor_E,
         state_monitor_E,
         plot_voltages=False):

    fig, ax = plt.subplots(3, figsize=(10, 8), sharex=True)

    ax[0].plot(spike_monitor_E.t/b2.ms, spike_monitor_E.i, '.k', ms=0.5)
    if plot_voltages:
        for i in range(num_E_cells):
            ax[3].plot(state_monitor_E.t/b2.ms,
                       state_monitor_E.vm[i]/b2.mV)

    ax[-1].set_xlabel("time (ms)")
    ax[0].set_ylabel("E cells")
    ax[3].set_ylabel("Voltages E")
    ax[-1].legend(loc="upper right")
    plt.savefig("data/E_cell.png", dpi=150)


if __name__ == "__main__":

    num_E_cells = 1

    sim_duration = 1000
    stimulus = np.sin(np.arange(0, sim_duration)*2*np.pi/sim_duration)
    state = "beta"

    integration_method = "rk2"
    record_volrages = False
    plot_voltages = record_volrages

    simulate()
    plot(plot_voltages)
