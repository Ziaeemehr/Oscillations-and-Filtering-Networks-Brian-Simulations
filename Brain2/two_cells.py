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

    param_E_syn = {
        "Erev_e": 0.0*b2.mV,
        "Tau_e": 4.0*b2.ms,
        "w_e": 1.1,  # ! *b2.nsiemens 0.1
        "p_e": 0.05,
    }

    E_cell_params = {'Ncells': num_E_cells,
                     'IXmean': 120*b2.pA,
                     'IXsd': 20*b2.pA}

    eqs_e = """
        Im = IX + 
            gL * (EL - vm) + 
            gL * DeltaT * exp((vm - VT) / DeltaT) : amp
        
        dvm/dt = (Im + I_syn_e) / C : volt
        VT : volt
        IX : amp
        I_syn_e : amp
        """
    eqs_syn_e = """
        dg_syn_e/dt = (s_e - g_syn_e) / Tau_e : siemens (event-driven)
        ds_e/dt = -s_e / Tau_e : siemens                (event-driven)
        I_syn_e_post =  g_syn_e * (Erev_e - vm): amp (summed)
        """

    E_cells = b2.NeuronGroup(E_cell_params['Ncells'],
                             model=eqs_e,
                             dt=dt0,
                             method=integration_method,
                             threshold="vm > 0.*mV",
                             reset="vm={}*mV".format(common_params['Vreset']),
                             refractory="vm > 0.*mV",
                             namespace={**common_params,
                                        **param_E_syn,
                                        })

    cEE = b2.Synapses(E_cells, E_cells, eqs_syn_e,
                      on_pre='g_syn_e += {}*nsiemens'.format(
                          param_E_syn["w_e"]),
                      method=integration_method,
                      namespace={**common_params,
                                 **param_E_syn,
                                 })
    cEE.connect(i=0, j=1)

    # Initialise random parameters.
    E_cells.VT = [common_params['VTmean']] * 2
    # (randn(len(E_cells)) *
    #               common_params['VTsd'] + common_params['VTmean'])
    E_cells.IX = [E_cell_params['IXmean']] * 2
    # (randn(len(E_cells)) *
    #               E_cell_params['IXsd'] + E_cell_params['IXmean'])

    E_cells.vm = randn(len(E_cells)) * 10 * b2.mV - 60 * b2.mV

    spike_monitor_E = b2.SpikeMonitor(E_cells)

    state_monitor_E = None
    if record_volrages:
        state_monitor_E = b2.StateMonitor(E_cells, "vm", record=True)

    net = b2.Network(E_cells)
    if record_volrages:
        net.add(state_monitor_E)
    net.add(cEE)
    net.add(spike_monitor_E)
    # Randomise initial membrane potentials.

    print('Simulation running...')
    start_time = time.time()
    b2.run(sim_duration*b2.ms)
    duration = time.time() - start_time
    print('Simulation time:', duration, 'seconds')

    return spike_monitor_E, state_monitor_E


def plot(spike_monitor_E,
         state_monitor_E,
         plot_voltages=False):

    fig, ax = plt.subplots(2, figsize=(10, 5), sharex=True)

    ax[0].plot(spike_monitor_E.t/b2.ms, spike_monitor_E.i, '.k', ms=3)

    if plot_voltages:
        for i in range(num_E_cells):
            ax[1].plot(state_monitor_E.t/b2.ms,
                       state_monitor_E.vm[i]/b2.mV, label=str(i+1))

    ax[0].set_ylabel("E cells")
    ax[-1].legend(loc="upper right")
    ax[1].set_ylabel("Voltages E")
    ax[-1].set_xlabel("time (ms)")
    plt.savefig("data/E_cell.png", dpi=150)
    plt.show()


if __name__ == "__main__":

    num_E_cells = 2
    dt0 = 0.1*b2.ms

    sim_duration = 200
    stimulus = np.sin(np.arange(0, sim_duration)*2*np.pi/sim_duration)
    state = "beta"

    integration_method = "rk2"
    record_volrages = True
    plot_voltages = record_volrages

    sp_mon, st_mon = simulate()
    plot(sp_mon, st_mon, plot_voltages)
