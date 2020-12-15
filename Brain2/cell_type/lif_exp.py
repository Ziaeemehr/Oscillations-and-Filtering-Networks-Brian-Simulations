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
b2.prefs.codegen.target = 'numpy'



def simulate(IXmean):

    common_params = {    # Parameters common to all neurons.
        'C': 100.*b2.pfarad,
        'tau_m': 10.*b2.ms,
        'EL': -60.*b2.mV,
        'DeltaT': 2.*b2.mV,
        'Vreset': -65.,  # *b2.mV
        'VTmean': -50.*b2.mV
    }

    common_params['gL'] = common_params['C'] / common_params['tau_m']

    E_cell_params = dict(common_params, **{'Ncells': num_E_cells,
                                           'IXmean': IXmean,  # 30
                                           'IXsd': 20*b2.pA})

    eqs = Equations(
        """
        Im = IX + 
            gL * (EL - vm) + 
            gL * DeltaT * exp((vm - VT) / DeltaT) : amp
        VT : volt
        IX : amp
        dvm/dt = Im / C : volt
        """
    )

    E_cells = b2.NeuronGroup(E_cell_params['Ncells'],
                             model=eqs,
                             method=integration_method,
                             threshold="vm > 0.*mV",
                             reset="vm={}*mV".format(E_cell_params['Vreset']),
                             refractory="vm > 0.*mV",
                             namespace=common_params)
    # Initialise random parameters.
    E_cells.VT = E_cell_params['VTmean']
    E_cells.IX = E_cell_params['IXmean']
    E_cells.vm = - 60 * b2.mV

    spike_monitor_E = b2.SpikeMonitor(E_cells)

    state_monitor_E = None
    if record_voltages:
        state_monitor_E = b2.StateMonitor(E_cells,
                                          "vm",
                                          record=True,
                                          dt=dt)

    net = b2.Network(E_cells)
    if record_voltages:
        net.add(state_monitor_E)
    net.add(spike_monitor_E)

    print('Simulation running...')
    start_time = time.time()
    b2.run(sim_duration*b2.ms)
    duration = time.time() - start_time
    print('Simulation time:', duration, 'seconds')

    return spike_monitor_E, state_monitor_E


def plot(spike_monitor_E,
         state_monitor_E,
         plot_voltages=False,
         filename="fig.png"):

    fig, ax = plt.subplots(2, figsize=(10, 4), sharex=True)

    ax[0].plot(spike_monitor_E.t/b2.ms, spike_monitor_E.i, '.k')
    if plot_voltages:
        for i in range(num_E_cells):
            ax[1].plot(state_monitor_E.t/b2.ms,
                       state_monitor_E.vm[i]/b2.mV, color="k")

    ax[-1].set_xlabel("time (ms)")
    ax[0].set_ylabel("E cells")
    ax[1].set_ylabel("Voltages E")
    ax[0].set_yticks([])
    ax[1].set_ylim(-70, -10)
    ax[0].margins(x=0)
    plt.savefig(filename, dpi=150)
    plt.show()
    plt.close()


if __name__ == "__main__":

    state = "beta"
    num_E_cells = 1
    sim_duration = 500
    dt = 0.01*b2.ms
    b2.defaultclock.dt = dt

    record_voltages = True
    integration_method = "rk2"
    plot_voltages = record_voltages

    spike_mon, state_mon = simulate(IXmean=90*b2.pA)

    plot(spike_mon,
         state_mon,
         plot_voltages,
         filename=f"E-{state:s}.png")
