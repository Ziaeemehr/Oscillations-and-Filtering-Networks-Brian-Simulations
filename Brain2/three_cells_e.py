import os
import time
import pylab as plt
import numpy as np
from numpy.random import randn
import brian2 as b2
from brian2.equations.equations import Equations
import logging
# logger = logging.getLogger('ftpuploader')


'''
connect three excitatory exponential LIF cells in feed forward loop
with alpha shape conductance base synapse
edges : ([0,1], [0,2], [1,2])
'''

b2.seed(2)
np.random.seed(2)


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
        "p_e": 1.0,
    }

    E_cell_params = {'Ncells': num_E_cells,
                     'IXmean': 120*b2.pA,
                     'IXsd': 20*b2.pA}

    eqs_e = """
        VT : volt
        IX : amp
        I_syn_e = g_syn_e * (Erev_e - vm): amp
        Im = IX + 
            gL * (EL - vm) + 
            gL * DeltaT * exp((vm - VT) / DeltaT) : amp
        
        ds_e/dt = -s_e / Tau_e : siemens                
        dg_syn_e/dt = (s_e - g_syn_e) / Tau_e : siemens 
        dvm/dt = (Im + I_syn_e) / C : volt
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

    cEE = b2.Synapses(E_cells, E_cells,
                      on_pre='s_e += {}*nsiemens'.format(
                          param_E_syn["w_e"]),
                      dt=dt0,
                      method=integration_method,
                      namespace={**common_params,
                                 **param_E_syn,
                                 })
    adj = np.array([[0, 0, 0],
                    [1, 0, 0],
                    [1, 1, 0]])
    cols, rows = np.nonzero(adj)
    # print(rows, cols)
    cEE.connect(i=rows, j=cols)    
    # for i, j in zip(cEE.i, cEE.j):
    #     print(i, j)

    # Initialise random parameters.
    E_cells.VT = [common_params['VTmean']] * E_cell_params["Ncells"]
    E_cells.IX = (randn(len(E_cells)) *
                  E_cell_params['IXsd'] + E_cell_params['IXmean'])

    E_cells.vm = randn(len(E_cells)) * 10 * b2.mV - 60 * b2.mV

    spike_monitor_E = b2.SpikeMonitor(E_cells)

    state_monitor_E = None
    if record_volrages:
        state_monitor_E = b2.StateMonitor(E_cells,
                                          ["vm", "g_syn_e", "I_syn_e"],
                                          record=True,
                                          dt=dt0)

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

    fig, ax = plt.subplots(4, figsize=(10, 5), sharex=True)

    ax[0].plot(spike_monitor_E.t/b2.ms, spike_monitor_E.i, '.k', ms=3)

    if plot_voltages:
        for i in range(num_E_cells):
            ax[1].plot(state_monitor_E.t/b2.ms,
                       state_monitor_E.vm[i]/b2.mV, label=str(i+1))
    # ax[2].plot(state_monitor_E.t/b2.ms,
    #            state_monitor_E.I_syn_e[1]/b2.amp,
    #            lw=1, color="r")
    ax[2].plot(state_monitor_E.t/b2.ms,
               state_monitor_E.g_syn_e[2]/b2.nsiemens,
               lw=1, color="b", ls="--")
    ax[3].plot(state_monitor_E.t/b2.ms,
               state_monitor_E.I_syn_e[2]/b2.pA,
               lw=1, color="b", ls="--")
    ax[2].set_ylabel(r"$g_{syn}$(nS)")
    ax[3].set_ylabel(r"$I_{syn}$(pA)")

    ax[0].set_ylabel("E cells")
    ax[1].legend(loc="upper right")
    ax[1].set_ylabel("Voltages E")
    ax[-1].set_xlabel("time (ms)")
    plt.savefig("data/E_cell.png", dpi=150)
    plt.show()


if __name__ == "__main__":

    num_E_cells = 3
    dt0 = 0.1*b2.ms

    sim_duration = 200
    state = "beta"

    integration_method = "rk2"
    record_volrages = True
    plot_voltages = record_volrages

    sp_mon, st_mon = simulate()
    plot(sp_mon, st_mon, plot_voltages)
