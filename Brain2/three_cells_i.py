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
connect three inhibitory exponential LIF cells in feed forward loop
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

    param_I_syn = {
        "Erev_i": -80.0*b2.mV,
        "Tau_i": 3.0*b2.ms,
        "w_i": 1.5,  # ! *b2.nsiemens 0.1
        "p_i": 1.0,
    }

    I_cell_params = {'Ncells': num_I_cells,
                     'IXmean': 150.*b2.pA,
                     'IXsd': 20*b2.pA}

    eqs = """
        VT : volt
        IX : amp
        I_syn_i = g_syn_i * (Erev_i - vm): amp
        Im = IX + 
            gL * (EL - vm) + 
            gL * DeltaT * exp((vm - VT) / DeltaT) : amp
        
        ds_i/dt = -s_i / Tau_i : siemens                
        dg_syn_i/dt = (s_i - g_syn_i) / Tau_i : siemens 
        dvm/dt = (Im + I_syn_i) / C : volt
        """

    I_cells = b2.NeuronGroup(I_cell_params['Ncells'],
                             model=eqs,
                             dt=dt0,
                             method=integration_method,
                             threshold="vm > 0.*mV",
                             reset="vm={}*mV".format(common_params['Vreset']),
                             refractory="vm > 0.*mV",
                             namespace={**common_params,
                                        **param_I_syn,
                                        })

    cII = b2.Synapses(I_cells, I_cells,
                      on_pre='s_i += {}*nsiemens'.format(
                          param_I_syn["w_i"]),
                      dt=dt0,
                      method=integration_method,
                      namespace={**common_params,
                                 **param_I_syn,
                                 })
    adj = np.array([[0, 0, 0],
                    [1, 0, 0],
                    [1, 1, 0]])
    cols, rows = np.nonzero(adj)
    cII.connect(i=rows, j=cols)    
    

    # Initialise random parameters.
    I_cells.VT = [common_params['VTmean']] * I_cell_params["Ncells"]
    I_cells.IX = (randn(len(I_cells)) *
                  I_cell_params['IXsd'] + I_cell_params['IXmean'])

    I_cells.vm = randn(len(I_cells)) * 10 * b2.mV - 60 * b2.mV

    spike_monitor_I = b2.SpikeMonitor(I_cells)

    state_monitor_I = None
    if record_volrages:
        state_monitor_I = b2.StateMonitor(I_cells,
                                          ["vm", "g_syn_i", "I_syn_i"],
                                          record=True,
                                          dt=dt0)

    net = b2.Network(I_cells)
    if record_volrages:
        net.add(state_monitor_I)
    net.add(cII)
    net.add(spike_monitor_I)
    # Randomise initial membrane potentials.

    print('Simulation running...')
    start_time = time.time()
    b2.run(sim_duration*b2.ms)
    duration = time.time() - start_time
    print('Simulation time:', duration, 'seconds')

    return spike_monitor_I, state_monitor_I


def plot(spike_monitor,
         state_monitor,
         plot_voltages=False):

    fig, ax = plt.subplots(4, figsize=(10, 5), sharex=True)

    ax[0].plot(spike_monitor.t/b2.ms, spike_monitor.i, '.k', ms=3)

    if plot_voltages:
        for i in range(num_I_cells):
            ax[1].plot(state_monitor.t/b2.ms,
                       state_monitor.vm[i]/b2.mV, label=str(i+1))
    # ax[2].plot(state_monitor_E.t/b2.ms,
    #            state_monitor_E.I_syn_e[1]/b2.amp,
    #            lw=1, color="r")
    ax[2].plot(state_monitor.t/b2.ms,
               state_monitor.g_syn_i[2]/b2.nsiemens,
               lw=1, color="b", ls="--")
    ax[3].plot(state_monitor.t/b2.ms,
               state_monitor.I_syn_i[2]/b2.pA,
               lw=1, color="b", ls="--")
    ax[2].set_ylabel(r"$g_{syn}$(nS)")
    ax[3].set_ylabel(r"$I_{syn}$(pA)")

    ax[0].set_ylabel("I cells")
    ax[1].legend(loc="upper right")
    ax[1].set_ylabel("Voltages I")
    ax[-1].set_xlabel("time (ms)")
    plt.savefig("data/I_cell.png", dpi=150)
    plt.show()


if __name__ == "__main__":

    num_I_cells = 3
    dt0 = 0.1*b2.ms

    sim_duration = 200
    state = "beta"

    integration_method = "rk2"
    record_volrages = True
    plot_voltages = record_volrages

    sp_mon, st_mon = simulate()
    plot(sp_mon, st_mon, plot_voltages)
