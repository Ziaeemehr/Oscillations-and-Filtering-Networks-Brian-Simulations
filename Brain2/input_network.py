'''
Class for simulating input networks from Akam et al Neuron 2010.  Network can be set to either asychronous, gamma oscillating, or beta oscillating state when it is initialised. The input networks were simulated in Nest simulator for the original paper and when I built them in Brian they showed substanitally lower firing rates for reasons that I do not fully understand but which may be due to the connection function working somewhat differently. I have adjusted a couple of synaptic weights (where indicated) to bring the firing rates back  to aproximately where they were in the Nest implementation. The resulting networks behave very similarly to those in the paper but have slightly lower oscillation frequencies.

# Copyright (c) Thomas Akam 2012.  Licenced under the GNU General Public License v3.
'''

import time
import logging
import numpy as np
import brian2 as b2
from numpy.random import randn
from numpy import pi, sin, arange, floor
from brian2.equations.equations import Equations
from utility import *

logger = logging.getLogger('ftpuploader')

b2.seed(2)
np.random.seed(2)


def simulate(to_file=True):

    common_params = {    # Parameters common to all neurons.
        'C': 100*b2.pF,
        'tau_m': 10*b2.ms,
        'EL': -60*b2.mV,
        'DeltaT': 2*b2.mV,
        'Vreset': -65,  # *b2.mV
        'VTmean': -50*b2.mV,
        'VTsd': 2*b2.mV,
        'delay': 0.*b2.ms,
    }

    common_params['gL'] = common_params['C'] / common_params['tau_m']

    E_cell_params = dict(common_params, **{'Ncells': num_E_cells,
                                           'IXmean': 30.*b2.pA,  # 30
                                           'IXsd': 20.*b2.pA})

    I_cell_params = dict(common_params, **{'Ncells': num_I_cells,
                                           'IXmean': 30.*b2.pA,
                                           'IXsd': 80.*b2.pA,
                                           'p_rate': 400.0*b2.Hz})

    param_I_syn = {"Erev_i": -80.0*b2.mV,
                   "Erev_x": 0.0*b2.mV,
                   "Erev_e": 0.0*b2.mV,
                   "Tau_i": 3.0*b2.ms,
                   "Tau_e": 4.0*b2.ms,
                   "Tau_x": 4.0*b2.ms,
                   "w_i": 1.5,  # *b2.nsiemens,  # Peak conductance
                   "w_x": 1.1,  # *b2.nsiemens,  # (0.8 in paper)
                   "w_e": 0.2,  # *b2.nsiemens,
                   "p_i": 0.05,
                   "p_e": 0.05,
                   }
    param_E_syn = {"Erev_i": -80.0*b2.mV,
                   "Erev_x": 0.0*b2.mV,
                   "Erev_e": 0.0*b2.mV,
                   "Tau_i": 3.5*b2.ms,
                   "Tau_e": 4.0*b2.ms,
                   "Tau_x": 4.0*b2.ms,
                   "w_i": 0.6*b2.nsiemens,  # *b2.nsiemens,  # Peak conductance
                   "w_x": 1.4*b2.nsiemens,
                   "w_e": 0.1*b2.nsiemens,
                   "p_i": 0.1,
                   "p_e": 0.05,
                   }
    if state == "gamma":
        print('Gamma oscillation state.')
        param_I_syn['w_x'] = 0.3*b2.nS
        param_I_syn['w_e'] = 0.4*b2.nS

    elif state == "beta":
        param_I_syn['w_x'] = 0.5 * b2.nS
        param_I_syn['Tau_x'] = 12. * b2.ms
        param_E_syn['w_x'] = 0.55 * b2.nS
        param_E_syn['Tau_x'] = 12. * b2.ms
        param_E_syn['w_e'] = 0.05 * b2.nS
        param_E_syn['Tau_e'] = 12. * b2.ms
        param_I_syn['w_e'] = 0.1 * b2.nS
        param_E_syn['w_i'] = 0.1 * b2.nS
        param_E_syn['Tau_i'] = 15. * b2.ms
        param_I_syn['w_i'] = 0.2 * b2.nS
        param_I_syn['Tau_i'] = 15. * b2.ms

    eqs = Equations(
        """
        VT : volt
        IX : amp
        I_syn_e = g_syn_e * (Erev_e - vm): amp
        I_syn_i = g_syn_i * (Erev_i - vm): amp
        I_syn_x = g_syn_x * (Erev_x - vm): amp
        Im = IX +
            gL * (EL - vm) +
            gL * DeltaT * exp((vm - VT) / DeltaT) : amp

        ds_e/dt = -s_e / Tau_e : siemens
        dg_syn_e/dt = (s_e - g_syn_e) / Tau_e : siemens

        ds_i/dt = -s_i / Tau_i : siemens
        dg_syn_i/dt = (s_i - g_syn_i) / Tau_i : siemens

        ds_x/dt = -s_x / Tau_x : siemens
        dg_syn_x/dt = (s_x - g_syn_x) / Tau_x : siemens

        dvm/dt = (Im + I_syn_e + I_syn_i + I_syn_x) / C : volt
        """
    )

    I_cells = b2.NeuronGroup(I_cell_params['Ncells'],
                             model=eqs,
                             dt=dt0,
                             method=integration_method,
                             threshold="vm > 0.*mV",
                             refractory="vm > 0.*mV",
                             reset="vm={}*mV".format(common_params['Vreset']),
                             namespace={**common_params,
                                        **param_I_syn})

    E_cells = b2.NeuronGroup(E_cell_params['Ncells'],
                             model=eqs,
                             dt=dt0,
                             method=integration_method,
                             threshold="vm > 0.*mV",
                             refractory="vm > 0.*mV",
                             reset="vm={}*mV".format(common_params['Vreset']),
                             namespace={**common_params,
                                        **param_E_syn})

    # rates = '400.0*(1 + 0.35 * cos(2*pi*sin(2*pi*t/(100*ms)) + pi + 2*pi/N + (1.0*i/N)*2*pi))*Hz'
    Poisson_to_E = b2.PoissonGroup(
        E_cell_params['Ncells'],
        rates='400.0*(1+0.35*cos(2*pi*sin(2*pi*t/({:d}*ms))+pi+'
        '2*pi/{:d} + (1.0*i/{:d})*2*pi))*Hz'.format(
            sim_duration,
            E_cell_params['Ncells'],
            E_cell_params['Ncells']))

    Poisson_to_I = b2.PoissonGroup(I_cell_params['Ncells'],
                                   rates=I_cell_params["p_rate"])

    # ---------------------------------------------------------------
    cEE = b2.Synapses(E_cells, E_cells,
                      dt=dt0,
                      delay=common_params['delay'],
                      on_pre='s_e+= {}*nS'.format(param_E_syn['w_e']),
                      namespace={**common_params, **param_E_syn})
    cEE.connect(p="{:g}".format(param_E_syn["p_e"])) #, condition='i!=j'

    cII = b2.Synapses(I_cells, I_cells,
                      dt=dt0,
                      delay=common_params['delay'],
                      method=integration_method,
                      on_pre='s_i+= {}*nS'.format(param_I_syn['w_i']),
                      namespace={**common_params, **param_I_syn})
    cII.connect(p="{:g}".format(param_I_syn["p_e"])) #, condition='i!=j'

    cIE = b2.Synapses(E_cells, I_cells,
                      dt=dt0,
                      method=integration_method,
                      on_pre='s_e+={}*nsiemens'.format(param_I_syn["w_e"]))
    cIE.connect(p=param_I_syn["p_e"])

    cEI = b2.Synapses(I_cells, E_cells,
                      dt=dt0,
                      delay=common_params['delay'],
                      method=integration_method,
                      on_pre='s_i+={}*nsiemens'.format(param_I_syn["w_i"]))
    cEI.connect(p=param_I_syn["p_i"])

    cEX = b2.Synapses(Poisson_to_E, E_cells,
                      dt=dt0,
                      delay=common_params['delay'],
                      method=integration_method,
                      on_pre="s_x += {}*nS".format(param_E_syn["w_x"]))
    cEX.connect(j='i')

    cIX = b2.Synapses(Poisson_to_I, I_cells,
                      dt=dt0,
                      delay=common_params['delay'],
                      method=integration_method,
                      on_pre="s_x += {}*nS".format(param_I_syn["w_x"]))
    cIX.connect(j='i')

    # Initialise random parameters.----------------------------------
    E_cells.VT = (randn(len(E_cells)) *
                  common_params['VTsd'] + common_params['VTmean'])
    I_cells.VT = (randn(len(I_cells)) *
                  common_params['VTsd'] + common_params['VTmean'])

    E_cells.IX = (randn(len(E_cells)) *
                  E_cell_params['IXsd'] + E_cell_params['IXmean'])
    I_cells.IX = (randn(len(I_cells)) *
                  I_cell_params['IXsd'] + I_cell_params['IXmean'])

    I_cells.vm = randn(len(I_cells)) * 10 * b2.mV - 60 * b2.mV
    E_cells.vm = randn(len(E_cells)) * 10 * b2.mV - 60 * b2.mV

    spike_mon_E = b2.SpikeMonitor(E_cells)
    spike_mon_I = b2.SpikeMonitor(I_cells)

    LFP_E = b2.PopulationRateMonitor(E_cells)
    LFP_I = b2.PopulationRateMonitor(I_cells)

    state_monitor_E = state_monitor_I = None
    if rocord_voltages:
        state_monitor_E = b2.StateMonitor(E_cells, "vm",
                                          record=True, dt=dt0)
        state_monitor_I = b2.StateMonitor(I_cells, "vm",
                                          record=True, dt=dt0)

    net = b2.Network(E_cells)
    net.add(I_cells)
    net.add(spike_mon_E)
    net.add(spike_mon_I)
    net.add(LFP_E)
    net.add(LFP_I)
    net.add(cEE)
    net.add(cII)
    net.add(cEI)
    net.add(cIE)
    net.add(cIX)
    net.add(cEX)

    if rocord_voltages:
        net.add(state_monitor_E)
        net.add(state_monitor_I)

    # ----------------------------------------------------------------
    print('Simulation running...')

    start_time = time.time()
    b2.run(sim_duration*b2.ms)

    duration = time.time() - start_time
    print('Simulation time:', duration, 'seconds')
    # ----------------------------------------------------------------

    if to_file:
        to_npz(spike_mon_E, LFP_E, "data/E")
        to_npz(spike_mon_I, LFP_I, "data/I")
    # ----------------------------------------------------------------
    # return (spike_monitor_E,
    #         rate_monitor_E,
    #         spike_monitor_I,
    #         rate_monitor_I)

# ------------------------------------------------------------------------------


def to_npz(spike_monitor, rate_monitor, filename, width=1*b2.ms):

    spikes_id = spike_monitor.i
    spike_times = spike_monitor.t/b2.ms

    rate_times = rate_monitor.t/b2.ms
    rate_amp = rate_monitor.smooth_rate(width=width) / b2.Hz

    np.savez(filename,
             spikes_time=spike_times,
             spikes_id=spikes_id,
             rate_amp=rate_amp,
             rate_times=rate_times,
             )
# ------------------------------------------------------------------------------


num_E_cells = 8000
num_I_cells = 2000
dt0 = 0.1*b2.ms
sim_duration = 1000  # ms

# ------------------------------------------------------------------------------

if __name__ == "__main__":

    state = "beta"

    integration_method = "rk2"
    plot_voltages = rocord_voltages = False

    # simulate(to_file=True)
    plot_raster_from_data("data/E", xlim=[100, sim_duration], title="E cells")
    plot_raster_from_data("data/I", xlim=[100, sim_duration], title="I cells")
