'''
Class for simulating input networks from Akam et al Neuron 2010.  Network can be set to 
either asychronous, gamma oscillating, or beta oscillating state when it is initialised.
The input networks were simulated in Nest simulator for the original paper and when I 
built them in Brian they showed substanitally lower firing rates for reasons that I do 
not fully understand but which may be due to the connection function working somewhat
differently.  I have adjusted a couple of synaptic weights (where indicated) to bring 
the firing rates back  to aproximately where they were in the Nest implementation. The
resulting networks behave very similarly to those in the paper but have slightly lower 
oscillation frequencies.

# Copyright (c) Thomas Akam 2012.  Licenced under the GNU General Public License v3.
'''

import brian2 as b2
from brian2.equations.equations import Equations
from numpy.random import randn
import numpy as np
import pylab as plt
# from brian import *
# from brian.library.IF import *
# from brian.library.synapses import *
import time
import os
from numpy import (sin,
                   cos,
                   pi,
                   floor,
                   linspace,
                   ones,
                   mean)
# ion()


def exp_IF_memb_eq(membrane_params):
    '''
    Returns exponential integrate and fire membrane potential equation with
    parameters specified by membrane_params.
    '''

    MemEq = Equations(
        """
    Im = IX + gL * (EL-vm) + gL* DeltaT * exp((vm-VT)/DeltaT) : amp
            VT:volt
            IX:amp
    """,
        gL=membrane_params["gL"],
        EL=membrane_params['EL'],
        DeltaT=membrane_params['DeltaT']
    )

    # MemEq = MembraneEquation(membrane_params['C']) +\
    #     Current('''Im=IX+gL*(EL-vm)+gL*DeltaT*exp((vm-VT)/DeltaT):amp
    #         VT:volt
    #         IX:amp
    #         ''',
    #             current_name='Im',
    #             gL=membrane_params['gL'],
    #             EL=membrane_params['EL'],
    #             DeltaT=membrane_params['DeltaT'])
    return MemEq


class input_network:
    def __init__(self, state='asynchronous'):

        self.stimulus = np.array([0])

        # Neuron parameters.

        self.common_params = {    # Parameters common to all neurons.
            'C': 100*b2.pF,
            'tau_m': 10*b2.ms,
            'EL': -60*b2.mV,
            'DeltaT': 2*b2.mV,
            'Vreset': -65*b2.mV,
            'VTmean': -50*b2.mV,
            'VTsd': 2*b2.mV}

        self.common_params['gL'] = self.common_params['C'] / \
            self.common_params['tau_m']

        self.E_cell_params = dict(self.common_params, **{'Ncells': 8000,
                                                         'IXmean': 30*b2.pA,
                                                         'IXsd': 20*b2.pA})

        self.I_cell_params = dict(self.common_params, **{'Ncells': 2000,
                                                         'IXmean': 30*b2.pA,
                                                         'IXsd': 80*b2.pA})
        # Synapse paramters.

        self.IXp = {  # External to I - cell
            'Erev_IX': 0*b2.mV,  # Reversal potential
            'Tau_IX': 4*b2.ms,  # Alpha function tau
            'g_IX': 1.1*b2.nS,  # Peak conductance  (0.8 in paper)
            'delay_IX': 0*b2.ms}

        self.EXp = {  # External to E - cell
            'Erev_EX': 0*b2.mV,  # Reversal potential
            'Tau_EX': 4*b2.ms,  # Alpha function tau
            'g_EX': 1.4*b2.nS,  # Peak conductance  (1 in paper)
            'delay_EX': 0*b2.ms}

        self.IIp = {  # I - cell to I - cell
            'Erev_II': -80*b2.mV,  # Reversal potential
            'Tau_II': 3*b2.ms,  # Alpha function tau
            'g_II': 1.5*b2.nS,  # Peak conductance
            'sparseness_II': 100./self.I_cell_params['Ncells'],
            'delay_II': 0*b2.ms}

        self.EIp = {  # I - cell to E - cell
            'Erev_EI': -80*b2.mV,  # Reversal potential
            'Tau_EI': 3.5*b2.ms,  # lpha function tau
            'g_EI': 0.6*b2.nS,  # Peak conductance
            'sparseness_EI': 200./self.I_cell_params['Ncells'],
            'delay_EI': 0*b2.ms}

        self.IEp = {  # E - cell to I - cell
            'Erev_IE': 0*b2.mV,  # Reversal potential
            'Tau_IE': 4*b2.ms,  # lpha function tau
            'g_IE': 0.2*b2.nS,  # Peak conductance
            'sparseness_IE': 400./self.E_cell_params['Ncells'],
            'delay_IE': 0*b2.ms}

        self.EEp = {  # E - cell to E - cell
            'Erev_EE': 0*b2.mV,  # Reversal potential
            'Tau_EE': 4*b2.ms,  # lpha function tau
            'g_EE': 0.1*b2.nS,  # Peak conductance
            'sparseness_EE': 400./self.E_cell_params['Ncells'],
            'delay_EE': 0*b2.ms}

        self.state = state

        if state == 'gamma':

            print('Gamma oscillation state.')
            self.IXp['g_IX'] = 0.3*b2.nS
            self.IEp['g_IE'] = 0.4*b2.nS

        elif state == 'beta':
            print('Beta oscillation state.')
            self.IXp['g_IX'] = 0.5*b2.nS
            self.IXp['Tau_IX'] = 12*b2.ms
            self.EXp['g_EX'] = 0.55*b2.nS  # (0.47 in paper)
            self.EXp['Tau_EX'] = 12*b2.ms
            self.EEp['g_EE'] = 0.05*b2.nS
            self.EEp['Tau_EE'] = 12*b2.ms
            self.IEp['g_IE'] = 0.1*b2.nS  # (0.08 in paper)
            self.IEp['Tau_IE'] = 12*b2.ms
            self.EIp['g_EI'] = 0.1*b2.nS
            self.EIp['Tau_EI'] = 15*b2.ms
            self.IIp['g_II'] = 0.2*b2.nS
            self.IIp['Tau_II'] = 15*b2.ms

        # I - cell equations.

        # self.eqs_I = exp_IF_memb_eq(self.I_cell_params)
        self.eqs = Equations(
            """
        Im = IX + gL * (EL-vm) + gL* DeltaT * exp((vm-VT)/DeltaT) - ge(vm-Erev_e) - gi(vm-Erev_i) - gx(vm-E_rev_x) : amp
            VT:volt
            IX:amp
            dg_i/dt = -g_i/Tau : siemens
            dg_x/dt = -g_x/Tau : siemens
            dg_e/dt = -g_e/Tau : siemens
        """
        )

        # self.eqs_I += alpha_conductance('gi',
        #                                 self.IIp['Erev'], self.IIp['Tau'])
        # self.eqs_I += alpha_conductance('gx',
        #                                 self.IXp['Erev'], self.IXp['Tau'])
        # self.eqs_I += alpha_conductance('ge',
        #                                 self.IEp['Erev'], self.IEp['Tau'])

        # E - cell equations.
        # self.eqs_E = exp_IF_memb_eq(self.E_cell_params)
        # self.eqs_E += alpha_conductance('gi',
        #                                 self.EIp['Erev'], self.EIp['Tau'])
        # self.eqs_E += alpha_conductance('gx',
        #                                 self.EXp['Erev'], self.EXp['Tau'])
        # self.eqs_E += alpha_conductance('ge',
        #                                 self.EEp['Erev'], self.EEp['Tau'])

        print('Building network')

        # create cell populations

        # self.I_cells = NeuronGroup(self.I_cell_params['Ncells'],
        #                            model=self.eqs_I,
        #                            threshold=0*b2.mV,
        #                            reset=self.I_cell_params['Vreset'],
        #                            refractory=0*b2.ms)

        # self.E_cells = NeuronGroup(self.E_cell_params['Ncells'],
        #                            model=self.eqs_E,
        #                            threshold=0*b2.mV,
        #                            reset=self.E_cell_params['Vreset'],
        #                            refractory=0*b2.ms)

        self.I_cells = b2.NeuronGroup(self.I_cell_params['Ncells'],
                                      model=self.eqs,
                                      threshold=0*b2.mV,
                                      reset=self.I_cell_params['Vreset'],
                                      refractory=0*b2.ms,
                                      namespace={**self.common_params,
                                                 **self.EEp,
                                                 **self.EXp,
                                                 **self.EIp})

        self.E_cells = b2.NeuronGroup(self.E_cell_params['Ncells'],
                                      model=self.eqs,
                                      threshold=0*b2.mV,
                                      reset=self.E_cell_params['Vreset'],
                                      refractory=0*b2.ms,
                                      namespace={**self.common_params, **self.IEp, **self.IXp, **self.IIp})

        # create input.

        # self.External_to_E = PoissonGroup(
        #     self.E_cell_params['Ncells'], self.input_rates)
        # self.External_to_I = PoissonGroup(
        #     self.I_cell_params['Ncells'], rates=400*b2.Hz)

        self.External_to_E = b2.PoissonGroup(self.E_cell_params['Ncells'],
                                             self.input_rates)
        self.External_to_I = b2.PoissonGroup(
            self.I_cell_params['Ncells'], rates=400*b2.Hz)

        # create recording devices

        self.E_cell_spike_monitor = b2.SpikeMonitor(self.E_cells)
        self.E_cell_rate_monitor = b2.PopulationRateMonitor(
            self.E_cells, bin=1*b2.ms)

        self.I_cell_spike_monitor = b2.SpikeMonitor(self.I_cells)
        self.I_cell_rate_monitor = b2.PopulationRateMonitor(
            self.I_cells, bin=1*b2.ms)

        print('Connecting devices.')

        # set up internal connections.

        print('Excitatory connections')

        # self.cEE = Connection(self.E_cells, self.E_cells, 'ge')
        # Connection.connect_random(
        #     self.cEE, sparseness=self.EEp['sparseness'], weight=self.EEp['g'], fixed=True)

        # self.cIE = Connection(self.E_cells, self.I_cells, 'ge')
        # Connection.connect_random(
        #     self.cIE, sparseness=self.IEp['sparseness'], weight=self.IEp['g'], fixed=True)

        self.cEE = b2.Synapses(self.E_cells,
                               self.E_cells,
                               on_pre='ge+=' + str(self.EEp["g_EE"]))
        self.cEE.connect(p=self.EEp['sparseness_EE'])

        self.cIE = b2.Synapses(self.E_cells,
                               self.I_cells,
                               on_pre='gi+=' + str(self.EEp["g_EE"]))
        self.cEE.connect(p=self.EEp['sparseness_EE'])


        print('Inhibitory connections')

        self.cEI = Connection(self.I_cells, self.E_cells, 'gi')
        Connection.connect_random(
            self.cEI, sparseness=self.EIp['sparseness'], weight=self.EIp['g'], fixed=True)

        self.cII = Connection(self.I_cells, self.I_cells, 'gi')
        Connection.connect_random(
            self.cII, sparseness=self.IIp['sparseness'], weight=self.IIp['g'], fixed=True)

        print('Input connections')

        self.cIX = IdentityConnection(
            self.External_to_I, self.I_cells, 'gx', weight=self.IXp['g'])
        self.cEX = IdentityConnection(
            self.External_to_E, self.E_cells, 'gx', weight=self.EXp['g'])

        # Initialise random parameters.

        self.I_cells.VT = (randn(len(self.I_cells)) *
                           self.I_cell_params['VTsd']+self.I_cell_params['VTmean'])
        self.I_cells.IX = (randn(len(self.I_cells)) *
                           self.I_cell_params['IXsd']+self.I_cell_params['IXmean'])

        self.E_cells.VT = (randn(len(self.E_cells)) *
                           self.E_cell_params['VTsd']+self.E_cell_params['VTmean'])
        self.E_cells.IX = (randn(len(self.E_cells)) *
                           self.E_cell_params['IXsd']+self.E_cell_params['IXmean'])

        # Create network object from neuron groups, connections and monitors.
        self.net = MagicNetwork()

        print('Network setup complete.')

    def input_rates(self, t):
        '''
        Returns firing rates for spatially patterned stimulus to E cells.  
        stimuluss is array of stimulus orientations as function of time in ms.
        Range of stimuluss is 0 - 1.
        '''
        return 400*(1+0.35 * cos(2*pi*self.stimulus[floor(t*1000)  # 360
                                                    % self.stimulus.size] + pi
                                 + linspace(2*pi/self.E_cell_params['Ncells'],
                                            2*pi, self.E_cell_params['Ncells'])))*b2.Hz

    def simulate(self, sim_duration=500, stimulus=0., to_plot=True):
        '''
        Simulate network:  
        - If stimulus is a single number, simulates network with fixed stimulus 
        orientation for duration specified by sim_duration.  
        - If stimulus is an array, it is treated as a time varying stimulus at 1ms time resolution
        and stim_duration is ignored.
        '''

        if stimulus.size > 1:
            sim_duration = stimulus.size()
            self.stimulus = stimulus
        else:
            self.stimulus = ones(sim_duration) * stimulus

        self.net.reinit(states=False)  # Reset recording devices.

        # Randomise initial membrane potentials.
        self.I_cells.vm = randn(len(self.I_cells)) * 10 * b2.mV - 60 * b2.mV
        self.E_cells.vm = randn(len(self.E_cells)) * 10 * b2.mV - 60 * b2.mV

        print('Simulation running...')

        start_time = time.time()
        self.net.run(sim_duration * b2.ms)
        duration = time.time() - start_time

        print('Simulation time:', duration, 'seconds')
        print('Principal cell rate: {:4.2f} Hz'.format(
            mean(self.E_cell_rate_monitor.rate[50::])))
        print('Interneuron rate: {:4.2f} Hz'.format(
            mean(self.I_cell_rate_monitor.rate[50::])))

        if to_plot:
            self.plot()

    def plot(self, fig_no=1):
        plt.figure(fig_no)
        plt.clf()
        plt.subplot(3, 1, 1)
        # raster_plot(self.E_cell_spike_monitor)
        # subplot(3, 1, 2)
        # raster_plot(self.I_cell_spike_monitor)
        # subplot(3, 1, 3)
        # plot(self.E_cell_rate_monitor.times, self.E_cell_rate_monitor.rate, 'b')
        # plot(self.I_cell_rate_monitor.times, self.I_cell_rate_monitor.rate, 'g')
        plt.ylabel('Population average rates')
        plt.show()


if __name__ == "__main__":
    gamma_network = input_network(state='gamma')
    gamma_network.simulate(stimulus=sin(arange(0, 1000)*2*pi/1000))
