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

Thomas Akam 2012
'''


from brian import *
from brian.library.IF import *
from brian.library.synapses import *
import time 
import os
ion()

def exp_IF_memb_eq(membrane_params): 
    '''
    Returns exponential integrate and fire membrane potential equation with
    parameters specified by membrane_params.
    '''                     
    MemEq=MembraneEquation(membrane_params['C'])+\
            Current('''Im=IX+gL*(EL-vm)+gL*DeltaT*exp((vm-VT)/DeltaT):amp
            VT:volt
            IX:amp
            ''',
            current_name = 'Im',
            gL = membrane_params['gL'],
            EL = membrane_params['EL'],
            DeltaT = membrane_params['DeltaT'])  
    return MemEq

class input_network:
    def __init__(self, state = 'asynchronous'):

        self.stimulus = array([0])

        # Neuron parameters.

        self.common_params={    # Parameters common to all neurons.   
                            'C' : 100*pF,    
                            'tau_m' : 10*msecond,
                            'EL' : -60*mV,
                            'DeltaT' : 2*mV,
                            'Vreset' : -65*mV,
                            'VTmean' : -50*mV,
                            'VTsd' : 2*mV}

        self.common_params['gL'] = self.common_params['C'] / self.common_params['tau_m']

        self.E_cell_params = dict(self.common_params, **{  'Ncells' : 8000,  
                                                           'IXmean' : 30*pA, 
                                                           'IXsd'   : 20*pA})

        self.I_cell_params = dict(self.common_params, **{ 'Ncells' : 2000,  
                                                          'IXmean' : 30*pA,
                                                          'IXsd' : 80*pA})
        # Synapse paramters.

        self.IXp = { # External to I - cell
                    'Erev' : 0*mvolt,    #Reversal potential
                    'Tau'  : 4*msecond,  #Alpha function tau
                    'g'    : 1.1*nS,     #Peak conductance  (0.8 in paper)
                    'delay' : 0*ms}    
          
        self.EXp = { # External to E - cell
                    'Erev' : 0*mvolt,    #Reversal potential
                    'Tau'  : 4*msecond,  #Alpha function tau
                    'g'    : 1.4*nS,     #Peak conductance  (1 in paper)
                    'delay' : 0*ms}
    
        self.IIp = { # I - cell to I - cell
                    'Erev' : -80*mvolt,    #Reversal potential
                    'Tau'  : 3*msecond,  #Alpha function tau
                    'g'    : 1.5*nS,     #Peak conductance
                    'sparseness' : 100./self.I_cell_params['Ncells'],  
                    'delay' : 0*ms}
          
        self.EIp = { # I - cell to E - cell
                    'Erev' : -80*mvolt,    #Reversal potential
                    'Tau'  : 3.5*msecond,  #lpha function tau
                    'g'    : 0.6*nS,     #Peak conductance
                    'sparseness' : 200./self.I_cell_params['Ncells'],
                    'delay' : 0*ms}
          
        self.IEp = { # E - cell to I - cell
                    'Erev' : 0*mvolt,    #Reversal potential
                    'Tau'  : 4*msecond,  #lpha function tau
                    'g'    : 0.2*nS,     #Peak conductance
                    'sparseness' : 400./self.E_cell_params['Ncells'],
                    'delay' : 0*ms}
          
        self.EEp = { # E - cell to E - cell
                    'Erev' : 0*mvolt,    #Reversal potential
                    'Tau'  : 4*msecond,  #lpha function tau
                    'g'    : 0.1*nS,     #Peak conductance
                    'sparseness' : 400./self.E_cell_params['Ncells'],
                    'delay' : 0*ms}

        self.state = state
        
        if state == 'gamma':  

            print 'Gamma oscillation state.'
            self.IXp['g'] = 0.3*nS
            self.IEp['g'] = 0.4*nS

        elif state == 'beta':
            print 'Beta oscillation state.'
            self.IXp['g']   = 0.5*nS
            self.IXp['Tau'] = 12*ms
            self.EXp['g']   = 0.55*nS # (0.47 in paper)  
            self.EXp['Tau'] = 12*ms
            self.EEp['g']   = 0.05*nS
            self.EEp['Tau'] = 12*ms
            self.IEp['g']   = 0.1*nS  # (0.08 in paper)
            self.IEp['Tau'] = 12*ms
            self.EIp['g']   = 0.1*nS
            self.EIp['Tau'] = 15*ms
            self.IIp['g']   = 0.2*nS
            self.IIp['Tau'] = 15*ms  

        # I - cell equations.
    
        self.eqs_I =  exp_IF_memb_eq(self.I_cell_params)
        self.eqs_I += alpha_conductance('gi', self.IIp['Erev'], self.IIp['Tau'])
        self.eqs_I += alpha_conductance('gx', self.IXp['Erev'], self.IXp['Tau'])
        self.eqs_I += alpha_conductance('ge', self.IEp['Erev'], self.IEp['Tau'])
    
        # E - cell equations.
    
        self.eqs_E =  exp_IF_memb_eq(self.E_cell_params)
        self.eqs_E += alpha_conductance('gi', self.EIp['Erev'], self.EIp['Tau'])
        self.eqs_E += alpha_conductance('gx', self.EXp['Erev'], self.EXp['Tau'])
        self.eqs_E += alpha_conductance('ge', self.EEp['Erev'], self.EEp['Tau'])

        print 'Building network'
    
        #create cell populations
    
        self.I_cells=NeuronGroup( self.I_cell_params['Ncells'],
                                  model = self.eqs_I,
                                  threshold = 0*mvolt,
                                  reset = self.I_cell_params['Vreset'],
                                  refractory = 0*ms)

        self.E_cells=NeuronGroup( self.E_cell_params['Ncells'],
                                  model = self.eqs_E,
                                  threshold = 0*mvolt,
                                  reset = self.E_cell_params['Vreset'],
                                  refractory = 0*ms)
    
        #create input.
    
        self.External_to_E = PoissonGroup(self.E_cell_params['Ncells'], self.input_rates )
        self.External_to_I = PoissonGroup(self.I_cell_params['Ncells'], rates = 400*Hz)

        #create recording devices
    
        self.E_cell_spike_monitor = SpikeMonitor(self.E_cells)
        self.E_cell_rate_monitor  = PopulationRateMonitor(self.E_cells, bin=1*ms)
    
        self.I_cell_spike_monitor = SpikeMonitor(self.I_cells)
        self.I_cell_rate_monitor  = PopulationRateMonitor(self.I_cells, bin=1*ms)

        print 'Connecting devices.'
    
        #set up internal connections.
    
        print 'Excitatory connections'
    
        self.cEE = Connection(self.E_cells, self.E_cells, 'ge')
        Connection.connect_random(self.cEE, sparseness = self.EEp['sparseness'], weight = self.EEp['g'], fixed=True)
    
        self.cIE = Connection(self.E_cells, self.I_cells, 'ge')
        Connection.connect_random(self.cIE, sparseness = self.IEp['sparseness'], weight=self.IEp['g'], fixed=True)
    
        print 'Inhibitory connections'
    
        self.cEI = Connection(self.I_cells, self.E_cells, 'gi')
        Connection.connect_random(self.cEI, sparseness = self.EIp['sparseness'], weight = self.EIp['g'], fixed=True)
    
        self.cII = Connection(self.I_cells,self.I_cells,'gi')
        Connection.connect_random(self.cII, sparseness = self.IIp['sparseness'], weight = self.IIp['g'], fixed=True)
    
        print 'Input connections'
    
        self.cIX = IdentityConnection(self.External_to_I, self.I_cells,'gx', weight = self.IXp['g'])
        self.cEX = IdentityConnection(self.External_to_E, self.E_cells,'gx', weight = self.EXp['g'])
    
        # Initialise random parameters.


        self.I_cells.VT = (randn(len(self.I_cells))*self.I_cell_params['VTsd']+self.I_cell_params['VTmean'])
        self.I_cells.IX = (randn(len(self.I_cells))*self.I_cell_params['IXsd']+self.I_cell_params['IXmean'])
        
        
        self.E_cells.VT = (randn(len(self.E_cells))*self.E_cell_params['VTsd']+self.E_cell_params['VTmean'])
        self.E_cells.IX = (randn(len(self.E_cells))*self.E_cell_params['IXsd']+self.E_cell_params['IXmean'])

        # Create network object from neuron groups, connections and monitors.
        self.net = MagicNetwork()

        print 'Network setup complete.'

    
    def input_rates(self, t):
        '''
        Returns firing rates for spatially patterned stimulus to E cells.  
        stimuluss is array of stimulus orientations as function of time in ms.
        Range of stimuluss is 0 - 1.
        '''
        return 400*(1+0.35* cos(2*pi*self.stimulus[floor(t*1000)   #360
                                % size (self.stimulus)] + pi 
                                + linspace(2*pi/self.E_cell_params['Ncells'], 
                                2*pi,self.E_cell_params['Ncells'])))*Hz

    def simulate(self, sim_duration = 500, stimulus = 0., to_plot = True):
        '''
        Simulate network:  
        - If stimulus is a single number, simulates network with fixed stimulus 
        orientation for duration specified by sim_duration.  
        - If stimulus is an array, it is treated as a time varying stimulus at 1ms time resolution
        and stim_duration is ignored.
        '''        
        
        if size(stimulus) > 1:
            sim_duration = size(stimulus)
            self.stimulus = stimulus
        else:
            self.stimulus = ones(sim_duration) * stimulus

        self.net.reinit(states = False)  # Reset recording devices.

        # Randomise initial membrane potentials.
        self.I_cells.vm =  randn(len(self.I_cells)) * 10 * mV - 60 * mV
        self.E_cells.vm =  randn(len(self.E_cells)) * 10 * mV - 60 * mV

        print 'Simulation running...'

        start_time = time.time()
        self.net.run(sim_duration * ms)
        duration = time.time() - start_time

        print 'Simulation time:',duration,'seconds'
        print 'Principal cell rate: %(Erate)2.2f Hz' %{'Erate' : mean(self.E_cell_rate_monitor.rate[50::])}
        print 'Interneuron rate: %(Irate)2.2f Hz' %{'Irate' :    mean(self.I_cell_rate_monitor.rate[50::])}

        if to_plot: self.plot()

    def plot(self, fig_no = 1):
        figure(fig_no)
        clf()
        subplot(3,1,1)
        raster_plot(self.E_cell_spike_monitor)
        subplot(3,1,2)
        raster_plot(self.I_cell_spike_monitor)
        subplot(3,1,3)
        plot(self.E_cell_rate_monitor.times, self.E_cell_rate_monitor.rate,'b')
        plot(self.I_cell_rate_monitor.times, self.I_cell_rate_monitor.rate,'g')
        ylabel('Population average rates')
        show()

if __name__ == "__main__":
    gamma_network = input_network(state = 'gamma')
    gamma_network.simulate(stimulus = sin(arange(0,1000)*2*pi/1000))