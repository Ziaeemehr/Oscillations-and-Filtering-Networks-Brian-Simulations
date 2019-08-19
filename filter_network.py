'''
Class for implementing filtering network from Akam et al, Neuron 2010, but lacking the spatial
organisation, such that it performs filtering on activity from networks with homogeneous activiy
rather than an input with a spatial population code.  This is the network used for figure S1.B2.

# Copyright (c) Thomas Akam 2012.  Licenced under the GNU General Public License v3.
'''

from brian import *
from brian.library.IF import *
from brian.library.synapses import *
import time 

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


class filter_network:
    def __init__(self):

        print 'Building network.'
    
        self.I_cell_params={    # Feed-forward interneuron parameters
                            'Ncells' : 400,
                            'C' : 100*pF,    
                            'taum' : 10*msecond,
                            'EL' : -60*mV,
                            'DeltaT' : 2*mV,
                            'IXmean' : 30*pA,
                            'IXsd' : 20*pA,
                            'Vreset' : -65*mV,
                            'VTmean' : -50*mV,
                            'VTsd' : 2*mV}
    
        self.I_cell_params['gL'] = self.I_cell_params['C'] / self.I_cell_params['taum']
    
        self.E_cell_params={   # E - cell parameters
                            'Ncells' : 400,
                            'C' : 100*pF,    
                            'taum' : 10*msecond,
                            'EL' : -60*mV,
                            'DeltaT' : 2*mV,
                            'IXmean' : 30*pA,
                            'IXsd' : 20*pA,
                            'Vreset' : -65*mV,
                            'VTmean' : -50*mV,
                            'VTsd' : 2*mV}
    
        self.E_cell_params['gL'] = self.E_cell_params['C'] / self.E_cell_params['taum']
    
        self.input_params = {    # Input activity parameters
                            'Ninputs'  : 800,
                            'DCrate'   : 10.*Hz,
                            'ACrate'   : 10.*Hz,
                            'Freq'     :  40*Hz,
                            'totalrate': 0.*Hz,
                            'amp'      : 0.}
    
    
        # Synapse parameters
    
        self.IXp={ # External to Interneuron
                  'Erev' : 0*mvolt,    #Reversal potential
                  'Tau'  : 4*msecond,  #Alpha function tau
                  'g'    : 0.3*nS,     #Peak conductance
                  'sparseness' : 110./self.input_params['Ninputs'],  
                  'delay' : 0.5*ms}
    
        self.IIp={ # Interneuron to Interneuron
                  'Erev' : -80*mvolt,    #Reversal potential
                  'Tau'  : 8*msecond,  #Alpha function tau
                  'g'    : 0.1*nS,     #Peak conductance
                  'sparseness' : 200./self.I_cell_params['Ncells'],  
                  'delay' : 0.5*ms}
    
        self.EXp={ # External to E - cell
                  'Erev' : 0*mvolt,    #Reversal potential
                  'Tau'  : 4*msecond,  #Alpha function tau
                  'g'    : 0.1*nS,     #Peak conductance
                  'sparseness' : 600./self.input_params['Ninputs'],  
                  'delay' : 0.5*ms}
          
        self.EIp={ # Interneuron to E - cell
                  'Erev' : -80*mvolt,    #Reversal potential
                  'Tau'  : 4*msecond,  #lpha function tau
                  'g'    : 0.75*nS,     #Peak conductance
                  'sparseness' : 200./self.I_cell_params['Ncells'],
                  'delay' : 0.5*ms}
    
        # Interneuron equations
    
        self.eqs_I  = exp_IF_memb_eq(self.I_cell_params)
        self.eqs_I += alpha_conductance('gi',self.IIp['Erev'], self.IIp['Tau'])
        self.eqs_I += alpha_conductance('ge',self.IXp['Erev'], self.IXp['Tau'])
    
        # E - cell equations
    
        self.eqs_E  = exp_IF_memb_eq(self.E_cell_params)
        self.eqs_E += alpha_conductance('gi',self.EIp['Erev'], self.EIp['Tau'])
        self.eqs_E += alpha_conductance('ge',self.EXp['Erev'], self.EXp['Tau'])
    
        #create cell populations and inputs
    
        self.I_cells = NeuronGroup(self.I_cell_params['Ncells'], model = self.eqs_I, threshold = -0 * mvolt, reset = self.I_cell_params['Vreset'], refractory = 1 * ms)
        self.E_cells = NeuronGroup(self.E_cell_params['Ncells'], model = self.eqs_E, threshold = -0 * mvolt, reset = self.E_cell_params['Vreset'], refractory = 1 * ms)
    
        self.ExtIn = PoissonGroup(self.input_params['Ninputs'], rates = lambda t:
                    (1 + self.input_params['amp'] * cos(t * 2 * pi * self.input_params['Freq'])) * self.input_params['totalrate'])
    
        self.cIX = Connection(self.ExtIn,   self.I_cells,'ge', weight = self.IXp['g'], sparseness = self.IXp['sparseness'], delay = self.IXp['delay'])
        self.cII = Connection(self.I_cells, self.I_cells,'gi', weight = self.IIp['g'], sparseness = self.IIp['sparseness'], delay = self.IIp['delay'])
        self.cEX = Connection(self.ExtIn,   self.E_cells,'ge', weight = self.EXp['g'], sparseness = self.EXp['sparseness'], delay = self.EXp['delay'])
        self.cEI = Connection(self.I_cells, self.E_cells,'gi', weight = self.EIp['g'], sparseness = self.EIp['sparseness'], delay = self.EIp['delay'])
    
        # Create recording  devices
    
        self.I_cell_spike_monitor = SpikeMonitor(self.I_cells)
        self.I_cell_rate_monitor  = PopulationRateMonitor(self.I_cells, bin=1*ms)
        self.I_cell_intracellular_monitor = MultiStateMonitor(self.I_cells[0], record=[0])
    
        self.E_cell_spike_monitor = SpikeMonitor(self.E_cells)
        self.E_cell_rate_monitor  = PopulationRateMonitor(self.E_cells,bin=1*ms)
        self.E_cell_intracellular_monitor = MultiStateMonitor(self.E_cells[0], record=[0])

        self.monitors = [self.I_cell_spike_monitor, self.I_cell_rate_monitor, self.I_cell_intracellular_monitor,
                         self.E_cell_spike_monitor, self.E_cell_rate_monitor, self.E_cell_intracellular_monitor]
  
        # Initialise random parameters.

        self.I_cells.VT = (randn(len(self.I_cells)) * self.I_cell_params['VTsd'] + self.I_cell_params['VTmean'])
        self.I_cells.IX = (randn(len(self.I_cells)) * self.I_cell_params['IXsd'] + self.I_cell_params['IXmean'])
        self.E_cells.VT = (randn(len(self.E_cells)) * self.E_cell_params['VTsd'] + self.E_cell_params['VTmean'])
        self.E_cells.IX = (randn(len(self.E_cells)) * self.E_cell_params['IXsd'] + self.E_cell_params['IXmean'])

        # Create network object from neuron groups, connections and monitors.
        self.net = MagicNetwork()

        print 'Network setup complete.'


    def simulate(self, DCrate, ACrate, sim_duration = 500, to_plot = True):    # Simulate and output some data to console.

        self.sim_duration = sim_duration * ms

        # Set input parameters.
        self.input_params['DCrate'] = DCrate
        self.input_params['ACrate'] = ACrate
        self.input_params['totalrate'] = self.input_params['ACrate'] + self.input_params['DCrate']
        self.input_params['amp'] = self.input_params['ACrate'] / (self.input_params['totalrate'] + 1e-15)

        #Reset recording devices.
        self.net.reinit(states = False)

        # Randomise initial membrane potentials.
        self.I_cells.vm =  randn(len(self.I_cells)) * 10 * mV - 60 * mV
        self.E_cells.vm =  randn(len(self.E_cells)) * 10 * mV - 60 * mV

        print 'Simulation running...'
        start_time=time.time()
        self.net.run(self.sim_duration)
        duration=time.time()-start_time
        print 'Simulation time:',duration,'seconds'
        print 'Interneuron rate: %(Irate)2.2f Hz' %{'Irate' :    mean(self.I_cell_rate_monitor.rate[50::])}
        print 'Principal cell rate: %(Erate)2.2f Hz' %{'Erate' : mean(self.E_cell_rate_monitor.rate[50::])}

        if to_plot: self.plot()

        return(mean(self.E_cell_rate_monitor.rate[50::]))


    def plot(self, fig_no = 1):   # Plot data from most recent simulation.
    
        figure(fig_no)
        clf()
        subplot(7,1,1)
        plot(self.E_cell_rate_monitor.times,self.E_cell_rate_monitor.rate,'b')
        plot(self.I_cell_rate_monitor.times,self.I_cell_rate_monitor.rate,'g')
        Inrate=(1+self.input_params['amp']*cos(self.I_cell_rate_monitor.times*2*pi*
                self.input_params['Freq']))*self.input_params['totalrate']
        plot(self.I_cell_rate_monitor.times,Inrate,'k')
        ylabel('Firing\nrates (Hz)', ha = 'center', labelpad = 30)

        subplot(7,1,2)
        raster_plot(self.E_cell_spike_monitor)
        ylabel('E - cell\nspikes.', ha = 'center', labelpad = 30)
        yticks([0,self.E_cell_params['Ncells']])
        xlim([0,self.sim_duration / ms])

        subplot(7,1,3)
        plot(self.E_cell_intracellular_monitor.times,self.E_cell_intracellular_monitor['vm'][0] * 1000.)
        ylabel('E - cell membrane\nvoltage (mV).', ha = 'center', labelpad = 30)

        subplot(7,1,4)
        PCellExCurrent=((self.EXp['Erev']-self.E_cell_intracellular_monitor['vm'][0])*self.E_cell_intracellular_monitor['ge'][0])*1e9
        PCellInCurrent=((self.EIp['Erev']-self.E_cell_intracellular_monitor['vm'][0])*self.E_cell_intracellular_monitor['gi'][0])*1e9
        plot(self.I_cell_intracellular_monitor.times, PCellInCurrent,'g')
        plot(self.I_cell_intracellular_monitor.times, PCellExCurrent,'b')
        plot(self.I_cell_intracellular_monitor.times, PCellExCurrent+PCellInCurrent,'r')
        plot([0,self.sim_duration],[0,0],'k')
        ylabel('E - cell\nmembrane\ncurrents (nA).', ha = 'center', labelpad = 40)

        subplot(7,1,5)
        raster_plot(self.I_cell_spike_monitor)
        ylabel('I - cell\nspikes.', ha = 'center', labelpad = 30)
        yticks([0,self.I_cell_params['Ncells']])
        xlim([0,self.sim_duration / ms])

        subplot(7,1,6)
        plot(self.I_cell_intracellular_monitor.times,self.I_cell_intracellular_monitor['vm'][0] * 1000)
        ylabel('E - cell membrane\nvoltage (mV).', ha = 'center', labelpad = 30)

        subplot(7,1,7)
        InCellExCurrent=((self.IXp['Erev']-self.I_cell_intracellular_monitor['vm'][0])*self.I_cell_intracellular_monitor['ge'][0])*1e9
        InCellInCurrent=((self.IIp['Erev']-self.I_cell_intracellular_monitor['vm'][0])*self.I_cell_intracellular_monitor['gi'][0])*1e9
        plot(self.I_cell_intracellular_monitor .times,InCellInCurrent,'g')
        plot(self.I_cell_intracellular_monitor .times,InCellExCurrent,'b')
        plot(self.I_cell_intracellular_monitor .times,InCellExCurrent+InCellInCurrent,'r')
        plot([0,self.sim_duration],[0,0],'k')
        ylabel('I - cell\nmembrane\ncurrents (nA).', ha = 'center', labelpad = 40)

        show()
        
if __name__ == "__main__":
    filt_net = filter_network()
    filt_net.simulate(10.,10.)