import os
import nest
import utility
import numpy as np
import pylab as plt
from time import time
import nest.raster_plot
from os.path import join
from copy import deepcopy


class NEURON(object):

    data_path = "data"

    # ---------------------------------------------------------------

    def __init__(self, dt, nthreads):

        self.name = self.__class__.__name__
        nest.ResetKernel()
        nest.set_verbosity('M_QUIET')
        self.dt = dt

        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)

        nest.SetKernelStatus({
            "resolution": dt,
            "print_time": False,
            "overwrite_files": True,
            "data_path": self.data_path,
            "local_num_threads": nthreads})

        np.random.seed(125472)

    # ---------------------------------------------------------------

    def set_params(self):

        # self.N = par['N']
        # self.t_simulation = par['t_simulation']
        # self.I_e = par['I_e']
        # self.p_rate = 400.  # Hz
        # tau_m= 10.,
        self.common_params = deepcopy(common_params)
        self.E_cell_params = deepcopy(E_cell_params)
        self.I_cell_params = deepcopy(I_cell_params)
        self.IIp = deepcopy(IIp)
        self.EEp = deepcopy(EEp)
        self.IEp = deepcopy(IEp)
        self.EIp = deepcopy(EIp)
        self.IXp = deepcopy(IXp)
        self.EXp = deepcopy(EXp)

        if state == 'gamma':

            print('Gamma oscillation state.')
            self.IXp['weight'] = 0.3
            self.IEp['weight'] = 0.4

        elif state == 'beta':
            print('Beta oscillation state.')
            self.IXp['g'] = 0.5
            self.IXp['Tau'] = 12.
            self.EXp['g'] = 0.55  # (0.47 in paper)
            self.EXp['Tau'] = 12.
            self.EEp['g'] = 0.05
            self.EEp['Tau'] = 12.
            self.IEp['g'] = 0.1  # (0.08 in paper)
            self.IEp['Tau'] = 12.
            self.EIp['g'] = 0.1
            self.EIp['Tau'] = 15.
            self.IIp['g'] = 0.2
            self.IIp['Tau'] = 15.

    # ---------------------------------------------------------------
    def build(self):

        self.E_cells = nest.Create("aeif_cond_alpha",
                                   self.E_cell_params['Ncells'])
        self.I_cells = nest.Create("aeif_cond_alpha",
                                   self.I_cell_params['Ncells'])
        nest.SetStatus(self.E_cells,
                       params=self.common_params)

        self.spikes_det_E = nest.Create("spike_detector")
        self.spikes_det_I = nest.Create("spike_detector")

        nest.SetStatus(self.spikes_det_E, [{"withtime": True,
                                            "withgid": True}])
        nest.SetStatus(self.spikes_det_I, [{"withtime": True,
                                            "withgid": True}])

        # self.multimeter = nest.Create("multimeter")
        # nest.SetStatus(self.multimeter, {"withtime": True,
        #                                  "record_from": ["V_m"],
        #                                  "interval": self.dt})
        # nest.Connect(self.multimeter, self.E_cells)
        nest.CopyModel("static_synapse", "EX_synapse",
                       {"weight": self.EXp['weight'],
                        "delay": self.EXp['delay']})
        nest.CopyModel("static_synapse", "IX_synapse",
                       {"weight": self.IXp['weight'],
                        "delay": self.IXp['delay']})
        nest.CopyModel("static_synapse", "EE_synapse",
                       {"weight": self.EEp['weight'],
                        "delay": self.EEp['delay']})
        nest.CopyModel("static_synapse", "II_synapse",
                       {"weight": self.IIp['weight'],
                        "delay": self.IIp['delay']})
        nest.CopyModel("static_synapse", "IE_synapse",
                       {"weight": self.IEp['weight'],
                        "delay": self.IEp['delay']})
        nest.CopyModel("static_synapse", "EI_synapse",
                       {"weight": self.EIp['weight'],
                        "delay": self.EIp['delay']})

        self.noise_to_E = nest.Create("poisson_generator",
                                      params={"rate": self.E_cell_params['p_rate'] * self.E_cell_params['Ncells']})
        self.noise_to_I = nest.Create("poisson_generator",
                                      params={"rate": self.I_cell_params['p_rate'] * self.I_cell_params['Ncells']})

    def connect(self):

        Nrec_E = self.E_cell_params["Nrec"]
        Nrec_I = self.I_cell_params["Nrec"]
        nest.Connect(self.E_cells[:Nrec_E], self.spikes_det_E)
        nest.Connect(self.I_cells[:Nrec_I], self.spikes_det_I)

        conn_dict_EE = {'rule': 'pairwise_bernoulli', 'p': self.EEp['p']}
        conn_dict_IE = {'rule': 'pairwise_bernoulli', 'p': self.IEp['p']}
        conn_dict_EI = {'rule': 'pairwise_bernoulli', 'p': self.EIp['p']}
        conn_dict_II = {'rule': 'pairwise_bernoulli', 'p': self.IIp['p']}

        nest.Connect(self.noise_to_E, self.E_cells, syn_spec="EX_synapse")
        nest.Connect(self.noise_to_I, self.I_cells, syn_spec="IX_synapse")

        nest.Connect(self.E_cells, self.E_cells,
                     conn_spec=conn_dict_EE,
                     syn_spec="EE_synapse")
        nest.Connect(self.I_cells, self.I_cells,
                     conn_spec=conn_dict_II,
                     syn_spec="II_synapse")
        nest.Connect(self.E_cells, self.I_cells,
                     conn_spec=conn_dict_IE,
                     syn_spec="IE_synapse")
        nest.Connect(self.I_cells, self.E_cells,
                     conn_spec=conn_dict_EI,
                     syn_spec="EI_synapse")

    # ---------------------------------------------------------------

    def simulate(self, t_simulation):
        
        self.set_params()
        self.build()
        self.connect()

        nest.Simulate(t_simulation)

        subname = ""  # str("%.3f-%.3f" % (self.g, self.eta))
        E_t, E_gid = utility.get_spike_times(self.spikes_det_E)
        I_t, I_gid = utility.get_spike_times(self.spikes_det_I)

        events_ex = nest.GetStatus(self.spikes_det_E, "n_events")[0]
        events_in = nest.GetStatus(self.spikes_det_I, "n_events")[0]

        rate_ex = events_ex / t_simulation * \
            1000.0 / self.E_cell_params['Nrec']
        rate_in = events_in / t_simulation * \
            1000.0 / self.I_cell_params['Nrec']

        print("rate_E = {:10.2f}".format(rate_ex))
        print("rate_I = {:10.2f}".format(rate_in))

        np.savez(join(self.data_path,"E"+subname),
                 t=E_t,
                 gid=E_gid,
                 rate=rate_ex)
        np.savez(join(self.data_path,"I"+subname),
                 t=I_t,
                 gid=I_gid,
                 rate=rate_in)

        return self.spikes_det_E, self.spikes_det_I
    # ---------------------------------------------------------------

# -------------------------------------------------------------------


if __name__ == "__main__":

    from config import *

    # fig, ax = plt.subplots(2, sharex=True)
    start_time = time()
    sol = NEURON(dt, nthreads)

    sp_mon_E, sp_mon_I = sol.simulate(t_simulation)
    utility.display_time(time()-start_time)
    # nest.raster_plot.from_device(sp_mon_E, hist=True, title="E cells")
    # plt.savefig("data/E_input_network.png")
    # plt.close()
    # nest.raster_plot.from_device(sp_mon_I, hist=True, title="I cells")
    # plt.savefig("data/I_input_network.png")
    
    # plt.show()
