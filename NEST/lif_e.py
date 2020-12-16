import os
import nest
import numpy as np
import pylab as plt
import nest.raster_plot
from os.path import join
from numpy.random import rand

np.random.seed(2)


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

    def set_params(self, **par):

        self.N = par['N']
        self.t_simulation = par['t_simulation']
        self.I_e = par['I_e']
        self.p_rate = par['p_rate']  # Hz
        # tau_m= 10.,

        self.common_params = {    # Parameters common to all neurons.
            'C_m': 100.,
            'E_L': -60.,
            'Delta_T': 2.,
            'V_reset': -65.,
            'V_th': -50.,
            "I_e": par['I_e'],
            'E_in': -80.0,  # Reversal potential
            'E_ex': 0.0,
            'tau_syn_ex': 4.,  # Alpha function tau
            'tau_syn_in': 3.,
            # "V_m": -70.0
        }

        self.EXp = {  # External to E - cell
            'weight': 1.4,  # Peak conductance  (1 in paper)
            'delay': self.dt}

        self.EEp = {  # E - cell to E - cell
            'weight': 0.1,  # Peak conductance
            'delay': self.dt}

        self.common_params['g_L'] = self.common_params['C_m'] / 10.0
    # ---------------------------------------------------------------
    def build(self):

        self.nodes = nest.Create("aeif_cond_alpha", self.N)
        nest.SetStatus(self.nodes, params=self.common_params)

        # V_reset, V_th = self.common_params["V_reset"], self.common_params["V_th"]
        # for node in self.nodes:
        # nest.SetStatus([node], {"V_m": V_reset + (V_th-V_reset)*rand()})
        nest.SetStatus(self.nodes, "V_m", [-65.0, -50.0])

        self.spikes_det = nest.Create("spike_detector", self.N)
        nest.SetStatus(self.spikes_det,
                       [{"withtime": True,
                         "withgid": True}])

        nest.Connect(self.nodes, self.spikes_det)

        self.multimeter = nest.Create("multimeter", self.N)
        nest.SetStatus(self.multimeter, {"withtime": True,
                                         "record_from": ["V_m"],
                                         "interval": self.dt})
        nest.Connect(self.multimeter, self.nodes, "one_to_one")

        nest.CopyModel("static_synapse",
                       "external_input",
                       {"weight": self.EXp['weight'],
                        "delay": self.EXp['delay']})
        nest.CopyModel("static_synapse",
                       "EE_synapse",
                       {"weight": self.EEp['weight'],
                        "delay": self.EEp['delay']}
                       )
        self.noise = nest.Create("poisson_generator",
                                 self.N,
                                 params={"rate": self.p_rate})
        nest.Connect(self.noise,
                     self.nodes,
                     "one_to_one",
                     syn_spec="external_input")
        nest.Connect([self.nodes[0]], [self.nodes[1]],
                     "one_to_one",
                     syn_spec="EE_synapse")

    def simulate(self):

        nest.Simulate(self.t_simulation)
    # ---------------------------------------------------------------
    def plot(self, ax):

        # fig, ax = plt.subplots(2, sharex=True)
        dSD = nest.GetStatus(self.spikes_det, keys='events')[0]
        evs = dSD['senders']
        tsd = dSD["times"]
        ax[0].plot(tsd, evs, '.', c="k", markersize=3)
        ax[1].set_xlabel("Time (ms)", fontsize=13)
        ax[0].set_ylabel("Neuron ID", fontsize=13)
        ax[0].tick_params(labelsize=13)
        ax[1].tick_params(labelsize=13)

        dmm = nest.GetStatus(self.multimeter)
        for i in range(self.N):
            Vms = dmm[i]['events']['V_m']
            ts = dmm[i]['events']['times']

            ax[1].plot(ts, Vms, lw=1, label=str(i+1))
        ax[1].legend()
        ax[1].margins(x=0.0)
        # plt.savefig("data/E.png")
        # plt.show()
# -------------------------------------------------------------------


if __name__ == "__main__":

    par = {"N": 2,
           "p_rate": 0.0,
           "t_simulation": 2000.0}
    dt = 0.01

    fig, ax = plt.subplots(2, sharex=True)

    for I_e in [121.]:
        par['I_e'] = I_e
        sol = NEURON(dt, 1)
        sol.set_params(**par)
        sol.build()
        sol.simulate()
        sol.plot(ax)

    plt.savefig("data/E.png")
    plt.show()
