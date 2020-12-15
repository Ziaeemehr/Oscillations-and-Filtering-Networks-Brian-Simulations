import os
import nest
import numpy as np
import pylab as plt
import nest.raster_plot
from os.path import join


class NEURON(object):

    data_path = "data"

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

    def set_params(self, **par):

        self.N = par['N']
        self.t_simulation = par['t_simulation']
        self.I_e = par['I_e']
        self.p_rate = 400.
        # tau_m= 10.,

        self.common_params = {    # Parameters common to all neurons.
            'C_m': 100.,
            'E_L': -60.,
            'Delta_T': 2.,
            'V_reset': -65.,
            'V_th': -50.,
            "I_e": par['I_e'],
            "V_m": -70.0
        }

        self.EXp = {  # External to E - cell
            'Erev': 0,  # Reversal potential
            'Tau': 4,  # Alpha function tau
            'g': 1.4,  # Peak conductance  (1 in paper)
            'delay': self.dt}

        self.common_params['g_L'] = self.common_params['C_m'] / 10.0

    def build(self):

        self.nodes = nest.Create("aeif_cond_alpha", self.N)
        nest.SetStatus(self.nodes, params=self.common_params)

        self.spikes_det = nest.Create("spike_detector")
        nest.SetStatus(self.spikes_det, [
            {"withtime": True,
             "withgid": True,
             }])

        nest.Connect(self.nodes, self.spikes_det)

        self.multimeter = nest.Create("multimeter")
        nest.SetStatus(self.multimeter, {"withtime": True,
                                         "record_from": ["V_m"],
                                         "interval": self.dt})
        nest.Connect(self.multimeter, self.nodes)

        nest.CopyModel("static_synapse", "excitatory_input",
                       {"weight": self.EXp['g'],
                        "delay": self.EXp['delay']})
        self.noise = nest.Create("poisson_generator", params={"rate": self.p_rate})
        nest.Connect(self.noise, self.nodes, syn_spec="excitatory_input")

    def simulate(self, ax):

        nest.Simulate(self.t_simulation)

        # fig, ax = plt.subplots(2, sharex=True)

        dSD = nest.GetStatus(self.spikes_det, keys='events')[0]
        evs = dSD['senders']
        tsd = dSD["times"]
        ax[0].plot(tsd, evs, '.', c="k", markersize=3)
        ax[1].set_xlabel("Time (ms)", fontsize=13)
        ax[0].set_ylabel("Neuron ID", fontsize=13)
        ax[0].tick_params(labelsize=13)
        ax[1].tick_params(labelsize=13)

        dmm = nest.GetStatus(self.multimeter)[0]
        Vms = dmm['events']['V_m']
        ts = dmm['events']['times']

        ax[1].plot(ts, Vms, lw=1, label=str(self.I_e))
        ax[1].legend()

        # plt.show()


if __name__ == "__main__":

    par = {"N": 1,
           "t_simulation": 2000.0}
    dt = 0.01

    fig, ax = plt.subplots(2, sharex=True)

    for I_e in [120.]:
        par['I_e'] = I_e
        sol = NEURON(dt, 1)
        sol.set_params(**par)
        sol.build()
        sol.simulate(ax)

    plt.show()
