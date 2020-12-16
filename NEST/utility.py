import os
import nest
import numpy as np
import pylab as plt
from time import time
import nest.raster_plot
from os.path import join
from copy import deepcopy


def make_plot(ts, gids, val, hist_binwidth, sel, xlim):
    """
    Generic plotting routine that constructs a raster plot along with
    an optional histogram (common part in all routines above)
    """

    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=(10, 8))
    gs1 = gridspec.GridSpec(4, 1, hspace=0.1)
    axs = []

    xlabel = "Time (ms)"
    ylabel = "Neuron ID"
    if xlim == None:
        xlim = plt.xlim()

    axs.append(fig.add_subplot(gs1[0:3]))
    axs.append(fig.add_subplot(gs1[3]))

    axs[0].plot(val[:, 1], val[:, 0], '.')
    axs[0].set_ylabel(ylabel)
    axs[0].set_xticks([])
    axs[0].set_xlim(xlim)

    t_bins = np.arange(
        np.amin(ts),
        np.amax(ts),
        float(hist_binwidth))
    n, bins = histogram(ts, bins=t_bins)
    num_neurons = len(np.unique(gids))
    heights = 1000 * n / (hist_binwidth * num_neurons)
    axs[1].bar(t_bins, heights, width=hist_binwidth,
               color='royalblue', edgecolor='black')
    axs[1].set_yticks([int(x) for x in np.linspace(
        0.0, int(max(heights) * 1.1) + 5, 4)])
    axs[1].set_ylabel("Rate (Hz)")
    axs[1].set_xlabel(xlabel)
    axs[1].set_xlim(xlim)


def extract_events(ts, gids, time=None, sel=None):
    """
    Extracts all events within a given time interval or are from a
    given set of neurons.
    - data is a matrix such that

    - time is a list with at most two entries such that
      time=[t_max] extracts all events with t< t_max
      time=[t_min, t_max] extracts all events with t_min <= t < t_max
    - sel is a list of gids such that
      sel=[gid1, ... , gidn] extracts all events from these gids.
      All others are discarded.
    Both time and sel may be used at the same time such that all
    events are extracted for which both conditions are true.
    """

    val = []

    if time is not None:
        t_max = time[-1]
        if len(time) > 1:
            t_min = time[0]
        else:
            t_min = 0
    else:
        print("provide time array")
        exit(0)

    for t, gid in zip(ts, gids):

        if time and (t < t_min or t >= t_max):
            continue
        if not sel or gid in sel:
            val.append([gid, t])

    return np.array(val)
#------------------------------------------------------------------------#


def histogram(a, bins=10, bin_range=None, normed=False):
    from numpy import asarray, iterable, linspace, sort, concatenate

    a = asarray(a).ravel()

    if bin_range is not None:
        mn, mx = bin_range
        if mn > mx:
            raise ValueError("max must be larger than min in range parameter")

    if not iterable(bins):
        if bin_range is None:
            bin_range = (a.min(), a.max())
        mn, mx = [mi + 0.0 for mi in bin_range]
        if mn == mx:
            mn -= 0.5
            mx += 0.5
        bins = linspace(mn, mx, bins, endpoint=False)
    else:
        if (bins[1:] - bins[:-1] < 0).any():
            raise ValueError("bins must increase monotonically")

    # best block size probably depends on processor cache size
    block = 65536
    n = sort(a[:block]).searchsorted(bins)
    for i in range(block, a.size, block):
        n += sort(a[i:i + block]).searchsorted(bins)
    n = concatenate([n, [len(a)]])
    n = n[1:] - n[:-1]

    if normed:
        db = bins[1] - bins[0]
        return 1.0 / (a.size * db) * n, bins
    else:
        return n, bins


def get_spike_times(detec):
    if nest.GetStatus(detec, "to_memory")[0]:
        if not nest.GetStatus(detec)[0]["model"] == "spike_detector":
            raise nest.NESTError("Please provide a spike_detector.")

        ev = nest.GetStatus(detec, "events")[0]
        ts = ev["times"]
        gids = ev["senders"]

    else:
        raise nest.NESTError(
            "No data to plot. Make sure that to_memory is set.")
    return ts, gids
# ---------------------------------------------------------------#


def raster_plot_from_data(
        ts, gids, hist_binwidth=5.0,
        xlim=None, sel=None):
    """
    Plot raster from data
    """

    if not len(ts):
        raise nest.NESTError("No events recorded!")

    val = extract_events(ts, gids, sel=sel)

    if val.shape[0] > 1:
        make_plot(ts, gids, val, hist_binwidth, sel, xlim)


def display_time(time):
    ''' 
    show real time elapsed
    '''
    hour = int(time/3600)
    minute = (int(time % 3600))//60
    second = time-(3600.*hour+60.*minute)
    print("Done in %d hours %d minutes %09.6f seconds"
          % (hour, minute, second))


# def plot(sp_mon_E, sp_mon_I, ax=None, filename="input_network"):

#     save_fig = False
#     if ax is None:
#         fig, ax = plt.subplots(1, sharex=True)
#         save_fig = True
    
#     dSD = nest.GetStatus(sp_mon_E, keys='events')[0]
#     evs = dSD['senders']
#     tsd = dSD["times"]
#     ax.plot(tsd, evs, '.', c="k", markersize=1)
#     ax.set_xlabel("Time (ms)", fontsize=13)
#     ax.set_ylabel("Neuron ID", fontsize=13)
#     ax.tick_params(labelsize=13)
#     # ax[1].tick_params(labelsize=13)

#     # dmm = nest.GetStatus(self.multimeter)[0]
#     # Vms = dmm['events']['V_m']
#     # ts = dmm['events']['times']

#     # ax[1].plot(ts, Vms, lw=1, label=str(self.I_e))
#     # ax[1].legend()
#     plt.tight_layout()
#     if save_fig:
#         plt.savefig(join("data", "{}.png".format(filename)))
#         plt.close()
#     # plt.show()


