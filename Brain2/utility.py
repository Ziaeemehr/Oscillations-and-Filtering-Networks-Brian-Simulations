from input_network import dt0
import numpy as np
import brian2 as b2
import pylab as plt
from scipy import fftpack
# from scipy.fftpack import fft, fftshift
# from scipy.stats.stats import pearsonr
# from scipy.signal import (welch, filtfilt, butter, hilbert)
import logging
logger = logging.getLogger('ftpuploader')


def fft_1d_real(signal, fs):
    """
    fft from 1 dimensional real signal

    :param signal: [np.array] real signal
    :param fs: [float] frequency sampling in Hz
    :return: [np.array, np.array] frequency, normalized amplitude

    -  example:

    >>> B = 30.0  # max freqeuency to be measured.
    >>> fs = 2 * B
    >>> delta_f = 0.01
    >>> N = int(fs / delta_f)
    >>> T = N / fs
    >>> t = np.linspace(0, T, N)
    >>> nu0, nu1 = 1.5, 22.1
    >>> amp0, amp1, ampNoise = 3.0, 1.0, 1.0
    >>> signal = amp0 * np.sin(2 * np.pi * t * nu0) + amp1 * np.sin(2 * np.pi * t * nu1) +
            ampNoise * np.random.randn(*np.shape(t))
    >>> freq, amp = fft_1d_real(signal, fs)
    >>> pl.plot(freq, amp, lw=2)
    >>> pl.show()

    """

    N = len(signal)
    F = fftpack.fft(signal)
    f = fftpack.fftfreq(N, 1.0 / fs)
    mask = np.where(f >= 0)

    freq = f[mask]
    amplitude = 2.0 * np.abs(F[mask] / N)

    return freq, amplitude


def plot_raster_from_device(spike_monitor_E,
                            rate_monitor_E,
                            filename,
                            width=5, **kwargs):

    fig, ax = plt.subplots(3, figsize=(10, 8))

    ax[0].plot(spike_monitor_E.t/b2.ms, spike_monitor_E.i, '.k', ms=1)

    # plot firing rate ----------------------------------------------
    rate = None
    try:
        rate = rate_monitor_E.smooth_rate(width=width*b2.ms)/b2.Hz
    except Exception as e:
        logger.error(str(e))

    ax[1].plot(rate_monitor_E.t / b2.ms,
               rate, color='b', label="e")

    freq, amp = fft_1d_real(rate, 1/(dt0))
    ax[2].semilogx(freq, amp, lw=2, color="k")

    ax[0].set_ylabel("E cells")
    ax[1].set_ylabel('Population average rates')
    ax[1].legend(loc="upper right")
    ax[1].set_xlabel("time (ms)")
    ax[0].margins(x=0)
    ax[1].margins(x=0)
    ax[2].set_xlim(0.01, 200)
    plt.tight_layout()

    plt.savefig("{:s}".format(filename), dpi=150)
    plt.close()
#--------------------------------------------------------------------

def plot_raster_from_data(filename, xlim=None, title=None):

    fig, ax = plt.subplots(3, figsize=(10, 8))

    data = np.load(filename+".npz")
    rate_times = data['rate_times']
    spikes_time = data['spikes_time']
    spikes_id = data['spikes_id']
    rate_amp = data['rate_amp']

    ax[0].plot(spikes_time, spikes_id, '.k', ms=1)

    # plot firing rate ----------------------------------------------

    ax[1].plot(rate_times, rate_amp, color='b')
    dt0 = (rate_times[1]-rate_times[0])*1e-3
    freq, amp = fft_1d_real(rate_amp, 1/(dt0))
    ax[2].plot(freq, amp, lw=2, color="k")

    ax[0].set_ylabel("Cell indices")
    ax[1].set_ylabel('Population average rates')
    # ax[1].legend(loc="upper right")
    ax[1].set_xlabel("Time (ms)")
    
    if xlim is not None:
        ax[0].set_xlim(xlim)
        ax[1].set_xlim(xlim)
    
    ax[2].set_xlim(0.01, 100)
    ax[2].set_xlabel("frequency [Hz]")
    ax[2].set_ylabel("amplitude")
    if title is not None:
        ax[0].set_title(title)

    plt.tight_layout()

    plt.savefig("{:s}.png".format(filename), dpi=150)
    plt.close()



# def plot(spike_monitor_E, rate_monitor_E):
#         #  spike_monitor_I,
#         #  state_monitor_E,
#         #  state_monitor_I,
#         #  rate_monitor_I,
#         #  plot_voltages=False):

#     fig, ax = plt.subplots(2, figsize=(10, 8), sharex=True)

#     ax[0].plot(spike_monitor_E.t/b2.ms, spike_monitor_E.i, '.k', ms=0.5)
#     # ax[1].plot(spike_monitor_I.t/b2.ms, spike_monitor_I.i, '.k', ms=0.5)
#     # if plot_voltages:
#     #     for i in range(num_E_cells):
#     #         ax[3].plot(state_monitor_E.t/b2.ms,
#     #                    state_monitor_E.vm[i]/b2.mV)
#     #     for i in range(num_I_cells):
#     #         ax[3].plot(state_monitor_I.t/b2.ms,
#     #                    state_monitor_I.vm[i]/b2.mV)
#     try:
#         ax[2].plot(rate_monitor_E.t/b2.ms,
#                    rate_monitor_E.smooth_rate(width=5*b2.ms)/b2.Hz,
#                    color='b',
#                    label="e")
#         # ax[2].plot(rate_monitor_I.t/b2.ms,
#         #            rate_monitor_I.smooth_rate(width=5*b2.ms)/b2.Hz,
#                 #    color='g',
#                 #    label="i")
#     except Exception as e:
#         logger.error(str(e))
#     ax[2].set_ylabel('Population average rates')
#     ax[-1].set_xlabel("time (ms)")
#     ax[0].set_ylabel("E cells")
#     ax[1].set_ylabel("I cells")
#     # ax[3].set_ylabel("Voltages E")
#     ax[-1].legend(loc="upper right")
#     plt.savefig("data/init.png", dpi=150)
