'''
Reproduces figure 6 from Akam et al Neuron 2010, though lacking the high gamma network.  Due to the
somewhat lower oscillation frequencies of the Brian implementations of the input networks, the 
frequencies at which the amplitude patterns are evaluated has been modified and a longer 200ms 
time window is used for the analysis.

# Copyright (c) Thomas Akam 2012.  Licenced under the GNU General Public License v3.
'''

from brian import *     # Used only for units.
import input_network as inp
ion()

def get_spike_arrays(spikes, time_window = [0 * ms, inf * ms] ):
    '''
    Takes spike object genrated by Brian spike monitor and returns numpy arrays
    of spike IDs and times in ms (relative to start of time range), for those 
    spikes that occured in the specified time range.
    '''
    spikes = [s for s in spikes if s[1] > time_window[0] and s[1] < time_window[1] ]
    spike_IDs     = array( [ s[0] for s in spikes ] )
    spike_times   = array( [ float((s[1] - time_window[0]) / ms) for s in spikes ] )
    return (spike_IDs, spike_times)


def amplitude_pattern_at_freq(spikes, time_window, frequency):
    '''
    Calculates spatial pattern of firing rate oscillation amplitude by method detailed in Akam et al.,
    Neuron 2010.  Briefly; the population of neurons is sub-divided into groups with similar stimulus 
    tuning.  The Fourier amplitude spectrum of the firing rate of each group is evaluated over the 
    specified time window  (using Hanning window and zero padding), and the patern of amplitude across
    the groups at the specified frequency is returned. 
    '''

    spike_IDs, spike_times  = get_spike_arrays(spikes, time_window)

    window_length = float((time_window[1] - time_window[0]) / ms)
    fft_length = 512
    n_neurons = 8000
    neurons_per_bin = 400

    bin_edges = arange(0, n_neurons, neurons_per_bin)
    n_bins = size(bin_edges)
    padding_length = (fft_length - window_length) / 2
    rate_pattern = zeros(n_bins)
    amplitude_pattern = zeros(n_bins)
    for i, bin_start in enumerate(bin_edges):
        spike_times_bin = spike_times[(spike_IDs >= bin_start) & (spike_IDs < (bin_start+neurons_per_bin))]
        spike_density , l = histogram(spike_times_bin, arange(0 , window_length + 1))
        spike_rate = 1000. * spike_density
        rate_pattern[i] = sum(spike_rate) / window_length
        rate_windowed_and_padded = hstack([ zeros(ceil(padding_length)),
                                   spike_rate * hanning(window_length),
                                   zeros(floor(padding_length))])
        amplitude_spectrum = abs(fft(rate_windowed_and_padded)) / (window_length / 2.)
        amplitude_pattern[i] = amplitude_spectrum[round(frequency * ceil((fft_length - 1)/2.) / 500)]

    return (amplitude_pattern, rate_pattern)

def plot_vert_raster( spikes, time_window, n_neurons ):
    '''
    Plot vertical spike raster with time going downwards.
    '''
    spike_IDs, spike_times  = get_spike_arrays(spikes, time_window)
    plot(spike_IDs, -spike_times, 'k.')
    xticks((1, n_neurons))
    yticks((0, (time_window[0]-time_window[1]) / ms) , (0 , int((time_window[1]-time_window[0]) / ms)) ) 
    xlim(1,n_neurons)
    ylim((time_window[0]-time_window[1]) / ms, 0)

def plot_amp_pattern(pattern, ymax, n_groups = 20, col = 'b' ):
    plot(pattern,'o', color = col)
    ylim((0, ymax))
    yticks((0,ymax))
    xlim((-1, n_groups))
    xticks((0, n_groups - 1), (1, n_groups))

def plot_amp_spec(spikes, time_window):
    spike_IDs, spike_times  = get_spike_arrays(spikes, time_window)
    window_length = float((time_window[1] - time_window[0]) / ms)
    fft_length = 1024
    padding_length = (fft_length - window_length) / 2
    spike_density , l = histogram(spike_times, arange(0 , window_length + 1))
    spike_rate = 1000. * spike_density
    rate_windowed_and_padded = hstack([ zeros(ceil(padding_length)),
                                   spike_rate * hanning(window_length),
                                   zeros(floor(padding_length))])
    amplitude_spectrum = abs(fft(rate_windowed_and_padded)) / (window_length / 2.)
    freq_vec = fftfreq(fft_length,0.001)
    plot(freq_vec[0:fft_length/2], amplitude_spectrum[0:fft_length/2])
   




print('Simulating networks')

gamma_network = inp.input_network(state = 'gamma')
gamma_network.simulate(sim_duration = 500, stimulus = 0.33, to_plot = False)

beta_network = inp.input_network(state = 'beta')
beta_network.simulate(sim_duration = 500, stimulus = - 0.33, to_plot = False)

asyn_network = inp.input_network(state = 'asynchronous')
asyn_network.simulate(sim_duration = 500, stimulus = 0, to_plot = False)

print('Running analysis')

time_window = [300.*ms, 500.*ms]  
n_neurons = 8000

gamma_net_spikes = gamma_network.E_cell_spike_monitor.spikes
gamma_net_35Hz_amp, gamma_net_rates = amplitude_pattern_at_freq(gamma_net_spikes, time_window, 35)
gamma_net_12Hz_amp = amplitude_pattern_at_freq(gamma_net_spikes, time_window, 12)[0]

beta_net_spikes = beta_network.E_cell_spike_monitor.spikes
beta_net_35Hz_amp, beta_net_rates = amplitude_pattern_at_freq(beta_net_spikes, time_window, 35)
beta_net_12Hz_amp = amplitude_pattern_at_freq(beta_net_spikes, time_window, 12)[0]

asyn_net_spikes = asyn_network.E_cell_spike_monitor.spikes
asyn_net_35Hz_amp, asyn_net_rates = amplitude_pattern_at_freq(asyn_net_spikes, time_window, 35)
asyn_net_12Hz_amp = amplitude_pattern_at_freq(asyn_net_spikes, time_window, 12)[0]

combined_spikes = gamma_net_spikes + beta_net_spikes + asyn_net_spikes
combined_35Hz_amp, combined_rates = amplitude_pattern_at_freq(combined_spikes, time_window, 35)
combined_12Hz_amp = amplitude_pattern_at_freq(combined_spikes, time_window, 12)[0]

print('Plotting')

figure(1)
clf()

subplot2grid((5,4), (0,0), rowspan = 2)
plot_vert_raster( gamma_net_spikes, time_window, n_neurons )
ylabel('Time\n(mS)', ha = 'center', labelpad = 30)

subplot2grid((5,4), (0,1), rowspan = 2)
plot_vert_raster( beta_net_spikes, time_window, n_neurons )

subplot2grid((5,4), (0,2), rowspan = 2)
plot_vert_raster( asyn_net_spikes, time_window, n_neurons )

subplot2grid((5,4), (0,3), rowspan = 2)
plot_vert_raster( combined_spikes, time_window, n_neurons )

subplot(5,4,9)
plot_amp_pattern(gamma_net_rates, 8000)
ylabel('Firing\nrate (Hz)', ha = 'center', labelpad = 30)

subplot(5,4,10)
plot_amp_pattern(beta_net_rates, 8000)

subplot(5,4,11)
plot_amp_pattern(asyn_net_rates, 8000)

subplot(5,4,12)
plot_amp_pattern(combined_rates, 10000, col = 'k')

subplot(5,4,13)
plot_amp_pattern(gamma_net_12Hz_amp, 5000)
ylabel('Amplitude\nat 12 Hz', ha = 'center', labelpad = 30)

subplot(5,4,14)
plot_amp_pattern(beta_net_12Hz_amp, 5000)

subplot(5,4,15)
plot_amp_pattern(asyn_net_12Hz_amp, 5000)

subplot(5,4,16)
plot_amp_pattern(combined_12Hz_amp, 5000, col = 'k')

subplot(5,4,17)
plot_amp_pattern(gamma_net_35Hz_amp, 3000)
ylabel('Amplitude\nat 35 Hz', ha = 'center', labelpad = 30)

subplot(5,4,18)
plot_amp_pattern(beta_net_35Hz_amp, 3000)

subplot(5,4,19)
plot_amp_pattern(asyn_net_35Hz_amp, 3000)

subplot(5,4,20)
plot_amp_pattern(combined_35Hz_amp, 3000, col = 'k')

show()

figure(2)
clf()
plot_amp_spec(gamma_net_spikes, time_window)
plot_amp_spec(beta_net_spikes, time_window)
plot_amp_spec(asyn_net_spikes, time_window)
ylabel('Amplitude')
xlabel('Frequency (Hz)')
plot([12,12],[0,50000],'k')
plot([35,35],[0,50000],'k')
xlim((0,80))
ylim((0,50000))
show()