## Oscillations and Filtering Networks Brian Simulations

This repository contains code for simulating some of the networks used in the manuscript:

**Thomas Akam and Dimitri Kullmann. ‘Oscillations and Filtering Networks Support Flexible Routing of Information’. *Neuron* 67, no. 2 (2010): 308–320.**

Although many of the simulations in the original paper used the [Nest](https://www.nest-simulator.org/) simulator, all the simulations in this repository use the [Brian](http://briansimulator.org/) simulator, so some are re-implementations of models used in the manuscript.

The file *input_network.py* is a Brian implementation of the input network that can be set to an asynchronous, gamma or beta oscillation state.

The file *filter_network.py* implements a version of the filtering network, but lacking spatial organization, such that it performs filtering on homogeneous activity rather than an input with a spatial population code.  This is the network used in figure S1.B2.

The file *multiplexing_figure.py* reproduces figure 6, which demonstrates  frequency division multiplexing by oscillatory modulation.

The file *filter_network_figure.py* reproduces figure S1.B2, showing the firing rate of E - cells in a filter network in response to an input consisting of an asynchronous component and oscillating component (40Hz sinusoidal modulation), as the firing rate  of the two components is varied.

### Dependencies:

- **Python 2**
- **Brian simulator** - I am not certain what version of Brian these files were created for, but they were made in 2012 so the version was <= 1.4.  

Please let me know if you update these files to Python 3 + Brian 2 as I would like to host updated versions here.

