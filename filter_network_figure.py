'''
Reproduces figure S1.B2, showing the firing rate of E - cells in a filter network in response to an 
input consisting of an asynchronous component and oscillating component (40Hz sinusoidal modulation),
as the firing rate  of the two components is varied.  In the paper a colormap was used to show the 
outpu firing rates but a surface plot of the same data is additionally plotted here.

Thomas Akam 2012
'''

import filter_network as fn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

DCrates=np.arange(0,26,5)  #  Rates of asychronous input. 
ACrates=np.arange(0,26,5)  #  Rates of sinusoidally moudlated input.
    
E_cell_rates=np.zeros([np.size(DCrates),np.size(ACrates)])
    
filt_net = fn.filter_network()    

for dcp, DCrate in enumerate(DCrates):
    for acp, ACrate in enumerate(ACrates):
        print 'DCrate: ',DCrate
        print 'ACrate: ',ACrate
        E_cell_rates[dcp,acp]=filt_net.simulate(DCrate,ACrate, to_plot = False)




fig = plt.figure()
fig.set_facecolor('white')
ax1 = fig.add_subplot(1,2,1)
ax1.imshow(E_cell_rates, cmap=cm.coolwarm)
ax1.set_xticks(np.arange(np.size(ACrates)))
ax1.set_xticklabels(ACrates)
ax1.set_xlabel('Oscillating input rate (Hz)')
ax1.set_ylabel('Asynchronous input rate (Hz)')


ax2 = fig.add_subplot(1,2,2, projection='3d')
DCR, ACR = np.meshgrid(DCrates, ACrates)
surf = ax2.plot_surface(ACR, DCR, E_cell_rates, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
fig.colorbar(surf, shrink=0.5, aspect=5)
ax2.set_xlabel('Asynchronous input rate (Hz)')
ax2.set_ylabel('Oscillating input rate (Hz)')
ax2.set_zlabel('Output rate (Hz)')
plt.show()