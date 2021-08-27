#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
from Distributions import TA_POR_Distribution

###############################################################################
### Load or calculate data
###############################################################################

px = np.arange(0,1.0,0.001)         # por-values for plotting
por_ex= [0.17,0.31]                 # selected por-values for plotting in Fig3

TA_con=TA_POR_Distribution(dp = 0.02)
TA_con.read_data(file_data='../data/FCC_2-1_por_ta_data_d2_r2.csv',compress2con = True)

ta_min=min(0.001,np.min(TA_con.ta))
ta_max=max(1,np.max(TA_con.ta))

###############################################################################
### Plottings
###############################################################################

lw = 2
textsize = 8
lso = ['--',(0, (5, 5))]
plt.close('all')

plt.figure(figsize=[3.75,2.5])
plt.scatter(TA_con.por,TA_con.ta,s=5,edgecolors ='k',linewidths=0.15,c='gold',zorder =2)

for ii,pi in enumerate(por_ex):
    plt.plot([pi,pi],[ta_min,ta_max],ls = lso[ii],color='0.5',lw=2,zorder=1,label=r'$\theta = {:.2f}$'.format(pi))
plt.plot(px,px,c='0.5',ls=':',label=r'$ta_{{con}}^{max} = \theta$')

plt.axis([0.05,0.9,ta_min,ta_max])
plt.yscale('log')
plt.grid(True)
plt.legend(loc = 'lower right',fontsize=textsize)
plt.xlabel(r'Porosity $\theta$',fontsize=textsize) 
plt.ylabel(r'Transport ability $ta_{{con}}(\theta)$',fontsize=textsize)
plt.tick_params(axis="both",which="major",labelsize=textsize)

plt.tight_layout()
# plt.savefig('../results/Fig02_Scatter_TA_Data.png',dpi=300)   
plt.savefig('../results/Fig02_Scatter_TA_Data.pdf')   

