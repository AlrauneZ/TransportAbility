#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
from Distributions import TA_POR_Distribution

###############################################################################
### Load or calculate data
###############################################################################

TA_con=TA_POR_Distribution(dp = 0.02)
TA_con.read_data(file_data='../data/FCC_2-1_por_ta_data_d2_r2.csv')
TA_con.statistics(compress = True) 

data = np.vstack((TA_con.por_compress,TA_con.stats['log-mean'],TA_con.stats['log-std']))
np.savetxt('../data/FCC_2-1_ta_por_ab.csv',data.T,fmt = '%.3f', delimiter = ' & ')

###############################################################################
### Plottings
###############################################################################
lw = 3
textsize = 8
plt.close('all')
  
fig = plt.figure(figsize = [7.5,2.5])
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)

ax1.plot(TA_con.por_compress,TA_con.stats['log-mean'],ls=':',c='C0',lw=1,marker='o')
ax2.plot(TA_con.por_compress,TA_con.stats['log-std'],ls=':',c='C1',lw=1,marker='s')

ax1.grid(True)
ax1.set_ylabel(r'$a$ - mean of $\log-\chi$',fontsize=textsize) 
ax1.set_xlabel(r'Porosity $\theta$',fontsize=textsize) 
ax1.tick_params(axis="both",which="major",labelsize=textsize)

ax2.set_xlabel(r'Porosity $\theta$',fontsize=textsize)
ax2.set_ylabel(r'$b$ - std of $\log-\chi',fontsize=textsize)
ax2.tick_params(axis="both",which="major",labelsize=textsize)
ax2.grid(True)

plt.tight_layout()
# plt.savefig('../results/Fig04_stats_TA.png',dpi=300)   
plt.savefig('../results/Fig04_stats_TA.pdf')   

