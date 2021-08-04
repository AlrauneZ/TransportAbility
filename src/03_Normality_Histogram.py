#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss

from Distributions import TA_POR_Distribution

###############################################################################
### Load or calculate data
###############################################################################

por_ex= [0.17,0.31]                 # selected por-values for plotting in Fig3

TA_con=TA_POR_Distribution(dp = 0.02)
TA_con.read_data(file_data='../data/FCC_2-1_por_ta_data_d2_r2.csv')
TA_con.statistics(compress = False) 

# TA_con.normality_tests()
# TA_con.write_stats(file_stats='../data/FCC_2-1_stats_ta_d2_r2.csv')

###############################################################################
### Plottings
###############################################################################
lw  = 3
textsize = 8
cc = ['goldenrod','darkgoldenrod']
plt.close('all')

fig=plt.figure(figsize=[7.5,4])

for ii,pi in enumerate(por_ex):
    ip = np.argmin(np.abs(TA_con.por_range - pi))

    data=TA_con.por_bin_data[TA_con.por_range[ip]]
    nn=TA_con.stats['number_in_bin'][ip]

    data_log=np.log(data)
    nbins=int(max(10,nn//10))

    data_range=np.linspace(0,max(data),10*nbins)
    data_log_range=np.linspace(min(data_log),max(data_log),50)

    Npdf=ss.norm.pdf(data_range,loc=TA_con.stats['mean'][ip],scale=TA_con.stats['std'][ip])
    LNpdf=ss.lognorm.pdf(data_range,s=TA_con.stats['log-std'][ip],scale=np.exp(TA_con.stats['log-mean'][ip]))
    Npdf_log=ss.norm.pdf(data_log_range,loc=TA_con.stats['log-mean'][ip],scale=TA_con.stats['log-std'][ip])

    ax=fig.add_subplot(2,2,ii+1)
    ax.hist(data,bins=nbins,color=cc[ii],edgecolor='k',density=1,label='data')
    ax.plot(data_range,Npdf,c='royalblue', lw=lw, label='Normal')
    ax.plot(data_range,LNpdf,c='forestgreen', lw=lw, label='Log-Normal')

    ax.set_xlabel('Transport ability',fontsize=textsize)
    ax.set_ylabel('Normalized Frequency/Density',fontsize=textsize)
    ax.legend(loc='best',fontsize=textsize)

    ax.set_title(r'${:.2f}\leq \theta \leq {:.2f}$     ($n={:.0f}$)'.format(pi-0.5*TA_con.dp,pi+0.5*TA_con.dp,nn),fontsize=textsize+1)
    ax.set_xlim([0.,data_range[-1]])
    ax.tick_params(axis="both",which="major",labelsize=textsize)

    ax=fig.add_subplot(2,2,ii+3)
    ax.hist(data_log,bins=nbins,color=cc[ii],edgecolor='k',density=1,label='data')
    ax.plot(data_log_range,Npdf_log,c='forestgreen', lw=lw, label='Log-Normal pdf')

    ax.set_xlabel('Log-transport ability',fontsize=textsize)
    ax.set_ylabel('Normalized Frequency/Density',fontsize=textsize)
    ax.set_xlim([data_log_range[0],data_log_range[-1]])
    ax.tick_params(axis="both",which="major",labelsize=textsize)


plt.tight_layout()
plt.savefig('../results/Fig03_Normality_Histogram',dpi=300)   
plt.savefig('../results/Fig03_Normality_Histogram.pdf')   

