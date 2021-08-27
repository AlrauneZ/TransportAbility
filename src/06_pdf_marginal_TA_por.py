#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
from Distributions import TA_POR_Distribution
from TA_Upscaling import TA_POR_Upscaling

###############################################################################
### Load or calculate data
###############################################################################
TA_con=TA_POR_Distribution(dp = 0.02)
TA_con.read_data(file_data='../data/FCC_2-1_por_ta_data_d2_r2.csv')
TA_con.statistics(compress = True) 

y0=np.logspace(-4,-0.4,200)         # ta values for plotting
px = np.arange(0,1.0,0.001)         # por-values for plotting

resolutions = [2,2,8,32,128]    
n_networks = [4,16,16,16,16]
file_data = '../data/FCC_2-1_d2_r{}-{}_N10000_ta_eff.csv'
ta_gmean= []

###############################################################################
### Plottings
###############################################################################
lw = 2
textsize = 8
plt.close('all')

plt.figure(figsize=[7.5,2.5])
ax1=plt.subplot(121)  
ax2=plt.subplot(122)

for ii,res in enumerate(resolutions):
    label = 'r = {}'.format(resolutions[ii]* n_networks[ii])        

    ENS_TA1 = TA_POR_Upscaling(
        dim = 2,
        res = res,
        n_network = n_networks[ii],
        )
    
    ENS_TA1.read_data(file_data = '../data/FCC_2-1_d2_r{}-{}_N10000_ta_eff.csv'.format(res,res*n_networks[ii]))
    ENS_TA1.ta_stats()

    print("\nEnsemble {}\n################".format(label))
    ### -------------------------------------------------------------------------------------------------------
    ### Statistics of ensemble porosity -> match to input statistics?

    ENS_TA1.set_por_dist_theory()
    ax1.plot(px,ENS_TA1.POR_theory.pdf_porosity(px),ls='-',c='C{}'.format(ii),alpha = 0.5,lw=lw,zorder = 6-ii) 
    print('Porosity stats (Theory): \n pm = {:.2f}, psdt = {:.5f}'.format(ENS_TA1.POR_theory.pmean,ENS_TA1.POR_theory.pstd))
    
    ### statistics of ensemble porosity
    ENS_TA1.set_por_dist_data() 
    ax1.plot(px,ENS_TA1.POR_data.pdf_porosity(px),ls='-',c='C{}'.format(ii),lw=lw,zorder = 12-ii,label=label) 
    print('Porosity stats (Ens): \n pm = {:.2f}, psdt = {:.5f}'.format(ENS_TA1.POR_data.pmean,ENS_TA1.POR_data.pstd))

    ### -------------------------------------------------------------------------------------------------------

    ENS_TA1.set_ta_dist_data()
    ENS_TA1.ta_hist(y0)
    ### Statistics of ensemble ta-values -> log-normal distribution?               
    print("TA-stats (Ens) \n mean = {:.3f}, log-std = {:.3f}".format(ENS_TA1.TA_data.ta_gmean, ENS_TA1.TA_data.ta_log_std))
    print('Connectivity: (Ens)\n p_con (ta) = {:.3f} '.format(ENS_TA1.TA_data.ta_pcon))

    ax2.plot(ENS_TA1.ta_hist_range,ENS_TA1.ta_hist,color='C{}'.format(ii),lw = lw,zorder = 12-ii,label=label)      
    ax2.plot(y0,ENS_TA1.TA_data.pdf_ta_con(y0),color='C{}'.format(ii),lw = lw,ls = '-',zorder = 6-ii,alpha=0.5 )
    ### highlight/plot disconnected elements
    ax2.plot([1e-3,1e-3],[0,100*(1-ENS_TA1.TA_data.ta_pcon)],color='C{}'.format(ii),lw = 4,ls = '-',zorder = 6-ii)
    
ax1.set_xlim([0.15,.45])  
ax1.set_xlabel(r'Porosity $\bar \theta$',fontsize=textsize)
ax1.set_ylabel(r'Ensemble pdf  $P_{\bar \theta}$',fontsize=textsize)
ax1.legend(loc='upper left',fontsize=textsize)#,ncol=2)
ax1.tick_params(axis="both",which="major",labelsize=textsize)
ax1.grid(True)
ax1.text(-0.13,-0.13,'(b)', bbox=dict(facecolor='w', alpha=1,boxstyle='round'),fontsize=textsize, transform=ax1.transAxes)

ax2.set_xlim([0.0009,0.1])
ax2.set_xscale('log')
ax2.set_xlabel(r'Transport ability $\bar {ta}$',fontsize=textsize)
ax2.set_ylabel(r'Ensemble pdf  $P_{\bar {ta}}(\bar \theta)$',fontsize=textsize)
ax2.tick_params(axis="both",which="major",labelsize=textsize)
ax2.legend(loc = 'upper left',fontsize=textsize)
ax2.grid(True)
ax2.text(-0.13,-0.13,'(c)', bbox=dict(facecolor='w', alpha=1,boxstyle='round'),fontsize=textsize, transform=ax2.transAxes)

plt.tight_layout()
# plt.savefig('../results/Fig06_pdf_marginal_TA_por_2D.png',dpi=300)   
plt.savefig('../results/Fig06_pdf_marginal_TA_por_2D.pdf')   
