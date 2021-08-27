#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

###############################################################################
### Load or calculate data
###############################################################################

data_2D = np.loadtxt('../data/FCC_2-1_ens_results_d2.csv',delimiter=',')
data_3D = np.loadtxt('../data/FCC_2-1_ens_results_d3.csv',delimiter=',')

pmean_r2 = 0.15
pstd_r2 = 0.3
ta_std_r2 = 1.96                                        # standard deviation
por_level,ta_level = 0.025,0.05

###############################################################################
### Plottings
###############################################################################
lw = 2
textsize = 8
plt.close('all')
plt.figure(figsize=[7.5,3])  
ax1=plt.subplot(121)  
ax2=plt.subplot(122)

for ii, data in enumerate([data_2D,data_3D]):

    dim = ii+2
    nn = data[:,1]
    nd = (0.5*nn)**dim

    ### coefficients of variation
    cv_por = data[:,8]/data[:,6]                        #   std/mean of por-ensemble values
    cv_log_ta = np.abs(data[:,10]/np.log(data[:,9]) )   #   log-ta std/ log-ta mean
    gcv_ta = np.sqrt(np.exp(data[:,10]**2)-1)             # coefficient of variation adapted to log-normal distributed data

    cv_por_theory = pmean_r2/pstd_r2/np.power(nn/2.,0.25*dim)  
    xl,yl = np.log(nn[:5]),np.log(gcv_ta[:5])
    a,b = np.polyfit(xl, yl, 1)
    c = np.exp(b)
    cv_log_ta_theory = ta_std_r2/np.power(nn/2,0.25*dim)

    ax1.plot(nn,cv_por,'d',c='C{}'.format(ii),zorder = 10,label = r'$\sigma_n/\mu\ (\theta)$ - {}D'.format(dim))    ### log-ta variance
    ax1.plot(nn,gcv_ta,'o',c='C{}'.format(ii),zorder = 10,label = r'$GCV\ (ta)$ - {}D'.format(dim))     ### log-ta variance   
    ax1.plot(nn,cv_por_theory,'k-',zorder = 2)     
    ax1.plot(nn,por_level*np.ones_like(nn),c='0.6',ls='--',zorder = 0)  
    ax1.plot(nn,cv_log_ta_theory,'k:',zorder = 3)   
    ax1.plot(nn,ta_level*np.ones_like(nn),c='0.6',ls='--',zorder = 0)  
        
    ax2.plot(nd,cv_por,'d',c='C{}'.format(ii),zorder = 10,label = r'$\sigma_n/\mu\ (\theta)$ - {}D'.format(dim))    ### log-ta variance
    ax2.plot(nd,gcv_ta,'o',c='C{}'.format(ii),zorder = 10,label = r'$GCV\ (ta)$ - {}D'.format(dim))     ### log-ta variance   
    ax2.plot(nd,cv_por_theory,'k-',zorder = 2)    
    ax2.plot(nd,por_level*np.ones_like(nn),c='0.6',ls='--',zorder = 0)  
    ax2.plot(nd,cv_log_ta_theory,'k:',zorder = 3) 
    ax2.plot(nd,ta_level*np.ones_like(nn),c='0.6',ls='--',zorder = 0)  
        
    print("Assymptotic value GCV(ta) {:.0f}D: {:.4f}".format(dim, gcv_ta[-1]))
    print("Assymptotic value CV(por) {:.0f}D: {:.4f}".format(dim, cv_log_ta[-1]))

labels = ['{:.0f}'.format(elem) for elem in nn]
ax1.set_xticks(nn[:-1],labels[:-1])
ax1.set_xlim([0.8*nn[0],1.2*nn[-1]])

ax1.set_yscale('log')
ax1.set_xscale('log')
ax1.set_ylabel(r'Coefficient of Variation',fontsize=textsize)
ax1.set_xlabel(r'domain length $r$ [$\mu$m]',fontsize=textsize) 

ax1.annotate(r'$\alpha = {}$'.format(por_level),(1.2*nn[0],0.4*por_level),fontsize=textsize, bbox=dict(facecolor='w',ec = '0.6', alpha=0.5,boxstyle='round'))
ax1.annotate(r'$\alpha = {}$'.format(ta_level),(3*nn[-3],1.7*ta_level),fontsize=textsize, bbox=dict(facecolor='w',ec = '0.6', alpha=0.5,boxstyle='round'))
ax1.tick_params(axis="both",which="major",labelsize=textsize)
ax1.grid(True)        
ax1.legend(loc = 'lower left',fontsize=textsize,ncol=2)
ax1.text(-0.13,-0.13,'(a)', bbox=dict(facecolor='w', alpha=1,boxstyle='round'),fontsize=textsize, transform=ax1.transAxes)

ax2.set_yscale('log')
ax2.set_xscale('log')
ax2.set_xlim([0.5*nd[0],2*nd[-1]])
ax2.set_xlabel(r'Number of network elements $N_r = (r/2)^d$',fontsize=textsize) 
ax2.set_ylabel(r'Coefficient of Variation',fontsize=textsize)
ax2.annotate(r'$\alpha = {}$'.format(por_level),(1.2*nd[0],0.4*por_level),fontsize=textsize, bbox=dict(facecolor='w',ec = '0.6', alpha=0.5,boxstyle='round'))
ax2.annotate(r'$\alpha = {}$'.format(ta_level),(0.5*nd[-2],1.7*ta_level),fontsize=textsize, bbox=dict(facecolor='w',ec = '0.6', alpha=0.5,boxstyle='round'))
ax2.tick_params(axis="both",which="major",labelsize=textsize)
ax2.grid(True)        
ax2.legend(loc = 'lower left',fontsize=textsize,ncol=2)
ax2.text(-0.13,-0.13,'(b)', bbox=dict(facecolor='w', alpha=1,boxstyle='round'),fontsize=textsize, transform=ax2.transAxes)

plt.tight_layout()
# plt.savefig('../results/Fig07_ens_stats_evolution.png',dpi=300)   
plt.savefig('../results/Fig07_ens_stats_evolution.pdf')   

