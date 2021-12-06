#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
from TA_Upscaling import TA_POR_Upscaling

###############################################################################
### Load or calculate data
###############################################################################

tay=np.logspace(-3,-0.4,100)         # ta values   
px = np.arange(0,0.5,0.001)      
xx,yy = np.meshgrid(px,tay)

pmean = 0.3
pstd = 0.15
means = [0.1,0.2,0.4]
stds = [pstd,pstd/4,pstd/16]

resolutions = [2,32]
n_network = 16

file_data = '../data/FCC_2-1_por_ta_data_d2_r2.csv'
file_ens = '../data/FCC_2-1_d2_r{}-{}_N10000_ta_eff.csv'#.format(resolutions,scale)

###############################################################################
### Plottings
###############################################################################
lw = 2
textsize = 8
plt.close('all')
plt.figure(figsize=[7.5,5])

print('\nVarying por statistics, constant ta-statistics:')
for ii, std in enumerate(stds):
    ax=plt.subplot(2,3,ii+1)

    TAP = TA_POR_Upscaling(
            dim = 2,
            res = 2,
            pmean = pmean, 
            pstd = std,
            scale_std = False,
            )

    TAP.read_data(file_data=file_data)  # read in observed data (r=2)
    TAP.set_ta_dist_data()              # create ta-distribution from observed data
    print('TA stats (Data): \n a = {:.4f}, b = {:.5f}'.format(TAP.TA_data.ta_gmean,TAP.TA_data.ta_log_std))

    TAP.ta_upscaling(tay,px,factor = 100)   
    ax.contourf(xx,yy,TAP.ta_por_cloud.T,cmap='Blues',levels = 15)

    if ii in [0,1]:
        ax.text(0.1,0.9,r'$\mu={}$'.format(TAP.POR_theory.pmean), bbox=dict(facecolor='w', alpha=0.5,boxstyle='round'),fontsize=textsize, transform=ax.transAxes)
    elif ii==2:    
        for im,mean in enumerate(means):
            TAP.POR_theory.pmean=mean
            print('Porosity stats (Theory): \n pm = {:.2f}, psdt = {:.5f}'.format(TAP.POR_theory.pmean,TAP.POR_theory.pstd))
            TAP.ta_upscaling(tay,px,factor = 100)       
            n0 = int(1000*mean)
            ax.contourf(xx[:,n0-50:n0+50],yy[:,n0-50:n0+50],TAP.ta_por_cloud[n0-50:n0+50,:].T,cmap='Blues',levels = 15)
        ax.text(0.1,0.9,r'$\mu=0.1; 0.2; 0.3; 0.4$', bbox=dict(facecolor='w', alpha=0.5,boxstyle='round'),fontsize=textsize, transform=ax.transAxes)

    if ii ==0:
        ax.set_ylabel('Transport-ability $\chi$',fontsize=textsize)
    ax.set_xlabel(r'Porosity $\theta$',fontsize=textsize)
    ax.set_yscale('log')
    ax.set_ylim([0.00098,0.2])
    ax.text(0.6,0.1,r'$r_{{\chi}} = {}\mu m$'.format(TAP.res), bbox=dict(facecolor='w', alpha=0.5,boxstyle='round'),fontsize=textsize, transform=ax.transAxes)
    ax.text(0.1,0.8,r'$\sigma={:.4f}$'.format(TAP.POR_theory.pstd), bbox=dict(facecolor='w', alpha=0.5,boxstyle='round'),fontsize=textsize, transform=ax.transAxes)
    ax.tick_params(axis="both",which="major",labelsize=textsize)

print('\nVarying por statistics, adapted ta-statistics:')

ax=plt.subplot(2,3,4)

TAP = TA_POR_Upscaling(
        dim = 2,
        res = 2,
        pmean = pmean, 
        pstd = pstd,
        scale_std = False,
        )

TAP.read_data(file_data=file_data)  # read in observed data (r=2)
TAP.set_ta_dist_data()              # create ta-distribution from observed data
print('TA stats (Data): \n a = {:.4f}, b = {:.5f}'.format(TAP.TA_data.ta_gmean,TAP.TA_data.ta_log_std))

TAP.ta_upscaling(tay,px,factor = 100)   
ax.contourf(xx,yy,TAP.ta_por_cloud.T,cmap='Blues',levels = 15)
ax.text(0.1,0.9,r'$\mu={}$'.format(TAP.POR_theory.pmean), bbox=dict(facecolor='w', alpha=0.5,boxstyle='round'),fontsize=textsize, transform=ax.transAxes)
ax.set_ylabel(r'Transport-ability $\chi$',fontsize=textsize)
ax.set_xlabel(r'Porosity $\theta$',fontsize=textsize)
ax.set_yscale('log')
ax.set_ylim([0.00098,0.2])
ax.text(0.6,0.1,r'$r_{{\chi}} = {}\mu m$'.format(TAP.res), bbox=dict(facecolor='w', alpha=0.5,boxstyle='round'),fontsize=textsize, transform=ax.transAxes)
ax.text(0.1,0.8,r'$\sigma={:.4f}$'.format(TAP.POR_theory.pstd), bbox=dict(facecolor='w', alpha=0.5,boxstyle='round'),fontsize=textsize, transform=ax.transAxes)
ax.tick_params(axis="both",which="major",labelsize=textsize)

for ii, res in enumerate(resolutions):
    ax=plt.subplot(2,3,5+ii)

    TAP = TA_POR_Upscaling(
            dim = 2,
            res = res,
            n_network= 16, 
            )

    TAP.read_data(file_data=file_ens.format(TAP.res,int(TAP.n_network*TAP.res)))
    print('Porosity stats (Theory): \n pm = {:.2f}, psdt = {:.5f}'.format(TAP.POR_theory.pmean,TAP.POR_theory.pstd))
    TAP.set_ta_dist_data()
    TAP.ta_upscaling(tay,px,factor = 100)
    ax.contourf(xx,yy,TAP.ta_por_cloud.T,cmap='Blues',levels = 15)

    if ii ==0:
        ax.text(0.1,0.9,r'$\mu={}$'.format(TAP.POR_theory.pmean), bbox=dict(facecolor='w', alpha=0.5,boxstyle='round'),fontsize=textsize, transform=ax.transAxes)
    elif ii==1:    
        for im,mean in enumerate(means):
            TAP.POR_theory.pmean=mean
            print('Porosity stats (Theory): \n pm = {:.2f}, psdt = {:.5f}'.format(TAP.POR_theory.pmean,TAP.POR_theory.pstd))
            TAP.ta_upscaling(tay,px,factor = 100)       
            n0 = int(1000*mean)
            ax.contourf(xx[:,n0-50:n0+50],yy[:,n0-50:n0+50],TAP.ta_por_cloud[n0-50:n0+50,:].T,cmap='Blues',levels = 15)
        ax.text(0.1,0.9,r'$\mu=0.1; 0.2; 0.3; 0.4$', bbox=dict(facecolor='w', alpha=0.5,boxstyle='round'),fontsize=textsize, transform=ax.transAxes)

    ax.set_xlabel(r'Porosity $\theta$',fontsize=textsize)
    ax.set_yscale('log')
    ax.set_ylim([0.00098,0.2])

    ax.text(0.6,0.1,r'$r_{{\chi}} = {}\mu m$'.format(TAP.res*TAP.n_network), bbox=dict(facecolor='w', alpha=0.5,boxstyle='round'),fontsize=textsize, transform=ax.transAxes)
    ax.text(0.1,0.8,r'$\sigma_{{{}}}={:.4f}$'.format(TAP.res*TAP.n_network,TAP.POR_theory.pstd), bbox=dict(facecolor='w', alpha=0.5,boxstyle='round'),fontsize=textsize, transform=ax.transAxes)
    ax.tick_params(axis="both",which="major",labelsize=textsize)

plt.tight_layout()
# plt.savefig('../results/Fig08_cloud_TA_pdf.png',dpi=300)   
plt.savefig('../results/Fig08_cloud_TA_pdf.pdf')   

