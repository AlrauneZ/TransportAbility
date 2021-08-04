#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
from Distributions import TA_POR_Distribution
from Class_TA_Upscaling import TA_POR_Upscaling

###############################################################################
### Load or calculate data
###############################################################################
TA_con=TA_POR_Distribution(dp = 0.02)
TA_con.read_data(file_data='../data/FCC_2-1_por_ta_data_d2_r2.csv')
TA_con.statistics(compress = True) 


resolutions = [2,2,8,32,128] #512    
n_networks = [4,16,16,16,16,16]
# resolutions = [2,2,8,32,128] #512    
# n_networks = [4,16,16,16,16,16]
file_data = '../data/FCC_2-1_d2_r{}-{}_N10000_ta_eff.csv'#.format()
# specials = [0,0,1,1,1,1]
ta_gmean= []

###############################################################################
### Plottings
###############################################################################
lw = 2
textsize = 8
plt.close('all')

plt.figure(figsize=[3.75,2.9])#[7.5,4])   
ax = plt.subplot(111)
plt.scatter(TA_con.por,TA_con.ta,s=5,edgecolors ='k',linewidths=0.15,c='gold',zorder =2,label='r = 2 (Data)')

 
for ii,res in enumerate(resolutions):
    label = 'r = {}'.format(resolutions[ii]* n_networks[ii])        
    # label = 'E {}-{}'.format(resolutions[ii],resolutions[ii]* n_networks[ii])        

    ENS_TA1 = TA_POR_Upscaling(
        dim = 2,
        res = res,
        n_network = n_networks[ii],
        )
    
    ENS_TA1.read_data(file_data = '../data/FCC_2-1_d2_r{}-{}_N10000_ta_eff.csv'.format(res,res*n_networks[ii]))
    ENS_TA1.ta_stats()

    ta_gmean.append(ENS_TA1.ta_gmean)

    plt.scatter(ENS_TA1.por,ENS_TA1.ta,s=5,edgecolors ='k',linewidths=0.15,c='C{}'.format(ii),zorder =3+ii,label = label)   
    plt.scatter(ENS_TA1.por,0.001*ENS_TA1.disconnected,s=5,edgecolors ='k',linewidths=0.15,c='C{}'.format(ii),zorder =3+ii)#,label = label)   
        
ta_gmean_ens = ta_gmean[-1]
plt.plot([0,1],[ta_gmean_ens,ta_gmean_ens],c='0.5')    
plt.plot([0.3,.3],[0,1],c='0.5')    
plt.axis([0.065,.6,0.0009,0.35])

plt.yscale('log')
plt.grid(True)

plt.text(-0.1,-0.1,'(a)', bbox=dict(facecolor='w', alpha=1,boxstyle='round'),fontsize=textsize, transform=ax.transAxes)
plt.text(0.1,0.9,'2D', bbox=dict(facecolor='w', alpha=0.5,boxstyle='round'),fontsize=textsize, transform=ax.transAxes)
plt.legend(bbox_to_anchor=(0.65, 0.515),fontsize=textsize,framealpha=1)
plt.xlabel(r'Porosity $\bar\theta$',fontsize=textsize) # plt.xlabel('Connected Porosity')
plt.ylabel(r'Transport ability $\bar{ta}(\bar\theta)$',fontsize=textsize)
plt.tick_params(axis="both",which="major",labelsize=textsize)

plt.tight_layout()
plt.savefig('../results/Fig05_Scatter_TA_eff_2D.png',dpi=300)   
plt.savefig('../results/Fig05_Scatter_TA_eff_2D.pdf')   
