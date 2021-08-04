#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np

from Distributions import Porosity_Distribution

###############################################################################
### Load or calculate data
###############################################################################

FCC_2_1=dict(
     r2 = dict(pmean=0.296,pstd=0.145),
     r8  = dict(pmean=0.29,pstd=0.09),
     r32 = dict(pmean=0.28,pstd=0.044),
    )

M1 = Porosity_Distribution(**FCC_2_1['r2'])
M2 = Porosity_Distribution(**FCC_2_1['r8'])
M3 = Porosity_Distribution(**FCC_2_1['r32'])
px = np.arange(0,1.0,0.001)         # por-values for plotting

M1.pdf_stats()

print("Mean of truncated normal distribution (n=2): {:.3f}".format(M1.pmean_trunc))
print("Mean of parent normal distribution (n=2): {:.3f}".format(M1.pmean))
print("Relative difference in mean: {:.1f} % \n".format(100*abs(M1.pmean_trunc-M1.pmean)/M1.pmean_trunc))

print("Std of truncated normal distribution (n=2): {:.4f}".format(M1.pstd_trunc))
print("Std of parent normal distribution (n=2): {:.4f}".format(M1.pstd))
print("Relative difference in std: {:.1f} % \n".format(100*abs(M1.pstd_trunc-M1.pstd)/M1.pstd_trunc))


###############################################################################
### Plottings
###############################################################################
lw = 2
textsize = 8
plt.close('all')

plt.figure(figsize=[3.75,2.5])
ax=plt.subplot(111)
    
ax.plot(px,M1.connectivity_func(px),ls='-',c='k',lw=lw+1,label=r'$p_\mathrm{con}(\theta)$',zorder=5)

ax.plot(px,M1.pdf_porosity(px),ls='-',c='C0',lw=lw,label=r'$2 \mu m$',zorder=4)
ax.plot(px,M1.pdf_porosity_connected(px),ls='--',c='C0',lw=lw)
ax.fill_between(px,M1.pdf_porosity(px),M1.pdf_porosity_connected(px),color='C0',zorder=1,alpha = 0.5)

ax.plot(px,M2.pdf_porosity(px),ls='-',c='C2',lw=lw,label=r'$8 \mu m$',zorder=3)
ax.plot(px,M2.pdf_porosity_connected(px),ls='--',c='C2',zorder=1,lw=lw)
ax.fill_between(px,M2.pdf_porosity(px),M2.pdf_porosity_connected(px),color='C2',zorder=1,alpha = 0.5)

ax.plot(px,M3.pdf_porosity(px),ls='-',c='C9',lw=lw,label=r'$32 \mu m$',zorder=2)
ax.plot(px,M3.pdf_porosity_connected(px),ls='--',c='C9',zorder=1,lw=lw)
ax.fill_between(px,M3.pdf_porosity(px),M3.pdf_porosity_connected(px),color='C9',zorder=1,alpha = 0.5)

ax.set_xlabel(r'Porosity $\theta$',fontsize=textsize)
ax.set_ylabel('Probability',fontsize=textsize)
ax.legend(loc='upper right',fontsize=textsize+2)
ax.tick_params(axis="both",which="major",labelsize=textsize)
ax.grid(True)
ax.set_xlim([0,0.6])  
ax.set_ylim([0,7])  

plt.tight_layout()
plt.savefig('../results/Fig01_pdf_porosity.png',dpi=300)   
plt.savefig('../results/Fig01_pdf_porosity.pdf')   



print("Total dis-connectivity for n=2:  {:.3f}".format(1-M1.connectivity_total))
print("Total dis-connectivity for n=8:  {:.3f}".format(1-M2.connectivity_total))
print("Total dis-connectivity for n=32: {:.3f}".format(1-M3.connectivity_total))

