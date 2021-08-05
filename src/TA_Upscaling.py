import numpy as np
import os
import copy
import scipy.stats as ss

from Distributions import Porosity_Distribution, TA_POR_Distribution, TA_Distribution

DEF_settings = dict(
     res0 = 2,
     pstd_r0 = 0.15,
     por_con_min = 0.07,
     nmin = 20,
     dp = 0.02,
     )
###############################################################################
###############################################################################
###############################################################################
    
class TA_POR_Upscaling():
    
    """
    Class to analyse ensemble of networks consisting of n_network^dim nodes
    with random porosity values from normal distribution, subsequent 
    connectivity and transport ability generated from porosity values
    and calculation of network porosity and transport ability for each network    
    """
    
    def __init__(self,
                 dim = 2,
                 res = 2,
                 n_network = 4,   
                 pmean = 0.3, 
                 pstd = 0.15,
                 scale_std = True,
                 **settings
                 ):

        self.dim = dim
        self.res = res
        self.pmean = pmean
        self.pstd = pstd
        
        self.n_network = n_network
        
        self.settings=copy.copy(DEF_settings) 
        self.settings.update(settings)   

        self.por = None
        self.ta = None
        
        self.ta_gmean = None        
        self.scale  = self.res/self.settings['res0']*self.n_network
        
        self.set_por_dist_theory(scale_std = scale_std)

    ###########################################################################

    def read_data(self,
                  file_data='..data/ta_eff.csv',
                  por_con_min = 0,
                  delimiter = ',',
                  ):

        """ read in ta-data as function of porosity """
        
        if not os.path.isfile(file_data):
            raise ValueError("File for ta-data not accessible: \n",file_data)
        # print(file_data)
        
        ta_data=np.loadtxt(file_data,delimiter=delimiter)

        if por_con_min > 0:
            """ compress to data above the connectivity level for porosity """
            condition = ta_data[:,0]>=por_con_min      
            ta_data = np.compress(condition,ta_data,axis=0)

        self.por = np.array(ta_data[:,0],ndmin=1)
        self.ta = np.array(ta_data[:,1],ndmin=1)

        self.connected = self.ta>0
        self.disconnected = (self.ta==0)
        ta_con_data = np.compress(self.connected,ta_data,axis=0)
        
        self.por_con = np.array(ta_con_data[:,0],ndmin=1)
        self.ta_con = np.array(ta_con_data[:,1],ndmin=1)     

        return self.por, self.ta

    ###########################################################################

    def ta_stats(self):

        """ ensemble level (from data):
            statistical analysis of TA-values integrated over all porosity values
            (marginal distribution of ta-por-scatter) 
            --> log-ta mean and variance
        """
        if self.ta is None:
            raise ValueError("read data first")
        
        ta_con_log = np.log(self.ta_con)

        self.ta_gmean = np.exp(np.mean(ta_con_log))
        self.ta_log_std = np.std(ta_con_log)       
        self.ta_pcon = np.mean(self.connected)

        return self.ta_gmean, self.ta_log_std,self.ta_pcon

    def ta_hist(self,
                tay,
                compress2connect = True,
                ):

        """ ensemble level (from data):
            histogram of TA-values integrated over all porosity values
            (marginal distribution of ta-por-scatter) 
        """

        if compress2connect:
            values = self.ta_con
        else:
            values = self.ta

        self.ta_hist, bin_edges = np.histogram(values,tay,density = 1)
        self.ta_hist_range = tay[:-1] + 0.5*np.diff(tay) 
        
        return self.ta_hist,self.ta_hist_range 

    ###########################################################################

    def set_por_dist_theory(self,
                            pmean = False,
                            pstd = False,
                            scale_std = True,
                            ):
        
        if pmean:
            self.pmean = pmean
        if pstd:
            self.pstd = pstd

        if scale_std:
            self.pstd = self.settings['pstd_r0']/np.power(self.scale,0.25*self.dim)      
  
        self.POR_theory = Porosity_Distribution(
            pmean = self.pmean,
            pstd = self.pstd,
            )
        
        return self.POR_theory.pmean, self.POR_theory.pstd

    def set_por_dist_data(self,
                          px = False,
                          dp = 0.02,
                          ):

        """ ensemble level (from data):
            statistical analysis of por-values 
            (marginal distribution of ta-por-scatter) 
            --> mean and variance of generated nodes
        """

        if px is False:
            px =np.arange(0.5*dp,1.+0.5*dp,dp)

        """ mean and variance of porosity values """
        self.por_hist, bin_edges = np.histogram(self.por,px,density = 1)
        self.por_hist_range = px[:-1] + 0.5*np.diff(px) 

        self.pmean_ens=np.mean(self.por)
        self.pstd_ens=np.std(self.por)
        
        self.POR_data = Porosity_Distribution(
            pmean = self.pmean_ens,
            pstd = self.pstd_ens)      
        
        return self.pmean_ens, self.pstd_ens 

    ###########################################################################

    def set_ta_dist_theory(self,
                           ta_gmean0 = 0.022,
                           ta_log_std0 = 2, 
                           ta_log_min = 0.05,
                           ):

        ta_log_std  = np.max(ta_log_min,ta_log_std0/np.power(self.scale,0.25*self.dim))
        ta_pcon  = self.POR_theory.total_connectivity()
     
        self.TA_theory=TA_Distribution(
            ta_gmean = ta_gmean0,
            ta_log_std = ta_log_std,
            ta_pcon = ta_pcon,
            **self.settings)                  

        return ta_gmean0, ta_log_std, ta_pcon

    def set_ta_dist_data(self):

        self.ta_stats()
        
        self.TA_data=TA_Distribution(
            ta_gmean = self.ta_gmean,
            ta_log_std = self.ta_log_std,
            ta_pcon = self.ta_pcon,
            **self.settings,
            )                  
        
    ###########################################################################

    def ta_upscaling(self,
                     tay,
                     px=False,
                     factor = False,
                     **kwargs,
                     ):

        """ ensemble level from por values (data+theory):
            pdf cloud ta vs. por for theoretical upscaling
        """
        self.settings.update(kwargs)   

        self.TAP_upscaling = TA_POR_Distribution(**self.settings)
        self.TAP_upscaling.set_data(self.ta,self.por, compress2con = False, **self.settings)
        self.TAP_upscaling.statistics(**self.settings)

        if px is False:
            px = self.POR_theory.por_range

        por_pdf = self.POR_theory.pdf_porosity_connected(px)   ### distribution of connected porosity values        
        pdf_ta_con = self.TAP_upscaling.pdf_ta_con(tay,px)
          
        por_extend = np.tile(por_pdf,(len(tay),1)).T
        self.ta_por_cloud = pdf_ta_con*por_extend

        if factor is not False:
            pdiscon = self.POR_theory.pdf_porosity_disconnected(px)
            self.ta_por_cloud[:,0] = factor*pdiscon

        self.ta_pdf_upscale = np.trapz(self.ta_por_cloud,x=px,axis=0)
       
        return self.ta_pdf_upscale

        
    def ta_upscaling_2LN(self,
                         tay,
                         px = False,
                         **kwargs,
                         ):
        
        self.ta_upscaling(tay,px,**kwargs)

        y0_log = np.log10(tay)
        mn0 = np.trapz(self.ta_pdf_upscale,x=y0_log) # zero moment
        mn1 = np.trapz(self.ta_pdf_upscale*y0_log,x=y0_log)/mn0 # first moment
        mn2 = np.trapz(self.ta_pdf_upscale*y0_log**2,x=y0_log)/mn0 - mn1**2 #second moment

        ta_gmean_upscale = 10**mn1
        ta_log_std_upscale = np.sqrt(mn2)
        
        self.ta_LN_upscale=ss.lognorm.pdf(tay,s=ta_log_std_upscale,scale=ta_gmean_upscale)

        return ta_gmean_upscale,ta_log_std_upscale


    def ta_upscale_moments(self,
                           tay,
                           px=False,
                           **kwargs,
                           ):

        self.ta_upscaling(tay,px,**kwargs)
               
        self.POR_theory.connectivity_func(px=px) ### --> self.pcon
        self.TAP_upscaling.interpolate_ta_log_mean(px=px) ### --> self.ta_log_mean_por (of log normal distrib for connected ta)
        self.TAP_upscaling.interpolate_ta_log_std(px=px) ### --> self.ta_log_std_por

        ### expectation value for adapted probability distrubtion including non-connected values
        self.ta_upscale_1moment = self.POR_theory.pcon*np.exp(self.ta_log_mean_por + 0.5*self.ta_log_std_por**2)
        
        return self.ta_pdf_1moment
    