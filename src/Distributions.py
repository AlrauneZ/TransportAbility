#!/usr/bin/env python3

import numpy as np
import os.path
import scipy.stats as ss

###############################################################################
###############################################################################
###############################################################################

class Porosity_Distribution():
    
    """
    Class defining porosity properties and statistics of a chosen material type
    at a specified resolution
    - mean and std of porosity measurements --> Gaussian pdf
    - probility function for connectivity given porosity value    
    """

    def __init__(self, 
                 pmean=0.3, 
                 pstd=0.15,
                 pmin = 0,
                 pmax = 1,
                 **kwargs):

        self.pmean = pmean
        self.pstd = pstd
        self.pmin = pmin
        self.pmax = pmax

        self.check()
        
    def check(self):
        
        if self.pmean <= 0 or self.pmean>=1:
            raise ValueError(
                "mean porosities smaller 0 or larger 1 are not allowed"
            )
        if self.pmin < 0 or self.pmin > 1:
            raise ValueError(
                "Lower and/or upper margin values in range which is not allowed"
            )

        ###Convert min and max for normal distribution to min and max for standard normal distribution
        self._a = ( self.pmin - self.pmean) / self.pstd
        self._b = ( self.pmax - self.pmean) / self.pstd

    def update_stats(self,
                     pmean=False,
                     pstd=False
                     ):
        if pmean:
            self.pmean = pmean
        if pstd:
            self.pstd = pstd
        self.check()

    def pdf_porosity(self, 
                     x
                     ):
        """pdf for truncated normal distribution of porosity values """

        return ss.truncnorm.pdf(x , self._a, self._b,loc=self.pmean,scale = self.pstd)

    def rvs_porosity(self, 
                     size = 1
                     ):
        """random value samples from truncated normal distribution of porosity values """

        return ss.truncnorm.rvs(self._a, self._b,loc=self.pmean,scale = self.pstd, size=size)

    def pdf_stats(self,
                  x = False,
                  ):

        if x is False:  
            term0 = (ss.norm.pdf(self._b) - ss.norm.pdf(self._a))/(ss.norm.cdf(self._b) - ss.norm.cdf(self._a))
            term1 = (self._b*ss.norm.pdf(self._b) - self._a*ss.norm.pdf(self._a))/(ss.norm.cdf(self._b) - ss.norm.cdf(self._a)) 

            pmean_trunc = self.pmean - self.pstd * term0                       
            pvar_trunc = self.pstd**2 * (1 - term1 - term0**2)
                
        elif x is not False:
            pdf = self.pdf_porosity(x)
            pmean_trunc = np.trapz(x*pdf,x=x)
            pvar_trunc = np.trapz((x-pmean_trunc)**2*pdf,x=x)

        self.pmean_trunc = pmean_trunc
        self.pstd_trunc = np.sqrt(pvar_trunc)

        return self.pmean_trunc, self.pstd_trunc


    def connectivity_func(self,
                          x,
                          por_con_min=0.07,
                          por_con_max=0.39,
                          fit_con=[27.362,-27.661,9.4256,-0.0927],
                          **kwargs):

        """percolation function: 
            for every value of porosity it gives the (bernoulli-distributed) value 
            of percolation probability
        """
        ### calculate percolation function, polynomial coefficients from regression analysis
        pp_range=np.polyval(fit_con,x)
    
        ### add threshold values
        pp1=np.where(x>=por_con_min,pp_range,0)
        pp2=np.where(x<=por_con_max,pp1,1)
    
        return pp2

    def pdf_porosity_connected(self,
                               x,
                               normalize=False,
                               **kwargs,
                               ):

        """Calculation of distribution of percolating porosity
        """
          
        ### distribution of percolating porosities (not normalized)
        self.pdf_por_con=self.pdf_porosity(x)*self.connectivity_func(x,**kwargs)         
        ### norm value of pdf of porosity values, should be 1
        self.connectivity_total=np.trapz(self.pdf_por_con,x=x)                           
        
        if normalize:
            self.pdf_por_con=self.pdf_por_con/self.connectivity_total

        return self.pdf_por_con


    def pdf_porosity_disconnected(self,
                                  x,
                                  normalize=False,
                                  **kwargs,
                                  ):

        """Calculation of distribution of percolating porosity
        """
          
        self.pdf_por_discon =self.pdf_porosity(x)*(1-self.connectivity_func(x,**kwargs))
        self.discon_total=np.trapz(self.pdf_por_discon,x=x)                              

        if normalize:
            self.pdf_por_discon=self.pdf_por_discon/self.discon_total

        return self.pdf_por_discon


  
###############################################################################
###############################################################################
###############################################################################

class TA_POR_Distribution():

    """
    Class for analysing connected transport ability data distributed over
    a range of porosity values
    """    

    def __init__(self, 
                 dp = 0.02,
                 **kwargs):
            
        self.dp = float(dp)
        
        self.ta = None
        self.ta_con = None
        self.por = None
        self.por_con = None

        self.stats = None
        self.por_compress = None

        self.check()
        self.porosity_range()

    def check(self):
        if self.dp <= 0 or self.dp >=1:
            raise ValueError(
                "Value of porosity step size must be between 0 and 1"
            )
        elif self.dp >=0.1:
            print("Warning: Value of porosity step size very coarse")

    def porosity_range(self):

        """ Set range of porosity values """
        self.por_range=np.arange(0.5*self.dp,1.+0.5*self.dp,self.dp)

        return self.por_range

    def read_data(self,
                  file_data='..data/por_ta_data.csv',      ### observed transport abilities vs. porosity
                  compress2con = False,
                  por_con_min = 0.07,
                  **kwargs,
                  ):

        self.por_con_min = por_con_min

        """ read in ta-data as function of porosity """

        if not os.path.isfile(file_data):
            raise ValueError("File for ta-data not accessible: \n",file_data)
            
        ta_data=np.loadtxt(file_data,delimiter=',')

        if compress2con:
            """ compress to data above the connectivity level for porosity """
            por_condition = ta_data[:,0]>=self.por_con_min      
            ta_data = np.compress(por_condition,ta_data,axis=0)

        ta_con_data = np.compress(ta_data[:,1]>0,ta_data,axis=0)
        
        self.por_con = np.array(ta_con_data[:,0],ndmin=1)
        self.ta_con = np.array(ta_con_data[:,1],ndmin=1)     

        self.por = np.array(ta_data[:,0],ndmin=1)
        self.ta = np.array(ta_data[:,1],ndmin=1)

        return self.por,self.ta

    def set_data(self,
                 ta,
                 por,
                 compress2con = False,
                 por_con_min = 0.07,
                 **kwargs,
                 ):
        
        self.ta = ta
        self.por = por
        self.por_con_min = por_con_min
        
        """ compress to data above the connectivity level for porosity """
            
        if compress2con:
            """ compress to data above the connectivity level for porosity """
            por_condition = (self.por>=self.por_con_min)      
            self.por = np.compress(por_condition,self.por,axis=0)
            self.ta = np.compress(por_condition,self.ta,axis=0)

        self.ta_con = np.compress(self.ta>0,self.ta,axis=0)
        self.por_con = np.compress(self.ta>0,self.por,axis=0)

        return self.por,self.ta

    def porosity_bin_data(self,
                          round_decimals=2,
                          ):

        """" resort ta-data into specified porosity bins """        
        if self.por_con is None:
            raise ValueError("read or set data first")

        self.por_bin_data=dict()
        self.por_bin_logdata=dict()
        
        for ip,pi in enumerate(self.por_range):   
            ### determine arguments in list of porosity values being in the range of interest
            data=np.compress((pi-0.5*self.dp<self.por_con)*(self.por_con<=pi+0.5*self.dp),self.ta_con)
            self.por_bin_data[pi]=data
            self.por_bin_logdata[pi]=np.log(data)

        return self.por_bin_data

    def statistics(self,
                   nmin = 20, 
                   compress = True,
                   **kwargs,
                  ):

        """
        Determine statistcs on TA distribution for each porosity bin

        Output
        ------    
        stats   :   array containing statistics on perm-distribution for every por-bin of size dpor
                    --> stat values specified in header
        """
        self.nmin = nmin
        
        if self.ta_con is None:
            raise ValueError("read or set data first")

        self.porosity_range()  
        self.stats=dict()
        self.stats['porosity']=self.por_range
        self.stats['number_in_bin'] = np.zeros(len(self.por_range))
        self.stats['mean'] = np.zeros(len(self.por_range))
        self.stats['std'] = np.zeros(len(self.por_range))
        self.stats['skewness'] = np.zeros(len(self.por_range))
        self.stats['log-mean'] = np.zeros(len(self.por_range))
        self.stats['log-std'] = np.zeros(len(self.por_range))

        self.porosity_bin_data()
        
        for ip,pi in enumerate(self.por_range):
        
            ### determine arguments in list of porosity values being in the range of interest

            data=self.por_bin_data[pi]
            log_data=np.log(data)                
            self.stats['number_in_bin'][ip]=len(data)
            
            if len(data)>=self.nmin:
                self.stats['mean'][ip]=np.mean(data)                ### mean of values 
                self.stats['std'][ip]=np.std(data)                  ### standard deviation of values 
                self.stats['skewness'][ip]=ss.skew(data)            ### skewness of values       
                self.stats['log-mean'][ip]=np.mean(log_data)        ### mean of log-scaled values 
                self.stats['log-std'][ip]=np.abs(np.std(log_data))  ### standard deviation of log-scaled values 

        if compress:
            self.stats_compress()

        self.stats_values = np.array(list(self.stats.values())).T

        return self.stats_values," , ".join(list(self.stats.keys()))

    def stats_compress(self,
                       **kwargs,
                       ):

        """" compress TA(por) data and statistics to bins of sufficient data """
        
        if self.stats is None:
            raise ValueError("read data and run statistical analysis first")

        compress_condition=(self.stats['number_in_bin']>self.nmin)*(self.por_range>self.por_con_min)
  
        self.por_compress = np.compress(compress_condition,self.por_range,axis=0)

        self.stats['mean']= np.compress(compress_condition,self.stats['mean'],axis=0)
        self.stats['std']= np.compress(compress_condition,self.stats['std'],axis=0)
        self.stats['skewness']= np.compress(compress_condition,self.stats['skewness'],axis=0)
        self.stats['log-mean']= np.compress(compress_condition,self.stats['log-mean'],axis=0)
        self.stats['log-std']= np.compress(compress_condition,self.stats['log-std'],axis=0)

        return compress_condition

    def write_stats(self,
                    file_stats='stats_ta.csv',
                    delimiter = ',',
                    fmt = '%.3f',
                    ):

        """ Write statistical results to file """        
        np.savetxt(file_stats,self.stats_values,header = " {}".format(delimiter).join(list(self.stats.keys())),fmt = fmt,delimiter=delimiter)

    def read_stats(self,
                   file_stats='stats_ta.csv',
                   delimiter = ',',
                   ):

        """ Read statistical results from file """        
                
        self.stats_values = np.loadtxt(file_stats,delimiter=delimiter,skiprow=1)
        
    def normality_tests(self,
                        alpha=0.05,
                        lognorm=False,
                        delimiter = ',',
                        **kwargs,
                        ):

        """
        Test TA-data in each porosity bin on normality and log-normality
        
       
        Optional
        --------
        alpha       :   p-value level to evaluate as matching normal distribution
        lognormal   :   transformation of data to log-normal (check on log-normality)
        nmin        :   minimal number of samples in bin to be statistically analysed
        
        Output
        ------    
        normality   :   array containing statistics on perm-distribution for every por-bin of size dpor
                    --> stat values specified in header
        """

        # if self.stats is None:
        self.statistics(compress = False,**kwargs)

        stats_normal=dict()
        stats_normal['porosity']=self.por_range
        
        if lognorm:
            test_data='log_'
        else:
            test_data=''

        tests=['{}shapiro_{:.0f}'.format(test_data,100*alpha),'{}dagostino_{:.0f}'.format(test_data,100*alpha),'{}anderson_15'.format(test_data),'{}anderson_5'.format(test_data),'{}anderson_1'.format(test_data)]
        for test in tests:
            stats_normal[test]=np.zeros(len(self.por_range))
        
        for ip,pi in enumerate(self.por_range):
            data=self.por_bin_data[pi]
#            data=np.compress((pi-0.5*self.dp<self.por_con)*(self.por_con<=pi+0.5*self.dp),self.ta_con)
            if lognorm:
                data=np.log(data)
            
            if len(data)>=self.nmin:
        
                ### Normality testing according to shapiro
                stat_shapiro_nr,p_shapiro_nr=ss.shapiro(data)           
                #print('Shapiro-Statistics: %.3f, p=%.3f' %(stat_shapiro_nr,p_shapiro_nr))   

                #self.stats['p_shapiro'][ip]=p_shapiro_nr
                if p_shapiro_nr>alpha:
                    stats_normal[tests[0]][ip]=1
                
                stat_dagostino_nr,p_dagostino_nr=ss.normaltest(data)           
                #print('Dagostino-Statistics: %.3f, p=%.3f' %(stat_dagostino_nr,p_dagostino_nr))
                #self.stats['p_dagostino'][ip]=p_dagostino_nr
                if p_dagostino_nr>alpha:
                    stats_normal[tests[1]][ip]=1
       
                stat_AD=ss.anderson(data)
                #print('Anderson-Darling Test, statistics={:.3f}'.format(stat_AD.statistic))
        
                if stat_AD.statistic < stat_AD.critical_values[0]:
                    #print('sign. level={:.0f}%, critical value={:.3f}: data not normal'.format(sl, cv))
                    stats_normal[tests[2]][ip]=1
                    #else:
                        #print('sign. level={:.0f}%,critical value={:.3f}: data looks normal '.format(sl, cv))
                if stat_AD.statistic < stat_AD.critical_values[2]:
                    stats_normal[tests[3]][ip]=1
                if stat_AD.statistic < stat_AD.critical_values[4]:
                    stats_normal[tests[4]][ip]=1

            else:
                for test in tests:
                    stats_normal[test][ip]=2

        self.stats.update(stats_normal)
        self.stats_values = np.array(list(self.stats.values())).T

        return stats_normal
        # return np.array(list(stats_normal.values())).T,'porosity {} {}'.format(delimiter, " {} ".format(delimiter).join(tests))

    def fit_stats2por(self,
                      fitting='ta_fit_poly',
                      por_min=0,
                      **kwargs,
                      ):

        """
        fit log-TA statistics (log-mean, log-std) to polynomial function of por
        """
        
        # if self.stats is None:
        self.statistics(compress = False,**kwargs)
        
        if fitting  == 'ta_fit_poly':
            self.ta_fit=dict(
                    mean_deg = 2,
                    std_deg = 3, 
                    )
        elif fitting == 'ta_fit_adapt':
            self.ta_fit=dict(
                    mean_deg = 1,
                    std_deg = 3, 
                    )       
        else:
            self.ta_fit=fitting

        ### Reduce to data which is has sufficient data points (len(data)>nmin) and is above percolation threshold (por>por_min)
        compress_condition=(self.stats['number_in_bin']>self.nmin)*(self.por_range>por_min)
        
        por_cond = np.compress(compress_condition,self.por_range,axis=0)
        mean_cond = np.compress(compress_condition,self.stats['log-mean'],axis=0)
        std_cond = np.compress(compress_condition,self.stats['log-std'],axis=0)
 
        """ fit of mean according to specified fitting function        """
        if fitting  == 'ta_fit_poly':        
            ### Polynomial fitting according to choice of degree of polynomial
            fp1=np.polyfit(por_cond, mean_cond,deg=self.ta_fit['mean_deg'])
            mean_fit=np.polyval(fp1,self.por_range)
            mean_fit [mean_fit>0] = 0
            self.ta_fit['mean_coeff'] = fp1
            self.ta_fit['mean_fit'] = mean_fit    # mean values of ta for por_range given fitting function

            self.ta_fit['por_min'] = np.compress(compress_condition,self.por_range)[0] # minimum porosity where fit is valid
            self.ta_fit['por_max'] = np.compress(compress_condition,self.por_range)[-1] # maximum porosity where fit is valid

        elif fitting =='ta_fit_adapt':        
            ### Polynomial fitting according to choice of degree of polynomial
            mean_adapt = np.where(por_cond >0,mean_cond/(1 - por_cond),0)
            fp1=np.polyfit(por_cond,mean_adapt ,deg=self.ta_fit['mean_deg'])
            mean_fit=(1-self.por_range)*np.polyval(fp1,self.por_range)

            self.ta_fit['mean_coeff'] = fp1
            self.ta_fit['mean_fit'] = mean_fit    # mean values of ta for por_range given fitting function

        else:
            print('Fitting type not specified')

        """ fitting of variance """
        fp2=np.polyfit(por_cond,std_cond,deg=self.ta_fit['std_deg'])
        std_fit=np.polyval(fp2,self.por_range)          
        std_fit = np.where(compress_condition,std_fit,0)
        std_fit [std_fit<0] = 0
        self.ta_fit['std_coeff'] = fp2
        self.ta_fit['std_fit'] = std_fit      # std values of ta for por_range given fitting function


    def interpolate_ta_log_mean(self,
                                px=False,
                                **kwargs,
                                ):

        """ mu_TA(por): 
        create function of log-mean for TA values as function of porosity values, either
        - specified range px (e.g. at finer resolution)
        - using the compressed porosity values (px = False)                
        """
           
        # if self.stats is None:
        self.statistics(compress=True,**kwargs)

        if px is False:
            px = self.por_compress

        mean_adapt = self.stats['log-mean']/(1 - self.por_compress)
        fp1=np.polyfit(self.por_compress,mean_adapt ,deg=1)

        self.ta_log_mean_por=(1-px)*np.polyval(fp1,px)

        return self.ta_log_mean_por

    def interpolate_ta_log_std(self,
                               px=False,
                               **kwargs,
                               ):

        """ sigma_TA(por): 
        create function of log-std for TA values as function of porosity values, either
        - specified range px (e.g. at finer resolution)
        - using the compressed porosity values (px = False)                
        """

        """ fitting of variance """
        # if self.stats is None:
        self.statistics(compress=True,**kwargs)

        if px is False:
            px = self.por_compress

        fp2=np.polyfit(self.por_compress,self.stats['log-std'],deg=3)
        std_fit=np.hstack([[0],np.polyval(fp2,self.por_compress),[0]])
        por_fit = np.hstack([[0],self.por_compress,[1]])

        self.ta_log_std_por=np.interp(px,por_fit,std_fit)

        return self.ta_log_std_por

    def rvs_ta(self, 
               sample_data = False, 
               nrand = 1600 , 
               **kwargs,
               ):

        """
        random sampling of ta values from
            - log-normal distribution (sample_data = False) using mean and var determined from log-data
            - directly from data by using the log-data as sample distribution
            
        random values are samples in log-transformed space (exponent) and transformed into data space         
        """


        self.interpolate_ta_log_mean(px = self.por_range,**kwargs)
        self.interpolate_ta_log_std(px = self.por_range,**kwargs)
        compress_condition  = self.stats_compress()

        ta_rand_values = np.ones([len(self.por_range),nrand])
        for ip,pi in enumerate(self.por_range):           
            if compress_condition[ip]:                

                if sample_data:
                    hist = np.histogram(self.por_bin_logdata[pi], bins=self.nmin) 
                    hist_dist = ss.rv_histogram(hist)               
                    ta_rand_values[ip,:] = hist_dist.rvs(size = [nrand])
             
                else:
                    ta_rand_values[ip,:] = ss.norm.rvs(loc = self.ta_log_mean_por[ip],scale = self.ta_log_std_por[ip],size =[nrand])
                
            else:
                ### porosity values with not sufficient data
                ta_rand_values[ip,:] = self.ta_log_mean_por[ip]
                
        return np.exp(ta_rand_values)

    def pdf_ta_con(self,
                   tay,
                   px = False,
                   **kwargs,
                   ):
        
        """ generate log-normal pdf of TA-values (in resolution tay) for each porosity value 
        compressed data range  
        """

        self.interpolate_ta_log_mean(px=px,**kwargs)
        self.interpolate_ta_log_std(px=px,**kwargs)
        
        if px is False:
            px = self.por_compress
       
        pdf = np.zeros((len(px),len(tay)))

        for ip in range(len(px)):           
            if px[ip] == 0:
                pdf[ip,:] = 0
            else:
                pdf[ip,:]=ss.lognorm.pdf(tay,s=self.ta_log_std_por[ip],scale=np.exp(self.ta_log_mean_por[ip]))
 
        self.pdf_ta_con = pdf
        return self.pdf_ta_con

    def connectivity_distribution(self,
                                  compress = True,
                                  ):

        """ por-resolution level (from data):
            analysis of connectivity of TA-values distributed for porosity values
            --> percolation function of particular ensemble
        """
         
        pcon = 2*np.ones(len(self.por_range))
        for ip,pi in enumerate(self.por_range):   
            ### determine arguments in list of porosity values being in the range of interest
            data=np.compress((pi-0.5*self.dp<self.por)*(self.por<=pi+0.5*self.dp),self.ta)
            if len(data)>0:
                pcon[ip] = np.mean(data>0)                   

        if compress:
            """ compress to data above the connectivity level for porosity """
            condition = pcon<2
            self.pcon_data = np.compress(condition,pcon)
            px = np.compress(condition,self.por_range)
        else:
            self.pcon_data =  pcon
            px = self.por_range
 
        return self.pcon_data, px

  
###############################################################################
###############################################################################
###############################################################################

class TA_Distribution():

    def __init__(self, 
                 dim = 2,
                 ta_gmean   = 0.022,    
                 ta_log_std = 2,    
                 ta_pcon    = 1,
                 **settings):

        self.dim = dim
        self.ta_gmean = ta_gmean
        self.ta_log_std = ta_log_std 
        self.ta_pcon = ta_pcon
        
    def pdf_ta_con(self,tay):

        """ ensemble level (theory):
            log-normal pdf of TA-values based on scale-dependent mean and std 
            of log-ta data from upscaling theory
            (marginal distribution of ta-por-scatter) 
        """

        self.pdf_ta_con= ss.lognorm.pdf(tay, s =  self.ta_log_std, scale = self.ta_gmean)

        return self.pdf_ta_con

    def pdf_ta(self,tay):

        """ ensemble level (theory):
            log-normal pdf of TA-values based on scale-dependent mean and std 
            of log-ta data from upscaling theory
            (marginal distribution of ta-por-scatter) 
        """
        self.pdf_ta_con(tay)

        self.pdf_ta = self.pdf_ta_con*self.ta_pcon
        self.pdf_ta[0] = 1-self.ta_pcon 
        
        return self.pdf_ta

    def moments_ta(self):
        
        ### expectation value for adapted probability distrubtion including non-connected values
        self.ta_1moment = self.ta_pcon*self.ta_gmean*np.exp(0.5*self.ta_log_std**2)
        ### variance value for adapted probability distrubtion including non-connected values
        self.ta_2moment = self.ta_pcon*self.ta_gmean**2*np.exp(self.ta_log_std**2)*(np.exp(self.ta_log_std**2)-self.ta_pcon)

        return self.ta_1moment,self.ta_2moment

