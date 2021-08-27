#!/usr/bin/env python3

import numpy as np
import os.path
import copy

from Distributions import Porosity_Distribution,TA_POR_Distribution
#### Load values of transport resistant from data file


DEF_DATA = dict(
     res0 = 2,
     pstd_r0=0.15,
     por_con_min = 0.07,
     nmin = 20,
     dp = 0.02,
     # scale_std = True,
     # compress2con = False,
    )

###############################################################################
###############################################################################
###############################################################################

class TA_Ensemble_Simulation:
    
    """
    Class to generate ensemble of networks consisting of n_network^dim nodes:
    with random porosity values from normal distribution, subsequent 
    connectivity and transport ability generated from porosity values
    and calculation of network porosity and transport ability for each network    
    """
    
    def __init__(self,
                 dim = 2,
                 res = 2,
                 n_network = 4,

                 n_ens = 100,        
                 ens_seed = False,

                 pmean=0.3, 
                 pstd=0.15,
                 task_root = './',
                 **settings):
        
        ### domain settings (of individual network)
        self.dim = dim
        self.res = res
        self.n_network = n_network

        ### Porosity statistics
        self.pmean = pmean
        self.pstd = pstd

        self.task_root = task_root
        if not os.path.isdir(self.task_root):
            os.mkdir(self.task_root)
        
        self.settings=copy.copy(DEF_DATA) 
        self.settings.update(settings)   

        self.set_seed(ens_seed)
        self.set_porosity()

        ### ensemble settings
        self.n_ens = n_ens
        self.size = (self.n_network**self.dim,self.n_ens)   #for random number generation        
        if self.dim==2:
            self.shape = (self.n_network,self.n_network)    # dimensional shape of upscaled network
        elif self.dim==3:
            self.shape = (self.n_network,self.n_network,self.n_network)

        self.ta_stats = dict()
               
    def set_seed(self,ens_seed=False):

        if ens_seed is not False:
            self.ens_seed = ens_seed
        else:
            self.ens_seed = np.random.randint(0, 10 ** 9)        

    def set_porosity(self,
                     pmean = False,
                     pstd = False,
                     scale_std = True,
                     **kwargs,
                     ):

        if pmean:
            self.pmean = pmean
        if pstd:
            self.pstd = pstd

        if scale_std:
            self.pstd = self.settings['pstd_r0']/np.power(self.res/self.settings['res0']*self.n_network,0.25*self.dim)

        ### create instance of material type with specific porosity distribution
        self.POR_in = Porosity_Distribution(pmean = self.pmean,
                                            pstd = self.pstd,
                                            **kwargs)           

    def set_ta(self,
               file_data='..data/por_ta_data.csv',
               **settings):

        self.settings.update(**settings)   

        self.TA_in=TA_POR_Distribution(
            dim = self.dim,
            res = self.res,
            n_network = self.n_network,
            pmean = self.pmean,
            pstd = self.pstd,
            **self.settings)                  

        # self.TA_in=TA_con_Distribution(**self.settings)                  
        self.TA_in.read_data(
            file_data = file_data,
            **settings)
        
        # self.TA_in.statistics(**self.settings)
        self.TA_in.interpolate_ta_log_mean(**self.settings)
        self.TA_in.interpolate_ta_log_std(**self.settings)

    def sample(self,**kwargs):
        
        self.sample_porosity(**kwargs)
        self.sample_connectivity(**kwargs)
        self.sample_ta_connect(**kwargs)
        self.sample_ta()
                       
    def sample_porosity(self,
                        reduce = True,
                        **kwargs):

        if reduce:
            rep = int(np.sqrt(self.n_network))
            nrep = rep 

            if (rep*nrep != self.n_network):
                rep = int(np.ceil(np.sqrt(self.n_network)))
                nrep = rep 

            if self.dim ==2:
                por1 = self.POR_in.rvs_porosity(size = (rep,rep,self.n_ens))
                por2 = np.repeat(np.repeat(por1,nrep,axis=0),nrep,axis = 1)
                por3 = por2[0:self.n_network,0:self.n_network]
            elif self.dim ==3:
                por1 = self.POR_in.rvs_porosity(size = (rep,rep,rep,self.n_ens))
                por2 = np.repeat(np.repeat(np.repeat(por1,nrep,axis=0),nrep,axis = 1),nrep,axis = 2)
                por3 = por2[0:self.n_network,0:self.n_network,0:self.n_network] 

            por_sample = por3.reshape((self.size))
        else:  
            por_sample = self.POR_in.rvs_porosity(self.size)

        self.por_sample = por_sample           
        return self.por_sample

    def sample_connectivity(self,**kwargs):

        self.settings.update(**kwargs)   
        
        ### calculate probability of connectivity for every porosity value in sample (value between 0 - 1)
        p_connect = self.POR_in.connectivity_func(self.por_sample,**self.settings)
        ### generate random number between 0 and 1 from uniform distribution
        rand_connect = np.random.uniform(0,1,size = self.size)
        ### generate connectivity as random number 0 or 1 
        self.connectivity_sample = p_connect > rand_connect
        
        return self.connectivity_sample

    def sample_ta_connect(self,
                          subens =100,
                          **kwargs
                          ):

        elem = self.size[0]

        ta_rvs = self.TA_in.rvs_ta(nrand=elem*subens,**kwargs).reshape([-1,elem,subens])

        self.ta_connect_sample = np.zeros(self.size)
        rand_index = np.random.randint(subens,size=self.n_ens)

#         ### WARNING: check when changing dp step size --> adapt to find nearest value
        for ii in range(self.n_ens):
            index = np.floor(self.por_sample[:,ii]/self.settings['dp']).astype(int)
            self.ta_connect_sample[:,ii] = ta_rvs[index,range(elem),rand_index[ii]]

        return self.ta_connect_sample

    def sample_ta(self):

        self.ta_sample = np.where(self.connectivity_sample,self.ta_connect_sample,0)
        
        return self.ta_sample
      
    def statistics(self): 
      
        ### network properties
        self.por_eff = np.mean(self.por_sample,axis = 0)          
        ta_total_connect = np.sum(self.connectivity_sample,axis = 0)    ### number of con elements per network
        
        ta_edges = np.zeros(self.n_ens)
        for ii in range(self.n_ens):
            ta_edges[ii] = number_of_edges(self.connectivity_sample[:,ii].reshape(self.shape))

        ta_networkfactor = ta_total_connect/ta_edges        
        ### network properties
        ta_log_connect_sample = np.log(self.ta_connect_sample)          ### log-transform of ta samples 
        ta_log_sample = np.where(
            self.connectivity_sample,
            ta_log_connect_sample,
            0
            )
        
        ### statistics of log-ta values
        ta_log_sample_mean = np.sum(ta_log_sample,axis = 0)/ta_total_connect
        ta_log_sample_var = np.sum((ta_log_sample-ta_log_sample_mean)**2,axis = 0)/ta_total_connect

        tr_sample = np.where(self.connectivity_sample,1./self.ta_connect_sample,0)
        ta_H = ta_total_connect/np.sum(tr_sample,axis = 0)
        
        self.ta_stats.update(dict(
            por_mean = self.por_eff,
            ta_total_connect = ta_total_connect, # number of percolating elements
            ta_total_non_connect = np.sum(np.logical_not(self.connectivity_sample),axis = 0), # number of percolating elements
            ta_A = np.mean(self.ta_sample,axis = 0),
            ta_H = ta_H,
            ta_G =  np.exp(ta_log_sample_mean),
            ta_log_mean = ta_log_sample_mean,
            ta_log_var = ta_log_sample_var,
            ta_edges = ta_edges,
            ta_networkfactor = ta_networkfactor,
             ))

        return self.ta_stats

    def ta_network_flow(self,
                        degenerated = True, 
                        **settings
                        ):

        self.settings.update(settings)    
        ta_eff = np.zeros(self.n_ens)
        
        for ii in range(self.n_ens):
           
            if degenerated: ### networks containing not-connected nodes
                ta = self.ta_sample[:,ii].reshape(self.shape)
            else: ### network with only connected nodes
                ta = self.ta_connect_sample[:,ii].reshape(self.shape)
                
            FlowSim = Network_Flow(ta,**self.settings)
            FlowSim.solve_flow(degenerated=degenerated)
 
            ta_eff[ii] = FlowSim.ta_eff
            if FlowSim.ta_eff==0:
                print('Disconnected network: i={}'.format(ii))
        
        self.por_eff = np.mean(self.por_sample,axis = 0)          
        self.ta_eff = ta_eff
        self.connected = self.ta_eff>0
        self.ta_eff_con = np.compress(self.connected,self.ta_eff)
        self.por_eff_con = np.compress(self.connected,self.por_eff)

        self.ta_stats.update(dict(ta_sim = self.ta_eff))

        return self.ta_eff

    def save_ensemble_taeff(self,
                            file_ta_eff= 'ta_eff.csv',
                            delimiter = ',',
                            fmt = '%.3e',
                            compress2connected = False,
                            ):

        try:
            self.ta_eff
        except AttributeError:
            raise ValueError("Run ensemble simulation first: ta_network_flow()")

        self.results = np.vstack((self.por_eff,self.ta_eff))

        if compress2connected:
            results = np.compress(self.connected,self.results,axis=1)
        else:
            results = self.results

        np.savetxt(os.path.join(self.task_root,file_ta_eff),results.T,delimiter=delimiter,fmt=fmt)               
            
        print('Ensemble ta_eff saved at \n {}'.format(os.path.join(self.task_root,file_ta_eff)))
        
        return self.results

    def save_ensemble_samples(self,
                              file_ta_stats='stats_ta.csv',
                              file_por_samples='samples_porosity.csv',
                              file_connectivity='samples_connectivity.csv',
                              file_ta_samples='samples_ta.csv',                      
                              delimiter = ',',
                              fmt = '%.2e',
                              ):

        try:
            self.ta_sample
        except AttributeError:
            raise ValueError("Run ensemble sample first: sample()")

        ### save statistics of ta-network samples               
        values=np.array(list(self.ta_stats.values())).T
        header= " , ".join(list(self.ta_stats.keys()))

        np.savetxt(os.path.join(self.task_root,file_ta_stats),values,delimiter=delimiter,header=header,fmt=fmt)        

        ### save por-network samples               
        np.savetxt( os.path.join(self.task_root,file_por_samples),self.por_sample.T,delimiter=delimiter,fmt=fmt)        

        ### save ta-network samples               
        np.savetxt( os.path.join(self.task_root,file_ta_samples),self.ta_sample.T,delimiter=delimiter,fmt=fmt)        

        ### save ta-network connnectivity samples               
        np.savetxt( os.path.join(self.task_root,file_connectivity),self.connectivity_sample.T,delimiter=delimiter,fmt='%.0f')        
        
        print('Ensemble samples saved in dir \n {}/'.format(self.task_root))
        return values,header

###############################################################################
###############################################################################
###############################################################################
 
class Network_Flow:
    
    def __init__(self,
                 ta,
                 pressure_in=1,
                 pressure_out=0,
                 **settings):

        self.ta = ta
        self.pressure_out = pressure_out
        self.pressure_in = pressure_in

        self.settings=copy.copy(DEF_DATA)
        self.settings.update(settings)            

        self.dim = len(ta.shape)
        self.set_dimension()

    def set_dimension(self):

        if self.dim==2:
            (self.Nx,self.Ny)=self.ta.shape
            self.nn = self.Nx*self.Ny
#            self.size=(self.n_network,self.n_network)
            self.ax_means=(1)
        elif self.dim==3:
            (self.Nx,self.Ny,self.Nz)=self.ta.shape
            self.nn = self.Nx*self.Ny*self.Nz
#            self.size=(self.n_network,self.n_network,self.n_network)
            self.ax_means=(1,2)

        self.condition=(self.ta>0).reshape((self.nn,1))[:,0]       

    def ta_elim_isolated(self):
           
        if self.dim == 2:
            t1 = np.zeros([self.Nx+2,self.Ny+2])
            t1[1:-1,1:-1] = self.ta[:,:]
            t2 = t1[:-2,1:-1]+t1[2:,1:-1]+t1[1:-1,:-2]+t1[1:-1,2:]
        elif self.dim == 3:
            t1 = np.zeros([self.Nx+2,self.Ny+2,self.Nz+2])
            t1[1:-1,1:-1,1:-1] = self.ta[:,:,:]
            t2 = t1[:-2,1:-1,1:-1]+t1[2:,1:-1,1:-1]+t1[1:-1,:-2,1:-1]+t1[1:-1,2:,1:-1]+t1[1:-1,1:-1,:-2]+t1[1:-1,1:-1,2:]
        
        self.ta = np.where(t2>0,self.ta,0)
        self.condition=(self.ta>0).reshape((self.nn,1))[:,0]       
       
        return t2>0

    def network_matrix(self):

        """ Creating Adjecency Matrix in 2D for solving 2D steady state diffusion 
            equation with 
                - heterogeneous values of transport ability
                - Dirichlet BC at left & right margin: constant input value (x-coordinate)
                - no flow BC at upper and lower margin (y-coordinates)
             
        """
        A_ta=np.zeros((self.nn,self.nn))
        b_ta=np.zeros(self.nn)

        if self.dim == 2:
            self.indeces=np.arange(self.Nx*self.Ny).reshape((self.Nx,self.Ny))
            
            ###########################################################################
            ### Create adjacency matrix for network with ta values        
            for ix in range(self.Nx):
                for iy in range(self.Ny):
                    index=self.indeces[ix,iy]
                    ### determine matrix entries for percolating nodes (ta>0) only 
                    if self.ta[ix,iy]:
                        if ix==0:
                            a1=self.ta[ix,iy]
                            b_ta[index]=a1*self.pressure_in
                        else:
                            a1=2.*self.ta[ix,iy]*self.ta[ix-1,iy]/(self.ta[ix,iy]+self.ta[ix-1,iy]) # 1/R
                            A_ta[index,self.indeces[ix-1,iy]]=-a1
                        if ix==self.Nx-1:
                            a2=self.ta[ix,iy]
                            b_ta[index]=a2*self.pressure_out
                        else:
                            a2=2.*self.ta[ix,iy]*self.ta[ix+1,iy]/(self.ta[ix,iy]+self.ta[ix+1,iy])
                            A_ta[index,self.indeces[ix+1,iy]]=-a2
                        if iy==0:
                            b1=0
                        else:
                            b1=2.*self.ta[ix,iy]*self.ta[ix,iy-1]/(self.ta[ix,iy]+self.ta[ix,iy-1])
                            A_ta[index,self.indeces[ix,iy-1]]=-b1
                        if iy==self.Ny-1:
                            b2=0
                        else:
                            b2=2.*self.ta[ix,iy]*self.ta[ix,iy+1]/(self.ta[ix,iy]+self.ta[ix,iy+1])
                            A_ta[index,self.indeces[ix,iy+1]]=-b2
            
                        A_ta[index,index]=a1+a2+b1+b2
        elif self.dim == 3:

            self.indeces=np.arange(self.nn).reshape((self.Nx,self.Ny,self.Nz))
                        
            for ix in range(self.Nx):
                for iy in range(self.Ny):
                    for iz in range(self.Nz):
                        index=self.indeces[ix,iy,iz]
                        ### determine matrix entries for percolating nodes (ta>0) only 
                        if self.ta[ix,iy,iz]:
                            if ix==0:
                                a1=self.ta[ix,iy,iz]
                                b_ta[index]=a1*self.pressure_in
                            else:
                                a1=2.*self.ta[ix,iy,iz]*self.ta[ix-1,iy,iz]/(self.ta[ix,iy,iz]+self.ta[ix-1,iy,iz]) # 1/R
                                A_ta[index,self.indeces[ix-1,iy,iz]]=-a1
                            if ix==self.Nx-1:
                                a2=self.ta[ix,iy,iz]
                                b_ta[index]=a2*self.pressure_out
                            else:
                                a2=2.*self.ta[ix,iy,iz]*self.ta[ix+1,iy,iz]/(self.ta[ix,iy,iz]+self.ta[ix+1,iy,iz])
                                A_ta[index,self.indeces[ix+1,iy,iz]]=-a2
        
                            if iy==0:
                                b1=0
                            else:
                                b1=2.*self.ta[ix,iy,iz]*self.ta[ix,iy-1,iz]/(self.ta[ix,iy,iz]+self.ta[ix,iy-1,iz])
                                A_ta[index,self.indeces[ix,iy-1,iz]]=-b1
                            if iy==self.Ny-1:
                                b2=0
                            else:
                                b2=2.*self.ta[ix,iy,iz]*self.ta[ix,iy+1,iz]/(self.ta[ix,iy,iz]+self.ta[ix,iy+1,iz])
                                A_ta[index,self.indeces[ix,iy+1,iz]]=-b2
        
                            if iz==0:
                                c1=0
                            else:
                                c1=2.*self.ta[ix,iy,iz]*self.ta[ix,iy,iz-1]/(self.ta[ix,iy,iz]+self.ta[ix,iy,iz-1])
                                A_ta[index,self.indeces[ix,iy,iz-1]]=-c1
                            if iz==self.Nz-1:
                                c2=0
                            else:
                                c2=2.*self.ta[ix,iy,iz]*self.ta[ix,iy,iz+1]/(self.ta[ix,iy,iz]+self.ta[ix,iy,iz+1])
                                A_ta[index,self.indeces[ix,iy,iz+1]]=-c2
                
                            A_ta[index,index]=a1+a2+b1+b2+c1+c2

        self.matrix = A_ta
        self.vector=b_ta

    def solve_uu(self,
                 degenerated=True,
                 **kwargs,
                 ):    

        if degenerated:
             self.uu=np.zeros(self.nn)           ### pressure for active nodes  
             ### condition for compression: active matrix values in flattened array
             a_compress=np.compress(self.condition,np.compress(self.condition,self.matrix, axis=0), axis=1)
             b_compress=np.compress(self.condition,self.vector, axis=0)
         
             ### calculate inverse matrix for compressed domain --> if still singular, no flow possible!

             uu_compress=self.solve_matrix_equation(
                 a_compress,
                 b_compress,
                 **kwargs,
                 )
             index_compress=np.compress(self.condition,self.indeces.reshape((self.nn,1)), axis=0)[:,0]
    
             self.uu[index_compress]=uu_compress     ### pressure for active nodes      

        else:
            self.uu=self.solve_matrix_equation(
                self.matrix,
                self.vector,
                **kwargs,
                ) ### solve matrix equation analytically    
            
        return self.uu

    def solve_matrix_equation(self,
                              A,
                              b,
                              matrix_approx = False,
                              n_iter = 25,
                              **kwargs):


        if matrix_approx is False:
            x = np.linalg.solve(A,b)
        else: 
            """Solves the equation Ax=b via the Jacobi iterative method."""
            # Create an initial guess if needed                                                                                                                                                            
            x = np.zeros(len(A[0]))

            # Create a vector of the diagonal elements of A                                                                                                                                                
            # and subtract them from A                                                                                                                                                                     
            D = np.diag(A)
            R = A - np.diagflat(D)
        
            # Iterate for N times                                                                                                                                                                          
            for i in range(n_iter):
                x = (b - np.dot(R,x)) / D
           
        return x

    def ta_from_u(self,
                  singular=False,
                  ):
        
        ### reshape solution into dimensional form
        self.u_solution=np.reshape(self.uu,self.ta.shape)

        if singular is True:
            q_out = 0
        else:
            ### calculate average influx = outflux as mean flux over boundary
            q_out=np.mean(self.ta*(self.pressure_out-self.u_solution),axis=self.ax_means)[-1]        

        self.q_out=q_out*(self.Nx+1)
        ### flux per node: q_x / dx where dx = 1./ (self.Nx+1)

        ###average gradient over domain given input and output pressure over distance        
        self.gradp = (self.pressure_out-self.pressure_in)

        ### Effective transport ability as ratio of simulated average flux over mean gradient        
        self.ta_eff = self.q_out/self.gradp

        return self.ta_eff

    def solve_flow(self,
                   degenerated=True
                   ):

        self.network_matrix()           
        
        singular=False
        if (np.sum(self.condition)==len(self.condition)) or (degenerated is False):        
            self.solve_uu(degenerated=False)    ### solve matrix equation (ODE)
        else:  
            self.ta_elim_isolated()
            try:
                self.solve_uu(degenerated = True)
            except:
                xcon = np.all(self.ta,axis = 0) # axis now adapted to flow direction    
                if np.any(xcon):            
                    print('Compressed matrix is singular: But flow in compressed domain')
                    if self.dim==2:
                        self.ta = np.compress(xcon,self.ta,axis=1)
                        degenerated=False
                    elif self.dim==3:
                        self.ta = np.compress(np.any(xcon,axis=0),np.compress(np.any(xcon,axis=1),self.ta,axis=1),axis=2)
                        degenerated=True
                    self.set_dimension()
                    self.network_matrix()           
                    try:
                        self.solve_uu(degenerated=degenerated)
                    except:
                        print('Compressed matrix is singular: No flow through domain')
                        singular = True
                else:
                    print('Compressed matrix is singular: No flow through domain')
                    singular = True

        self.ta_from_u(singular=singular)    

        return not np.isnan(self.ta_eff) 


###############################################################################
###############################################################################
###############################################################################

def number_of_edges(matrix):

    dim = len(matrix.shape)
    """ number of edges in a 2D or 3D matrix representing a (degenerated) 
    network of nodes where values = 0 represents missing node and 
    value > 0 network node 
    """

    if dim == 2:
        ne = np.sum(matrix[1:,:]*matrix[:-1,:]>0)+np.sum(matrix[:,1:]*matrix[:,:-1]>0)
    elif dim == 3:
        ne = np.sum(matrix[1:,:,:]*matrix[:-1,:,:]>0)+np.sum(matrix[:,1:,:]*matrix[:,:-1,:]>0)+np.sum(matrix[:,:,1:]*matrix[:,:,:-1]>0)
    return ne

def network_factor(matrix,degenerated = True):

    """ ratio of nodes to edges in a 2D/3D matrix representing a (degenerated) 
    network where values = 0 represents missing node and value > 0 nrtwork node
    """

    dim = len(matrix.shape)

    if degenerated:
        ne = number_of_edges(matrix)
        nnodes = np.sum(matrix>0)
    else:
        if dim == 2:
            nnodes = matrix.shape[0]*matrix.shape[1]
            ne = (matrix.shape[0]-1)*matrix.shape[1] + matrix.shape[0]*(matrix.shape[1]-1) 
        elif dim == 3:
            nnodes = matrix.shape[0]*matrix.shape[1]*matrix.shape[2]
            ne = (matrix.shape[0]-1)*matrix.shape[1]*matrix.shape[2] + matrix.shape[0]*(matrix.shape[1]-1)*matrix.shape[2]+ matrix.shape[0]*matrix.shape[1]*(matrix.shape[2]-1)

    return nnodes/ne
