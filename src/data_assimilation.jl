using LinearAlgebra

export DataAssimilation

"""
parameters of the filtering method
 - method :chosen method ('AnEnKF', 'AnEnKS', 'AnPF')
 - N : number of members (AnEnKF/AnEnKS) or particles (AnPF)
"""
struct DataAssimilation{T}

    method :: Symbol
    N      :: Int
    xb     :: Vector{T}
    B      :: Array{Bool,    2}
    H      :: Array{Bool,    2}
    R      :: Array{Float64, 2}
    m      :: AnalogForecasting

    function DataAssimilation( method :: Symbol, 
                               N      :: Integer, 
                               xt     :: TimeSeries{T},
                               sigma2 :: Float64 ) where T
                
        xb = xt.values[1,:]
        B  = 0.1 * Matrix(I, xt.nv, xt.nv)
        H  = Matrix( I, xt.nv, xt.nv)

        R  = sigma2 .* H

        new{T}(  method, N, xb, B, H, R, m )

    end

    
end

#=
    
from scipy.stats import multivariate_normal
from analog_data_assimilation.stat_functions import resampleMultinomial, inv_using_SVD
from tqdm import tqdm
import sys

def data_assimilation(yo, DA):
    """ 
    Apply stochastic and sequential data assimilation technics using 
    model forecasting or analog forecasting. 
    """

    # dimensions
    n = len(DA.xb)
    T, p = yo.values.shape
    # check dimensions
    assert p == DA.R.shape[0]

    # initialization
    class x_hat:
        part = np.zeros([T,DA.N,n])
        weights = np.zeros([T,DA.N])
        values = np.zeros([T,n])
        loglik = np.zeros([T])
        time = yo.time

    m_xa_part = np.zeros([T,DA.N,n])
    xf_part = np.zeros([T,DA.N,n])
    Pf = np.zeros([T,n,n])
    for k in tqdm(range(T)):
        # update step (compute forecasts)            
        if k==0:
            xf = np.random.multivariate_normal(DA.xb, DA.B, DA.N)
        else:
            xf, m_xa_part_tmp = DA.m(x_hat.part[k-1,:,:])
            m_xa_part[k,:,:] = m_xa_part_tmp         
        xf_part[k,:,:] = xf
        Ef = np.dot(xf.T,np.eye(DA.N)-1/DA.N)
        Pf[k,:,:] = np.dot(Ef,Ef.T)/(DA.N-1)
        # analysis step (correct forecasts with observations)          
        i_var_obs = np.nonzero(~np.isnan(yo.values[k,:]))[0]            
        if (len(i_var_obs)>0):                
            eps = np.random.multivariate_normal(np.zeros_like(i_var_obs),DA.R[np.ix_(i_var_obs,i_var_obs)],DA.N)
            yf = np.dot(DA.H[i_var_obs,:],xf.T).T
            SIGMA = np.dot(np.dot(DA.H[i_var_obs,:],Pf[k,:,:]),DA.H[i_var_obs,:].T)+DA.R[np.ix_(i_var_obs,i_var_obs)]
            SIGMA_INV = np.linalg.inv(SIGMA)
            K = np.dot(np.dot(Pf[k,:,:],DA.H[i_var_obs,:].T),SIGMA_INV)             
            d = yo.values[k,i_var_obs][np.newaxis]+eps-yf
            x_hat.part[k,:,:] = xf + np.dot(d,K.T)           
            # compute likelihood
            innov_ll = np.mean(yo.values[k,i_var_obs][np.newaxis]-yf,0)
            loglik = -0.5*(np.dot(np.dot(innov_ll.T,SIGMA_INV),innov_ll))-0.5*(n*np.log(2*np.pi)+np.log(np.linalg.det(SIGMA)))
        else:
            x_hat.part[k,:,:] = xf          
        x_hat.weights[k,:] = 1.0/DA.N
        x_hat.values[k,:] = np.sum(x_hat.part[k,:,:]*x_hat.weights[k,:,np.newaxis],0)
        x_hat.loglik[k] = loglik
    # end AnEnKF
    
    
    for k in tqdm(range(T-1,-1,-1)):           
        if k==T-1:
            x_hat.part[k,:,:] = x_hat.part[T-1,:,:]
        else:
            m_xa_part_tmp = m_xa_part[k+1,:,:]
            tej, m_xa_tmp = DA.m(np.mean(x_hat.part[k,:,:],0)[np.newaxis])
            tmp_1 =(x_hat.part[k,:,:]-np.mean(x_hat.part[k,:,:],0)).T
            tmp_2 = m_xa_part_tmp - m_xa_tmp                   
            Ks = 1.0/(DA.N-1)*np.dot(np.dot(tmp_1,tmp_2),inv_using_SVD(Pf[k+1,:,:],0.9999))                    
            x_hat.part[k,:,:] = x_hat.part[k,:,:]+np.dot(x_hat.part[k+1,:,:]-xf_part[k+1,:,:],Ks.T)
        x_hat.values[k,:] = np.sum(x_hat.part[k,:,:]*x_hat.weights[k,:,np.newaxis],0)             
    
    
    return x_hat       
=#
