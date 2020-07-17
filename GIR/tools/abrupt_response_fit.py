# this module provides the tools to fit a 3-layer Geoffrey energy balance model to TOA energy imbalnce and tas from an abrupt-4xCO2 experiment

import numpy as np
import pandas as pd
import scipy as sp

def BuildMat(params):
#     param_names = ['gamma', 'C1', 'C2', 'C3', 'kap1', 'kap2', 'kap3', 'epsilon', 'stds', 'stdx','F_2x']
    A = np.array([[-1*params[0],0,0,0],\
                     [1/params[1],-1*(params[4]+params[5])/params[1],params[5]/params[1],0],\
                     [0,params[5]/params[2],-1*(params[5]+params[7]*params[6])/params[2],params[7]*params[6]/params[2]],\
                     [0,0,params[6]/params[3],-1*params[6]/params[3]]])
    k = A.shape[0]
    b = np.array([params[0],0,0,0]).T
    u = params[10]
    Q = np.zeros((4,4))
    Q[0,0] = params[8]**2
    Q[1,1] = (params[9]/params[1])**2
    A_d = sp.linalg.expm(A)
    b_d = sp.linalg.solve(A, (A_d - np.identity(k)) @ b)
    ## use Van Loan (1978) to compute the matrix exponential
    H = np.zeros((k*2,k*2))
    H[:k,:k] = -A
    H[:k,k:] = Q
    H[k:,k:] = A.T
    G = sp.linalg.expm(H)
    Q_d = G[k:,k:].T @ G[:k,k:]
    C_d = np.array([[0,1,0,0],\
                   [1,-1*params[4],(1-params[7])*params[6],-1*(1-params[7])*params[6]]])
    gamma0 = (sp.linalg.solve(np.identity(k**2)-np.kron(A_d,A_d),Q_d.flatten())).reshape(4,4)
    x0 = np.array([params[10],0,0,0])
    a0 = A_d@x0 + b_d*params[10]
    
    return A,b,Q,A_d,b_d,Q_d,gamma0,a0,C_d,u

def Kalman(a0, P0, dt, ct, Tt, Zt, HHt, GGt, yt):
    
    # computes a Kalman filter recursively over all the timesteps of the input data, yt
    # returns the Negative log likelihood for fitting
    
    n = yt.shape[1] # measurements
    k = a0.shape[0] # state dimension size
    d = yt.shape[0] # measurement dimensions
    vt = np.zeros((d,n))
    Ft = np.zeros((d,d,n))
    Kt = np.zeros((k,d,n))
    at = np.zeros((k,n+1))
    Pt = np.zeros((k,k,n+1))
    nll = 0

    at[...,0] = a0
    Pt[...,0] = P0

    for i in np.arange(n):
        vt[...,i] = yt[...,i] - ct - Zt @ at[...,i]
        Ft[...,i] = Zt @ Pt[...,i] @ Zt.T + GGt
        Kt[...,i] = Pt[...,i] @ Zt.T @ np.linalg.inv(Ft[...,i])
        att = at[...,i] + Kt[...,i] @ vt[...,i]
        Ptt = Pt[...,i] - Pt[...,i] @ Zt.T @ Kt[...,i].T

        at[...,i+1] = dt + Tt@att
        Pt[...,i+1] = Tt @ Ptt @ Tt.T + HHt
        
        nll += np.log(2*np.pi) + (1/2) * (  np.log(np.linalg.det(Ft[...,i])) + vt[...,i].T @ np.linalg.inv(Ft[...,i]) @ vt[...,i] )
        
    return nll


def fit_kbox_kal(params,yt,Transform=True):
    
    # this builds the matrices as required and passes them to the Kalman Filter function
    
    if Transform:
        params = np.exp(params)
    
    A,b,Q,A_d,b_d,Q_d,gamma0,a0,C_d,u = BuildMat(params)
    
    P0 = gamma0.copy()
    dt = b_d*u
    ct = np.zeros(2)
    Tt = A_d.copy()
    Zt = C_d.copy()
    HHt = Q_d.copy()
    GGt = np.identity(2)*1e-15
    
    return Kalman(a0, P0, dt, ct, Tt, Zt, HHt, GGt, yt)

def fit_model(data,method='bobyqa',nfev=10000):
    
    from pdfo import pdfo
    
    # data is an 2 x t array: array([temp,N])
    
    x0 = np.array([2, 5, 20, 100, 1, 2, 1, 1, 0.5, 0.5, 5])
    soln_pdfo = pdfo(lambda x: fit_kbox_kal(x,data,True),x0=np.log(x0),method=method,options={'maxfev':nfev})
    
    if soln_pdfo.success:
        print('fit converged')
        return np.exp(soln_pdfo.x)
    else:
        print('fit failed after '+str(nfev)+' iterations')
        return None

def convert_geoffrey_to_FaIR(params):
#     ['gamma', 'C1', 'C2', 'C3', 'kap1', 'kap2', 'kap3', 'epsilon', 'stds', 'stdx','F_4x']

### transfer function relations ###
# tau = -1/np.linalg.eig(A[1:,1:])[0]
# kap = np.array([1.21,1.7,0.79])
# C = np.array([5.3,12.3,49])
# ep = 1.28

# q_C = np.array([[1,1,1],\
#                 [tau[1]+tau[2],tau[0]+tau[2],tau[0]+tau[1]],\
#                 [1/tau[0],1/tau[1],1/tau[2]]])

# q_b = np.array([np.product(kap[1:])*np.product(tau)/np.product(C),((kap[1]+ep*kap[2])/(C[0]*C[1])+kap[2]/(C[0]*C[2]))*np.product(tau),1/C[0]])

# q = np.linalg.solve(q_C,q_b)

    A,b,Q,A_d,b_d,Q_d,gamma0,a0,C_d,u = BuildMat(params)
    eigval,eigvec = np.linalg.eig(A[1:,1:])
    tau = -1/eigval
    q = tau * ( eigvec[0,:] * np.linalg.inv(eigvec)[:,0] ) / params[1]
    
    return pd.DataFrame([tau,q],index=['d','q'],columns=[1,2,3])