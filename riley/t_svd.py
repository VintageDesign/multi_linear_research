#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 14:35:23 2020

@author: rhoover
"""

import numpy as np
from scipy import fftpack
from scipy import linalg

a = np.arange(27).reshape(3,3,3) + 1 
b = np.arange(24).reshape(3,2,4) + 1 
c = np.arange(24).reshape(2,3,4) + 1 
##  Tensor functions for evaluation  ##

def t_svd(A):
    n1,n2,n3 = A.shape
    F = np.fft.fft(A,axis=0) # Python ordering (axis=0 implies the "depth" axis)
    tempU = np.zeros((n1,n2,n2),dtype=complex)
    tempS = np.zeros((n1,n2,n3),dtype=complex)
    tempVt = np.zeros((n1,n3,n3),dtype=complex)
    for i in range(0,n1):
        M = F[i,0:]
        U,S,Vt = linalg.svd(M,full_matrices=True)
        tempU[i,0:] = U
        tempS[i,0:] = linalg.diagsvd(S,n2,n3) # need to re-case as matrix #
        tempVt[i,0:] = Vt.T
     
    Ut = np.real(np.fft.ifft(tempU,axis=0))
    St = np.real(np.fft.ifft(tempS,axis=0))
    Vt_t = np.real(np.fft.ifft(tempVt,axis=0))
    return Ut,St,Vt_t

def t_eig(A):
    n1,n2,n3 = A.shape
    if(n2 != n3):
        raise RuntimeError("Faces are not square")
    else:
        F = fftpack.fft(A,axis=0) # Python ordering (axis=0 implies the "depth" axis)
        tempD = np.zeros((n1,n2,n3),dtype=complex)
        tempV = np.zeros((n1,n2,n3),dtype=complex)
        for i in range(0,n1):
            M = F[i,0:]
            Dt,Vt = np.linalg.eig(M)
            ##  need to sort in decending order for consistancy ##
            delta = np.abs(Dt)
            idx = delta.argsort()[::-1]
            Dt = np.diag(Dt[idx])
            Vt = Vt[:,idx]
            tempD[i,0:] = Dt
            tempV[i,0:] = Vt
            DD = np.real(fftpack.ifft(tempD,axis=0))
            VV = np.real(fftpack.ifft(tempV,axis=0))
        return DD,VV

def unfold(A):
    n1,n2,n3 = A.shape
    return A.reshape(n1*n2,n3)

def tcirc(A):
    n1,n2,n3 = A.shape
    Av = unfold(A)
    temp = np.zeros((n1*n2,n1*n3))
    Ar = np.roll(Av,n2,axis=0)
    temp[:,0:n3] = Av
    for i in range(1,n1):
        temp[:,i*n3:(i+1)*n3] = Ar
        Ar = np.roll(Ar,n2,axis=0)
    return temp

def fold(A,r,c,d):
    ##  Expects a r*d x c block matrix  ##
    return A.reshape(r,c,d)

def tprod(A,B):
    n1,n2,n3 = A.shape
    na,n4,nb = B.shape
    #print(B.shape)
    if(n1 != na) or (n3 != n4):
        raise RuntimeError("Incompatable Dimensions")
    else:
        return fold(tcirc(A) @ unfold(B),n1,n2,nb)

def ttran(A):
    n1,n2,n3 = A.shape
    Ac = tcirc(A).T
    Ac = Ac[:,0:n3]
    return fold(Ac,n1,n2,n3)

def tinv(A):
    n1,n2,n3 = A.shape
    Ac = tcirc(A)
    Ac = np.linalg.inv(Ac)
    Ac = Ac[:,0:n3]
    return fold(Ac,n1,n2,n3)

def teye(n):
    I = np.zeros((n,n,n))
    I[0,:,:] = np.eye(n)
    return I
    
def tfronorm(A):
    temp = A*A
    return np.sqrt(np.sum(np.abs(temp)))

## ---------------- Print test routines  -----------------  ##
print(a)

print('Evaluate Tensor Eigenvalue Decomposition')
D,V = t_eig(a)
##  reconstruct a from its eigenvalue decomposition  ##
print(tprod(tprod(V,D),tinv(V)))

print('Evaluate Tensor Singular Value Decomposition')
print(c)
U,S,V = t_svd(c)
##  reconstruct a from its eigenvalue decomposition  ##
print(tprod(tprod(U,S),ttran(V)))

print('Evaluate Tensor Identity')
print(tprod(a,teye(3)))

print('Evaluate Tensor Norm')
print(tfronorm(a))



















