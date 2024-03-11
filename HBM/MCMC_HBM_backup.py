# -*- coding: utf-8 -*-


"""
Created on Tue Nov  5 15:27:57 2019
@author: Luyuting
"""


import numpy as np
import matplotlib.pyplot as plt
import scipy.special as ss
import matplotlib.gridspec as gridspec
import os
import glob
import pandas as pd
import scipy.io as sio
import corner
import emcee
from numpy.linalg import inv,det
from multiprocessing import Pool
import time
import _thread


os.environ["OMP_NUM_THREADS"] = "1"


#parameter info
"""
kappa-concentration
mu-location
J- upper limit of j
i- inclination
WARNING: DON'T USE i AS INDEX IN THIS CODE
"""


n = 0
    
def Phi(J,kappa,mu):    
    I1 = ss.iv(1,kappa)
    phi = I1*(np.pi*np.sin(mu)+2*np.cos(mu))/4
    for j in range(2,J+1):
        Ij = ss.iv(j,kappa)
        phi = phi + Ij*(j*np.sin(j*(np.pi*0.5-mu))-np.cos(mu*j))/(j**2-1)
    return phi
    
def P_alpha(kappa,mu,i):

    ##i,mu radian

    I0 = ss.iv(0,kappa)
    phi = Phi(16,kappa,mu)    
    C = 1/(I0+2*phi)

    
    y = np.cos(i*np.pi/180)
    term2 = np.exp(kappa*y*np.cos(mu)+kappa*np.sqrt(1-y**2)*np.sin(mu))
    P_modified = C*term2
    P_modified[np.isnan(P_modified)] = 0

    return P_modified

def Likelihood_single(kappa,mu,i):
    K = len(i)
    if 0 <= mu <= np.pi/2 and 0 < kappa:
        P_modified = P_alpha(kappa,mu,i)
        for item in P_modified:
            if np.isnan(item):
                print('Problem from P_modified:NaN')   
                
        
        L = sum(P_modified)/K
    else:
        L = 0
    if L <0:
        print('Problem from L:negative problem')
    elif np.isnan(L):
        print('Problem from L:NaN problem')
    return L




def likelihood(theta):
    number = 20  
    kappa, mu = theta
    L_alpha = 1
    for s in range(0, number):
        i = simu_i[s,:]
        L = Likelihood_single(kappa,mu,i)
        L_alpha = L_alpha*L
    if L_alpha <0: 
        print('Problem from L_alpha:negative log')
    elif np.isnan(L_alpha):
        print('Problem from L_alpha:L_alpha NaN')

      
    return L_alpha




def prior_kappa(kappa):
    gamma = 50
    p = (gamma**2/(gamma**2+kappa**2))/(np.pi*gamma)

    
    return p


def prior(theta):   
    kappa, mu = theta
    if 0 <= mu <= np.pi/2 and 0 < kappa <700:
        pmu = np.sin(mu)
        pkappa = prior_kappa(kappa)
        p = pmu*pkappa
        if p<0:
            print(p)
        return p
        
    return 0

def log_probability(theta):
    lp = prior(theta)
    lh = likelihood(theta)
    if lp == 0 and np.isinf(lh):
        pdf = 0
    else:
        pdf = lp * lh
    if pdf<0:
        print('problem from pdf: negative in log')
    if np.isinf(pdf):
        print('possible problem from pdf:Inf')
    if np.isnan(pdf):
        print('problem from pdf:NaN')    
    log_pdf = np.log(pdf)
    if np.isnan(log_pdf):
        print('problem from log_prob')
        print(lh)
        print(lp)
        print(pdf)
    return log_pdf

def distribution(flat_samples,median_i,m):
    kappa,mu = flat_samples[:,0],flat_samples[:,1]
    dis = []
    angle = np.linspace(0,90.000000001,1000)*np.pi/180
    degree_angle = angle*180/(np.pi)
    y = np.cos(angle)
    n = 0
    for item in y:
        n = n+1
        if n%100==0:
            print('distribution %d/100 done'%(n/100))
        I0 = ss.iv(0,kappa)
        phi = Phi(16,kappa,mu)
        C = 1/(I0+2*phi)
        P_modified = C*np.exp(kappa*item*np.cos(mu)+kappa*np.sqrt(1-item**2)*np.sin(mu))
        dis.append(sum(P_modified))
    plt.subplot(211)
    plt.plot(y,dis)
    plt.subplot(212)
    plt.plot(degree_angle, dis)
    plt.axvline(degree_angle[np.argmax(dis)], color='k', linestyle='--')
    plt.text(degree_angle[np.argmax(dis)], max(dis)/2, '%.2f'%(degree_angle[np.argmax(dis)]),color = 'k', fontsize=22)
    plt.axvline(median_i, color='r', linestyle='--')
    plt.text(median_i, max(dis), '%.2f'%(median_i),color = 'r', fontsize=22)
    figpath2 = r'../distribution.png'%(m+1)
    plt.savefig(figpath2)
    plt.close()
    return 



median = [0,25,50,75,90]
for m in range(0,5):
    #80 degree
    
    median_i = median[m]
    number = 20
    deg_i = np.random.normal(median_i, 1, number)
    simu_i = np.zeros((number,10000))
    for s in range(0,number):
        std = np.random.normal(1, 1, 1)
        pdf1 = np.random.normal(deg_i[s], std, 10000)
        simu_i[s,:] = pdf1    
    
    nwalkers = 100
    ndim = 2
    startkappa = 50 + np.random.randn(nwalkers)*0.01*50
    startpi = (57.2*np.pi)/180+np.random.randn(nwalkers)*(np.pi/2)*0.01
    pos = np.zeros((nwalkers,ndim))
    pos[:,0] = startkappa
    pos[:,1] = startpi
    nsteps = 400
    
    with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool=pool)
        sampler.run_mcmc(pos, nsteps, progress=True)
        
#=================
    burnin = 200
    
    ###ploting FIG1
    fig, axes = plt.subplots(2, figsize=(10, 7), sharex=True)
    samples = sampler.get_chain()
    labels = ["kappa", "mu"]
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])

        #ax.yaxis.set_label_coords(-0.1, 0.5)
    
    axes[-1].set_xlabel("step number")
    figpath = r'../MCMC.png'%(m+1)
    plt.savefig(figpath)
    plt.close()    
    
    ###plotting FIG2 
    flat_samples = np.zeros((nwalkers*nsteps,ndim))
    kappa_result = []
    mu_result = []
    for s in range(nwalkers):
        kappa_result.append(samples[burnin:, s, 0])
        mu_result.append(samples[burnin:, s, 1])
    flat_samples[:,0]= kappa_result
    flat_samples[:,1]= mu_result

    #flat_samples = sampler.get_chain(flat=True)
    labels = ["kappa", "mu"]
    fig = corner.corner(
        flat_samples, labels=labels
    )
    figpath = r'../contour.png'%(m+1)
    plt.savefig(figpath)
    plt.close()
    
    ###plotting FIG3
    distribution(flat_samples,median_i,m)

    ####save csv
    data = pd.DataFrame({'kappa':flat_samples[:,0],'mu':flat_samples[:,1]})
    csvpath = r'../data.csv'%(m+1)
    data.to_csv(csvpath,index=False,sep=',')
    

        
        
        
        
        
        