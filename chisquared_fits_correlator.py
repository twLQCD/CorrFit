# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 09:54:49 2020

@author: Admin
"""

import numpy as np
import numpy.linalg as la
import math
import matplotlib.pyplot as plt
import glob
from lmfit.models import ExpressionModel
from lmfit.model import load_modelresult

def eff_mass(CN):
    
    meff = list()
    for i in range(len(CN)-1):
        tmp = -np.log(CN[i+1]/CN[i])
        meff.append(tmp)
        
    return meff


def jackknife_err(testlist):

    #avg = np.mean(testlist)
    avg = np.sum(testlist)/len(testlist)
    val = list()
    jkl = list()
    
    for i in range(len(testlist)):
        
        tmp = testlist[:]
        tmp.pop(i)
        
        ell = np.mean(tmp)
        jkl.append(ell)
        
        sig = (ell-avg)**2
        
        val.append(sig)
     
    
    err = math.sqrt((len(testlist)-1)*np.mean(val))
    
    
    return err, avg, jkl

def linfit(x,y):
    
    tmp1 = 0
    tmp2 = 0
    
    xbar = np.mean(x)
    ybar = np.mean(y)
    
    for i in range(len(x)):
        
        tmp1 = tmp1 + x[i]*y[i]
        tmp2 = tmp2 + x[i]**2
        
    tmp1 = tmp1 - len(x)*xbar*ybar
    tmp2 = tmp2 - len(x)*xbar**2
    
    m = tmp1/tmp2
    
    b = ybar-m*xbar
    
    return m, b

def sigma_mat(Nconf,Nt,Cavg,corr):
    
    eof = Nt*Nconf
    
    sigma = np.zeros((Nt,Nt))
    print(sigma.shape)
    for i in range(Nt):
        tmp1 = corr[i:eof:Nt,1]-Cavg[i]*np.ones((len(Cavg.tolist()),1))
        for j in range(Nt):
            tmp2 = corr[j:eof:Nt,1]-Cavg[j]*np.ones((len(Cavg.tolist()),1))
            sigma[i][j] = np.sum(np.multiply(tmp1,tmp2))

    
    return sigma

def ground_state():
    
    masses = list()
    #this reads in the params of the run
    #list_of_files = glob.glob('./Data/*.dat')           # create the list of file
    list_of_files = glob.glob('*.dat')
    for file_name in list_of_files:
        print(file_name)
        f = open(file_name, 'r')
    #f = open('test.dat', 'r')
        params = f.readline()
        params = list(params.split())
        params = list(map(int,params))
        f.close()
    
    
        Nconf = params[0]
        Nt = params[1]
        eof = Nconf*Nt
        numeffs = Nt-1
    
    
        corr = np.loadtxt(file_name,skiprows=1) # read in correlator data
    
        jack = list()
        jackerr = list()
        jackavg = list()
    
    #it = t-1;
        for i in range(Nt):
        
            tmp = list(corr[i:eof:Nt,1])
        
    
            err, avg, jkl = jackknife_err(tmp)
        
            jack.append(jkl)
            jackerr.append(err)
            jackavg.append(avg)
        
        
            Cavg = np.array(jackavg)
    
            Cerr = np.array(jackerr)
        



        tmin,tmax = 12 , Nt//2
        times = np.arange(tmin,tmax)
        x = times
        mod = ExpressionModel('A1 * exp(-m1*x)')# note 'x' is the default independent variable in lmfit
        # lmfit is smart with the ExpressionModel function to be able to determine
        # the fit parameters, A1 and m1. 


        params = mod.make_params(A1=1., m1=0.3)


        dof = x.shape[0] - len(params)

        out = mod.fit(Cavg[tmin:tmax], params, x=times, weights=Cerr[tmin:tmax])


        print(out.fit_report())

        plt.errorbar(x,Cavg[tmin:tmax], Cerr[tmin:tmax],fmt='b.')
        plt.scatter(x,Cavg[tmin:tmax])
        plt.plot(x, out.best_fit, 'r-')
        plt.show()
        
        
def first_excited_state():
    
    masses = list()
    #this reads in the params of the run
    #list_of_files = glob.glob('./Data/*.dat')           # create the list of file
    list_of_files = glob.glob('*.dat')
    for file_name in list_of_files:
        
        print(file_name)
        f = open(file_name, 'r')
        params = f.readline()
        params = list(params.split())
        params = list(map(int,params))
        f.close()
    
    
        Nconf = params[0]
        Nt = params[1]
        eof = Nconf*Nt
        numeffs = Nt-1
    
    
        corr = np.loadtxt(file_name,skiprows=1) # read in correlator data
    
        jack = list()
        jackerr = list()
        jackavg = list()
    
    #it = t-1;
        for i in range(Nt):
        
            tmp = list(corr[i:eof:Nt,1])
        
    
            err, avg, jkl = jackknife_err(tmp)
        
            jack.append(jkl)
            jackerr.append(err)
            jackavg.append(avg)
        
        
            Cavg = np.array(jackavg)
    
            Cerr = np.array(jackerr)
        


        sigma = sigma_mat(Nconf, Nt, Cavg, corr)
        tmin,tmax = 12, Nt//2
        times = np.arange(tmin,tmax)
        x = times
        mod = ExpressionModel('A1 * exp(-m1*x) + A2 * exp(-m2*x)')
        
        params = mod.make_params(A1=1., A2=1., m1=0.3, m2=0.6)
        
        dof = x.shape[0] - len(params)

        out = mod.fit(Cavg[tmin:tmax], params, x=times, weights=Cerr[tmin:tmax])


        print(out.fit_report())

        plt.errorbar(x,Cavg[tmin:tmax], np.diag(np.sqrt(sigma[tmin:tmax,tmin:tmax])),fmt='b.')
        plt.scatter(x,Cavg[tmin:tmax])
        plt.plot(x, out.best_fit, 'r-')
        plt.show()
        
    

if __name__ == "__main__":
    
    first_excited_state()
    ground_state()
