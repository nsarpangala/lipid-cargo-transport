import numpy as np
from src.analysis.klump_expressions import *
from decimal import *
import math

def nCr(n,r):
    f = math.factorial
    fn=Decimal(f(n))
    fr=Decimal(f(r))
    fn_r=Decimal(f(n-r))
    return float(fn/(fr*fn_r))
    #return f(n) / f(r) / f(n-r)
#print(nCr(4,1),nCr(4,2),nCr(4,3),nCr(4,4))
def prob(n,alp,N):
    return nCr(N,n)*alp**(n)*(1-alp)**(N-n)

def av_run(N,alp,v,eps,pi_ad):
    sums=0
    nor=0
    for ii in range(1,N+1):
        nor+=prob(ii,alp,N)
    for ii in range(1,N+1):
        rn=klump_run(ii,v,pi_ad,eps)
        #if N==4 and v<0.8e-6:
            #print(rn,prob(ii,alp,N)/nor)

        #print(prob(ii,alp,N),rn)
        sums+=(prob(ii,alp,N)/nor)*rn
    return sums


