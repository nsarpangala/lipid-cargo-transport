import os
import matplotlib
def mat_behaviour(show):
    if show==0:
        #matplotlib.use('Agg') #for png
        matplotlib.use('PS') #for vector graphics
import matplotlib.pyplot as plt
from numpy import fft
import numpy as np
from math import sqrt,atan2,acos
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp
from random import random
from matplotlib.pyplot import close
home =os.getenv("HOME")
from random import random
from math import acos
import pandas as pd
import matplotlib.ticker as ticker
from scipy import stats
import pyarrow.parquet as pq

class quick_plot_tools:
    @staticmethod
    def savefig(show,fig,fnam):
        fig.savefig(fnam+".svg",format='svg', dpi=1200)
        #fig.savefig(fnam+".eps",format='eps', dpi=1000)
        fig.savefig(fnam+".png",format='png')
        fig.savefig(fnam+".pdf",format='pdf')
    @staticmethod
    def add_hist(fig_arr,array,range_v,bin_number,*args, **kwargs):
        #not tested - still under development
        cumulative=kwargs.get('cumulative',0)
        label=kwargs.get('label',None)
        fig, ax =fig_arr
        val3,bin02=np.histogram(array,bins=int(bin_number),range=range_v)
        p2=len(bin02)
        bin2=[0.5*(bin02[num]+bin02[num+1]) for num in range(p2-1)]
        val4=np.divide(val3,sum(val3)*1.0)
        cum_val=np.cumsum(val4)
        if cumulative==1:
            ax.plot(bin2,cum_val,ls='-',marker='o',ms=4,label=label)
        else:
            ax.plot(bin2,val4,ls='-',marker='o',ms=4,label=label)
        return fig, ax

class distinct_colors:
    def color(self):
        direc=home+"/microtubule_roadblock_project/support_files/"
        colr=open(direc+"distinct_colors.txt","r").read()
        lines=colr.split()
        return lines
c=distinct_colors()
lines=c.color()

    
        
                       