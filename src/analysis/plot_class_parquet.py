import os
import matplotlib
def mat_behaviour(show):
    if show==0:
        #matplotlib.use('Agg') #for png
        matplotlib.use('PS') #for vector graphics
import matplotlib.pyplot as plt
from numpy import fft
import numpy as np
from src.simulation.class_parameter_reader import *
from math import sqrt,atan2,acos
from src.simulation.class_pseudo_single import *
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp
from random import random
from matplotlib.pyplot import close
#from joblib import Parallel, delayed
#import multiprocessing
#import seaborn as sns; sns.set()
home =os.getenv("HOME")
from random import random
from math import acos
import pandas as pd
import matplotlib.ticker as ticker
from scipy import stats
import pyarrow.parquet as pq

# class quick_plot_tools:
#     @staticmethod
#     def savefig(show,fig,fnam):
#         fig.savefig(fnam+".svg",format='svg', dpi=1200)
#         #fig.savefig(fnam+".eps",format='eps', dpi=1000)
#         #fig.savefig(fnam+".png",format='png')
#         fig.savefig(fnam+".pdf",format='pdf')
#     @staticmethod
#     def add_hist(fig_arr,array,range_v,bin_number,*args, **kwargs):
#         #not tested - still under development
#         cumulative=kwargs.get('cumulative',0)
#         label=kwargs.get('label',None)
#         fig, ax =fig_arr
#         val3,bin02=np.histogram(array,bins=int(bin_number),range=range_v)
#         p2=len(bin02)
#         bin2=[0.5*(bin02[num]+bin02[num+1]) for num in range(p2-1)]
#         val4=np.divide(val3,sum(val3)*1.0)
#         cum_val=np.cumsum(val4)
#         #ax.bar(bin2,val4,width=1000,label=labels)
#         if cumulative==1:
#             ax.plot(bin2,cum_val,ls='-',marker='o',ms=4,label=label)
#         else:
#             ax.plot(bin2,val4,ls='-',marker='o',ms=4,label=label)
#         return fig, ax

# class distinct_colors:
#     def color(self):
#         direc=home+"/3dtransport/support_files/"
#         colr=open(direc+"distinct_colors.txt","r").read()
#         lines=colr.split()
#         return lines
# c=distinct_colors()
# lines=c.color()
class result:
    def __init__(self,sim):
        self.proj_res=home+"/data/results/3dtransport/"
        self.resdir=self.proj_res+str(sim)
        self.curdir="empty"
    def mkdir(self,direc):
        if os.path.isdir(direc)*1 ==0:
            os.mkdir(direc)
class analyse:
    def __init__(self,sim,data_directory):
        self.sim=sim
        self.proj_data=home+"/data/"+data_directory+"/"
        self.sim_datadir=self.proj_data+self.sim
        self.in_file_path=self.sim_datadir+"/"+str(sim)+".txt"
        self.p=Params(self.in_file_path)
        self.cm=cargo_motor(self.in_file_path)
        self.sample_files=[]
        #print("analyse class initiated for simulation:%s"%(sim))
        
    def get_sample_files(self):
        for sample in range(int(self.p.sampleSiz)):
            spath=self.sim_datadir+"/"+str(sample)+".parquet"
            if os.path.isfile(spath):
                self.sample_files.append(spath)
        #print("Number of sample files: %i"%(len(self.sample_files)))

    def runlength_lifetime(self):
        '''
        returns runlength and lifetime for the given simulation
        '''
        run=[]
        time=[]
        for sfile in self.sample_files:
            dp = pq.read_table(sfile,columns=["time","cx"])
            df=dp.to_pandas()
            run.append(df["cx"].iloc[-1]) #last element of column cx dataframe df
            time.append(df["time"].iloc[-1])
        run=np.array(run)
        mean_r=np.mean(run)
        sem_r=stats.sem(run)
        
        time=np.array(time)
        mean_t=np.mean(time)
        sem_t=stats.sem(time)
        
        #vel=mean_r/mean_t
        #sem_vel=vel*np.sqrt((sem_r/mean_r)**2+(sem_t/mean_t)**2)
        return [mean_r,sem_r,mean_t,sem_t,len(run)]
    
    def step_distribution(self,delta_n,**kwargs):
        '''
        returns array of microscopic steps and array of change in number of bound motors
        '''
        bound_on=kwargs.get('bound_on', None)
        bound_mot=kwargs.get('bound_mot', None)
        step_arr=np.empty(0)
        if bound_on==1:
            bound_arr=np.empty(0)
        for sfile in self.sample_files:
            traject_list=['cx']
            col_list=traject_list
            if bound_on is not None:
                bind_list=["bind"+str(i) for i in range(int(self.p.N))]
                col_list=traject_list+bind_list
            dp = pq.read_table(sfile,columns=col_list)
            df=dp.to_pandas()
            df.drop(df.tail(1).index,inplace=True)
            cx=np.array(df["cx"])
            length=len(cx)
            ls=length-delta_n
            if length>delta_n:
                arr=cx[delta_n:length]-cx[:ls]
                step_arr=np.concatenate((step_arr, arr), axis=None)
                if bound_on ==1:
                    df['tot_bound']=df[bind_list].sum(axis=1)
                    tot_bound=np.array(df["tot_bound"])
                    tot_bound.astype('int')
                    idetify_location=tot_bound==bound_mot #Replaces locations in no_of_bound_mot with value==bound_mot
                    idetify_location=idetify_location.astype('float')
                    idetify_location[idetify_location == 0] = float('nan')
                    cargo_x_=cx*idetify_location

                    steps=cargo_x_[delta_n:length]-cargo_x_[:length-delta_n]
                    steps = steps[~np.isnan(steps)]
                    bound_arr=np.concatenate((bound_arr, steps), axis=None)
        if bound_on==1:
            return [step_arr,bound_arr]
        else:
            return step_arr
    
    
    def pause_times(self,**kwargs):
        delta_n=kwargs.get('delta_n', None)
        pause_time_combined=np.empty(0)
        for sfile in self.sample_files:
            traject_list=['cx']
            col_list=traject_list
            hold_list=["Hold"+str(i) for i in range(self.p.N)]
            col_list=traject_list+hold_list
            dp = pq.read_table(sfile,columns=col_list)
            df=dp.to_pandas()
            df['tot_hold']=df[hold_list].sum(axis=1)
            import scipy.ndimage as sp
            tot_bound=np.array(df['tot_hold'])
            labeled_arr,num_pause=sp.measurements.label(tot_bound, structure=None, output=None)
            pause_times=np.array([len(labeled_arr[labeled_arr==num]) for num in range(1,num_pause+1)])
            pause_time_combined=np.concatenate((pause_time_combined, pause_times), axis=None)
        return pause_time_combined*self.p.samtime*delta_n
    
    def hold_data(self):
        run=[]
        for sfile in self.sample_files:
            dp = pq.read_table(sfile,columns=["Hold0","Hold1","Hold2","Hold3","bind0","bind1","bind2","bind3"])
            df=dp.to_pandas()
            print(df)
                        
    def motor_location_array(self,**kwargs):
        '''
        returns array of motor anchor, which='anchor',and head, which='head', of slow, mot_type='slow' and fast,'fast' motors with respect to the center of mass of the cargo.
        '''
        which=kwargs.get('which', None)
        mot_type=kwargs.get('mot_type', None)
        
        traject_list=['cx']
        col_list=traject_list
        mot_dict={'slow':3.5e-7,'fast':8e-7}
        import re
        mot_loc_arr=np.empty(0)
        for sfile in self.sample_files:
            mot_vel_file = re.sub('\.parquet$', '', sfile)
            motvelocity=np.loadtxt(mot_vel_file+"motor_velocity.txt")
            req_list=[]
            
            for i in range(int(self.p.N)):
                if abs(motvelocity[i]-mot_dict[mot_type])<1e-9:
                    req_list.append(i)
            #bind_list=["bind"+str(i) for i in req_list]
            which_list=[which+str(i)+"x" for i in req_list]
            col_list=traject_list+which_list
            
            dp = pq.read_table(sfile,columns=col_list)
            df=dp.to_pandas()
            df.drop(df.tail(1).index,inplace=True)
            
            for i in req_list:
                df["rel_"+which+str(i)]=df[which+str(i)+"x"]-df["cx"]
            req_list_arr=["rel_"+which+str(i) for i in req_list]
            arr=np.array(df[req_list_arr])
            flat=arr.flatten()
            flat=flat[~np.isnan(flat)]
            mot_loc_arr=np.concatenate((mot_loc_arr, flat), axis=None)
        return mot_loc_arr
    
    def motor_forces(self,bound_number,**kwargs):
        '''
        returns array of motor forces, mot_type='slow' and fast,'fast' motors.
        '''
        #which=kwargs.get('which', None)
        #mot_type=kwargs.get('mot_type', None)
        mot_force_arr=np.empty(0)
        traject_list=['cx','cy','cz']
        Dimen=["x","y","z"]
        Var=["head","anchor"]
        bind_list=[]
        for i in range(int(self.p.N)):
            bind_list.append("bind"+str(i))
        col_list=traject_list+bind_list
        for vari in Var:
                for i in range(int(self.p.N)):
                    for dim in Dimen:
                        col_list.append(vari+str(i)+dim)
        mot_force_mag_arr=np.empty(0)
        force_corr_arr=np.empty(0)
        variance_arr=np.empty(0)
        
        force_mag_arr=[]
        for i in range(int(self.p.N)):
            force_mag_arr.append("force_mag"+str(i))
            
            
        pair_count_list=[]
        pair_count=0
        for ii in range(self.p.N):
                for jj in range(ii+1,self.p.N):
                    if ii!=jj:
                        pair_count_list.append(str(pair_count))
                        pair_count+=1
                    
                    
        for sfile in self.sample_files:
            
            dp = pq.read_table(sfile,columns=col_list)
            df_all=dp.to_pandas()
            df_all.drop(df_all.tail(1).index,inplace=True)
            df_all['total_bound'] = df_all[bind_list].sum(axis=1)
            df_all=df_all.astype({'total_bound': 'int'})
            df=df_all.loc[df_all['total_bound']==bound_number]
            for i in range(int(self.p.N)):
                for dim in Dimen:
                    if dim=="x":
                        distance=(df["head"+str(i)+dim]-df["anchor"+str(i)+dim])**2
                    else:
                        distance+=(df["head"+str(i)+dim]-df["anchor"+str(i)+dim])**2
                    distance_sqrt=np.sqrt(distance)
                df["distance"+str(i)]=distance_sqrt
                df["force_mag"+str(i)]=(self.p.kmot)*(distance_sqrt-self.p.Lmot)
                df["force_mag"+str(i)].loc[df["distance"+str(i)]<=self.p.Lmot]=0
                #print(df["force_mag"+str(i)])
                #for dim in Dimen:
                dim="x"
                df["force"+str(i)+dim]=-df["force_mag"+str(i)]*(df["head"+str(i)+dim]-df["anchor"+str(i)+dim])/df["distance"+str(i)] #value is negative for hindring and positive for assistive.
                #print(df["force"+str(i)+dim])
                #df["force_mag"+str(i)]=np.sqrt((df["force"+str(i)+"x"])**2+(df["force"+str(i)+"y"])**2+(df["force"+str(i)+"z"])**2)
                #df["force_mag"+str(i)] = df["force_mag"+str(i)].fillna(0)
                df["force_mag"+str(i)]=np.sign(df["force"+str(i)+"x"])*df["force_mag"+str(i)]
            mag_arr=np.array(df[force_mag_arr])
            
            flat=mag_arr.flatten()
            flat=flat[~np.isnan(flat)]
            mot_force_mag_arr=np.concatenate((mot_force_mag_arr, flat), axis=None)
            
            #variance 
            df['variance']=df[force_mag_arr].var(axis=1,ddof=0)
            variance_arr_sfile=np.array(df['variance'])
            variance_arr=np.concatenate((variance_arr, variance_arr_sfile), axis=None)
            
            #correlations
            pair_count=0
            for ii in range(self.p.N):
                for jj in range(ii+1,self.p.N):
                    if ii!=jj:
                        df[str(pair_count)]=np.array(df["force"+str(ii)+"x"])*np.array(df["force"+str(jj)+"x"])
                        pair_count+=1
                    
                    
            cor_arr=np.array(df[pair_count_list])
            flat_cor=cor_arr.flatten()
            flat_cor=flat_cor[~np.isnan(flat_cor)]
            force_corr_arr=np.concatenate((force_corr_arr, flat_cor), axis=None)
        #print(selected_mag_vals)
        #print(len(mot_force_mag_arr))
        if len(mot_force_mag_arr)>=10000:
            cut_len=10000
        else:
            cut_len=len(mot_force_mag_arr)
        selected_mag_vals=np.random.choice(mot_force_mag_arr, cut_len, replace=False)
        
        #variance measurements
        if len(variance_arr)>=10000:
            cut_len=10000
        else:
            cut_len=len(variance_arr)
        selected_variance_vals=np.random.choice(variance_arr, cut_len, replace=False)
        variance=np.mean(selected_variance_vals)
        var_err=stats.sem(selected_variance_vals)
        variance_sample_size=cut_len
        
        if len(force_corr_arr)>=10000:
            cut_len=10000
            selected_corr_vals=np.random.choice(force_corr_arr, cut_len, replace=False)
        else:
            cut_len=len(force_corr_arr)
            selected_corr_vals=force_corr_arr
        
        correlation=np.mean(selected_corr_vals)
        corr_err=stats.sem(selected_corr_vals)
        return selected_mag_vals,correlation,corr_err,len(selected_corr_vals), variance, var_err, variance_sample_size 
    
    def motor_offrates(self,**kwargs):
        '''
        returns array of motor off rates
        '''
        sizes=np.empty(0)
        col_list=[]
        from scipy.ndimage import measurements
        for i in range(self.p.N):
            col_list.append("bind"+str(i))
        for sfile in self.sample_files:

            dp = pq.read_table(sfile,columns=col_list)
            df=dp.to_pandas()
            df.drop(df.tail(1).index,inplace=True)
            for col in col_list:
                #req_list_arr.append("bind"+str(i))
                arr=df[col]
                lw, num = measurements.label(arr)
                sizes_min = measurements.sum(arr, lw, index=np.arange(1,lw.max() + 1))
                sizes=np.concatenate((sizes, sizes_min), axis=None)
        
        if len(sizes)>=580:
            cut_len=580
        else:
            cut_len=len(sizes)
        selected_sizes=np.random.choice(sizes, cut_len, replace=False) 
        mean_time=np.mean(selected_sizes)*self.p.samtime
        mean_rate=1/mean_time
        from scipy import stats
        mean_rate_err=stats.sem(selected_sizes)*self.p.samtime/(mean_time**2)
        return mean_rate,mean_rate_err,len(selected_sizes)
    
    
    def fraction_of_time(self,**kwargs):
        '''
        returns fraction of time slow or fast motor is bound when just one motor is bound
        '''
        mot_type=kwargs.get('mot_type', None)
        no_bound=kwargs.get('no_bound', None)
        col_list=[]
        mot_dict={'slow':3.5e-7,'fast':8e-7}
        import re
        mast_tot=0
        mast_spec=0
        grand_tot=0
        grand_bound=0
        for i in range(int(self.p.N)):
            col_list.append("bind"+str(i))
        for sfile in self.sample_files:
            mot_vel_file = re.sub('\.parquet$', '', sfile)
            motvelocity=np.loadtxt(mot_vel_file+"motor_velocity.txt")
            req_list=[]
            for i in range(int(self.p.N)):
                if abs(motvelocity[i]-mot_dict[mot_type])<1e-9:
                    req_list.append("bind"+str(i))
            dp = pq.read_table(sfile,columns=col_list)
            df=dp.to_pandas()
            df.drop(df.tail(1).index,inplace=True)
            df['bound_no'] = df[col_list].sum(axis=1)
            df['specific_type_bound'] = df[req_list].sum(axis=1)
            boun=np.array(df['bound_no'])
            spec=np.array(df['specific_type_bound'])
            boun=boun.astype(int)
            spec=spec.astype(int)
            tot=np.sum(boun[boun==no_bound])
            spec_no=np.sum(spec[boun==no_bound])
            mast_tot+=tot
            mast_spec+=spec_no
            grand_tot+=len(boun)
            grand_bound+=len(boun[boun==no_bound])
        print(mast_tot)
        return mast_spec/mast_tot,np.sqrt(mast_spec)/mast_tot,(grand_bound/grand_tot)
        

    def average_bound_motors(self,**kwargs):
            '''
            returns average number of bound motors
            '''
            col_list=["bind"+str(i) for i in range(int(self.p.N))]
            bound_arr=np.array([])
            for sfile in self.sample_files:
                dp = pq.read_table(sfile,columns=col_list)
                df=dp.to_pandas()
                df.drop(df.tail(1).index,inplace=True)
                df['total_bound'] = df.sum(axis=1)
                tot_bound=np.array(df['total_bound'])
                bound_arr = np.concatenate((bound_arr, tot_bound))    
            mean=np.mean(bound_arr)
            error=stats.sem(bound_arr)
            return [mean,error,len(bound_arr)]
        
        
    def average_height_of_cargo(self,bound_number):
        '''
        returns average_height_of_cargo
        '''

        mot_force_arr=np.empty(0)
        traject_list=['cx','cy','cz']

        bind_list=[]
        for i in range(int(self.p.N)):
            bind_list.append("bind"+str(i))
        col_list=traject_list+bind_list

        height_array=np.empty(0)
                    
                    
        for sfile in self.sample_files:
            
            dp = pq.read_table(sfile,columns=col_list)
            df_all=dp.to_pandas()
            df_all.drop(df_all.tail(1).index,inplace=True)
            df_all['total_bound'] = df_all[bind_list].sum(axis=1)
            df_all=df_all.astype({'total_bound': 'int'})
            df=df_all.loc[df_all['total_bound']==bound_number]
            
            h_sam=np.sqrt(df["cy"]*df["cy"]+df["cz"]*df["cz"])
            height_array=np.concatenate((height_array, h_sam), axis=None)
        #print(selected_mag_vals)
        #print(len(mot_force_mag_arr))
        
        if len(height_array)>=10000:
            cut_len=10000
            selected_h_vals=np.random.choice(height_array, cut_len, replace=False)
        else:
            cut_len=len(height_array)
            selected_h_vals=height_array
        
        mean=np.mean(selected_h_vals)
        error=stats.sem(selected_h_vals)
        return [mean,error,len(selected_h_vals)]
    
    
    
    def motor_radial_normal_forces(self,file_no,**kwargs):
        '''
        This function will return a data frame with all motor forces, motor positions, tangential and normal components of motor forces for given number of bound motors and sample number.
        '''
        #which=kwargs.get('which', None)
        #mot_type=kwargs.get('mot_type', None)
        bound_number=kwargs.get('bound_number', None)
        time_max=kwargs.get('time_max', None)
        mot_force_arr=np.empty(0)
        traject_list=['time','cx','cy','cz']
        Dimen=["x","y","z"]
        Var=["head","anchor"]
        bind_list=[]
        for i in range(int(self.p.N)):
            bind_list.append("bind"+str(i))
        col_list=traject_list+bind_list
        for vari in Var:
                for i in range(int(self.p.N)):
                    for dim in Dimen:
                        col_list.append(vari+str(i)+dim)
                    
                    
        for sfile in [self.sample_files[file_no]]:
            
            dp = pq.read_table(sfile,columns=col_list)
            df_all=dp.to_pandas()
            df_all.drop(df_all.tail(1).index,inplace=True)
            df_all['total_bound'] = df_all[bind_list].sum(axis=1)
            df_all=df_all.astype({'total_bound': 'int'})
            if bound_number is not None:
                df=df_all.loc[df_all['total_bound']==bound_number]
            else:
                df=df_all.loc[df_all['time']<time_max]
            for i in range(int(self.p.N)):
                for dim in Dimen:
                    if dim=="x":
                        distance=(df["head"+str(i)+dim]-df["anchor"+str(i)+dim])**2
                    else:
                        distance+=(df["head"+str(i)+dim]-df["anchor"+str(i)+dim])**2
                    distance_sqrt=np.sqrt(distance)
                df["distance"+str(i)]=distance_sqrt
                df["force_mag"+str(i)]=(self.p.kmot)*(distance_sqrt-self.p.Lmot)
                df["force_mag"+str(i)].loc[df["distance"+str(i)]<=self.p.Lmot]=0
                #print(df["force_mag"+str(i)])
                for dim in Dimen:
                    df["force"+str(i)+dim]=df["force_mag"+str(i)]*(df["head"+str(i)+dim]-df["anchor"+str(i)+dim])/df["distance"+str(i)] #value is negative for hindring and positive for assistive.
                    df["radial_unit_vec"+str(i)+dim]=(df["anchor"+str(i)+dim]-df["c"+dim])/self.p.radius
                for dim in Dimen:
                    if dim=="x":
                        df['radial_component'+str(i)]=df["force"+str(i)+dim]*df["radial_unit_vec"+str(i)+dim]
                    else:
                        df['radial_component'+str(i)]+=df["force"+str(i)+dim]*df["radial_unit_vec"+str(i)+dim]
                for dim in Dimen:
                    df["tangential_component"+str(i)+dim]=df["force"+str(i)+dim]-df['radial_component'+str(i)]*df["radial_unit_vec"+str(i)+dim]
                    
                    if dim=="x":
                        tan_compon=(df["tangential_component"+str(i)+dim])**2
                    else:
                        tan_compon+=(df["tangential_component"+str(i)+dim])**2
                df["tangential_component"+str(i)]=np.sqrt(tan_compon)
                
                df["force_mag"+str(i)]=np.sign(df["force"+str(i)+"x"])*df["force_mag"+str(i)]
                
        return df
    
    
    def theta_phi_compute(self):
        '''
        compute theta and phi arrays and plot the histogram
        '''
        #which=kwargs.get('which', None)
        #mot_type=kwargs.get('mot_type', None)
        mot_force_arr=np.empty(0)
        traject_list=['time','cx','cy','cz']
        Dimen=["x","y","z"]
        Var=["head","anchor"]
        bind_list=[]
        for i in range(int(self.p.N)):
            bind_list.append("bind"+str(i))
        col_list=traject_list+bind_list
        for vari in Var:
                for i in range(int(self.p.N)):
                    for dim in Dimen:
                        col_list.append(vari+str(i)+dim)                  
        theta_array=np.empty(0)
        phi_array=np.empty(0)
        for sfile in self.sample_files:
            
            dp = pq.read_table(sfile,columns=col_list)
            df_all=dp.to_pandas()
            df_all.drop(df_all.tail(1).index,inplace=True)
            df=df_all
            #df_all['total_bound'] = df_all[bind_list].sum(axis=1)
            #df_all=df_all.astype({'total_bound': 'int'})
            #df=df_all.loc[df_all['total_bound']==bound_number]
            cargo_orien=np.arctan2(df['cz'],df['cy'])
            df['cyprime']=np.sin(cargo_orien)*df['cy']-np.cos(cargo_orien)*df['cz']
            df['czprime']=np.cos(cargo_orien)*df['cy']+np.sin(cargo_orien)*df['cz']
            df['cxprime']=df['cx']
            for i in range(int(self.p.N)):
                df_dum=pd.DataFrame()
                df_dum['anchor_y']=np.sin(cargo_orien)*df["anchor"+str(i)+"y"]-np.cos(cargo_orien)*df["anchor"+str(i)+"z"]
                df_dum['anchor_z']=np.cos(cargo_orien)*df["anchor"+str(i)+"y"]+np.sin(cargo_orien)*df["anchor"+str(i)+"z"]
                df_dum['anchor_x']=df["anchor"+str(i)+"x"]
                for dim in Dimen:
                    df[dim+str(i)]=np.array(df_dum["anchor_"+dim]-df["c"+dim+'prime'])
                the=np.arccos(df["z"+str(i)]/self.p.radius)
                phi=np.arctan2(df["y"+str(i)],df["x"+str(i)])
                theta_array=np.concatenate((theta_array, the), axis=None)
                phi_array=np.concatenate((phi_array, phi), axis=None)
        
        cut_len=100000       
        selected_theta=np.random.choice(theta_array, cut_len, replace=False)
        selected_phi=np.random.choice(phi_array, cut_len, replace=False)   
        return selected_theta,selected_phi
                
                
                
                
                
                


        
