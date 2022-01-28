#this script will collect data of average height as a function of number of bound motors, runlength, average number of bound motors, off rates of motors


import numpy as np
import os
import sys
import copy

project_name="3dtransport"
project_folder=os.getenv("HOME")+"/lipid_cargo_transport_v2/"
sys.path.insert(0,project_folder)
from src.simulation.class_pseudo_single import *
from src.simulation.data_aquisition_module_bose import *
from src.analysis.plot_class_parquet import *
from random import seed, randrange
import pdb
from time import *
import shutil as sh
from joblib import Parallel, delayed
import multiprocessing
import cProfile
from math import ceil,log10
start_time_float=gmtime()
#INPUT FILE
data_folder_name="3dtransport"
res_folder=project_folder+"data_mini/impact_of_rotation_data_v3/"
res_folder=res_folder+"offrate/"
os.makedirs(res_folder)

#simulations={"AS1RHNR_v3":[0,0,1],"AS1D4HNR_v3":[1,0,1],
#             "AS1RHR_v3":[0,1,1],"AS1D4HR_v3":[1,1,1]}
#             "AS16RHNR_v3":[0,0,16],"AS16D4HNR_v3":[1,0,16],
#             "AS16RHR_v3":[0,1,16],"AS16D4HR_v3":[1,1,16]}
Leg_D=["Rigid","Lipid"]
Leg_R=["With rotation","Without rotation"]
simulations={"AS1RHR_v3":[0,1,1],"AS1D4HR_v3":[1,1,1],
             "AS16RHR_v3":[0,1,16],"AS16D4HR_v3":[1,1,16],
            "AS1RHNR_v3":[0,0,1],"AS1D4HNR_v3":[1,0,1],
             "AS16RHNR_v3":[0,0,16],"AS16D4HNR_v3":[1,0,16]}

def collect_average_data(simulations,res_folder):
    df_data=pd.DataFrame(columns=["simname","D","Rotation","N","mean_runlength","run_error","mean_boundmotors","boundmotors_error","mean_lifetime","lifetime_error","mean_offrate","offrate_error"])
    df_sample_sizes=pd.DataFrame(columns=["simname","D","Rotation","N","mean_runlength","run_error","mean_boundmotors","boundmotors_error","mean_lifetime","lifetime_error","mean_offrate","offrate_error"])
    for ele in simulations.keys():
        a=analyse(ele,data_folder_name)

        r=result(ele)
        #r.mkdir(r.resdir)
        #r.curdir=r.resdir+"/../"+folder_name
        #print(r.curdir)
        #r.mkdir(r.curdir)

        a.get_sample_files()
        [mean_r,sem_r,mean_t,sem_t,sample_size_runs]=a.runlength_lifetime()
        [mean_bound,bounderror,sample_size_bound]=a.average_bound_motors()
        [mean_offrate,offrate_error,sample_size_offrate]=a.motor_offrates()
        
        df_data.loc[len(df_data.index)]=[ele,simulations[ele][0],simulations[ele][1],simulations[ele][2],mean_r,sem_r,mean_bound,bounderror,mean_t,sem_t,mean_offrate,offrate_error]
        
        df_sample_sizes.loc[len(df_sample_sizes.index)]=[ele,simulations[ele][0],simulations[ele][1],simulations[ele][2],                                                                                    sample_size_runs,sample_size_runs,sample_size_bound,sample_size_bound,sample_size_runs,sample_size_runs,sample_size_offrate,sample_size_offrate]
    df_data.to_csv(res_folder+"average_values.csv")

    df_sample_sizes.to_csv(res_folder+"average_values_sample_size.csv")
    return df_data,df_sample_sizes
df_sample_sizes,df_data=collect_average_data(simulations,res_folder)                                           


def collect_correlation_variance(simulations,res_folder):
    df_correlation=pd.DataFrame(columns=["simname","D","Rotation","no_bound","mean_fi_fj (pN2)","error_corr (pN2)","correlation_sample_size","mean_variance (pN2)","error_variance (pN2)", "variance_sample_size"])
    for ele in simulations.keys():
        if simulations[ele][2]>1:
            a=analyse(ele,data_folder_name)
            r=result(ele)
            a.get_sample_files()

            for bound_no in [2,3,4]:
                force_arr,correlation_mean,correlation_err,cor_sam_size, variance_mean, error_variance, variance_ssize = a.motor_forces(bound_no)
                correlation_mean=1e24*correlation_mean
                correlation_err=1e24*correlation_err
                
                variance_mean=1e24*variance_mean
                error_variance=1e24*error_variance
                df_correlation.loc[len(df_correlation.index)]=[ele,simulations[ele][0],simulations[ele][1],bound_no,correlation_mean,correlation_err,cor_sam_size, variance_mean, error_variance, variance_ssize]
    df_correlation.to_csv(res_folder+"force_correlations_variance_with_rotation.csv")
    return df_correlation

#res_sub=res_folder+"correlation_variance/"
#os.makedirs(res_sub)
#df_correlation=collect_correlation_variance(simulations,res_sub)



def collect_cargo_heights(simulations,res_folder):
    df_height=pd.DataFrame(columns=["simname","D","Rotation","no_bound","average_height","error","sample_size"])
    for ele in simulations.keys():
        if simulations[ele][2]>1:
            a=analyse(ele,data_folder_name)
            r=result(ele)
            a.get_sample_files()

            for bound_no in [1,2,3,4,5]:
                mean_height,err,sample_size=a.average_height_of_cargo(bound_no)
                df_height.loc[len(df_height.index)]=[ele,simulations[ele][0],simulations[ele][1],bound_no,mean_height,err,sample_size]
    df_height.to_csv(res_folder+"average_height_with_rotation.csv")

#collect_cargo_heights(simulations,res_folder)


def force_distributions(simulations,res_folder):
    df_force=pd.DataFrame()
    bound_no_list=[1,3]
    for bound_no in bound_no_list:
        for ele in simulations.keys():
            a=analyse(ele,data_folder_name)
            r=result(ele)
            a.get_sample_files()
            force_arr,correlation_mean,correlation_err,cor_sam_size=a.motor_forces(bound_no)
            print(len(force_arr))
            df_force[ele+str(bound_no)]=force_arr
    df_force.to_csv(res_folder+'force_distributions/force_distributions.csv')

#simulations_N16_only={"AS16RHR_v3":[0,1,16],"AS16D4HR_v3":[1,1,16],
#             "AS16RHNR_v3":[0,0,16],"AS16D4HNR_v3":[1,0,16]}
#force_distributions(simulations_N16_only,res_folder)
