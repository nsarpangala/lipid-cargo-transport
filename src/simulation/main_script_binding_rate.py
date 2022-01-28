#!/usr/bin/python3
#Shree Durga Devi Namah#
#Project: Study of the effect of lipid membrane on cargo transport.
#Niranjan S(nsarpangala@ucmerced.edu), Dr. David Quint, Prof. Ajay Gopinathhan
#Prof. Ajay Gopinathan Group, Physics Department, UC Merced

import sys
import os
project_name="3dtransport"
project_folder=os.getenv("HOME")+"/lipid_cargo_transport_v2/"
sys.path.insert(0,project_folder)
from src.simulation.class_pseudo_single import *
from src.simulation.data_aquisition_module_bose import *
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
f_in=project_folder+"inputs/"+str(sys.argv[1])
#INSTANTIATE PARENT
main=parent(f_in)
d=data_aq(main)
d.mk_main_dir()
#COPY FILES TO DIR

d.cp_uniqdir(f_in)
#print(src.simulation.class_pseudo_single.__file__)
path=project_folder+"src/simulation/"
d.cp_uniqdir(path+"class_pseudo_single.py")
d.cp_uniqdir(path+"data_aquisition_module_bose.py")
d.cp_uniqdir(path+"class_parameter_reader.py")
d.cp_uniqdir(str(sys.argv[0]))
sh.copy(f_in,d.uniqdir+str(main.p.simname)+".txt")

#INSTANTIATE CARGO MOTOR
mc=cargo_motor(f_in)

@profile
def run(sample):
    start_time_floats=gmtime()
    #GENERATE AND SAVE SEED
    if job_nature==1 or job_nature==2:
        print("-------------------------")
        print("Production Run")
        sd=randrange(int(2**32 - 1))
        seed(sd)
        np.random.seed(sd)
        d.save_var("seed"+str(sample),[sd])
    if job_nature==0:
        print("-------------------------")
        print("Test/Optimization Run")
        sd=100
        seed(sd)
        np.random.seed(sd)
    if job_nature==3:
        print("-------------------------")
        print("Test/Optimization Run: multi core")
        seed(int(sample))
        np.random.seed(int(sample))
    #RESET
    mc.reset()
    mc
    mc.initialize()
    bind_time=mc.initial_engagement_lipid_brate_calculation()
    d.save_var("bind"+str(sample),[bind_time])
    end_time_floats=gmtime()
    #d.create_logfile(start_time_floats,end_time_floats,sample)
    
job_nature=int(sys.argv[2])

if job_nature==0: #for optimization and testing purposes
    #d.mk_sample_dir(0)
    run(0)
    #cProfile.run('run(0)')

if job_nature==1:   #for merced cluster production run
    tsk_id=int(os.environ.get('SLURM_ARRAY_TASK_ID'))
    sam_p=int(sys.argv[3])
    for sample in range(sam_p*(tsk_id-1),sam_p*(tsk_id)):
        run(sample)
#     proc_id=int(os.environ.get('SLURM_PROCID'))
#     ntasks=int(os.environ.get('SLURM_NTASKS'))
#     sam_p=int(main.p.sampleSiz/ntasks)
#     for sample in range(sam_p*proc_id,sam_p*(proc_id+1)):
#         run(sample)




def pack(sample): #useful for job nature 2 runs
    #d.mk_sample_dir(sample)
    run(sample)
if job_nature==2: #for Burrata/Biotheory/Softbio/Chandra/Fermi production runs
    num_cores = int(sys.argv[3])
    Parallel(n_jobs=num_cores)(delayed(pack)(sample) for sample in range(int(main.p.sampleSiz)))
if job_nature==3: #for Burrata/Biotheory/Softbio/Chandra/Fermi production test optimization
    num_cores = int(sys.argv[3])
    Parallel(n_jobs=num_cores)(delayed(pack)(sample) for sample in range(int(main.p.sampleSiz)))

end_time_float=gmtime()
d.create_logfile(start_time_float,end_time_float,"final")

