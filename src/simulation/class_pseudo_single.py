from os import getcwd, path, rename, mkdir, path
import numpy as np
from numpy.linalg import norm
import sys
from src.simulation.class_parameter_reader import *
from math import sqrt, pi, sin, cos, exp, acos, atan, atan2
from random import random, seed, normalvariate
from pdb import *
from numba import jit
import time
from math import ceil,log10
try:
    profile
except NameError:
    def profile(x):
        return x

class parent:
    def __init__(self,f_in):
        p=Params(f_in)
        self.p=p
        self.health=0

#DEFINE MICROTUBULE 2
     # y for second MT: y=mx+C   
    def y(self,x):
        return self.p.m*x+self.p.yintercept
    #r gives the distance from point of intersection to a given point on second MT
    def r(self,x):
        return ((x-self.p.xintercept)**2+(self.y(x))**2)**(0.5)
    # zfun gives z value dz/dx, d2z/dx2 for given x on second MT
#    def zfun(self,x):
#        r=self.r(x)
#        if r<self.p.L_beam:
#            z=(self.p.ho-(self.p.f_bind*(r**2)*(3*self.p.L_beam-r))/(6*self.p.fl_rgdty))
#            dz=(self.p.f_bind/self.p.fl_rgdty)*(r*0.5-self.p.L_beam)*(x-self.p.xintercept)
#            if (abs(self.p.xintercept-x)<1e-15):
#                d2z=0.0
#            else:
#                d2z=(self.p.f_bind/self.p.fl_rgdty)*((r*0.5-self.p.L_beam)+((x-self.p.xintercept)**2)/(2*r))
#            return [z,dz,d2z]
#        else:
#            return [0.0,0.0,0.0]
    #z value for given x for 2nd MT
    def z(self,x):
        r=self.r(x)
        if r<self.p.L_beam:
            return (self.p.ho-(self.p.f_bind*(r**2)*(3*self.p.L_beam-r))/(6*self.p.fl_rgdty))
        else:
            return 0.0
   #vec_r gives position vecor on 2nd microtubule for a given x         
    def vec_r(self,x):
        return np.array([x, self.y(x), self.zfun(x)[0]])
    #dz is untested, use only after testing
    def dz(self,x):
        r=self.r(x)
        if r<self.p.L_beam:
            return (self.p.f_bind/self.p.fl_rgdty)*(r*0.5-self.p.L_beam)*(x-self.p.xintercept)
        else:
            return 0.0
    # r_c_m1(B) gives minimum distance from point B to the MT1(x axis)
    @staticmethod
    @jit(nopython=True)
    def r_c_m1(B):
        return sqrt(B[1]**2+B[2]**2)
    #Rcm2=r_c_m2(B) gives minimum distance from point B to MT2; Rcm2[1] and unit vecor from
    # point to the closest point on MT2
    def r_c_m2(self,B):
        xapp=(B[0]-self.p.m*(self.p.yintercept-B[1]))/(1+self.p.m**2)
        x0=self.root(xapp,B,self.p.nr_sf_accuracy,100)
        rmin=B-self.vec_r(x0)
        return self.unit_vec_mag(rmin)
   
    #haven't tested the functions poly and root. Test it before finalising
    def poly(self,x,B):
        Z=self.zfun(x)
        po1=(B[0]-x)+(B[1]-self.y(x))*self.p.m+(B[2]-Z[0])*Z[1]
        po2=-(1+self.p.m**2)+(B[2]-Z[0])*Z[2]-Z[1]**2
        return [po1,po2]
    
    def root(self,x0,B,e,maxiter):
        P=self.poly(x0,B)
        delta = abs(P[0])
        counter=0
        while delta > e:
            P=self.poly(x0,B)
            if abs(P[1])<1e-24:
                x0=float('nan')
                self.health=1
                delta=e
            else:
                x0 = x0 - P[0]/P[1]
                delta = abs(P[0])
            counter+=1
            if counter>maxiter:
                self.health=2
                return x0
        return x0

class cargo_motor(parent):
    def __init__(self,f_in):
        parent.__init__(self,f_in)
        self.nana=float('nan')
        self.bind_stat=np.array([int(0) for i in range(self.p.N)])
        self.A=np.array([[self.nana for i in range(self.p.dim)] for i in range(self.p.N)])
        self.H=np.array([[self.nana for i in range(self.p.dim)] for i in range(self.p.N)])
        self.F=np.array([[self.nana for i in range(self.p.dim)] for i in range(self.p.N)])
        self.B=np.array([self.nana for i in range(self.p.dim)])
        self.mult_lipid_diff=self.p.dt/self.p.spdgcf
        self.mult_cargo_diff=self.p.dt/self.p.ksi
        self.zero_vec=np.zeros(3)
        self.health=0
        
    def reset(self):
        self.bind_stat=np.array([int(0) for i in range(self.p.N)])
        self.A=np.array([[self.nana for i in range(self.p.dim)] for i in range(self.p.N)])
        self.H=np.array([[self.nana for i in range(self.p.dim)] for i in range(self.p.N)])
        self.F=np.array([[self.nana for i in range(self.p.dim)] for i in range(self.p.N)])
        self.B=np.array([self.nana for i in range(self.p.dim)])
        self.health=0
        self.pointer_noise_lipid=float('nan')
        self.pointer_noise_cargo=float('nan')
        self.gauss_noise_lipid=[]
        self.gauss_noise_cargo=[]
    def create_gauss_noise_lipid(self,max_num_tstep):
        self.gauss_noise_lipid=np.random.normal(0.0, self.p.lip_rwf,int(2*self.p.N*max_num_tstep))
        #assert abs(np.std(self.gauss_noise_lipid)-sqrt(2*self.p.lipD*self.p.dt))<1e-10
        self.pointer_noise_lipid=0
        #self.gauss_noise_cargo=np.loadtxt(getcwd()+"/test_inputs/normalvariate.txt")
    def create_gauss_noise_cargo(self,max_num_tstep):
        self.gauss_noise_cargo=np.random.normal(0.0, self.p.cargo_rwf,int(3*max_num_tstep))
        #assert abs(np.std(self.gauss_noise_cargo)-sqrt(2*self.p.D*self.p.dt))<1e-10
        self.pointer_noise_cargo=0
    def create_thermal_rotation_vectors(self,max_num_tstep):
        self.gauss_noise_cargo_rotation=np.random.normal(0.0, self.p.cargo_rotation_rwf,max_num_tstep)
        #assert abs(np.std(self.gauss_noise_cargo_rotation)-sqrt(4*self.p.cargo_rot_D*self.p.dt))<1e-4
        #self.random_unitvectors=np.random.random((3,max_num_tstep))
        uniform=np.random.random((2,max_num_tstep))
        phi=2*np.pi*uniform[0]
        theta=np.arccos(2*uniform[1]-1)
        self.random_unitvectors=np.array([np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)])
        self.thermal_rotation=(self.gauss_noise_cargo_rotation*self.random_unitvectors).T
        self.pointer_thermal_rotation_vector=0
        
    #U=unit_vec_mag(R) gives : U[0]=unit vector of R, U[1]=magnitude of R
    @staticmethod
    @jit(nopython=True)
    def divide(a,b):
        return a/b
    @staticmethod
    @jit(nopython=True)
    def unit_vec_mag(R):
        a,b,c=R
        mag=sqrt(a**2+b**2+c**2)
        unit_vec=R/mag
        return (unit_vec,mag)
    #vec_diff(R1,R2) gives the vector R1-R2
    @staticmethod
    @jit(nopython=True)
    def vec_diff(R1,R2):
        return R1-R2
        #return np.array(R1)-np.array(R2)
    
    @staticmethod
    @jit(nopython=True)
    def mag(R):
        a,b,c=R
        return sqrt(a**2+b**2+c**2)
    @staticmethod
    @jit(nopython=True)
    def get_angles(vector,mag):
        return [acos(vector[2]/mag), atan2(vector[1],vector[0])]

    def cart_polar(self,cart):
        vector=cart-self.B
        mag=self.mag(vector)
        #assert abs(mag-self.p.radius)<1e-18
        return self.get_angles(vector,mag)

    def polar_cart(self,pol):
        [c_theta, c_phi]=pol
        xc=self.p.radius*sin(c_theta)*cos(c_phi)
        yc=self.p.radius*sin(c_theta)*sin(c_phi)
        zc=self.p.radius*cos(c_theta)
        cor=np.array([xc,yc,zc])
        return np.add(cor,self.B)
        
#Pick a point on the cargo: cpick_bd is in frame of reference of bead
        # cpick is in the frame of reference of lab
    def pick_Cargo(self):
        c_theta=acos(2*random()-1)
        c_phi=2*pi*random()
        xc=self.p.radius*sin(c_theta)*cos(c_phi)
        yc=self.p.radius*sin(c_theta)*sin(c_phi)
        zc=self.p.radius*cos(c_theta)
        cpick_bd=np.array([xc,yc,zc])
        cpick=np.add(cpick_bd,self.B)
        return [cpick,[c_theta,c_phi]]

 #point of intersections of sphere with radius 2L with microtubules1 and 2
    def int_m1(self,cpick):
        r_c_m1=self.r_c_m1(cpick)
        if r_c_m1>self.p.access_r:
            m1p1=np.array([self.nana for i in range(self.p.dim)])
            m1p2=np.array([self.nana for i in range(self.p.dim)])
        else:
            x2=cpick[0]+sqrt(self.p.access_r**2-r_c_m1**2)
            x1=cpick[0]-sqrt(self.p.access_r**2-r_c_m1**2)
            m1p1=np.array([x1,0.0,0.0])
            m1p2=np.array([x2,0.0,0.0])
        return np.array([m1p1,m1p2])

    def int_m2(self,cpick):
        [R, mag_r_c_m2]=self.r_c_m2(cpick)
        if mag_r_c_m2>self.p.access_r:
            m2p1=[self.nana for i in range(self.p.dim)]
            m2p2=[self.nana for i in range(self.p.dim)]
        else:
            bx=cpick[0]
            lm=self.p.access_r/2.0
            X0s = [bx+2.0*lm, bx+lm, bx, bx-lm, bx-2.0*lm]
            rootlist=[]
            for x0 in X0s:
                x0=self.in_root(x0,cpick,self.p.nr_sf_accuracy,100)
                x0=round(x0,10) #round of  to 10^-10 m
                rootlist.append(x0)
            rootlist=[root for root in set(rootlist) if root==root]
            rootlist.sort()
            if len(rootlist)==1:
                [x1,x2]=[rootlist[0],rootlist[0]]
            elif len(rootlist)==2:
                [x1,x2]=[rootlist[0], rootlist[1]]
            elif len(rootlist)>2:
                [x1,x2]=[rootlist[0], rootlist[len(rootlist)-1]]
                print("error")
                print(rootlist,cpick)
            m2p1=[x1,self.y(x1),self.z(x1)]
            m2p2=[x2,self.y(x2),self.z(x2)]
        return [m2p1,m2p2]
            
            
    def f(self,x,cpick):
        Z=self.zfun(x)
        f=(x-cpick[0])**2+(self.y(x)-cpick[1])**2+(Z[0]-cpick[2])**2-(self.p.access_r)**2
        df=2*(x-cpick[0])+2*(self.y(x)-cpick[1])*self.p.m+2*(Z[0]-cpick[2])*Z[1]
        return [f, df]
    def in_root(self,x0, cpick,e, maxiter):
        [f,df]=self.f(x0,cpick)
        counter=0
        while abs(f) > e:
            [f,df]=self.f(x0,cpick)
            if abs(df)<1e-18:
                x0=float('nan')
                return x0
            else:
                x0 = x0 - f/df
                counter+=1
                if counter>maxiter:
                    self.health=2
                    return x0
        return x0
#Pick a point in the region of access:
#    def pick_MT(self,cpick):
#        [m1p1,m1p2]=self.int_m1(cpick)
#        [m2p1,m2p2]=self.int_m2(cpick)
#        [l1,l2]=[abs(m1p2[0]-m1p1[0]), sqrt((m2p2[0]-m2p1[0])**2+(m2p2[1]-m2p1[1])**2)]
#        if np.isnan(l1) and np.isnan(l2):
#            return [[self.nana, self.nana, self.nana],0]
#        if np.isnan(l1):
#            l1=0
#        if np.isnan(l2):
#            l2=0
#        L=l1+l2
#        rl=L*random()
#        if rl<l1:
#            xpick=m1p1[0]+rl
#            return [[xpick,0.0,0.0],1]
#        else:
#            xpick=min(m2p1[0],m2p2[0])+(rl-l1)*cos(self.p.rtheta)
#            return [[xpick, self.y(xpick),self.z(xpick)],2]
    def pick_MT_single(self,cpick):
        [m1p1,m1p2]=self.int_m1(cpick)
        l1=abs(m1p2[0]-m1p1[0])
        if np.isnan(l1):
            return [[self.nana, self.nana, self.nana],0]
        xpick=m1p1[0]+l1*random()
        return [[xpick,0.0,0.0],1]
    
    @staticmethod
    @jit(nopython=True)
    def steric_allowance(mpick,cpick,B,radius):
        bm=sqrt((B[0]-mpick[0])**2+(B[1]-mpick[1])**2+(B[2]-mpick[2])**2)
        pm=sqrt((cpick[0]-mpick[0])**2+(cpick[1]-mpick[1])**2+(cpick[2]-mpick[2])**2) #similar equation in other place
        if (radius**2+bm**2)>=pm**2:
            return 1
        else:
            return 0

    def boltzman(self,mpick,cpick):
        ext=sqrt((cpick[0]-mpick[0])**2+(cpick[1]-mpick[1])**2+(cpick[2]-mpick[2])**2)-self.p.Lmot
        return sqrt(self.p.kmot/(2*self.p.kbolt*self.p.temp*pi))*exp(-self.p.kmot*ext**2/(2*self.p.kbolt*self.p.temp))*self.p.deltax
    
    def check_attach_general(self,num):
        [mpick,mt_num]=self.pick_MT_single(self.A[num])
        if mt_num!=0:
            st_all=self.steric_allowance(mpick,self.A[num],) #update this line
            if random()< self.p.Pon*self.p.dt and st_all==1:
                self.bind_stat[num]=int(mt_num)
                self.H[num]=mpick
                
    def check_attach_single(self,num):
        r_c_m1=self.r_c_m1(self.A[num])
        if r_c_m1<=self.p.access_r:
            [mpick,mt_num]=self.pick_MT_single(self.A[num])
            st_all=self.steric_allowance(np.array(mpick),self.A[num],self.B,self.p.radius)
            if random()< self.p.Pon*self.p.dt and st_all==1:
                self.bind_stat[num]=int(mt_num)
                self.H[num]=mpick
    @staticmethod
    @jit(nopython=True)
    def spring_force(ha_cap,ha,Lmot,kmot):
        return -kmot*(Lmot-ha)*ha_cap
    
    def mot_force(self,num):
        if (self.bind_stat[num] != 0):
            [ha_cap,ha]=self.unit_vec_mag(self.vec_diff(self.H[num],self.A[num]))
            if ha>self.p.Lmot:
                self.F[num]=self.spring_force(ha_cap,ha,self.p.Lmot,self.p.kmot)
            else:
                self.F[num]=self.zero_vec

    def mot_force_loop(self):
        for num in range(self.p.N):
            self.mot_force(num)

    def check_force_type(self,num):
        if self.bind_stat[num]==1:
            if self.F[num][0]>0:
                return [1,self.F[num][0]]
            else:
                return [0,self.F[num][0]]
        elif self.bind_stat[num]==2:
            proj=self.F[num][0]+self.p.m*self.F[num][1]
            if proj>0:
                return [1,proj]
            else:
                return [0,proj]
    def k_hind_off(self,F):
        if F>4.9e-9:
            return self.p.eps*1e100
        return self.p.eps*exp(F/(self.p.Fs))
    def k_ast_off(self,F):
        return self.p.eps+1.56*F*1e12 #converting F to pN by multiplying with1e12
        
#use check_force_type(num) for two microtubules. for single microtubule just use if condition
    def check_detach(self,num,vs,Fmag,force_type):
        vs=0.8e-6*(vs/self.p.velMean) # if 0.8e-6 is the velocity at high atp, if you change that change here
        pstep=(1-exp(-vs*self.p.dt/self.p.dx))
        if force_type==1 and random()< (self.k_hind_off(Fmag)*self.p.dt/pstep):
            self.detach(num)
        elif force_type==0 and random()< (self.k_ast_off(Fmag)*self.p.dt/pstep):
            self.detach(num)
    def detach(self,num):
        self.bind_stat[num]=0
        self.H[num]=[self.nana for i in range(self.p.dim)]
        self.F[num]=[self.nana for i in range(self.p.dim)]
        
#two microtubules, change this function
    def stepping_velocity(self,num,Fmag,force_type):
        if  force_type==1:
            return max(self.p.velMean*(1-(Fmag/self.p.Fs)**2),0)
        else:
            return self.p.velMean
    def move_forward(self,num):
        if self.bind_stat[num]==1:
            self.H[num][0]+=self.p.dx
        else:
            self.H[num][0]+=self.p.dx*cos(self.p.rtheta)
            self.H[num][1]+=self.p.dx*sin(self.p.rtheta)
            self.H[num][2]=self.z(self.H[num][0])
    
    def forward_step(self,num):
        Fmag=self.mag(self.F[num])
        if self.F[num][0]>=0:
            force_type=1
        elif self.F[num][0]<0:
            force_type=0
        vs=self.stepping_velocity(num,Fmag,force_type)
        if random()<(1-exp(-vs*self.p.dt/self.p.dx)): #if this condition is satisfied motor tries to make a step. 
            self.check_detach(num,vs,Fmag,force_type) #check motor detachment
            if self.bind_stat[num]!=0: # if motor is not unbound then move forward 
                self.move_forward(num)

    def forward_step_loop(self):
        for num in range(self.p.N):
            if (self.bind_stat[num] !=0):
                self.forward_step(num) # forward step checks if the bound motor tries to make a step irrespective of whether motor is blocked or not 

    def check_attach_loop(self):
        for num in range(self.p.N):
            if self.bind_stat[num]==0:
                self.check_attach_single(num)
#Forces
    def gauss_noise(self):
        return [normalvariate(0,1.0) for i in range(self.p.dim)]

#when two microtubules are used modify steric force

    def steric_force(self):
        rmin_m1=self.r_c_m1(self.B)
        return self.f1s(self.B,rmin_m1)
    def f1s(self,B,rmin):
        if rmin<self.p.brmr:
            mag1=self.p.ksf*(self.p.brmr-
                              (B[1]**2+B[2]**2)**(0.5))/(B[1]**2+B[2]**2)**(0.5) #positve so that f1s has correct mag and direction
            f1s=np.multiply(mag1,B)
            f1s[0]=0
            return f1s
        else:
            return self.zero_vec
    
    #Calculation of steric force due to second microtubule
    def f2s(self,B,rmin):
        if rmin[1]<self.p.brmr:
            return np.multiply(rmin[0],self.p.ksf*(self.p.brmr-rmin[1]))
        else:
            return [0,0,0]
        

 #add motor forces
    @staticmethod
    @jit(nopython=True)
    def sum_motor_force(F,N):
        [Fm_x,Fm_y,Fm_z]=[0,0,0]
        for i in range(N):
            if F[i][0]==F[i][0]:
                Fm_x+=F[i][0]
                Fm_y+=F[i][1]
                Fm_z+=F[i][2]
        return [Fm_x,Fm_y,Fm_z]
    
    @staticmethod
    @jit(nopython=True)
    def shift_adder(Fm_x,Fm_y,Fm_z,Fs,gauss_noise,mult_cargo_diff):
        del_x=mult_cargo_diff*(Fm_x+Fs[0])+gauss_noise[0]
        del_y=mult_cargo_diff*(Fm_y+Fs[1])+gauss_noise[1]
        del_z=mult_cargo_diff*(Fm_z+Fs[2])+gauss_noise[2]
        return (del_x,del_y,del_z)
        
    @profile
    def shift(self):
        [Fm_x,Fm_y,Fm_z]=self.sum_motor_force(self.F,self.p.N)
        Fs=self.steric_force()
        jj=self.pointer_noise_cargo
        gauss_noise=self.gauss_noise_cargo[jj:jj+3] #retrieves normally distributed a, b values from array gauss_noise_cargo
        self.pointer_noise_cargo=jj+3

        return self.shift_adder(Fm_x,Fm_y,Fm_z,Fs,gauss_noise,self.mult_cargo_diff)

    def update_pos(self,shift):
        self.B=self.B+shift
        self.A=self.A+shift
        
    @staticmethod
    @jit(nopython=True)
    def force_ind(Tran,force,mult_fact):
        return np.multiply(mult_fact,np.dot(Tran,force))
    @staticmethod
    @jit(nopython=True)
    def tranf_mat(t,p):
        return np.array([[cos(t)*cos(p),cos(t)*sin(p),-sin(t)],[-sin(p),cos(p),0]])
    @staticmethod
    @jit(nopython=True)
    def update_A(a,b,tranf,anchor):
        #A_new=A_old+vector step in the tangent plane
        return anchor+a*tranf[0]+b*tranf[1]

    def mot_diffuse(self,num):
        [t,p]=self.cart_polar(self.A[num]) #uses numba
        
        jj=self.pointer_noise_lipid
        [a,b]=self.gauss_noise_lipid[jj:jj+2] #retrieves normally distributed a, b values from array gauss_noise_lipid
        self.pointer_noise_lipid=jj+2
        
        tranf=self.tranf_mat(t,p) #uses numba, cretes the matrix with unit vectors in theta phi direction
        if self.bind_stat[num] != 0:
            [a2,b2]=self.force_ind(tranf,self.F[num],self.mult_lipid_diff) #uses numba, returns force associated shift in anchor position of motor
            a+=a2
            b+=b2
        self.A[num]=self.update_A(a,b,tranf,self.A[num]) #uses numba, updates anchor position
        self.project_back(num) #anchor will be slightly out of the sphere in previous step, we project the motor back to sphere

    def mot_lang_step_loop(self):
        for num in range(self.p.N):
            self.mot_diffuse(num)
    @staticmethod
    @jit(nopython=True)
    def proj_adder(b,rad,rhat):
        return b+rad*rhat
    
    def project_back(self,num):
        [r_hat,del_r]=self.unit_vec_mag(self.vec_diff(self.A[num],self.B))
        self.A[num]=self.proj_adder(self.B,self.p.radius,r_hat)

    def rot_diff_step(self,theta,g):
        [t,p]=np.multiply(self.p.rot_rwf,g)
        if abs(sin(theta))>1e-50:
            p=p/sin(theta)
        if abs(t)>(0.5*pi):
            t=(0.5*pi)
        if abs(p)>pi:
            p=pi
        return [t,p]

    def rot_diff_update(self):
        g=np.array([normalvariate(0,1.0) for i in range(3)])
        [dz,dt,dp]=np.multiply(self.p.rot_rwf,g)
        A=[[cos(dz)*cos(dp)-sin(dz)*cos(dt)*sin(dp),
        -cos(dz)*sin(dp)-sin(dz)*cos(dt)*cos(dp),
        sin(dz)*sin(dt)],
        [sin(dz)*cos(dp)+cos(dz)*cos(dt)*sin(dp),
        -sin(dz)*sin(dp)+cos(dz)*cos(dt)*cos(dp),
        -cos(dz)*sin(dt)],
        [sin(dt)*sin(dp),
        sin(dt)*cos(dp),
        cos(dt)]]
        for num in range(self.p.N):
            rel_A=self.A[num]-self.B
            rel_A2=np.dot(A,rel_A)
            self.A[num]=rel_A2+self.B
            
    def initialize(self):
        rad=self.p.beadposz
        thet=2*pi*random()
        self.B=np.array([self.p.beadposx,rad*cos(thet),rad*sin(thet)])
        for num in range(self.p.N):
            [cpick,cpickr]=self.pick_Cargo()
            self.A[num]=np.array(cpick)
            self.bind_stat[num]=0
            
    def initialize_motor_velocity(self):
        self.mot_vel=np.random.normal(loc=self.p.velMean, scale=self.p.velStdev, size=self.p.N)


    def initial_engagement_lipid(self):
        a=ceil(log10(10**8/(2*self.p.N)))
        array_freq=int(10**a)
        count=1
        max_count=int(self.p.tmax/self.p.dt)+1
        while(sum(self.bind_stat)==0 and count<=max_count):
            if count%array_freq==1:
                self.create_gauss_noise_lipid(array_freq)
            for num in range(self.p.N):
                self.mot_diffuse(num)
                self.check_attach_single(num)
            count+=1
        time=(count-1)*self.p.dt
        return time
        
    def initial_engagement_lipid_brate_calculation(self):
        a=ceil(log10(10**8/(2*self.p.N)))
        array_freq=int(10**a)
        count=1
        max_count=int(self.p.tmax/self.p.dt)+1
        while(sum(self.bind_stat)==0 and count<=max_count):
            if count%array_freq==1:
                self.create_gauss_noise_lipid(array_freq)
            self.mot_diffuse(0)
            self.check_attach_single(0)
            count+=1
        time=(count-1)*self.p.dt
        return time
    
    def initial_engagement_rigid(self):
        count=0
        while(sum(self.bind_stat)==0):
            count+=1
            self.rot_diff_update()
            for num in range(self.p.N):
                self.check_attach_single(num)
        return count    

    def rebinding_lipid(self,destn):
        count=0
        base=destn+"/"+"rebind_anch"
        while(sum(self.bind_stat)==0):
            count+=1
            if (count%self.p.samrate==0):
                np.savetxt(base+str(count)+".txt",self.A)
            self.rot_diff_update()
            for num in range(self.p.N):
                self.mot_lang_step(num)
                self.check_attach_single(num)
        return count
    
    def assign_polar(self):
        for num in range(self.p.N):
            if self.bind_stat[num]==0:
                [cpick,cpickr]=self.pick_Cargo()
                self.A[num]=cpick
    
    #Rodrigus formula
    def Rotation_matrix(self,theta_vector,theta):
        [wx,wy,wz] = theta_vector / theta
        omega_tilde=np.matrix([[0,-wz,wy],[wz,0,-wx],[-wy,wx,0]])
        return np.identity(3)+omega_tilde*np.sin(theta)+(omega_tilde**2)*(1-np.cos(theta))
    
    def calculate_torque(self):
        ''' torque = RX F'''
        R_vector=(self.A-self.B)
        torque=np.zeros(3)
        for num,bind in enumerate(self.bind_stat):
            if bind==1:
                torque=np.cross(R_vector[num],self.F[num])
        return torque

    
    def cargo_rotation_from_torque(self):
        tor=self.calculate_torque()
        theta_vector=tor*self.p.torque_multiplier
        
        theta_vector+=self.thermal_rotation[self.pointer_thermal_rotation_vector]
        self.pointer_thermal_rotation_vector+=1
        
        theta=np.linalg.norm(theta_vector)
        if abs(theta)>0:
            Rmat=self.Rotation_matrix(theta_vector,theta)
            At=np.matrix(self.A)
            At=At.T
            Bt=np.matrix(self.B)
            Bt=Bt.T
            At=Bt+Rmat*(At-Bt)
            self.A=np.array(At.T)
    

        
                
        
