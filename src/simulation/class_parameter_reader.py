label_attr_map = {
    "N": ["N", int],
    "sampleSiz": [ "sampleSiz", int],
    "kmot": [ "kmot", float],
    "velMean": [ "velMean", float],
    "velStdev": [ "velStdev", float],
    "dt": [ "dt", float],
    "tmax": [ "tmax", float],
    "samrate": [ "samrate", int],
    "Ktrap": [ "Ktrap", float],
    "load": [ "load", float],
    "Pon": [ "Pon", float],
    "load": [ "load", float],
    "dx": [ "dx", float],
    "Lmot": [ "Lmot", float],
    "Fs": [ "Fs", float],
    "w": [ "w", float],
    "radius": [ "radius", float],
    "eps": [ "eps", float],
    "theta": [ "theta", float],
    "xintercept": [ "xintercept", float],
    "beadposx": [ "beadposx", float],
    "beadposy": [ "beadposy", float],
    "beadposz": [ "beadposz", float],
    "viscosity": ["viscosity", float],
    "temp": ["temp", float],
    "fl_rgdty":["fl_rgdty",float],
    "ho":["ho",float],
    "f_bind":["f_bind",float],
    "mr":["mr",float],
    "ksf":["ksf",float],
    "nr_sf_accuracy":["nr_sf_accuracy",float],
    "dim":["dim",int],
    "kbolt":["kbolt",float],
    "access_r":["access_r",float],
    "lipD":["lipD",float],
    "deltax":["deltax",float],
    "simname":["simname",str],
    "lipon":["lipon",int],
    "rotD":["rotD",float],
    "samtime":["samtime",float],
}

import numpy as np
from math import pi, tan, radians, sqrt
tan, radians

class Params(object):
    def __init__(self, input_file_name):
        with open(input_file_name, 'r') as input_file:
            for line in input_file:
                
                row = line.split(",")
        
                data = row[2]  # rest of row is data list
                label = row[1]
                

                attr = label_attr_map[label][0]
                datatypes = label_attr_map[label][1]

                value= (datatypes(data))
                self.__dict__[attr] = value
        self.nana=float('nan')
        self.rtheta=radians(self.theta)
        self.m=tan(self.rtheta)
        self.ksi=6*pi*self.viscosity*self.radius*3  #viscosity is twice bulk near surface and three times right at the surface
        self.noise_std=sqrt(2*self.ksi*self.kbolt*self.temp)
        self.D=1.38065e-23*self.temp/self.ksi  #Einstein Diffusion constant - teathered diffusion ?
        self.L_beam=(3*self.fl_rgdty*self.ho/self.f_bind)**(1/3)
        self.B=np.array([self.beadposx,self.beadposy,self.beadposz])
        self.brmr=self.radius+self.mr
        self.yintercept=-self.m*self.xintercept
        self.spdgcf=self.kbolt*self.temp/self.lipD
        self.lip_rwf=sqrt(2*self.lipD*self.dt)
        self.cargo_rwf=sqrt(2*self.D*self.dt)
        #self.rot_rwf=sqrt(2*self.rotD*self.dt)/self.radius
        
        #rotation business
        if self.lipon==1:
            self.ksi_rotation=8*pi*self.viscosity*self.radius**3
            self.cargo_rot_D=self.kbolt*self.temp/self.ksi_rotation
        elif self.lipon==0:
            self.cargo_rot_D=self.rotD
            self.ksi_rotation=self.kbolt*self.temp/self.cargo_rot_D
        self.cargo_rot_D=2.106*self.cargo_rot_D #Rotational diffusion correction to get the correct Mean square displacement for thermal rotation    
        self.torque_multiplier=self.dt/self.ksi_rotation    
        self.cargo_rotation_rwf=sqrt(4*self.cargo_rot_D*self.dt)
        self.simname=self.simname.rstrip()
#The code to use this class is.
##params = Params('input.txt')

#Following are the codes to test whether this class is working correctly
##print params.N
##print params.sampleSiz
##print params.kmot
##print params.velMean
##print params.velStdev
##print params.dt
##print params.tmax
##print params.samrate
##print params.Ktrap
##print params.load
##print params.Pon
##print params.dx
##print params.Lmot
##print params.Fs
##print params.w
##print params.radius
##print params.eps
##print params.theta
##print params.xintercept
##print params.beadposx
##print params.beadposy
##print params.viscosity
##print params.temp



