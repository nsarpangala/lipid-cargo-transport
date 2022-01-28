# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 11:49:49 2019

@author: Niranjan
"""

from mayavi import mlab
mlab.options.offscreen = True
import numpy as np

class figure_handling():
    def __init__(self):
        self.nana=float('nan')
    def sphere_surface(self):
        phi, theta = np.mgrid[0:np.pi:100j, 0:2*np.pi:100j]
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        self.x=x
        self.y=y
        self.z=z
    def microtubule_surface(self,fg,extent,col,radius):
        x, theta = np.mgrid[extent[0]:extent[1]:500j, 0:2*np.pi:100j]
        y = radius*np.sin(theta)
        z = radius*np.cos(theta)
        mlab.mesh(x,y,z,color=col,figure=fg)
        return fg    
    def radius_values(self,cargo,motor):
        self.c_r=cargo
        self.m_r=motor
        self.rx=self.x*self.c_r
        self.ry=self.y*self.c_r
        self.rz=self.z*self.c_r
        self.mx=self.x*self.m_r
        self.my=self.y*self.m_r
        self.mz=self.z*self.m_r
    def generate_figure(self):
        mlab.clf()
        fg=mlab.figure(size=(800, 700))
        return fg
    def cargo(self,fg,spc,col):
        mlab.mesh(spc[0]+self.rx, spc[1]+self.ry, spc[2]+self.rz,color=col,figure=fg)
        return fg
    def motor(self,fg,mc,col):
        mlab.mesh(mc[0]+self.mx, mc[1]+self.my, mc[2]+self.mz,color=col,figure=fg)
        return fg
    def bound_motor(self,fg,anchor,head,col,rad):
        x=np.array([anchor[0],head[0]])
        y=np.array([anchor[1],head[1]])
        z=np.array([anchor[2],head[2]])
        mlab.plot3d(x,y,z,figure=fg,color=col,tube_radius=rad)
        return fg
    def time_stamp(self,fg,time,loc):
        #ts='{:05.3f}'.format(time)
        ts='{:07.5f}'.format(time)
        ts =ts+ " s"
        mlab.text(loc[0],loc[1],ts, width=0.2)
        return fg