# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 16:20:24 2020

optional funtions which may be used.

@author: Admin
"""
import numpy as np
from random import  choice
def signle_flip(image,q):
    """
    Usage: this function use single flip proposal to decide matrix Q
    Parameters:
        image:N*N matrix,which record the state of  Potts model.
    
        q    : each point in matrix image have q states.
    """
    N=int(image.shape[0])
    random_x=np.random.randint(N)
    random_y=np.random.randint(N)
    q_xy=image[random_x,random_y]
    state_list=list(range(1,q+1))
    state_list.remove(q_xy)
    # old_state=image[random_x,random_y]
    image[random_x,random_y]=choice(state_list)
    # delta_state=image[random_x,random_y]-old_state
    # new_image=image.copy()
    # new_image[random_x,random_y]=delta_state
    # #delta_hamilton=hamiltonian(new_image, J, h)
    return image

def wolf(grid,q,beta,J,h):
    """
    Usage: this function use swendsen_wang_wolf proposal to decide matrix Q
    Parameters:
        image:N*N matrix,which record the state of  Potts model
        beta ,J:hamiltonian parameter
        q    : each point in matrix image have q states.
    """
    N=grid.shape[0]
    pro=1 -np.exp(-2*beta*J)
    edges=np.random.choice(a=[True,False],size=(N,N,4),p=[pro,1-pro])
    #left
    edges[np.roll(a=grid,shift=1,axis=1)!=grid,0] =False
    #right
    edges[np.roll(a=grid,shift=-1,axis=1)!=grid,1]=False
    #top
    edges[np.roll(a=grid,shift=1,axis=0)!=grid,2] =False
    #bottom
    edges[np.roll(a=grid,shift=-1,axis=0)!=grid,3]=False
    #random postion
    rand_po=np.random.randint(N,size=(1,2))[0]
    rand_xy=[rand_po]
    #find cluster
    i=0
    reach_end=False
    while (not reach_end) and (i<N*N-1):
        this_x=rand_xy[i][0]
        this_y=rand_xy[i][1]
        neighbors=np.where(edges[this_x,this_y,:])
        if(np.any(edges[this_x,this_y,:])):
            for neighboring in np.nditer(neighbors):
            #left
                if neighboring==0:
                    new_y=this_y-1
                    new_xy=np.array([this_x,new_y%N])
                    if  not any((new_xy==old).all() for old in rand_xy):
                        rand_xy.append(new_xy)
                elif neighboring==1:
                    new_y=this_y+1
                    new_xy=np.array([this_x,new_y%N])
                    if  not any((new_xy==old).all() for old in rand_xy):
                        rand_xy.append(new_xy)
                elif neighboring==2:
                    new_x=this_x-1
                    new_xy=np.array([new_x%N,this_y])
                    if  not any((new_xy==old).all() for old in rand_xy):
                        rand_xy.append(new_xy)
                else:
                    new_x=this_x+1
                    new_xy=np.array([new_x%N,this_y])
                    if  not any((new_xy==old).all() for old in rand_xy):
                        rand_xy.append(new_xy)
        i+=1
        if i ==len(rand_xy):
            reach_end=True
    # flip entire cluster
    state_list=list(range(1,q+1))
    state_list.remove(grid[this_x,this_y])
    new_state=choice(state_list)
    idx = [xy[0] for xy in rand_xy]
    idy = [xy[1] for xy in rand_xy]
    grid[idx, idy] = new_state
    ham =hamiltonian(grid,J,h)
    return ham,grid
            
def initialize(N,p):
    """
    initialize image random

    Parameters
    ----------
    N : image size.
    p : point state number.

    Returns
    -------
    initialized image.

    """
    return np.random.randint(1,4,size=(N,N))
def hamiltonian(image,J,h):
    """"
    Usage: this function is used to compute the hamiltonian of system image
    """
    x_fw=np.roll(image,1,axis=1)
    x_bc=np.roll(image,-1,axis=1)
    y_fw=np.roll(image,1,axis=0)
    y_bc=np.roll(image,-1,axis=0)
    ham_matrix=-J*((image==x_fw).astype(int)+(image==x_bc).astype(int)\
                   +(image==y_fw).astype(int)+(image==y_bc).astype(int))
    hamilton=np.sum(ham_matrix)
    if h!=0:
        hamilton+=-h*(np.sum(image))
    return hamilton

def metropolis(image,J=1,h=0,q=3,k_B=1,init_temp=1,proposal_type="signle_flip"):
    """
    Usage:use to metropolis algorithm to solve MC intergal.
    proposal_maxtix is default "signle_flip".
    Parameters:
        image        :the state of system
        J            :hamiltonian parameters
        h            :hamiltonian parameters
        q            :the number of particle states
        k_B          :pdf parameters
        init_temp    :initial temperature.
        proposal_type:proposal matrix type
    """
    beta=1/(k_B*init_temp)
    N=int(image.shape[0])
    new_image=np.zeros_like(image)
    hamilton_list=[]
    hamilton_list.append(hamiltonian(image, J, h))
    if proposal_type =="signle_flip":
        new_image=signle_flip(image,q)
        hamilton_list.append(hamiltonian(new_image,J,h))
        delta_hamilton=hamilton_list[-1]-hamilton_list[-2]
        if delta_hamilton<=0:
            pass
        else:
            r=np.random.random()
            if r<=np.exp(-beta*delta_hamilton):
                pass
            else:
                hamilton_list[-1]=hamilton_list[-2]
                new_image=image.copy()
    return hamilton_list[-1],new_image

def temper_choice(temper_now,temper_list):
    """
    choose next temper in simulated tempering

    Parameters
    ----------
    temper_now : current temper.
    temper_list : temper optional list.

    Returns
    -------
    next temper.

    """
    if (temper_now in temper_list):
        if temper_now==temper_list[0]:
            return temper_list[1]
        elif temper_now==temper_list[-1]:
            return temper_list[-2]
        else:
            diretion=np.random.choice([-1,1])
            return temper_list[temper_list.index(temper_now)+diretion]
    else:
        print("error ,temper_now is not in temper_list")
        return 
    
def simulated_temp(x,temper_now,temp_list,alpha,J=1,h=0,q=3,k_B=1,proposal_type="signle_flip"):
    """
    simulated tempering methods when temper is low
    Parameters
    ----------
    x : current state.
    temp_list : optional temper list.
    alpha : mixture-type.

    Returns
    -------
    next point.
    """        
    u=np.random.rand()
    if u<alpha:
        temper=temper_now
        ham,x=metropolis(x,J,h,q,k_B,temper,proposal_type)            
        return temper,ham,x
    else:  
        temper_propose=temper_choice(temper_now, temp_list)
        ham_next=hamiltonian(x, J, h)
        pro_now=np.exp(-(1/(k_B*temper_propose)-1/(k_B*temper_now))*ham_next)
       
        if temper_now==temp_list[0] or temper_now==temp_list[-1]:
            divide_a=0.5
        elif temper_propose==temp_list[0] or temper_propose==temp_list[-1]:
            divide_a=2
        else:
            divide_a=1
        r=min(1,divide_a*pro_now)
        if np.random.rand()<r:
            temper=temper_propose
            return temper,ham_next,x
        else:
            return temper_now,ham_next,x
        
def square_sum_list(l):
    """
    sum the square of  elements in list l

    Parameters
    ----------
    l : list.

    Returns
    -------
    sum.
    """        
    y=0.0
    for x in l:
        y+=x*x
    return y
        
        