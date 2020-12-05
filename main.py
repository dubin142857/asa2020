# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 16:15:00 2020

@author: Admin
"""

import  sys
import numpy as np
import optargv
import optfun
import matplotlib.pyplot as plt

# read data from command line
if len(sys.argv) > 1 and (sys.argv[1] == "--help"or sys.argv[1] == "-h"):
    options = optargv.default_options()
    print("Usage:")
    print("      python main.py opt1=value1 opt2=value2...")
    print("      e.g python main.py --image_path=image/peppers256.png\n")
    print("Available options:default values")
    for k, v in options.items():
        print("", k, ":", v)
    sys.exit()
options, globalnames = optargv.setoptions(argv=sys.argv[1:], kw=None)
globals().update(globalnames)

#initial image
N2                  =N*N
Tc                  =1./(k_B*np.log(1+np.sqrt(q)))

internal_energy     =[]
heat                =[]

magnetization       =[]
temper_range        =np.linspace(0.1,2,num=50)
beta_range          =(1/(k_B*np.array(temper_range))).tolist()
#image               = optfun.initialize(N,q)
for temper in temper_range:
    image               = optfun.initialize(N,q)
    beta                =1./(k_B*temper)
    hamilton_list       =[]
    magnetization_list  =[]
    for i in range(warm_iter):
        hamilton,image=optfun.metropolis(image,J,h,q,k_B,init_temp=temper,proposal_type="signle_flip")
    for i in range(max_iter):
        hamilton,image=optfun.metropolis(image,J,h,q,k_B,init_temp=temper,proposal_type="signle_flip")
        hamilton_list.append(hamilton)
        magnetization_list.append(np.sum(image)/N2)
    # if temper <low_temper:
    #     temp_list=(np.linspace(temper,temper+3,num=100)).tolist()
    #     temper_now=temper.copy()+3
    #     for i in range(max_iter):
    #         temper_now,hamilton,image=optfun.simulated_temp(image,temper_now,temp_list,alpha,J,h,q,k_B,proposal_type="signle_flip")
    #         hamilton_list.append(hamilton)
    #         magnetization_list.append(np.sum(image)/N2)
    # else:
    #     for i in range(max_iter):
    #         #hamilton,image =optfun.wolf(image,q,beta,J,h)
    #         hamilton,image=optfun.metropolis(image,J,h,q,k_B,init_temp=temper,proposal_type="signle_flip")
    #         hamilton_list.append(hamilton)
    #         magnetization_list.append(np.sum(image)/N2)
    exp_ham =sum(hamilton_list)/max_iter
    exp_ham2=optfun.square_sum_list(hamilton_list)/max_iter
    internal_energy.append(exp_ham/N2)
    C=k_B*beta**2*(exp_ham2-exp_ham**2)/N2
    heat.append(C)
    magnetization.append(sum(magnetization_list)/max_iter)
    print("temperature now is   {:.4f}".format(temper))
if output_image=="None":
    fig=plt.figure()
    ax1=fig.add_subplot()
    ax1.set_title('energy and heat under different temperature')
    lns1=ax1.plot(temper_range,internal_energy,color='red',label='energy')
    ax1.set_ylabel('internal energy')
    #plt.show()
    
    ax2=ax1.twinx()
    lns2=ax2.plot(temper_range,heat,color='green',label='heat')
    ax2.set_ylabel('heat')
    ax2.set_xlabel('temperature')
    lns=lns1+lns2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc=0)
    plt.show()
    fig.savefig('h0'+output_image,dpi=200)
    
    fig2=plt.figure()
    ax=fig2.add_subplot()
    ax.set_title('magnetization under diff beta')
    ax.plot(beta_range,(np.array(magnetization)-2).tolist(),color='red',label='magnet')
    ax.set_xlabel('beta')
    ax.set_ylabel('magnetization')
    plt.legend()
    plt.show()
    fig2.savefig('magnetization'+output_image,dpi=200)



# if not output_image == "None":
#     import matplotlib.pyplot as plt
#     fig = plt.figure()
#     ax = fig.add_subplot(1,3,1)
#     ax.imshow(f, cmap='gray')
#     ax.set_title("$u_{true}$", fontsize=20); ax.set_xticks([]); ax.set_yticks([])
#     ax = fig.add_subplot(1,3,2)
#     ax.imshow(f_noise, cmap='gray')
#     ax.set_title("f", fontsize=20); ax.set_xticks([]); ax.set_yticks([])
#     ax = fig.add_subplot(1,3,3)
#     ax.imshow(u1_heat, cmap='gray')
#     ax.set_title("$u_{compute}$", fontsize=20); ax.set_xticks([]); ax.set_yticks([])
#     fig.set_size_inches([15,5])
#     fig.savefig('heat-'+output_image, dpi=200)
