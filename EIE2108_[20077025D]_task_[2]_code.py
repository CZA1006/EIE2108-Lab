#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 21:54:00 2021

@author: CAI Zhuoang
"""
#%%
import numpy as np
import matplotlib.pyplot as plt

fn='eie2108-lab-2021-datafile-04.txt'
data=[]
with open(fn)as f:
    for line in f:
        data.append(line.rstrip())
no_of_data=len(data)
data=np.array(data,dtype=np.float32)

'''cost function'''
def SE(a2,a3,data):
    return sum(np.power(data[3:]-a2*data[1:-2]-a3*data[0:-3],2))

a2=1  # initialize a2, a3
a3=1
alpha=0.005
max_no_of_iter=50

rec_J=np.zeros(max_no_of_iter)  # declare an array to record the J @ iter i
rec_a2a3=np.zeros([max_no_of_iter,2])  # declare an array to record the a2, a3 @ iter i

J=SE(a2,a3,data)  # compute the initial J

rec_J[0]=J  # save the initial estimates of a2 and a3
rec_a2a3[0,:]=[a2,a3]  # save the initial J

for i in range(1,50):
    temp=data[3:]-a2*data[1:-2]-a3*data[0:-3]

    a2+=alpha*sum(data[1:-2]*temp)
    a3+=alpha*sum(data[0:-3]*temp)

    J=SE(a2,a3,data)

    rec_J[i]=J  # save the estimates of a2 and a3 after iter i
    rec_a2a3[i,:]=[a2,a3]  # save the J after iter i

#-- show how J converges to the minimum with the gradient descent method
plt.plot(range(max_no_of_iter),rec_J,'r',linewidth=1)
plt.xlabel('iter i')
plt.ylabel('J')
plt.title('Cost J @ iter i')
plt.grid()
plt.show()

#-- show how a2 and a3 converge
plt.plot(range(max_no_of_iter),rec_a2a3[:,0],'g')
plt.plot(range(max_no_of_iter),rec_a2a3[:,1],'b')
plt.xlabel('iter i')
plt.title('a2 and a3 @ iter i')
plt.legend(['a2','a3'])
plt.grid()
plt.show()

print(a2)
print(a3)
print(J)

