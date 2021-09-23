#!/usr/bin/env python
# coding: utf-8

# In[1]:


from control import *
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import *
from scipy.signal import cont2discrete, lti, dlti, dstep


# In[2]:


#reference signal
t = np.linspace(0, 1, 50)

yd = 9*np.sin(3.1*t) ; yd[49] = 0

plt.xlabel('Waktu (s)')
plt.ylabel('Posisi sendi (derajat)')
plt.plot(t,yd,'-')


# In[3]:


#example transfer function

numj1 = np.array([0, 0.2622, 1624])
denj1 = np.array([1, 39.42, 1761])

numj2 = np.array([0, 0.2101, 1312])
denj2 = np.array([1, 35.15, 1374])

numj3 = np.array([0, 0.6385, 1285])
denj3 = np.array([1, 35.74, 1380])

Aj1,Bj1,Cj1,Dj1 = tf2ss(numj1,denj1)

Aj2,Bj2,Cj2,Dj2 = tf2ss(numj2,denj2)

Aj3,Bj3,Cj3,Dj3 = tf2ss(numj3,denj3)


# In[4]:


#tf2zpk(num,den)


# In[5]:


dt = 1/len(t)

Adj1,Bdj1,Cdj1,Ddj1,_ = cont2discrete((Aj1, Bj1, Cj1, Dj1), dt, method='bilinear')

Adj2,Bdj2,Cdj2,Ddj2,_ = cont2discrete((Aj2, Bj2, Cj2, Dj2), dt, method='bilinear')

Adj3,Bdj3,Cdj3,Ddj3,_ = cont2discrete((Aj3, Bj3, Cj3, Dj3), dt, method='bilinear')


# In[6]:


zj1,pj1 = ss2tf(Adj1,Bdj1,Cdj1,Ddj1)

zj2,pj2 = ss2tf(Adj2,Bdj2,Cdj2,Ddj2)

zj3,pj3 = ss2tf(Adj3,Bdj3,Cdj3,Ddj3)


# In[7]:


#from plot_zplane import zplane
#fig, axs = plt.subplots(3, 1,figsize=(30,25),squeeze=False)


#zplane(zj1[0],pj1,'plotj1.png',title='Plot Z-plane J1')

#zplane(zj2[0],pj2,'plotj2.png',title='Plot Z-plane J2')

#zplane(zj3[0],pj3,'plotj3.png',title='Plot Z-plane J3')


# In[150]:


#markov parameters H for minimum phase
#np.set_printoptions(threshold=np.inf)
from numpy.linalg import matrix_power
from numpy.linalg import multi_dot

H = np.zeros(shape=(len(t),len(t)))

for i in range(len(t)):
    for j in range(len(t)):
        if i == j :
            H[i][j] = np.dot(Cdj3,Bdj3)
        for k in range(len(t)):
            if i - j == k:
                H[i][j] = multi_dot([Cdj3,matrix_power(Adj3, k),Bdj3])
print(H)


# In[194]:


#derivative gain
def d_error(e,t=len(t),dt = 0.3):
    de = np.zeros(t)
    for i in range(t):
        if i == 1:
            de[i] = (e[i+1] - e[i])/(dt)
        elif i == t-1:
            de[i] = (e[i] - e[i-1])/(dt)
        else :
            de[i] = (e[i+1] - e[i-1])/(2*dt)
            
    return de


# In[195]:


plt.plot(t,yd,'o')
#first test signal
Kp = np.zeros(shape=(len(t),len(t))) #matriks NxN proportional gain
np.fill_diagonal(Kp, 0.5)

Kd = np.zeros(shape=(len(t),len(t))) #matriks NxN derivative gain
np.fill_diagonal(Kd, 0.3)

#new input uk
u0 = np.zeros(shape=(len(t),1)) #matriks Nx1

#error reference
error = []

iterasi = 1
iter_save = []
yk_save = []
while True :
    Yk = np.dot(H,u0)
    e0 = yd - np.transpose(Yk)[0] ; e0[0] = 0
    e0 = np.reshape(e0,(len(t),1))
    errval = np.sqrt(np.sum(e0**2)/(len(e0)))
    error.append(errval)
    
    #try:
        #u0 = u0 + np.dot(Lp[iterasi-1],e0) #+ np.dot(Kd,d_error(e0).reshape((len(t),1)))
    #except IndexError :
        #u0 = u0 + np.dot(np.zeros(shape=(len(t),len(t))),e0)
    
    u0 = u0 + np.dot(Kp,e0) + np.dot(Kd,d_error(e0).reshape((len(t),1)))
    #u0 = np.dot(Q, (u0 + np.dot(Kp,e0))) 
    
    yk_save.append(Yk)
    iter_save.append(iterasi)
    iterasi = iterasi + 1
    plt.plot(t,Yk)
    #if errval < 1e-2 : break
    if iterasi > 15 : break

print(errval)


# In[182]:


plt.plot(iter_save, error,'--')
#plt.title('Root Mean Square Error untuk Kp = 0.1')
plt.xlabel('Iteration')
plt.ylabel('RMSE')


# In[184]:


plt.plot(t,yd,'ok',label='Desired trajectory')
plt.plot(t,yk_save[1],'-r',label='1st Iteration')
plt.plot(t,yk_save[5],'-g',label='5th Iteration')
plt.plot(t,yk_save[10],'-b',label='10th Iteration')
#plt.plot(t,yk_save[50],'-y',label='50th Iteration')

plt.title('Hasil Simulasi J3 dengan nilai Kd=0.2')
plt.xlabel('Waktu (s)')
plt.ylabel('Posisi sendi (derajat)')
plt.legend(loc = 'upper right',fontsize='7.4')

#plt.savefig('third_simulation_kp_0.5.png')
#plt.savefig('first_simulation.png')
#plt.savefig('second_simulation.png')


# In[196]:


#calculate eigenvalue
Kp = np.zeros(shape=(len(t),len(t))) #matriks NxN proportional gain
np.fill_diagonal(Kp, 0.5)

Kd = np.zeros(shape=(len(t),len(t))) #matriks NxN derivative gain
np.fill_diagonal(Kd, 0.2)

#np.dot((Kp+Kd),H)
spectral = np.identity(len(t)) - np.dot((Kp+Kd),H)

eig,_ = np.linalg.eig(spectral)
print (np.max(eig))






