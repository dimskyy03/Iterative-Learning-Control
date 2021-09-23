#!/usr/bin/env python
# coding: utf-8

# In[9]:


from control import *
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import *


# In[10]:


#example transfer function

num = np.array([0, 1, 2.19, 1.174]) * 7.634e-4
den = np.array([1, -1.079, -0.541, 0.623])

#numplus = np.array([1, 0])
#nummin = num = np.array([0, 1, 1.5])

#denplus = np.array([1, -0.5])
#denmin = np.array([0, 1, -0.8])

A,B,C,D = tf2ss(num,den)


# In[11]:


tf2zpk(num,den)


# In[12]:


def trapezoid_signal(t, width=2., slope=1., amp=1., offs=0):
    a = slope*width*signal.sawtooth(2*np.pi*t/width, width=0.5)/4.
    a[a>amp/2.] = amp/2.
    a[a<-amp/2.] = -amp/2.
    return a + amp/2. + offs


# In[13]:


#reference signal
#t = np.arange(0, 50)
t = np.linspace(0,1,50)
#yd = np.sin(np.pi * t/20)
yd = -5 * t * np.sin((2 * np.pi *t))
#yd = 1 + signal.sawtooth(np.pi * t/25,width=0.5) #; yd[49] = 0
#yd = trapezoid_signal(t, width=10., slope=2., amp=1., offs=0.)

plt.xlabel('Waktu (s)')
plt.ylabel('Posisi sudut (derajat)')
plt.plot(t,yd,'-')


# In[6]:


#markov parameters J
from numpy.linalg import matrix_power
from numpy.linalg import multi_dot

degree = 1
nmp = 1

J = np.zeros(shape=(len(t),len(t)))

for i in range(len(t)):
    for j in range(len(t)):
        if i == j :
            J[i][j] = multi_dot([C,matrix_power(A, degree+nmp-1),B])
        elif j-i == 1:
            J[i][j] = multi_dot([C,matrix_power(A, degree+nmp-2),B])
        for k in range(len(t)):
            if j < i and i - j == k:
                J[i][j] = multi_dot([C,matrix_power(A, k+degree+nmp-1),B])
print(J)


# In[14]:


#markov learning function Lp
#np.set_printoptions(threshold=np.inf)

num1 = np.array([0, 1446, -2670.729, 1228.06])
den1 = np.array([1, 1.950, 0.951, 0])

A1,B1,C1,D1 = tf2ss(num1,den1)

Lp = np.zeros(shape=(len(t),len(t)))

for i in range(len(t)):
    for j in range(len(t)):
        if i == j :
            Lp[i][j] = np.dot(C1,B1)
        for k in range(len(t)):
            if j < i and i - j == k:
                Lp[i][j] = multi_dot([C1,matrix_power(A1, k),B1])
print(Lp)


# In[27]:


tf2zpk(num1,den1)


# In[15]:


#matriks Q filter

Q = np.zeros(shape=(len(t),len(t)))

val = np.array([0.25, 0.5, 0.25])

for i in range(len(t)):
    for j in range(len(t)):
        if i - j == 1:
            Q[i][j] = 0.25
        elif i == j :
            Q[i][j] = 0.5
        elif j - i == 1:
            Q[i][j] = 0.25

print(Q)
        


# In[57]:


plt.plot(t,yd,'-k',label='Yd')
#first test signal
Kp = np.zeros(shape=(len(t),len(t))) #matriks NxN proportional gain
np.fill_diagonal(Kp, 0.5)

Kd = np.zeros(shape=(len(t),len(t))) #matriks NxN derivative gain
np.fill_diagonal(Kd, 0.01)

#new input uk
u0 = np.zeros(shape=(len(t),1)) #matriks Nx1

#error reference
error = []

iterasi = 1
iter_save = []
yk_save = []
while True :
    Yk = np.dot(J,u0)
    e0 = yd - np.transpose(Yk)[0] ; e0[0] = 0
    e0 = np.reshape(e0,(len(t),1))
    errval = np.sqrt(np.sum(e0**2)/(len(e0)))
    error.append(errval)
    
    #try:
        #u0 = u0 + np.dot(Lp[iterasi-1],e0) #+ np.dot(Kd,d_error(e0).reshape((len(t),1)))
    #except IndexError :
        #u0 = u0 + np.dot(np.zeros(shape=(len(t),len(t))),e0)
    
    u0 = np.dot(Q, (u0 + np.dot(Lp,e0))) 
    #u0 = u0 + np.dot(Lp,e0)
    
    iter_save.append(iterasi)
    yk_save.append(Yk)
    iterasi = iterasi + 1
    plt.plot(t,Yk,'--',label=f"Yk ke {iterasi-1}")
    #if errval < 1e-2 : break
    if iterasi > 10 : break


plt.xlabel('Iteration')
plt.ylabel('Yd & Yk')
#print(Yk)


# In[58]:


plt.plot(iter_save, error,'--')
#plt.title('Kp = 0.5')
plt.xlabel('Iteration')
plt.ylabel('RMSE')

#plt.savefig('first_rmse.png')


# In[54]:


plt.plot(t,yd,'ok',label='Desired trajectory')
plt.plot(t,yk_save[1],'-r',label='1st Iteration')
plt.plot(t,yk_save[2],'-g',label='2nd Iteration')
plt.plot(t,yk_save[5],'-b',label='5th Iteration')

#plt.title('Simulasi ILC dengan Kp=0.5 dan fungsi Q-filter')
plt.xlabel('Waktu (s)')
plt.ylabel('Posisi sudut (derajat)')
plt.legend()

#plt.savefig('third_simulation_kp_0.5.png')
#plt.savefig('first_simulation.png')
#plt.savefig('second_simulation.png')


# In[ ]:





# In[ ]:




