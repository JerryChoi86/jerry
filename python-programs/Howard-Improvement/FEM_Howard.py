# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 23:32:47 2014

@author: Jerry
"""

from __future__ import division
from numpy import linspace, zeros, empty
from math import log
from pylab import plot, show, legend, title, xlabel
from scipy.interpolate import interp1d
from scipy.optimize import fminbound
import time

#----- Timer to calculate total running time
start = time.time()

#----- Parameters
A, alpha, beta = 2.0, 0.3, 0.99 

#----- The number of grid
n_grid = 30

#----- Tolerance level
tol = 1e-7

#----- Capital grid
k_min, k_max = 0.1, 0.9
k_grid = linspace(k_min,k_max,n_grid)


def linear_interp(x,y,xi) :
    """
    This function implements linear interpolation
    """
    
    f = interp1d(x,y,kind='linear')
    yi = f(xi)
    
    return yi


def quad_interp(x,y,xi) :
    """
    This function implements linear interpolation
    """
    f = interp1d(x,y,kind='quadratic')
    yi = f(xi)
    
    return yi


def cubic_interp(x,y,xi) :
    """
    This function implements linear interpolation
    """
    
    f = interp1d(x,y,kind='cubic')
    yi = f(xi)
    
    return yi


def runtime_cal(start,end) :
    """
    This function calculates total running time for given starting and ending time. 
    
    Input : start (starting time)
            end (ending time)
            
    Output mm (minute)
           ss (second)
    """
    run_time = end - start
    mm = int(run_time/60)
    ss = round(run_time%60)
    return mm, ss

def Howard(policy,value_next,n_h,method) :
    """
    This function implements Howard's Improvement Algorithm. 
    
    Input : policy (computed policy function for each state)
            value_next (computed value function for each state)
            n_h (The number of repeatition)
            
    Output : value_next (Computed value function resulting from Howard's improvement)
             policy (Corresponding policy function)
    """
    
    v_hat = empty(n_grid)
    for repeat in range(n_h) :
        for index in range(n_grid) :
            if method==1 :
                v_hat[index] = log(A*k_grid[index]**(alpha) - policy[index]) + beta*linear_interp(k_grid,value_next,policy[index])
            elif method==2 :
                v_hat[index] = log(A*k_grid[index]**(alpha) - policy[index]) + beta*quad_interp(k_grid,value_next,policy[index])
            elif method==3 :
                v_hat[index] = log(A*k_grid[index]**(alpha) - policy[index]) + beta*cubic_interp(k_grid,value_next,policy[index])
        value_next = v_hat
        v_hat = empty(n_grid)
    return value_next, policy
    

def VFI(method) :
    """
    This Function implements value function iteration 
    using three different interpolation method ; linear, spline, cubic.
    
    Input : method (1=linear, 2=spline, 3=cubic)
    
    Output : v_k (value function)
             g_k (policy function : saving function)
             runtime (total runnning time)
    """
    iteration=0     # Iteration Counter
    converged = 0     # Convergence Flag|
    
#----- Initial Settings 
    v_update = zeros(n_grid)
    v_func = empty(n_grid)
    k_next_vec = empty(n_grid)
    run_time = empty(2)
    
    def obj(k_next) :
        """
        This function is used in value function iteration.
        It represents the objective function to be maximized for one node (state) of current capitals.
        Resulting value is maximized one corresponding to next period's capital as a maximizer. 
        Next period's value is computed by interpolation.
        
        Input : k_next (next period's capital)
        
        Output : value_vec (maximized value resulting from choosing optimal capital in the next period)
        """ 
        
        if method==1 :
            value_vec = -1 * (log(A*k_current**(alpha) - k_next) + beta*linear_interp(k_grid,v_update,k_next))
        elif method==2 :
            value_vec = -1 * (log(A*k_current**(alpha) - k_next) + beta*quad_interp(k_grid,v_update,k_next))
        elif method==3 :
            value_vec = -1 * (log(A*k_current**(alpha) - k_next) + beta*cubic_interp(k_grid,v_update,k_next))
            
        return value_vec

#----- Value function iteration
    start = time.time() # start time
    while converged==0 :
        index = 0
        for k_current in k_grid :
            k_next = fminbound(obj,k_grid[0],k_grid[-1])
            v_func[index] = (-1) * obj(k_next)
            k_next_vec[index] = k_next
            index = index + 1
        v_func, k_next_vec = Howard(k_next_vec,v_func,3,method)
        dist = abs(max(v_func - v_update))
        if dist<tol :
            converged = 1
            v_k, g_k = v_func, k_next_vec
        v_update = v_func
        print "Iteration : ",iteration,"","Distance : ",dist    # convergence process
        iteration = iteration + 1
        v_func = empty(n_grid)  
        k_next_vec = empty(n_grid)
    
    end = time.time()   # end time
    run_time[0],run_time[1] = runtime_cal(start,end)    # total running time
        
    return v_k, g_k, run_time, iteration

    
#-----
#-----True Value Function
#-----
v_true = empty(n_grid)
E = (1/(1-beta))*(log(A*(1-alpha*beta))+((alpha*beta)/(1-alpha*beta))*log(alpha*beta*A))
F = (alpha/(1-alpha*beta))
for i in range(n_grid) :
    v_true[i] = E + F*log(k_grid[i])

#-----
#-----True Policy Function
#-----
g_true = alpha*beta*A*k_grid**(alpha)



#-----
#-----Value Function Iteration for 3 Different Interpolation Methods
#-----
v_linear, g_linear, run_linear, iter_linear = VFI(method=1)
v_spline, g_spline, run_spline, iter_quad = VFI(method=2)
v_cubic, g_cubic, run_cubic, iter_cubic = VFI(method=3)


#-----
#-----Approximation Errors
#-----

app_linear = abs(v_true-v_linear)
app_spline = abs(v_true-v_spline)
app_cubic = abs(v_true-v_cubic)



#-----
#-----Figures
#-----

#----- Value Functions
plot(k_grid,v_linear,'b*',label='Value Function (linear)')
plot(k_grid,v_true,'r',label = 'True Value Function')
xlabel('k')
title('Linear Interpolation')
legend()
show()

plot(k_grid,v_spline,'g+',label='Value Function (spline)')
plot(k_grid,v_true,'r',label = 'True Value Function')
xlabel('k')
title('Quadratic Interpolation')
legend()
show()

plot(k_grid,v_cubic,'ko',label='Value Function (cubic)')
plot(k_grid,v_true,'r',label = 'True Value Function')
xlabel('k')
title('Cubic Interpolation')
legend()
show()

plot(k_grid,g_linear,'b*',label='Policy Function (linear)')
plot(k_grid,g_true,'r',label='True Policy Function')
xlabel('k')
title('Linear Interpolation')
legend()
show()

#----- Policy Functions
plot(k_grid,g_spline,'g+',label='Policy Function (spline)')
plot(k_grid,g_true,'r',label='True Policy Function')
xlabel('k')
title('Quadratic Interpolation')
legend()
show()

plot(k_grid,g_cubic,'ko',label='Policy Function (cubic)')
plot(k_grid,g_true,'r',label='True Policy Function')
xlabel('k')
title('Cubic Interpolation')
legend()
show()

plot(k_grid,app_linear,'b',label='Approximation Error (linear)')
title('Linear Interpolation')
xlabel('k')
legend()
show()

#----- Approximation Error
plot(k_grid,app_spline,'g',label='Approximation Error (spline)')
title('Quadratic Interpolation')
xlabel('k')
legend()
show()

plot(k_grid,app_cubic,'k',label='Approximation Error (cubic)')
xlabel('k')
title('Cubic Interpolation')
legend()
show()
