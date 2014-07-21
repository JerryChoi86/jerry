from __future__ import division
from numpy import linspace, zeros, empty
from math import log
from matplotlib.mlab import find
from pylab import plot, show, legend, xlabel
import time


start = time.time()

## Parameters
A, alpha, beta = 2.0, 0.3, 0.99 

## The number of grid
n_grid = 30

## Tolerance
tol = 1e-7

## Grid
k_min, k_max = 0.1, 0.9
k_grid = linspace(k_min,k_max,n_grid)

## Iteration Counter
it=0

## Convergence Flag|
converged = 0
iteration = 0

## Initials 
v_update = zeros(n_grid)
v_func = empty(n_grid)
k_star_index = empty(n_grid)



def v_grid_search(k_current,grid,value_next) :
    value_vec = empty(len(grid))
    for i in range(len(grid)) :
        k_next = grid[i]
        value_vec[i] = log(A*k_current**(alpha) - k_next) + beta*value_next[i]
    value = max(value_vec)
    max_index = find(value==value_vec[:])
    return value, max_index


while converged == 0 :
	k_index = 0
	for k_state in k_grid :
		v_func[k_index], k_star_index[k_index] = v_grid_search(k_state,k_grid,v_update)
		k_index = k_index + 1
	
	dist = abs(max(v_func-v_update))
	v_update = v_func
	if dist<tol :
		converged = 1
		v_k, g_k = v_func, k_grid[list(k_star_index)]

	print "Iteration : ",iteration,"","Distance : ",dist
	iteration = iteration + 1
	v_func = empty(n_grid)

end = time.time() 
print "Total Running Time: ", end-start,"Sec"


# True Value Function
v_true = empty(n_grid)
E = (1/(1-beta))*(log(A*(1-alpha*beta))+((alpha*beta)/(1-alpha*beta))*log(alpha*beta*A))
F = (alpha/(1-alpha*beta))
for i in range(n_grid) :
    v_true[i] = E + F*log(k_grid[i])
end

# True Policy Function
g_true = alpha*beta*A*k_grid**(alpha)


## Approximation Error
app_err = v_true-v_k

plot(k_grid,v_k,'*',label='Value Function')
plot(k_grid,v_true,label = 'True Value Function')
legend()
xlabel('k')
show()

plot(k_grid,g_k,'o',label='Policy Function')
plot(k_grid,g_true,label='True Policy Function')
xlabel('k')
legend()
show()

plot(k_grid,app_err,label='Approximation Error')
xlabel('k')
legend()
show()
