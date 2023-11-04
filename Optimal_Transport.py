import numpy as np

from scipy.optimize._milp import milp
from scipy.optimize._constraints import LinearConstraint

def find_optimal_transport(C, G, P):
    """"Inputs:

    City information C: Array[N,4]. A table listing the price and amount of commodities that a city wants to buy or sell. 
    Example row: (99, 101, 100, 200), saying the city buys the commodity at a unit price of 99 up to 100 units and sells the commodity 
    at a unit price of 101 up to 200 units. (the buying price, the selling price, the maximum demand and the maximum supply of the commodity.)

    Transportation costs G: Array[N,N]. A table which represents the cost to move a unit of commodity from one city to another. G[i,j] 
    in the array represents the cost to move a unit of commodity from city i to city j, where i and j are between 0 to N-1.

    Failure probability P: Array[N,N]. A table which represents the probability of a route being closed, so the gain loss from that 
    route is set to zero. P[i,j] in the array represents the failure probability of the route from city i to city j, where i and j are 
    between 0 and N-1.

    Output:

    Transportation plans: Array[X,3]. A table that describes how many units to move from one city to another.
    Example row: (0, 1, 500), saying we move 500 units from city 0 to city 1.

    All inputs are represented as numpy arrays and the output should also be returned as a numpy array."""

    # 
    buy_price = C[:,0]
    sell_price = C[:,1]
    max_demand = C[:,2]
    max_supply = C[:,3]
    N = len(C)

    # 
    A_ub = np.zeros((2*N, N*(N-1)))
    b_u = np.zeros(2*N)

    #
    for i in range(N):
        A_ub[i, i*(N-1):(i+1)*(N-1)] = 1
        temp_2 = np.zeros((N,N-1))
        temp_matrix = np.zeros((N,N))
        temp_matrix[:, i] = 1
        for k in range(N):
            temp_2[k,:] = np.concatenate((temp_matrix[k,:k], temp_matrix[k,k+1:]))
        
        A_ub[N+i, :] = temp_2.reshape((-1))
               
    b_u[:N] = max_demand
    b_u[N:] = max_supply
    b_l = np.full_like(b_u, -np.inf)

    c = np.zeros((N,N-1))
    for i in range(N):
        for j in range(N-1):
            if i <= j:
                k = j + 1
            else:
                k = j
            c[i,j] = -max((sell_price[k] - buy_price[i] - G[i,k]) * (1-P[i,k]),0)
    
    integrality = np.ones(N *(N-1))
    constraints = LinearConstraint(A_ub, b_l, b_u)
    x_opt = milp(c.reshape(-1), constraints = constraints, integrality = integrality).x.reshape((N,N-1))

    output = []

    for i, row in enumerate(x_opt):
        for j, x in enumerate(row):
            if x > 1e-10:
                if i <= j:
                    j += 1
                output.append([i,j,x])

    return np.array(output)

if __name__ == "__main__":
    C = np.array([[98,100,100,100],
[98,100,100,100],
[102,103,100,100],
[102,103,100,100],
[102,103,100,100],])
    G = np.array([[0,1,1,1,1],
[1,0,1,1,1],
[1,1,0,1,1],
[1,1,1,0,1],
[1,1,1,1,0],])
    P = np.array([[0,0,0.5,0,0],
[0,0,0,0,0],
[0.5,0,0,0,0],
[0,0,0,0,0],
[0,0,0,0,0],])
    print(find_optimal_transport(C, G, P))
