using JuMP
using Gurobi
using LinearAlgebra
using Distributions

## Data

# Number of flexible generators and wind farms
n_g = 10;
n_w = 6;

# Flexible generation parameters
p_g_max = 200.0
p_g_min = 0.0

# Wind production parameters
p_w_max = 200.0;
std = 200*0.075;
var = std^2;
cov_mat = var .* convert.(Float64, Matrix(I, n, n));
cov = sum(cov_mat);
W =

# Demand parameters
D = 1000

# Risk parameters
epsilon = 0.05


# Cost parameters
C_Q = [10 for i = 1:n]
C_L = [3 for i = 1:n]


## Model

model = Model(Gurobi.Optimizer)

@variable(model, p_g[i = 1:n] .>= 0)
@variable(model, alpha_g[1=1:n] .>= 0)
