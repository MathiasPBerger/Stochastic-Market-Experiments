using JuMP
using Gurobi
using LinearAlgebra
using Distributions

## Data

# Number of flexible generators and wind farms
n_g = 3;
n_w = 1;

# Flexible generation parameters
p_max = 200.0;
p_min = 0.0;
R_D_max = 200.0;
R_U_max = 200.0;

# Wind production parameters
p_w_max = 20.0;
std = p_w_max*0.075;
var = std^2;
cov_mat = var .* convert.(Float64, Matrix(I, n, n));
cov = sum(cov_mat);
agg_cov = sqrt(cov);
W = 0.8*p_w_max*n_w;

# Demand parameters
D = 1000;

# Risk parameters
epsilon = 0.05;
phi = quantile(Normal(), (1-epsilon));

# Cost parameters
v_Q, s_Q = 0.1, 0.05;
v_L, s_L = 1.0, 0.1;
C_Q = [(v_Q + s_Q*g) for g = 1:n];
C_L = [(v_L + s_L*g) for g = 1:n];


## Model

model = Model(Gurobi.Optimizer)

@variable(model, 0.0 <= p[g = 1:n])
@variable(model, 0.0 <= alpha[g = 1:n] <= 1.0)

@constraint(model, power_balance, sum(p[g] for g = 1:n) + W == D)
@constraint(model, reserve_allocation, sum(alpha[g] for g = 1:n) == 1.0)
@constraint(model, min_prod[g = 1:n], p_min[g] <= p[g] - phi*agg_cov*alpha[g])
@constraint(model, max_prod[g = 1:n], p[g] + phi*agg_cov*alpha[g] <= p_max[g])
@constraint(model, down_ramp[g = 1:n], -R_D_max <= phi*agg_cov*alpha[g])
@constraint(model, up_ramp[g = 1:n], phi*agg_cov*alpha[g] <= R_U_max)

@objective(model, Min, sum(C_Q[g]*(p[g]^2 + cov*alpha[g]^2) + C_L[g]*p[g] for g = 1:n))
