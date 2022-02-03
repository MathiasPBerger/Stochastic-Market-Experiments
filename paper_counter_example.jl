using JuMP
using Gurobi
using LinearAlgebra
using Distributions

## Data

# Number of flexible generators and wind farms
n_g = 3;
n_w = 1;

# Flexible generation parameters
p_max = [200.0 for g = 1:n_g];
p_min = [0.0 for g = 1:n_g];
R_up_max = [p_max[g] for g = 1:n_g];
R_down_max = [p_max[g] for g = 1:n_g];

# Wind production parameters
p_w_max = 250.0;
std = p_w_max*0.075;
var = std^2;
cov_mat = var .* convert.(Float64, Matrix(I, n_g, n_g));
cov = sum(cov_mat);
agg_cov = sqrt(cov);
W = 0.8*p_w_max*n_w;

# Demand parameters
D = 700;

# Risk parameters
epsilon = 0.05;
phi = quantile(Normal(), (1-epsilon));

# Cost parameters
v_Q, s_Q = 0.1, 0.05;
v_L, s_L = 1.0, 0.1;
true_C_Q = [(v_Q + s_Q*g) for g = 1:n_g];
true_C_L = [(v_L + s_L*g) for g = 1:n_g];

truthful_bidding = true;

if truthful_bidding == true
    C_Q = deepcopy(true_C_Q);
    C_L = deepcopy(true_C_L);
else
    fake_C_L = zeros(Float64, n_g);
    fake_C_L[n_g] += 1.0;
    C_Q = deepcopy(true_C_Q);
    C_L = deepcopy(true_C_L) .+ fake_C_L;
end


## Model

model = Model(Gurobi.Optimizer)

@variable(model, 0.0 <= p[g = 1:n_g])
@variable(model, 0.0 <= alpha[g = 1:n_g] <= 1.0)

@constraint(model, power_balance, sum(p[g] for g = 1:n_g) + W == D)
@constraint(model, reserve_allocation, sum(alpha[g] for g = 1:n_g) == 1.0)
@constraint(model, min_prod[g = 1:n_g], p_min[g] <= p[g] - phi*agg_cov*alpha[g])
@constraint(model, max_prod[g = 1:n_g], p[g] + phi*agg_cov*alpha[g] <= p_max[g])
@constraint(model, max_up_ramp[g = 1:n_g], phi*agg_cov*alpha[g] <= R_up_max[g])
@constraint(model, max_down_ramp[g = 1:n_g], phi*agg_cov*alpha[g] <= R_down_max[g])

@objective(model, Min, sum(C_Q[g]*(p[g]^2 + cov*alpha[g]^2) + C_L[g]*p[g] for g = 1:n_g))

## Solve

optimize!(model)

## Post-process

p_scheduled = value.(p);
alpha_scheduled = value.(alpha);
electricity_price = dual(power_balance);
reserve_price = dual(reserve_allocation);
energy_revenue = [(electricity_price*p_scheduled[g]) for g = 1:n_g];
reserve_revenue = [(reserve_price*alpha_scheduled[g]) for g = 1:n_g];
true_cost = [(true_C_Q[g]*(p_scheduled[g]^2 + cov*alpha_scheduled[g]^2) + true_C_L[g]*p_scheduled[g]) for g = 1:n_g];
reported_cost = [(C_Q[g]*(p_scheduled[g]^2 + cov*alpha_scheduled[g]^2) + C_L[g]*p_scheduled[g]) for g = 1:n_g];
true_profit = [(energy_revenue[g] + reserve_revenue[g] - true_cost[g]) for g = 1:n_g];
reported_profit = [(energy_revenue[g] + reserve_revenue[g] - reported_cost[g]) for g = 1:n_g];

println("\n")
println("Power generation: ", p_scheduled)
println("Reserve procurement: ", alpha_scheduled)
println("Electricity price: ", electricity_price)
println("Reserve price: ", reserve_price)
println("Energy revenue: ", energy_revenue)
println("Reserve revenue: ", reserve_revenue)
println("True cost: ", true_cost)
println("Reported cost: ", reported_cost)
println("True profit: ", true_profit)
println("Reported profit: ", reported_profit)
println("\n")
