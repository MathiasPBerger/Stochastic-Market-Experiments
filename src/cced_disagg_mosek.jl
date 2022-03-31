using JuMP
using LinearAlgebra
using Distributions
using MosekTools

## Comments
#
# This script formulates the problem as a convex program with a convex quadratic
# objective to be minimised subject to affine and second-order cone constraints.
# It then solves the problem with Mosek. This has non-trivial implications
# for post-processing because of the primal and dual forms used by the solver.
# More precisely, Mosek uses conic programming duality (rather than Lagrangian
# duality), which means that the dual variables associated with second-order
# cone constraints are vector variables (belonging to the dual SOCP cone) and
# not scalar variables (as would be the case in Lagrangian duality for nonlinear
# programs). Theoretical results in the tex documents were derived based on
# Lagrangian duality and it is not entirely clear how conic dual variables
# relate to Lagrangian dual variables. For affine constraints, the
# interpretation of dual variables remains unchanged.

## Data

# Number of flexible generators and wind farms
n_g = 3;
n_w = 2;

# Flexible generation parameters
p_max = [200.0 for g = 1:n_g];
p_min = [0.0 for g = 1:n_g];
R_up_max = [p_max[g] for g = 1:n_g];
R_down_max = [p_max[g] for g = 1:n_g];

# Wind production parameters
p_w_max = [250.0 for i = 1:n_w];
mu = [0.0*p_w_max[i] for i = 1:n_w];
std = [0.1*p_w_max[i] for i = 1:n_w];
std[2]+= 0.25*p_w_max[2];
rho = -0.75;
corr_mat = [[1. rho]; [rho 1.]];
cov_mat = Diagonal(std) * corr_mat * Diagonal(std);
cov_sqrt = sqrt(cov_mat); # computes the matrix square root of covariance matrix to define SOC constraint
W = [0.8*p_w_max[i] for i = 1:n_w];

# Demand parameters
D = 700;

# Risk parameters
epsilon = 0.05;
phi = quantile(Normal(), (1-epsilon));
phi_inv = 1/phi;

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

model = Model(Mosek.Optimizer)

@variable(model, 0.0 <= p[g = 1:n_g])
@variable(model, 0.0 <= alpha[g = 1:n_g, i = 1:n_w] <= 1.0)

@constraint(model, power_balance, sum(p[g] for g = 1:n_g) + sum(W[i] for i = 1:n_w) == D)
@constraint(model, reserve_allocation[i = 1:n_w], sum(alpha[g, i] for g = 1:n_g) == 1.0)
@constraint(model, max_prod[g = 1:n_g], [phi_inv * (p_max[g] - p[g] + sum(alpha[g, i]*mu[i] for i = 1:n_w)); cov_sqrt*alpha[g, :]] in SecondOrderCone())
@constraint(model, min_prod[g = 1:n_g], [phi_inv * (p[g] - p_min[g] - sum(alpha[g, i]*mu[i] for i = 1:n_w)); cov_sqrt*alpha[g, :]] in SecondOrderCone())

@objective(model, Min, sum(C_Q[g]*((p[g] - alpha[g,:]'*mu)^2 + alpha[g,:]'*cov_mat*alpha[g,:]) + C_L[g]*(p[g] - alpha[g,:]'*mu) for g = 1:n_g))

## Solve

optimize!(model)

println(termination_status(model))
println(dual_status(model))

## Post-processing

p_scheduled = value.(p);
alpha_scheduled = value.(alpha);
electricity_price = dual(power_balance);
reserve_price = dual.(reserve_allocation);
dual_min_prod = dual.(min_prod);
dual_max_prod = dual.(max_prod);
load_payment = D*electricity_price;
wind_energy_revenue = [W[i]*electricity_price for i = 1:n_w];
wind_profit = [(wind_energy_revenue[i]-reserve_price[i]) for i = 1:n_w];
generator_energy_revenue = [(electricity_price*p_scheduled[g]) for g = 1:n_g];
generator_reserve_revenue = [(reserve_price'*alpha_scheduled[g,:]) for g = 1:n_g];
true_generator_cost = [(true_C_Q[g]*((p_scheduled[g] - alpha_scheduled[g,:]'*mu)^2 + alpha_scheduled[g,:]'*cov_mat*alpha_scheduled[g,:]) + true_C_L[g]*(p_scheduled[g] - alpha_scheduled[g,:]'*mu)) for g = 1:n_g];
reported_generator_cost = [(C_Q[g]*((p_scheduled[g] - alpha_scheduled[g,:]'*mu)^2 + alpha_scheduled[g,:]'*cov_mat*alpha_scheduled[g,:]) + C_L[g]*(p_scheduled[g] - alpha_scheduled[g,:]'*mu)) for g = 1:n_g];
true_generator_profit = [(generator_energy_revenue[g] + generator_reserve_revenue[g] - true_generator_cost[g]) for g = 1:n_g];
reported_generator_profit = [(generator_energy_revenue[g] + generator_reserve_revenue[g] - reported_generator_cost[g]) for g = 1:n_g];

sensitivity_coeffs = zeros(n_w, n_g);
for i = 1:n_w
    for k = 1:n_g
        sensitivity_coeffs[i, k] = std[i] * alpha_scheduled[k, i]^2 + sum((rho * std[j] * alpha_scheduled[k, j] * alpha_scheduled[k, i]) for j = 1:n_w if j != i)
    end
end

println("\n")
println("Power generation: ", p_scheduled)
println("Reserve procurement: ", alpha_scheduled)
println("Electricity price: ", electricity_price)
println("Reserve price: ", reserve_price)
println("Load Payment: ", load_payment)
println("Wind energy revenue: ", wind_energy_revenue)
println("Wind profit: ", wind_profit)
println("Generators energy revenue: ", generator_energy_revenue)
println("Generators reserve revenue: ", generator_reserve_revenue)
println("True generator cost: ", true_generator_cost)
println("Sensitivity Coefficients: ", sensitivity_coeffs)
if !truthful_bidding
    println("Reported generator cost: ", reported_generator_cost)
end
println("True generator profit: ", true_generator_profit)
if !truthful_bidding
    println("Reported generator profit: ", reported_generator_profit)
end
println("\n")
