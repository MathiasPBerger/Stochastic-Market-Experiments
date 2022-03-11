using JuMP
using Gurobi
#using MosekTools
using LinearAlgebra
using Distributions

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
#mu_pc = [0.05 for i = 1:n_w];
#std_pc = [0.075 for i = 1:n_w];
#mu = [p_w_max[i]*mu_pc[i] for i = 1:n_w];
#std = [p_w_max[i]*std_pc[i] for i = 1:n_w];
#var = std^2;
mu = [0.05*p_w_max[i] for i = 1:n_w];
std = [0.01*p_w_max[i] for i = 1:n_w];
rho = 0.0;
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

model = Model(Gurobi.Optimizer)
#set_optimizer_attribute(model, "QCPDual", 1)

@variable(model, 0.0 <= p[g = 1:n_g])
@variable(model, 0.0 <= alpha[g = 1:n_g, i = 1:n_w] <= 1.0)

@constraint(model, power_balance, sum(p[g] for g = 1:n_g) + sum(W[i] for i = 1:n_w) == D)
@constraint(model, reserve_allocation[i = 1:n_w], sum(alpha[g, i] for g = 1:n_g) == 1.0)
@constraint(model, max_prod[g = 1:n_g], [phi_inv * (p_max[g] - p[g] + sum(alpha[g, i]*mu[i] for i = 1:n_w)); cov_sqrt*alpha[g, :]] in SecondOrderCone())
@constraint(model, min_prod[g = 1:n_g], [phi_inv * (p[g] - p_min[g] - sum(alpha[g, i]*mu[i] for i = 1:n_w)); cov_sqrt*alpha[g, :]] in SecondOrderCone())

@objective(model, Min, sum(C_Q[g]*((p[g] - alpha[g,:]'*mu)^2 + alpha[g,:]'*cov_mat*alpha[g,:]) + C_L[g]*(p[g] - alpha[g,:]'*mu) for g = 1:n_g))

## Solve

optimize!(model)

## Post-processing

p_scheduled = value.(p);
alpha_scheduled = value.(alpha);
electricity_price = dual(power_balance);
reserve_price = dual.(reserve_allocation);
load_payment = D*electricity_price;
wind_energy_revenue = [W[i]*electricity_price for i = 1:n_w];
wind_profit = [(wind_energy_revenue[i]-reserve_price[i]) for i = 1:n_w];
generator_energy_revenue = [(electricity_price*p_scheduled[g]) for g = 1:n_g];
generator_reserve_revenue = [(reserve_price'*alpha_scheduled[g,:]) for g = 1:n_g];
true_generator_cost = [(true_C_Q[g]*((p_scheduled[g] - alpha_scheduled[g,:]'*mu)^2 + alpha_scheduled[g,:]'*cov_mat*alpha_scheduled[g,:]) + true_C_L[g]*(p_scheduled[g] - alpha_scheduled[g,:]'*mu)) for g = 1:n_g];
reported_generator_cost = [(C_Q[g]*((p_scheduled[g] - alpha_scheduled[g,:]'*mu)^2 + alpha_scheduled[g,:]'*cov_mat*alpha_scheduled[g,:]) + C_L[g]*(p_scheduled[g] - alpha_scheduled[g,:]'*mu)) for g = 1:n_g];
true_generator_profit = [(generator_energy_revenue[g] + generator_reserve_revenue[g] - true_generator_cost[g]) for g = 1:n_g];
reported_generator_profit = [(generator_energy_revenue[g] + generator_reserve_revenue[g] - reported_generator_cost[g]) for g = 1:n_g];

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
if !truthful_bidding
    println("Reported generator cost: ", reported_generator_cost)
end
println("True generator profit: ", true_generator_profit)
if !truthful_bidding
    println("Reported generator profit: ", reported_generator_profit)
end
println("\n")
