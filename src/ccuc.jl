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
std = p_w_max*0.025;
var = std^2;
cov_mat = var .* convert.(Float64, Matrix(I, n_w, n_w));
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
true_C_f = [0.5*v_L*g for g = 1:n_g];

truthful_bidding = true;

if truthful_bidding == true
    C_Q = deepcopy(true_C_Q);
    C_L = deepcopy(true_C_L);
    C_f = deepcopy(true_C_f);
else
    fake_C_L, fake_C_f = zeros(Float64, n_g), zeros(Float64, n_g);
    fake_C_L[n_g] += 1.0;
    fake_C_f[n_g] += 0.5;
    C_Q = deepcopy(true_C_Q);
    C_L = deepcopy(true_C_L) .+ fake_C_L;
    C_f = deepcopy(true_C_f) .+ fake_C_f;
end


## Model

# MILP model (commitment decisions)

MILP_model = Model(Gurobi.Optimizer)

@variable(MILP_model, 0.0 <= p[g = 1:n_g])
@variable(MILP_model, 0.0 <= alpha[g = 1:n_g] <= 1.0)
@variable(MILP_model, z_MILP[g = 1:n_g], Bin)

@constraint(MILP_model, pb, sum(p[g] for g = 1:n_g) + W == D)
@constraint(MILP_model, rall, sum(alpha[g] for g = 1:n_g) == 1.0)
@constraint(MILP_model, ract[g = 1:n_g], alpha[g] <= z_MILP[g])
@constraint(MILP_model, minprd[g = 1:n_g], p_min[g] * z_MILP[g] <= p[g] - phi*agg_cov*alpha[g])
@constraint(MILP_model, maxprd[g = 1:n_g], p[g] + phi*agg_cov*alpha[g] <= p_max[g] * z_MILP[g])
@constraint(MILP_model, uprmp[g = 1:n_g], phi*agg_cov*alpha[g] <= R_up_max[g] * z_MILP[g])
@constraint(MILP_model, dwnrmp[g = 1:n_g], phi*agg_cov*alpha[g] <= R_down_max[g] * z_MILP[g])

@objective(MILP_model, Min, sum(C_Q[g]*(p[g]^2 + cov*alpha[g]^2) + C_L[g]*p[g] + C_f[g]*z_MILP[g] for g = 1:n_g))

optimize!(MILP_model)

z_opt = value.(z_MILP)

#  LP model (pricing)

LP_model = Model(Gurobi.Optimizer)

@variable(LP_model, 0.0 <= p[g = 1:n_g])
@variable(LP_model, 0.0 <= alpha[g = 1:n_g] <= 1.0)
@variable(LP_model, 0.0 <= z[g = 1:n_g] <= 1.0)

@constraint(LP_model, power_balance, sum(p[g] for g = 1:n_g) + W == D)
@constraint(LP_model, reserve_allocation, sum(alpha[g] for g = 1:n_g) == 1.0)
@constraint(LP_model, reserve_activation[g = 1:n_g], alpha[g] <= z[g])
@constraint(LP_model, min_prod[g = 1:n_g], p_min[g] * z[g] <= p[g] - phi*agg_cov*alpha[g])
@constraint(LP_model, max_prod[g = 1:n_g], p[g] + phi*agg_cov*alpha[g] <= p_max[g] * z[g])
@constraint(LP_model, max_up_ramp[g = 1:n_g], phi*agg_cov*alpha[g] <= R_up_max[g] * z[g])
@constraint(LP_model, max_down_ramp[g = 1:n_g], phi*agg_cov*alpha[g] <= R_down_max[g] * z[g])
@constraint(LP_model, commitment_cuts[g = 1:n_g], z[g] == z_opt[g])

@objective(LP_model, Min, sum(C_Q[g]*(p[g]^2 + cov*alpha[g]^2) + C_L[g]*p[g] + C_f[g]*z[g] for g = 1:n_g))

optimize!(LP_model)

## Post-processing

p_scheduled = value.(p);
alpha_scheduled = value.(alpha);
z_scheduled = z_opt;
electricity_price = dual(power_balance);
reserve_price = dual(reserve_allocation);
commitment_price = dual.(commitment_cuts);
load_payment = D*electricity_price;
wind_energy_revenue = W*electricity_price;
generator_energy_revenue = [(electricity_price*p_scheduled[g]) for g = 1:n_g];
generator_reserve_revenue = [(reserve_price*alpha_scheduled[g]) for g = 1:n_g];
generator_commitment_revenue = [(commitment_price[g]*z_scheduled[g]) for g = 1:n_g];
true_cost = [(true_C_Q[g]*(p_scheduled[g]^2 + cov*alpha_scheduled[g]^2) + true_C_L[g]*p_scheduled[g] + true_C_f[g]*z_scheduled[g]) for g = 1:n_g];
reported_cost = [(C_Q[g]*(p_scheduled[g]^2 + cov*alpha_scheduled[g]^2) + C_L[g]*p_scheduled[g] + C_f[g]*z_scheduled[g]) for g = 1:n_g];
true_profit = [(generator_energy_revenue[g] + generator_reserve_revenue[g] + generator_commitment_revenue[g] - true_cost[g]) for g = 1:n_g];
reported_profit = [(generator_energy_revenue[g] + generator_reserve_revenue[g] + generator_commitment_revenue[g] - reported_cost[g]) for g = 1:n_g];

println("\n")
println("Power generation: ", p_scheduled)
println("Reserve procurement: ", alpha_scheduled)
println("Commitment decisions: ", z_scheduled)
println("Electricity price: ", electricity_price)
println("Reserve price: ", reserve_price)
println("Commitment price: ", commitment_price)
println("Load Payment: ", load_payment)
println("Wind energy revenue: ", wind_energy_revenue)
println("Generators energy revenue: ", generator_energy_revenue)
println("Generators reserve revenue: ", generator_reserve_revenue)
println("Generators commitment revenue: ", generator_commitment_revenue)
println("True cost: ", true_cost)
println("Reported cost: ", reported_cost)
println("True profit: ", true_profit)
println("Reported profit: ", reported_profit)
println("\n")
