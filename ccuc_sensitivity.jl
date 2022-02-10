using JuMP
using Gurobi
using LinearAlgebra
using Distributions
using Plots

## Comments
#
# This formulation works under the assumption that the source of uncertainty
# (here the forecast error) follows a Gaussian distribution whose mean and
# variance are known.

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
W = 0.8*p_w_max*n_w;

# Demand parameters
D = 700;

# Risk parameters
epsilon = 0.05;

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

# Sensitivity parameters

S = 10; # number of scenarios

mu = 0.0; # mean forecast error
agg_cov_vec = [p_w_max*(0.1 + 0.01*s) for s = 1:S]; # standard deviation of forecast error
cov_vec = agg_cov_vec.^2;
Q_eps = quantile(Normal(), (1-epsilon));

## Sensitivity Analysis

MILP_model = Model(Gurobi.Optimizer)
LP_model = Model(Gurobi.Optimizer)

p_scheduled = zeros(Float64, n_g, S);
alpha_scheduled = zeros(Float64, n_g, S);
z_scheduled = zeros(Float64, n_g, S);
electricity_price = zeros(Float64, S);
reserve_price = zeros(Float64, S);
commitment_price = zeros(Float64, n_g, S);
load_payment = zeros(Float64, S);
wind_energy_revenue = zeros(Float64, S);
generator_energy_revenue = zeros(Float64, n_g, S);
generator_reserve_revenue = zeros(Float64, n_g, S);
generator_commitment_revenue = zeros(Float64, n_g, S);
true_cost = zeros(Float64, n_g, S);
reported_cost = zeros(Float64, n_g, S);
true_profit = zeros(Float64, n_g, S);
reported_profit = zeros(Float64, n_g, S);

for s = 1:S

    phi_n = Q_eps * agg_cov_vec[s] - mu;
    phi_p = Q_eps * agg_cov_vec[s] + mu;

    # Define MILP model for parameters p
    @variable(MILP_model, 0.0 <= p[g = 1:n_g])
    @variable(MILP_model, 0.0 <= alpha[g = 1:n_g] <= 1.0)
    @variable(MILP_model, z_MILP[g = 1:n_g], Bin)
    @constraint(MILP_model, pb, sum(p[g] for g = 1:n_g) + W == D)
    @constraint(MILP_model, rall, sum(alpha[g] for g = 1:n_g) == 1.0)
    @constraint(MILP_model, ract[g = 1:n_g], alpha[g] <= z_MILP[g])
    @constraint(MILP_model, minprd[g = 1:n_g], p_min[g] * z_MILP[g] <= p[g] - phi_n*alpha[g])
    @constraint(MILP_model, maxprd[g = 1:n_g], p[g] + phi_n*alpha[g] <= p_max[g] * z_MILP[g])
    @constraint(MILP_model, uprmp[g = 1:n_g], phi_p*alpha[g] <= R_up_max[g] * z_MILP[g])
    @constraint(MILP_model, dwnrmp[g = 1:n_g], phi_n*alpha[g] <= R_down_max[g] * z_MILP[g])
    @objective(MILP_model, Min, sum(C_Q[g]*(p[g]^2 + cov_vec[s]*alpha[g]^2) + C_L[g]*p[g] + C_f[g]*z_MILP[g] for g = 1:n_g))

    # Solve MILP model, extract relevant information and empty model
    optimize!(MILP_model)
    z_opt = value.(z_MILP)
    empty!(MILP_model)

    # Define LP model for parameters p
    @variable(LP_model, 0.0 <= p[g = 1:n_g])
    @variable(LP_model, 0.0 <= alpha[g = 1:n_g] <= 1.0)
    @variable(LP_model, 0.0 <= z[g = 1:n_g] <= 1.0)
    @constraint(LP_model, power_balance, sum(p[g] for g = 1:n_g) + W == D)
    @constraint(LP_model, reserve_allocation, sum(alpha[g] for g = 1:n_g) == 1.0)
    @constraint(LP_model, reserve_activation[g = 1:n_g], alpha[g] <= z[g])
    @constraint(LP_model, min_prod[g = 1:n_g], p_min[g] * z[g] <= p[g] - phi_n*alpha[g])
    @constraint(LP_model, max_prod[g = 1:n_g], p[g] + phi_n*alpha[g] <= p_max[g] * z[g])
    @constraint(LP_model, max_up_ramp[g = 1:n_g], phi_n*alpha[g] <= R_up_max[g] * z[g])
    @constraint(LP_model, max_down_ramp[g = 1:n_g], phi_p*alpha[g] <= R_down_max[g] * z[g])
    @constraint(LP_model, commitment_cuts[g = 1:n_g], z[g] == z_opt[g])
    @objective(LP_model, Min, sum(C_Q[g]*(p[g]^2 + cov_vec[s]*alpha[g]^2) + C_L[g]*p[g] + C_f[g]*z[g] for g = 1:n_g))

    # Solve LP model, extract information and empty it
    optimize!(LP_model)
    p_scheduled[:,s] .= value.(p);
    alpha_scheduled[:,s] .= value.(alpha);
    z_scheduled[:,s] .= z_opt;
    electricity_price[s] = dual(power_balance);
    reserve_price[s] = dual(reserve_allocation);
    commitment_price[:,s] .= dual.(commitment_cuts);
    load_payment[s] = D*electricity_price[s];
    wind_energy_revenue[s] = W*electricity_price[s];
    generator_energy_revenue[:,s] .= [(electricity_price[s]*p_scheduled[g, s]) for g = 1:n_g];
    generator_reserve_revenue[:,s] .= [(reserve_price[s]*alpha_scheduled[g, s]) for g = 1:n_g];
    generator_commitment_revenue[:,s] .= [(commitment_price[g, s]*z_scheduled[g, s]) for g = 1:n_g];
    true_cost[:,s] .= [(true_C_Q[g]*(p_scheduled[g, s]^2 + cov_vec[s]*alpha_scheduled[g, s]^2) + true_C_L[g]*p_scheduled[g, s] + true_C_f[g]*z_scheduled[g, s]) for g = 1:n_g];
    reported_cost[:,s] .= [(C_Q[g]*(p_scheduled[g, s]^2 + cov_vec[s]*alpha_scheduled[g, s]^2) + C_L[g]*p_scheduled[g, s] + C_f[g]*z_scheduled[g, s]) for g = 1:n_g];
    true_profit[:,s] .= [(generator_energy_revenue[g, s] + generator_reserve_revenue[g, s] + generator_commitment_revenue[g, s] - true_cost[g, s]) for g = 1:n_g];
    reported_profit[:,s] .= [(generator_energy_revenue[g, s] + generator_reserve_revenue[g, s] + generator_commitment_revenue[g, s] - reported_cost[g, s]) for g = 1:n_g];
    empty!(LP_model)

end

## Post-processing

fig_electricity_price = plot(agg_cov_vec./D, electricity_price, title="Electricity Price vs Standard Deviation of Forecast Error", label="Electricity Price", color="red", legend=:bottomright)
plot!(xlabel="Standard Deviation of Forecast Error (fraction of demand)", ylabel="Electricity Price")
savefig(fig_electricity_price, "electricity_price_sensitivity_fd.png")

fig_electricity_price_2 = plot(agg_cov_vec./W, electricity_price, title="Electricity Price vs Standard Deviation of Forecast Error", label="Electricity Price", color="red", legend=:bottomright)
plot!(xlabel="Standard Deviation of Forecast Error (fraction of expected output)", ylabel="Electricity Price")
savefig(fig_electricity_price_2, "electricity_price_sensitivity_feo.png")

fig_reserve_price = plot(agg_cov_vec./D, reserve_price, title="Reserve Price vs Standard Deviation of Forecast Error", label="Reserve Price", color="red", legend=:bottomright)
plot!(xlabel="Standard Deviation of Forecast Error (fraction of demand)", ylabel="Reserve Price")
savefig(fig_reserve_price, "reserve_price_sensitivity.png")
