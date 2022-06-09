include("vcg_utils.jl")

## Comments
#
# This script formulates the problem as a convex program with a convex quadratic
# objective to be minimised subject to affine and second-order cone constraints.
# It then solves the problem with Ipopt. This has non-trivial implications
# for post-processing because of the primal and dual forms used by the solver.
# More precisely, Ipopt relies on Lagrangian duality, which means that the dual
# variables associated with second-order cone constraints are scalar variables.
# Theoretical developments are also based on Lagrangian duality and the
# formulation used in the theoretical analysis can be solved as such, which
# implies that the meaning of dual variables of second-order cone constraints
# retrieved from the solver is roughly the same (they are the opposite of the
# ones used in our developments). They can therefore be used to check our
# results numerically.

## Data

# Number of flexible generators and wind farms
n_g = 5;
n_w = 3;

# Flexible generation parameters
true_p_g_max = [200.0 for g = 1:n_g];
p_g_min = [0.0 for g = 1:n_g];

# Wind production parameters
p_w_max = [250.0 for i = 1:n_w];
mu = [0.0*p_w_max[i] for i = 1:n_w];
#mu[1] = 0.1*p_w_max[1]
#mu[2] += 0.25*p_w_max[2];
#mu[3] -= 0.1*p_w_max[3];
true_std = [0.1*p_w_max[i] for i = 1:n_w];
true_std[2] += 0.05*p_w_max[2];
true_std[3] = 1. + true_std[3]/5.;
#std[3] = 40.;
#std[3] += 0.1*p_w_max[3];
#rho = -0.75;
#corr_mat = [[1. rho]; [rho 1.]];
#rho_12, rho_13, rho_23 = -0.65, 0.1, -0.1;
rho_12, rho_13, rho_23 = 0.0, 0.0, 0.0;
corr_mat = [[1. rho_12 rho_13]; [rho_12 1. rho_23]; [rho_13 rho_23 1.]];
true_cov_mat = Diagonal(true_std) * corr_mat * Diagonal(true_std);
p_w_exp = [0.8 * p_w_max[i] for i = 1:n_w]; # expected wind production

# Demand parameters
D = 1000;

# Risk parameters
epsilon = 0.05;

# Cost parameters
v_Q, s_Q = 0.1, 0.05;
v_L, s_L = 1.0, 0.1;
true_C_Q = [(v_Q + s_Q*g) for g = 1:n_g];
true_C_L = [(v_L + s_L*g) for g = 1:n_g];

# Truthfulness of bids
truthful_bidding = true;

if truthful_bidding == true
    cov_mat = deepcopy(true_cov_mat);
    p_g_max = deepcopy(true_p_g_max);
    C_Q = deepcopy(true_C_Q);
    C_L = deepcopy(true_C_L);
else
    fake_std = zeros(Float64, n_w);
    fake_std[1] = 0.9*true_std[1];
    fake_cov_mat = Diagonal(fake_std) * corr_mat * Diagonal(fake_std);
    cov_mat = deepcopy(true_cov_mat) .- fake_cov_mat;
    fake_p_g_max = zeros(Float64,n_g);
#    fake_p_g_max[1] = 0.5 * p_g_max[1];
    p_g_max = deepcopy(true_p_g_max) .- fake_p_g_max;
    fake_C_L = zeros(Float64, n_g);
#    fake_C_L[1] -= 1.0;
    C_Q = deepcopy(true_C_Q);
    C_L = deepcopy(true_C_L) .+ fake_C_L;
end

# Bidder out
bidder_out_1 = Dict(["type", "number"] .=> ["disp", 1])
bidder_out_2 = Dict()
bidder_out_3 = Dict(["type", "number"] .=> ["wind", 2])
bidder_out_disp = [Dict(["type", "number"] .=> ["disp", g]) for g = 1:n_g]
bidder_out_wind = [Dict(["type", "number"] .=> ["wind", i]) for i = 1:n_w]
bidder_out = vcat(bidder_out_disp, bidder_out_wind)

## Simulation

# Calculate allocation
obj, p_g_scheduled, alpha_scheduled, p_w_scheduled, electricity_price, reserve_price = allocation(C_L, C_Q, p_g_max, p_w_exp, p_w_max, mu, cov_mat; D=D)

## Post-processing

load_payment = D*electricity_price;
energy_revenue_wind = [p_w_scheduled[i]*electricity_price for i = 1:n_w];
reserve_revenue_wind = reserve_price;
LMP_payment_wind = [(energy_revenue_wind[i]-reserve_revenue_wind[i]) for i = 1:n_w];
true_cost_wind = [0.0 for i = 1:n_w];
#true_cost_wind = [(1/sqrt(cov_mat[i,i]))*p_w_scheduled[i] for i = 1:n_w];
#true_cost_wind = [sqrt(cov_mat[i,i])*p_w_scheduled[i] for i = 1:n_w];
true_LMP_profit_wind = [(LMP_payment_wind[i] - true_cost_wind[i]) for i = 1:n_w];
reported_cost_wind = [0.0 for i = 1:n_w];
#reported_cost_wind = [(1/sqrt(cov_mat[i,i]))*p_w_scheduled[i] for i = 1:n_w];
#reported_cost_wind = [sqrt(cov_mat[i,i])*p_w_scheduled[i] for i = 1:n_w];
reported_LMP_profit_wind = [(LMP_payment_wind[i] - reported_cost_wind[i]) for i = 1:n_w];
LMP_energy_revenue_disp = [(electricity_price*p_g_scheduled[g]) for g = 1:n_g];
LMP_reserve_revenue_disp = [(reserve_price'*alpha_scheduled[g,:]) for g = 1:n_g];
LMP_payment_disp = [(LMP_energy_revenue_disp[g]+LMP_reserve_revenue_disp[g]) for g = 1:n_g]
true_cost_disp = [(true_C_Q[g]*((p_g_scheduled[g] - alpha_scheduled[g,:]'*mu)^2 + alpha_scheduled[g,:]'*cov_mat*alpha_scheduled[g,:]) + true_C_L[g]*(p_g_scheduled[g] - alpha_scheduled[g,:]'*mu)) for g = 1:n_g];
true_LMP_profit_disp = [(LMP_payment_disp[g] - true_cost_disp[g]) for g = 1:n_g];
reported_cost_disp = [(C_Q[g]*((p_g_scheduled[g] - alpha_scheduled[g,:]'*mu)^2 + alpha_scheduled[g,:]'*cov_mat*alpha_scheduled[g,:]) + C_L[g]*(p_g_scheduled[g] - alpha_scheduled[g,:]'*mu)) for g = 1:n_g];
reported_LMP_profit_disp = [(LMP_payment_disp[g] - reported_cost_disp[g]) for g = 1:n_g];

# Calculate VCG payments (using Clarke's pivot rule)
h_cpr, VCG_payment_wind, VCG_payment_disp = zeros(Float64, 0), zeros(Float64, n_w), zeros(Float64, n_g);
for bo in bidder_out
    h = clarke_pivot(bo; D=D);
    push!(h_cpr, h)
    if bo["type"] == "disp"
        VCG_payment_disp[bo["number"]] = reported_cost_disp[bo["number"]]+h-obj;
    elseif bo["type"] == "wind"
        VCG_payment_wind[bo["number"]] = reported_cost_wind[bo["number"]]+h-obj;
    end
end

true_VCG_profit_disp = [(VCG_payment_disp[g] - true_cost_disp[g]) for g = 1:n_g];
reported_VCG_profit_disp = [(VCG_payment_disp[g] - reported_cost_disp[g]) for g = 1:n_g];
true_VCG_profit_wind = [(VCG_payment_wind[i] - true_cost_wind[i]) for i = 1:n_w];
reported_VCG_profit_wind = [(VCG_payment_wind[i] - reported_cost_wind[i]) for i = 1:n_w];

println("\n")
println("Power generation: ", p_g_scheduled)
println("Reserve procurement: ", alpha_scheduled)
println("System cost: ", obj)
println("Electricity price: ", electricity_price)
println("Reserve price: ", reserve_price)
println("Load payment: ", load_payment)
println("LMP wind payments: ", LMP_payment_wind)
println("VCG wind payments: ", VCG_payment_wind)
println("LMP generator payments: ", LMP_payment_disp)
println("VCG generator payments: ", VCG_payment_disp)
println("True LMP wind profit: ", true_LMP_profit_wind)
println("True VCG wind profit: ", true_VCG_profit_wind)
println("True LMP generator profit: ", true_LMP_profit_disp)
println("True VCG generator profit: ", true_VCG_profit_disp)

println("\n")
