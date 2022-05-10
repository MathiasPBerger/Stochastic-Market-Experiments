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
p_max = [200.0 for g = 1:n_g];
p_min = [0.0 for g = 1:n_g];

# Wind production parameters
p_w_max = [250.0 for i = 1:n_w];
mu = [0.0*p_w_max[i] for i = 1:n_w];
#mu[1] = 0.1*p_w_max[1]
#mu[2] += 0.25*p_w_max[2];
#mu[3] -= 0.1*p_w_max[3];
std = [0.1*p_w_max[i] for i = 1:n_w];
#std[1] = std[1];
std[2] += 0.05*p_w_max[2];
std[3] = 1. + std[3]/5.;
#std[3] = 40.;
#std[3] += 0.1*p_w_max[3];
#rho = -0.75;
#corr_mat = [[1. rho]; [rho 1.]];
#rho_12, rho_13, rho_23 = -0.65, 0.1, -0.1;
rho_12, rho_13, rho_23 = 0.0, 0.0, 0.0;
corr_mat = [[1. rho_12 rho_13]; [rho_12 1. rho_23]; [rho_13 rho_23 1.]];
cov_mat = Diagonal(std) * corr_mat * Diagonal(std);
W = [0.8*p_w_max[i] for i = 1:n_w];

# Demand parameters
D = 1000;

# Risk parameters
epsilon = 0.05;
dr = false
if dr == true
    phi = sqrt((1-epsilon)/epsilon); # one-sided distributionally-robust CC
else
    phi = quantile(Normal(), (1-epsilon)); # one-sided CC using normal distrib.
end

# Cost parameters
v_Q, s_Q = 0.1, 0.05;
v_L, s_L = 1.0, 0.1;
true_C_Q = [(v_Q + s_Q*g) for g = 1:n_g];
true_C_L = [(v_L + s_L*g) for g = 1:n_g];

# Truthfulness of bids
truthful_bidding = false;

if truthful_bidding == true
    C_Q = deepcopy(true_C_Q);
    C_L = deepcopy(true_C_L);
else
#    std[1] = 0.5*std[1];
#    cov_mat = Diagonal(std) * corr_mat * Diagonal(std);
    fake_C_L = zeros(Float64, n_g);
    fake_p_max = zeros(Float64,n_g);
    fake_p_max[1] = 0.5 * p_max[1];
    p_max .-= fake_p_max;
    #    fake_C_L[1] += 1.0;
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
obj, p_scheduled, alpha_scheduled, electricity_price, reserve_price, dual_max_prod, dual_min_prod = allocation(C_L, C_Q, p_max, W, mu, cov_mat; D=D)

## Post-processing

load_payment = D*electricity_price;
energy_revenue_wind = [W[i]*electricity_price for i = 1:n_w];
reserve_revenue_wind = reserve_price;
LMP_payment_wind = [(energy_revenue_wind[i]-reserve_revenue_wind[i]) for i = 1:n_w];
energy_revenue_disp = [(electricity_price*p_scheduled[g]) for g = 1:n_g];
reserve_revenue_disp = [(reserve_price'*alpha_scheduled[g,:]) for g = 1:n_g];
LMP_payment_disp = [energy_revenue_disp[g]+reserve_revenue_disp[g] for g = 1:n_g]
true_cost_disp = [(true_C_Q[g]*((p_scheduled[g] - alpha_scheduled[g,:]'*mu)^2 + alpha_scheduled[g,:]'*cov_mat*alpha_scheduled[g,:]) + true_C_L[g]*(p_scheduled[g] - alpha_scheduled[g,:]'*mu)) for g = 1:n_g];
true_LMP_profit_disp = [(LMP_payment_disp[g] - true_cost_disp[g]) for g = 1:n_g];
reported_cost_disp = [(C_Q[g]*((p_scheduled[g] - alpha_scheduled[g,:]'*mu)^2 + alpha_scheduled[g,:]'*cov_mat*alpha_scheduled[g,:]) + C_L[g]*(p_scheduled[g] - alpha_scheduled[g,:]'*mu)) for g = 1:n_g];
reported_LMP_profit_disp = [(LMP_payment_disp[g] - reported_cost_disp[g]) for g = 1:n_g];

# Calculate VCG payments (using Clarke's pivot rule)
h_cpr, VCG_payment_wind, VCG_payment_disp = zeros(Float64, 0), zeros(Float64, 0), zeros(Float64, 0);
for bo in bidder_out
    h = clarke_pivot(C_L, C_Q, W, mu, cov_mat, bo; D=D);
    push!(h_cpr, h)
    if bo["type"] == "disp"
        push!(VCG_payment_disp, reported_cost_disp[bo["number"]]+h-obj)
    elseif bo["type"] == "wind"
        push!(VCG_payment_wind, h-obj)
    end
end

true_VCG_profit_disp = [(VCG_payment_disp[g] - true_cost_disp[g]) for g = 1:n_g];
reported_VCG_profit_disp = [(VCG_payment_disp[g] - reported_cost_disp[g]) for g = 1:n_g];
true_VCG_profit_wind = VCG_payment_wind;


println("\n")
println("Power generation: ", p_scheduled)
println("Reserve procurement: ", alpha_scheduled)
println("System cost: ", obj)
println("Electricity price: ", electricity_price)
println("Reserve price: ", reserve_price)
println("Load payment: ", load_payment)
println("LMP payments to wind producers: ", LMP_payment_wind)
println("VCG payments to wind producers: ", VCG_payment_wind)
println("LMP payments to flexible generators: ", LMP_payment_disp)
println("VCG payments to flexible generators: ", VCG_payment_disp)
println("True LMP generator profit: ", true_LMP_profit_disp)
println("True VCG generator profit: ", true_VCG_profit_disp)

println("\n")
