using JuMP
using LinearAlgebra
using Distributions
using Ipopt

# Function defining the social choice function of the mechanism and returning
# the optimal objective value, allocation (primal variables) and prices (dual variables)

function allocation(C_L, C_Q, p_g_max, p_w_max, mu, cov_mat; n_g=5, n_w=3, p_g_min=p_g_min, D=900., epsilon=0.05, dr=false, solve=true)

    if dr == true
        phi = sqrt((1-epsilon)/epsilon); # one-sided distributionally-robust CC
    else
        phi = quantile(Normal(), (1-epsilon)); # one-sided CC using normal distribution
    end

    model = Model(Ipopt.Optimizer)

    @variable(model, 0.0 <= p_g[g = 1:n_g])
    @variable(model, 0.0 <= p_w[i = 1:n_w])
    @variable(model, 0.0 <= alpha[g = 1:n_g, i = 1:n_w] <= 1.0)

    @constraint(model, power_balance, sum(p_g[g] for g = 1:n_g) + sum(p_w[i] for i = 1:n_w) == D)
    @constraint(model, reserve_allocation[i = 1:n_w], sum(alpha[g, i] for g = 1:n_g) == 1.0)
    @expression(model, qt[g = 1:n_g], alpha[g,:]'*cov_mat*alpha[g, :])
    @NLconstraint(model, gen_max_prod[g = 1:n_g], phi * sqrt(qt[g]) <= p_g_max[g] - p_g[g] + sum(alpha[g, i]*mu[i] for i = 1:n_w))
    @NLconstraint(model, gen_min_prod[g = 1:n_g], phi * sqrt(qt[g]) <= p_g[g] - p_g_min[g] - sum(alpha[g, i]*mu[i] for i = 1:n_w))
    @constraint(model, wind_max_prod[i = 1:n_w], p_w[i] <= p_w_max[i] - mu[i] - phi * sqrt(cov_mat[i, i]))

    @objective(model, Min, sum(C_Q[g]*((p_g[g] - alpha[g,:]'*mu)^2 + alpha[g,:]'*cov_mat*alpha[g,:]) + C_L[g]*(p_g[g] - alpha[g,:]'*mu) for g = 1:n_g))

    if solve
        optimize!(model)
        println(termination_status(model))
        println(dual_status(model))
        return (objective_value(model), value.(p_g), value.(alpha), value.(p_w), dual(power_balance), dual.(reserve_allocation), dual.(gen_max_prod), dual.(gen_min_prod), dual.(wind_max_prod))
    else
        return model
    end
end

# Function computing the Clarke pivot term used in VCG payments for a given bidder

function clarke_pivot(bidder_out; n_g=5, n_w=3, C_L=C_L, C_Q=C_Q, mu=mu, cov_mat=cov_mat, p_g_max=p_g_max, p_g_min=p_g_min, p_w_max=p_w_max, D=900., epsilon=0.05, dr=false, solve=true)

    if dr == true
        phi = sqrt((1-epsilon)/epsilon); # one-sided distributionally-robust CC
    else
        phi = quantile(Normal(), (1-epsilon)); # one-sided CC using normal distribution
    end

    model = Model(Ipopt.Optimizer)

    if haskey(bidder_out, "type") && bidder_out["type"] == "disp"
        C_L = [C_L[g] for g = 1:n_g if g != bidder_out["number"]]
        C_Q = [C_Q[g] for g = 1:n_g if g != bidder_out["number"]]
        p_g_max = [p_g_max[g] for g = 1:n_g if g != bidder_out["number"]]
        p_g_min = [p_g_min[g] for g = 1:n_g if g != bidder_out["number"]]
        n_g -= 1
        @variable(model, 0.0 <= p_g[g = 1:n_g])
        @variable(model, 0.0 <= p_w[i = 1:n_w])
        @variable(model, 0.0 <= alpha[g = 1:n_g, i = 1:n_w] <= 1.0)
    elseif haskey(bidder_out, "type") && bidder_out["type"] == "wind"
        mu = [mu[i] for i = 1:n_w if i != bidder_out["number"]]
        cov_mat = get_submatrix(cov_mat, bidder_out["number"])
        p_w_max = [p_w_max[i] for i = 1:n_w if i != bidder_out["number"]]
        n_w -= 1
        @variable(model, 0.0 <= p_g[g = 1:n_g])
        @variable(model, 0.0 <= p_w[i = 1:n_w])
        @variable(model, 0.0 <= alpha[g = 1:n_g, i = 1:n_w] <= 1.0)
    else
        @variable(model, 0.0 <= p_g[g = 1:n_g])
        @variable(model, 0.0 <= p_w[i = 1:n_w])
        @variable(model, 0.0 <= alpha[g = 1:n_g, i = 1:n_w] <= 1.0)
    end

    @constraint(model, power_balance, sum(p_g[g] for g = 1:n_g) + sum(p_w[i] for i = 1:n_w) == D)
    @constraint(model, reserve_allocation[i = 1:n_w], sum(alpha[g, i] for g = 1:n_g) == 1.0)
    @expression(model, qt[g = 1:n_g], alpha[g,:]'*cov_mat*alpha[g, :])
    @NLconstraint(model, gen_max_prod[g = 1:n_g], phi * sqrt(qt[g]) <= p_g_max[g] - p_g[g] + sum(alpha[g, i]*mu[i] for i = 1:n_w))
    @NLconstraint(model, gen_min_prod[g = 1:n_g], phi * sqrt(qt[g]) <= p_g[g] - p_g_min[g] - sum(alpha[g, i]*mu[i] for i = 1:n_w))
    @constraint(model, wind_max_prod[i = 1:n_w], p_w[i] <= p_w_max[i] - mu[i] - phi * sqrt(cov_mat[i, i]))

    @objective(model, Min, sum(C_Q[g]*((p_g[g] - alpha[g,:]'*mu)^2 + alpha[g,:]'*cov_mat*alpha[g,:]) + C_L[g]*(p_g[g] - alpha[g,:]'*mu) for g = 1:n_g))

    if solve
        optimize!(model)
        println(termination_status(model))
        println(dual_status(model))
        return objective_value(model)
    else
        return model
    end
end

# Function retrieving a n-1 x n-1 submatrix from a n x n matrix

function get_submatrix(mat, ind_out)
    n = size(mat)[1]
    submat = zeros(n-1, n-1)
    i, k = 1, 1
    @inbounds while i <= n
        j, m = 1, 1
        if i != ind_out
            @inbounds while j <= n
                if j != ind_out
                    submat[k,m] = mat[i, j]
                    m+=1
                end
                j+=1
            end
            k+=1
        end
        i+=1
    end
    return submat
end
