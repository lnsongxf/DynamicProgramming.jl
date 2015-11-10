abstract DynamicProgramming{T<:AbstractFloat}

typealias ValueFunction Any

immutable UnconstrainedDynamicProgramming{T} <: DynamicProgramming{T}
    reward::Function          # two argument function of the form reward(state, control)
    transition::Function      # three argument mutating function of the form transition(state, control, shock)
    initial::Function         # specifies a feasible point
    beta::Real                # discounting factor
    grid::GridSpace{T}
    state_dim::Int            # dimension of the state space
    control_dim::Int          # dimension of the control space
    solver::MathProgBase.AbstractMathProgSolver
end

num_const(d::DynamicProgramming) = 0
grid_range{T}(grid::Tuple{Vararg{FloatRange{T}}}) = [minimum(r) for r in grid], [maximum(r) for r in grid]
function expected_bellman_value{T}(d::DynamicProgramming{T},
                                   valuefn::ValueFunction,
                                   samples::Vector,
                                   state,
                                   control)
    new_state = Vector{T}(d.state_dim)
    v = zero(T)
    for (i,s) in enumerate(samples)
        new_state = d.transition(state, control, s)
        v += try valgrad(valuefn, new_state...)[1] catch;  v end
    end
    return v / length(samples)
end


function expected_bellman_value_gradient{T}(d::DynamicProgramming{T},
                                            valuefn::ValueFunction,
                                            samples::ValueFunction,
                                            state,
                                            control)

    new_state = Vector(d.state_dim)
    g  = zeros(T, d.state_dim)
    for shock in samples
        new_state = d.transition(state, control, shock)
        g += try valgrad(valuefn, new_state...)[2] catch; g end
    end
    return g / length(samples)
end

@inline function optimize_bellman(d::DynamicProgramming,
                          value_fn::ValueFunction,
                          samples::Vector,
                          state::Vector)
    n, m = d.control_dim, num_const(d)
    l  = fill(-Inf, n)
    u  = fill(+1e4, n)
    lb = fill(-Inf, m)
    ub = fill(+Inf, m)

    model   = MathProgBase.model(d.solver)
    problem = BellmanIteration(d, valuefn, samples, state)

    MathProgBase.loadnonlinearproblem!(model, n, m, l, u, lb, ub, :Max, problem)

    # solve the model
    MathProgBase.setwarmstart!(model, d.initial(state))
    MathProgBase.optimize!(model)

    MathProgBase.status(model) == :Optimal || warn("Possible optimisation failure")
    val     = MathProgBase.getobjval(model)
    control = MathProgBase.getsolution(model)

    return val, control
end


# initial_value{T}(d::DynamicProgramming{T}) = zero(T)

# function bellman_recursion(d::DynamicProgramming,
#                           samples::Vector,   # iid samples of the exogenous shocks
#                           iterates::Integer) # number of times we iterate with the Bellman operator

#     (iterates == 1) && return bellman_operator(d, samples, initial_value(d))

#     value_fn, policy_fn = bellman_recursion(d, samples, iterates-1)

#     return bellman_operator(d, samples, value_fn)
# end



# function bellman_operator(d::DynamicProgramming,
#                           value_fn::ValueFunction,
#                           samples::Vector)  # iid samples of the exogenous shocks

#     vals = evaluate_points(x->optimize_bellman(d, samples, value_fn, x), d.grid)
#     itp  = interpolate!(grid, vals, Gridded(Linear()))

#     return itp
# end
