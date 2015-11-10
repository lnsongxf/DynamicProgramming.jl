abstract DynamicProgramming{T<:AbstractFloat}

typealias ValueFunction Any

immutable UnconstrainedDynamicProgramming{T} <: DynamicProgramming{T}
    reward::Function          # two argument function of the form reward(state, control)
    transition!::Function      # three argument mutating function of the form transition(state, control, shock)
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
    for shock in samples
        d.transition!(state, control, shock, new_state)
        v += valuefn[ new_state... ]
    end
    return v / length(samples)
end


function expected_bellman_gradient{T}(d::DynamicProgramming{T},
                                      valuefn::ValueFunction,
                                      samples::ValueFunction,
                                      state,
                                      control)

    new_state = Vector(d.state_dim)
    g  = zeros(T, d.state_dim)
    _g = zeros(T, d.state_dim)
    for shock in samples
        d.transition!(state, control, shock, new_state)
        g += Interpolations.gradient!(_g, valuefn, new_state...)
    end
    return g / length(samples)
end

function optimize_bellman(d::DynamicProgramming,
                          valuefn::ValueFunction,
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

function evaluate_bellman_on_grid{T}(d::DynamicProgramming{T}, valuefn, samples)
    points_per_dimension = [ length(j) for j in d.grid ]
    vals = Array{T}(points_per_dimension...)
    args = Array{Vector{T}}(points_per_dimension...)

    for (i, state) in enumerate(product(d.grid...))
        @time vals[i], args[i] = optimize_bellman(d, valuefn, samples, collect(state))
    end

    return vals, args
end

function approximate_bellman(d::DynamicProgramming, valuefn, samples)
    vals, args  = evaluate_bellman_on_grid(d, valuefn, samples)
    new_valuefn = scale(interpolate(vals, BSpline(Linear()), OnGrid()), d.grid...)
    return new_valuefn
end
