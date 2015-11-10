typealias GridSpace{T<:AbstractFloat} Tuple{Vararg{Range{T}}}

typealias ValueFunction Interpolations.AbstractInterpolation

abstract AbstractDynamicProgramming{T<:AbstractFloat}

type UnconstrainedDynamicProgramming{T} <: AbstractDynamicProgramming{T}
    reward::Function          # two argument function of the form reward(state, control)
    transition!::Function      # three argument mutating function of the form transition(state, control, shock)
    initial::Function         # specifies a feasible point
    beta::Real                # discounting factor
    grid::GridSpace{T}
    state_dim::Int            # dimension of the state space
    control_dim::Int          # dimension of the control space
    solver::MathProgBase.AbstractMathProgSolver

    function UnconstrainedDynamicProgramming{T}(reward,
                                                transition!,
                                                initial,
                                                beta,
                                                grid::GridSpace{T},
                                                state_dim::Int,
                                                control_dim::Int,
                                                solver::MathProgBase.AbstractMathProgSolver = IpoptSolver(print_level=0, tol=1e-3))

        @assert state_dim == length(grid) "State discretization dimension must match no. of state dimensions: got
            grid dimensions = $(length(grid)), state dimensions = $state_dim"

        new{T}(reward, transition!, initial, beta, grid, state_dim, control_dim, solver)
    end
end

function dynamic_programming(reward::Function,
                             transition!::Function,
                             initial::Function,
                             beta::Real,
                             grid::Gridspace{T}
                             solver::MathProgBase.AbstractMathProgSolver = IpoptSolver(print_level=0, tol=1e-3)
                             )

    state_dim   = length(grid)
    control_dim = state_dim
    UnconstrainedDynamicProgramming{T}(reward, transition!, initial, beta, grid, state_dim, control_dim)
end


num_const(d::AbstractDynamicProgramming) = 0
grid_range{T}(grid::Tuple{Vararg{FloatRange{T}}}) = [minimum(r) for r in grid], [maximum(r) for r in grid]

function expected_bellman_value{T}(d::AbstractDynamicProgramming{T},
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

function expected_bellman_gradient{T}(d::AbstractDynamicProgramming{T},
                                      valuefn::ValueFunction,
                                      samples::Vector,
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

function optimize_bellman(d::AbstractDynamicProgramming,
                          valuefn::ValueFunction,
                          samples::Vector,
                          state::Vector)

    n, m = d.control_dim, num_const(d)
    l  = fill(-Inf, n)
    u  = fill(+Inf, n)
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

function evaluate_bellman_on_grid{T}(d::AbstractDynamicProgramming{T}, valuefn, samples; verbose = false)
    points_per_dimension = [ length(j) for j in d.grid ]
    vals = Array{T}(points_per_dimension...)
    args = Array{Vector{T}}(points_per_dimension...)

    for (i, state) in enumerate(product(d.grid...))
        vals[i], args[i] = optimize_bellman(d, valuefn, samples, collect(state))
    end

    return vals, args
end

function approximate_bellman(d::AbstractDynamicProgramming, valuefn, samples)
    vals, args  = evaluate_bellman_on_grid(d, valuefn, samples)
    new_valuefn = scale(interpolate(vals, BSpline(Linear()), OnGrid()), d.grid...)
    return new_valuefn
end

# method to supply the initial guess, use the reward function with some state
function approximate_bellman{T}(d::AbstractDynamicProgramming{T}, samples::Vector)
    some_state   = T[ rand(r) for r in d.grid ]
    some_control = d.initial(some_state)

    points_per_dimension = [ length(j) for j in d.grid ]
    vals = Array{T}(points_per_dimension...)
    for (i,x) in enumerate(product(d.grid...))
        vals[i] = d.reward(collect(x), some_control)
    end

    new_valuefn = scale(interpolate(vals, BSpline(Linear()), OnGrid()), d.grid...)

    return new_valuefn
end
