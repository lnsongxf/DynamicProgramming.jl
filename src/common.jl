abstract AbstractDynamicProgramming{T<:AbstractFloat}

typealias GridSpace{T<:AbstractFloat} Tuple{Vararg{Range{T}}}
typealias ControlBounds{T} Tuple{Vector{T}, Vector{T}}

type UnconstrainedDynamicProgramming{T} <: AbstractDynamicProgramming{T}
    reward::Function                   # two argument function, must return a real number   reward(state::Vector, control::Vector)
    transition::Function               # three argument function, must return a vector      transition(state::Vector, control::Vector, shock)
    initial::Function                  # specifies a feasible point, must return a vector   initial(state::Vector) = control
    beta::Real                         # discounting factor

    grid::GridSpace{T}                 # discretization of the state space
    state_dim::Int                     # dimension of the state space
    control_dim::Int                   # dimension of the control space

    control_bounds::ControlBounds{T}   # left and right bounds on the control variable, defulat ([typemin(T)], [typemax(T)])
    solver::MathProgBase.AbstractMathProgSolver
    interp::Interpolations.BSpline

    function UnconstrainedDynamicProgramming{T}(reward,
                                                transition,
                                                initial,
                                                beta,
                                                grid::GridSpace{T},
                                                state_dim::Int,
                                                control_dim::Int,
                                                control_bounds::ControlBounds{T},
                                                solver::MathProgBase.AbstractMathProgSolver,
                                                interp::Interpolations.BSpline)

        @assert 0 < beta < 1 "Beta must be between zero and one: got beta = $beta"
        @assert state_dim == length(grid) "State discretization dimension must match specified state dimension: got
        grid dimensions = $(length(grid)), state dimensions = $state_dim"

        new{T}(reward, transition, initial, beta, grid, state_dim, control_dim, control_bounds, solver, interp)
    end
end

num_const(d::UnconstrainedDynamicProgramming) = 0

function Base.show(io::IO, d::AbstractDynamicProgramming)
    @printf io "%s\n" typeof(d)

    grid_range, grid_points = if length(d.grid) == 1
        d.grid[1], length(d.grid[1])
    else
        d.grid, length(product(d.grid...))
    end

    @printf io " discount factor: %.2f\n" d.beta
    @printf io " discretization: %s (%d points)\n" grid_range grid_points
    @printf io " approximation: %s\n" typeof(d.interp)
    @printf io " dimension: %d state(s), %d control(s)\n"   d.state_dim d.control_dim
    @printf io " control bounds: %sᵀ ≤ 𝑢 ≤ %sᵀ\n" first(d.control_bounds) last(d.control_bounds)
    @printf io " solver: %s\n" typeof(d.solver)

    if !isempty(d.solver.options)
        maxl = maximum([length(string(k)) for (k,v) in d.solver.options])
        for (k,v) in d.solver.options
            @printf io "    %s = %s\n" lpad(string(k), maxl,' ') v
        end
    end
end

# user friendly interface
function dynamic_programming{T}(reward::Function,
                                transition::Function,
                                initial::Function,
                                beta::Real,
                                grid::Union{Range{T}, GridSpace{T}},
                                control_dim::Int = isa(grid, GridSpace) ? length(grid) : 1; # defaults to the state dimension, as inferred via the grid
                                control_bounds::ControlBounds{T} = (fill(typemin(T), control_dim), fill(typemax(T), control_dim)),
                                solver::MathProgBase.AbstractMathProgSolver = Ipopt.IpoptSolver(print_level = 0, tol = 1e-2, max_iter=500),
                                interpolation::Interpolations.Interpolations.BSpline = BSpline(Linear())
                                )
    _grid = isa(grid, GridSpace) ? grid : (grid,)
    state_dim   = length(_grid)

    return UnconstrainedDynamicProgramming{T}(reward, transition, initial, beta, _grid, state_dim, control_dim, control_bounds, solver, interpolation)
end

