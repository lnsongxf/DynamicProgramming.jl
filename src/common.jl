abstract AbstractDynamicProgramming{T<:AbstractFloat}
typealias ValueFunction Interpolations.ScaledInterpolation
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

type ConstrainedDynamicProgramming{T} <: AbstractDynamicProgramming{T}
    reward::Function                   # two argument function, must return a real number                         reward(state::Vector, control::Vector)
    transition::Function               # three argument function, must return a vector                            transition(state::Vector, control::Vector, shock)
    constraint::Function               # two argument function, must return a vector ≤ 0 when control is feasible constraint(state::Vector, control::Vector)
    initial::Function                  # specifies a feasible point, must return a vector                         initial(state::Vector) = control
    beta::Real                         # discounting factor

    grid::GridSpace{T}                 # discretization of the state space
    state_dim::Int                     # dimension of the state space
    control_dim::Int                   # dimension of the control space

    control_bounds::ControlBounds{T}   # left and right bounds on the control variable, defulat ([typemin(T)], [typemax(T)])
    solver::MathProgBase.AbstractMathProgSolver
    interp::Interpolations.BSpline

    function ConstrainedDynamicProgramming{T}(reward,
                                              transition,
                                              constraint,
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

        new{T}(reward, transition, constraint, initial, beta, grid, state_dim, control_dim, control_bounds, solver, interp)
    end
end


function superscript(i)
    sup_script = Dict{AbstractString, AbstractString}("0"=>"⁰","1"=>"¹","2"=>"²","3"=>"³","4"=>"⁴","5"=>"⁵","6"=>"⁶","7"=>"⁷","8"=>"⁸","9"=>"⁹")
    i_string = string(i)
    for (k,v) in sup_script
        i_string = replace(i_string, k, v)
    end
    return i_string
end

function Base.show(io::IO, d::AbstractDynamicProgramming)
    @printf io "%s\n" typeof(d)

    grid_range, grid_points = if length(d.grid) == 1
        d.grid[1], length(d.grid[1])
    else
        d.grid, length(product(d.grid...))
    end
    @printf io " 𝑇𝑣 = 𝑥 ↦ max{ 𝑟(𝑥,𝑢) + 𝛽∫𝑣[𝑓(𝑥,𝑢,𝜀)]𝜇(d𝜀) }\n"
    @printf io " discount factor: 𝛽 = %.2f\n" d.beta
    @printf io "      dimensions: 𝑥 ∈ ℝ%s (state), 𝑢 ∈ ℝ%s (control)\n" superscript(d.state_dim) superscript(d.control_dim)
    @printf io "  control bounds: %sᵀ ≤ 𝑢 ≤ %sᵀ\n" first(d.control_bounds) last(d.control_bounds)
    @printf io "\n"
    @printf io " approximation: %s\n" typeof(d.interp)
    @printf io "     %s (%d points)\n" grid_range grid_points
    @printf io " solver: %s\n" typeof(d.solver)
    if !isempty(d.solver.options)
        maxl = maximum([length(string(k)) for (k,_) in d.solver.options])
        for (k,v) in d.solver.options
            @printf io "     %s = %s\n" lpad(string(k), maxl,' ') v
        end
    end
end

# this is not correct, need
num_const(d::ConstrainedDynamicProgramming) = 1

# user friendly interface
function dynamic_programming{T}(reward::Function,
                                transition::Function,
                                initial::Function,
                                beta::Real,
                                grid::Union{Range{T}, GridSpace{T}},
                                state_dim::Int,
                                control_dim::Int = state_dim; # defaults to the state dimension
                                constraint = nothing,
                                control_bounds::ControlBounds{T} = (fill(typemin(T), control_dim), fill(typemax(T), control_dim)),
                                solver::MathProgBase.AbstractMathProgSolver = Ipopt.IpoptSolver(print_level = 0, tol = 1e-2, max_iter=500),
                                interpolation::Interpolations.Interpolations.BSpline = BSpline(Linear())
                                )

    _grid = isa(grid, GridSpace) ? grid : (grid,)
    @assert state_dim == length(_grid) "State discretization dimension must match specified state dimension: got
    grid dimensions = $(length(_grid)), state dimensions = $state_dim"
    if constraint == nothing
        return UnconstrainedDynamicProgramming{T}(reward, transition, initial, beta, _grid, state_dim, control_dim, control_bounds, solver, interpolation)
    else
        return ConstrainedDynamicProgramming{T}(reward, transition, constraint, initial, beta, _grid, state_dim, control_dim, control_bounds, solver, interpolation)
    end
end

