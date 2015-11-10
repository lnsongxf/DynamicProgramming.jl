typealias GridSpace{T<:AbstractFloat} Tuple{Vararg{Range{T}}}

typealias ValueFunction Interpolations.AbstractInterpolation

abstract AbstractDynamicProgramming{T<:AbstractFloat}

typealias ADP{T} AbstractDynamicProgramming{T}

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

function dynamic_programming{T}(reward::Function,
                             transition!::Function,
                             initial::Function,
                             beta::Real,
                             grid::GridSpace{T},
                             solver::MathProgBase.AbstractMathProgSolver = IpoptSolver(print_level=0, tol=1e-3)
                             )

    state_dim   = length(grid)
    control_dim = state_dim
    UnconstrainedDynamicProgramming{T}(reward, transition!, initial, beta, grid, state_dim, control_dim)
end


num_const(d::AbstractDynamicProgramming) = 0
grid_range{T}(grid::Tuple{Vararg{FloatRange{T}}}) = [minimum(r) for r in grid], [maximum(r) for r in grid]