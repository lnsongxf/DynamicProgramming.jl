module DynamicProgramming

using MathProgBase, ForwardDiff, Iterators, Ipopt, Interpolations

# types
export UnconstrainedDynamicProgramming

# bellman things
export expected_bellman_value, expected_bellman_gradient, optimize_bellman, evaluate_bellman_on_grid, approximate_bellman

include("common.jl")
include("bellmanoperator.jl")
include("optimisation.jl")

end # module
