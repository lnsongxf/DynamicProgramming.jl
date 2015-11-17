module DynamicProgramming

using MathProgBase, Ipopt, ForwardDiff, Iterators, Interpolations

# types
export dynamic_programming
export UnconstrainedDynamicProgramming

# bellman things
export bellman_value, bellman_gradient, optimize_bellman
export approximate_bellman_operator, iterate_bellman_operator

include("common.jl")
include("optimisation.jl")
include("bellmanoperator.jl")

end # module
