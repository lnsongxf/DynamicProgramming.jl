using MathProgBase, ForwardDiff, Grid, Iterators, Ipopt, Interpolations

include("approximation.jl")
include("dynamic_programming.jl")
include("optimisation.jl")

# samples = abs(randn(100))
# reward(state, control)            = log(state[1] - control[1])
# transition(state, control, shock) = shock[1]*state[1]*control[1]^(1/3)
# initial(state) = zero(state) # specifies a feasible control for a given state

# d = UnconstrainedDynamicProgramming(reward, transition, initial, 0.8, (10.0:1:100.0,), 1, 1, IpoptSolver(print_level=0, tol=1e-3))

# state = [1.0]
# control = [0.5]

# vals      = evaluate_points(x->d.reward(x, 0.0), d.grid)
# valuefn   = CoordInterpGrid(d.grid, vals, BCnil, InterpQuadratic)
# new_state = Vector{Float64}(d.state_dim)

# @time expected_bellman_value(d, valuefn, samples, 2.0, 1.0)



samples = abs(randn(100))
reward(state, control)                        = - dot(control, control) - dot(state,state)
transition!(state, control, shock, new_state) = (new_state[:] = state)
initial(state)                                = zero(state) # specifies a feasible control for a given state

d2 = UnconstrainedDynamicProgramming(reward, transition!, initial, 0.8, (-5:1.0:5.0,-5:1.0:5.0), 2, 2, IpoptSolver(print_level=0, tol=1e-2))

init_state = [0.0, 0.0]
vals      = evaluate_points(x->d2.reward(x, init_state), d2.grid)
valuefn   = scale(interpolate(vals, BSpline(Linear()), OnGrid()), d2.grid...)


@time expected_bellman_value(d2, valuefn, samples, [2.0, 1.0], zeros(2))
@time expected_bellman_gradient(d2, valuefn, samples, [2.0, 1.0], zeros(2))
@time optimize_bellman(d2, valuefn, samples, zeros(2))
@time evaluate_bellman_on_grid(d2, valuefn, samples)
@time approximate_bellman(d2, valuefn, samples)

v, x = optimize_bellman(d2, valuefn, samples, [0.0, 0.0])

@time expected_bellman_value(d2, valuefn, samples, [2.0, 1.0], zeros(2))
@time expected_bellman_value_gradient(d2, valuefn, samples, [2.0, 1.0], zeros(2))
