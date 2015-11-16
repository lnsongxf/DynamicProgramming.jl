# Brock-Mirman (1972) problem

# without constraint
using DynamicProgramming, Gadfly, Distributions, Interpolations

# it can help troubleshooting to write functions like this
# reward(x, u) = try
#         log(x[1] - u[1])
#     catch
#         warn("Reward function not defined")
#         @show x[1], u[1], x[1]- u[1]
#         return 0
#     end

β = 0.95
A = 1.00
reward(x, u)        = log(x[1] - u[1])
transition(x, u, ε) = cbrt(A*ε^3 .* u)
initial(state)      = collect(state[1]/2) # specifies a feasible control for a given state

bounds = fill(0.0, 1), fill(Inf, 1)
grid   = 0.1:0.1:20.0
interp = BSpline(Linear())
br     = dynamic_programming(reward, transition, initial, β, grid, 1, 1, control_bounds = bounds, interpolation = interp)
shocks = rand(LogNormal(0.0, 0.25), 50)

hist   = Vector{Any}()
br_vfn = iterate_bellman_operator(br, shocks, 20, verbose = true, history = hist)

l = [ layer(x->y[x], 0.001, 2.0) for y in (hist[1:end])]
plot(l...)


# using constraint
using DynamicProgramming, Gadfly, Distributions, Interpolations

β = 0.95
A = 1.00
r(x, u)        = log(x[1] - u[1])
f(x, u, ε)     = cbrt(A*ε^3 .* u)
initial(state) = collect(state[1]/2) # specifies a feasible control for a given state
g(x, u)        = u - x

bounds = fill(0.0, 1), fill(Inf, 1)
grid   = 0.1:0.1:20.0
interp = BSpline(Quadratic(Reflect()))
br     = dynamic_programming(r, f, initial, β, grid, 1, 1, control_bounds = bounds, interpolation = interp, constraint = g)
shocks = rand(LogNormal(0.0, 0.25), 50)

hist   = Vector{Any}()
br_vfn = iterate_bellman_operator(br, shocks, 20, verbose = true, history = hist)

l = [ layer(x->y[x], 0.001, 20.0) for y in (hist[1:end])]
plot(l...)

model   = MathProgBase.model(d.solver)
problem = DynamicProgramming.BellmanIteration(br, valuefn, shocks, state)

MathProgBase.loadnonlinearproblem!(model, d.control_dim, num_const(d), l, u, lb, ub, :Max, problem)
