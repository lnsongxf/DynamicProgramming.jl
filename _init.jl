using DynamicProgramming

Gadfly.plot(f::Interpolations.AbstractInterpolation, args...) = Gadfly.plot(x->f[x], args...)

α = 1/3
β = 0.98

# reward(x, u) = try
#         log(x[1] - u[1])
#     catch
#         warn("Reward function not defined")
#         @show x[1], u[1], x[1]- u[1]
#         return 0
#     end

reward(x, u)        = log(x[1] - u[1])
transition(x, u, ε) = cbrt(ε^3 .* u)
initial(state)      = collect(state[1]/2) # specifies a feasible control for a given state

rti     = (reward, transition, initial)
solver  = Ipopt.IpoptSolver(print_level=0, tol=1e-2, max_iter=500)
d1      = UnconstrainedDynamicProgramming{Float64}(rti..., β, 0.5:0.1:10.0, 1, 1,
                control_bounds=(fill(0.0, 1), fill(Inf, 1)), solver=solver)

samples = rand(LogNormal(), 100)
vfn     = iterate_bellman_operator(d1, samples, 10, print_level = 1)

plot(vfn, 0.5,10)

Iterations.

