using DynamicProgramming, Gadfly, Distributions

# it can help troubleshooting to write functions like this
# reward(x, u) = try
#         log(x[1] - u[1])
#     catch
#         warn("Reward function not defined")
#         @show x[1], u[1], x[1]- u[1]
#         return 0
#     end

α = 1/3
β = 0.98

reward(x, u)        = log(x[1] - u[1])
transition(x, u, ε) = cbrt(ε^3 .* u)
initial(state)      = collect(state[1]/2) # specifies a feasible control for a given state

bounds         = fill(0.0, 1), fill(Inf, 1)
discretization = 0.5:0.1:10.0
interp = Interpolations.BSpline(Interpolations.Linear())
d1     = dynamic_programming(reward, transition, initial, β, discretization, 1, 1, control_bounds = bounds, interpolation = interp)
shocks = rand(LogNormal(), 100)
vfn    = iterate_bellman_operator(d1, shocks, 50, verbose = true)

Gadfly.plot(f::Interpolations.Interpolations.ScaledInterpolation, args...) = Gadfly.plot(x->f[x], args...)
plot(vfn, 0.5,10)
