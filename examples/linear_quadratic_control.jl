using DynamicProgramming, Distributions, Gadfly

# linear-quadratic control problem

β = 0.90
δ = 0.90
γ = 1.10
r = inv(β)-1
b = 30.0
μ = 29.0
ρ = 0.40

reward(x, i) = -(r*x[1] + x[2] - i[1] - b)^2 - γ*i[1]^2
@inline function transition(x, i, ε)
    w = δ*x[1] + i[1]
    y = (1-ρ)*μ + ρ*x[2] + ε
    return [w, y]
end
initial(state) = zeros(1) # specifies a feasible control for a given state

bounds         = fill(0.0, 1), fill(Inf, 1)
discretization = (-10.0:10.0, -10.0:10.0)
interp = Interpolations.BSpline(Interpolations.Quadratic(Reflect()))
solver = Ipopt.IpoptSolver(print_level = 0, tol = 1e-2, max_iter=1000)
lq     = dynamic_programming(reward, transition, initial, β, discretization, 2, 1, interpolation=interp, solver=solver)
shocks = rand(Normal(0,0.1), 100)
vfn    = iterate_bellman_operator(lq, shocks, 20, verbose = true)

plot((x,y)->vfn[x,y],-10.0,10,-10.0,10)

