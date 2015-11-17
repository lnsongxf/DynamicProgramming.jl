# Brock-Mirman (1972) problem
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
solver = Ipopt.IpoptSolver(print_level = 3, tol = 1e-2, max_iter=1000)
br     = dynamic_programming(r, f, initial, β, grid, 1, 1, control_bounds = bounds, interpolation = interp, constraint = g, solver = solver)
shocks = rand(LogNormal(0.0, 0.25), 100)

using Colors


br_vfn = iterate_bellman_operator(br, shocks, 2, verbose = true)
hist   = Vector{Any}()
br_vfn = iterate_bellman_operator(br, shocks, 50, verbose = true, history = hist)

hist_gs   = Vector{Any}()
br_vfn_gs = iterate_bellman_operator(br, shocks, 50, verbose = true, history = hist_gs, gridsearch = true)

function average_time(v::Vector)
    avg = zero(Float64)
    for t in v[1:end]
        avg += t[2][2]
    end
    avg/length(v)
end

average_time(hist[2:end])
average_time(hist_gs[2:end])

plot_vfn(hist, title::AbstractString) = begin
    n = length(hist) - 1
    cmap = colormap("blues", n)
    plot(x->br_vfn_gs[x], 0, 20.0, )
    m = [ layer(x->first(y)[x], 0.1, 20.0, Theme(default_color=cmap[i], plot_padding=0cm)) for (i,y) in enumerate(hist[2:end])]
    p = plot(m..., Guide.xlabel(""), Guide.ylabel(""), Theme(plot_padding = 0cm, default_point_size = 10pt))
    pltdir = "/Users/zaccranko/Dropbox/University/Honours/ECON8001/Research Assignment/"
    draw(PDF(joinpath(pltdir, replace(title," ", "_")*".pdf"), 10cm, 8cm), p)
end


p = plot_vfn(hist, "Value Function Iteration (Gradient Optimisation)")
q = plot_vfn(hist_gs, "Value Function Iteration (Grid Search)")

