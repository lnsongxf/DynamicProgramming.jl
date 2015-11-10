function expected_bellman_value{T}(d::ADP{T},
                                   valuefn::ValueFunction,
                                   samples::Vector,
                                   state,
                                   control
                                   )
    new_state = Vector{T}(d.state_dim)
    v = zero(T)
    for shock in samples
        d.transition!(state, control, shock, new_state)
        v += valuefn[ new_state... ]
    end
    return v / length(samples)
end

# n.b. these return beta*∇E[v(...)] since this is more useful
# in building ∇reward(...) + beta*∇E[v(...)]
function expected_bellman_gradient!{T}(d::ADP{T},
                                      valuefn::ValueFunction,
                                      samples::Vector,
                                      state,
                                      control,
                                      g::Vector{T}
                                      )

    new_state = Vector(d.state_dim)
    _g = similar(g)
    coeff = d.beta / length(samples)
    g[:] /= coeff

    for shock in samples
        d.transition!(state, control, shock, new_state)
        g[:] += Interpolations.gradient!(_g, valuefn, new_state...)
    end

    g[:] *= coeff

    return g
end

function expected_bellman_gradient{T}(d::ADP{T},
                                      valuefn::ValueFunction,
                                      samples::Vector,
                                      state,
                                      control
                                      )
    g  = zeros(T, d.state_dim)
    return expected_bellman_gradient!(d, valuefn, samples, state, control, g)
end

function optimize_bellman(d::ADP,
                          valuefn::ValueFunction,
                          samples::Vector,
                          state::Vector
                          )

    n, m = d.control_dim, num_const(d)
    l  = fill(-Inf, n)
    u  = fill(+Inf, n)
    lb = fill(-Inf, m)
    ub = fill(+Inf, m)

    model   = MathProgBase.model(d.solver)
    problem = BellmanIteration(d, valuefn, samples, state)

    MathProgBase.loadnonlinearproblem!(model, n, m, l, u, lb, ub, :Max, problem)

    # solve the model
    MathProgBase.setwarmstart!(model, d.initial(state))
    MathProgBase.optimize!(model)

    MathProgBase.status(model) == :Optimal || warn("Possible optimisation failure")
    val     = MathProgBase.getobjval(model)
    control = MathProgBase.getsolution(model)

    return val, control
end

# this function is the most equivalent in definition to the mathematical bellman operator
function approximate_bellman{T}(d::ADP{T}, valuefn, samples; verbose = false)
    length_per_dim = map(length, d.grid)
    vals = Array{T}(length_per_dim...)
    args = Array{Vector{T}}(length_per_dim...)
    for (i, state) in enumerate(product(d.grid...))
        vals[i], args[i] = optimize_bellman(d, valuefn, samples, collect(state))
        verbose && @show i, vals[i], state[i]
    end

    new_valuefn = scale(interpolate!(vals, BSpline(Linear()), OnGrid()), d.grid...)
    return new_valuefn
end

# method to supply the initial guess for the bellman,
# use the reward function with some state
function approximate_bellman{T}(d::ADP{T}, samples::Vector)
    some_state     = collect(map(rand,d.grid))

    some_control   = d.initial(some_state)
    length_per_dim = map(length, d.grid)
    vals           = Array{T}(length_per_dim...)

    for (i,x) in enumerate(product(d.grid...))
        @inbounds vals[i] = d.reward(collect(x), some_control)
    end

    new_valuefn = scale(interpolate!(vals, BSpline(Linear()), OnGrid()), d.grid...)

    return new_valuefn
end

function fixed_iter(f::Function, t::Integer, init)
    state = init
    for _ = 1:t
        state = f(state)
    end
    return state
end

function iterate_bellman_operator{T}(d::ADP{T}, samples::Vector, t::Integer)
    init  = approximate_bellman(d, samples)

    final = fixed_iter(t, init) do current
        return approximate_bellman(d, current, samples)
    end

    return final
end
