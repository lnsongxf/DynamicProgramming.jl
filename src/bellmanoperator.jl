typealias ValueFunction Interpolations.AbstractInterpolation

function bellman_value{T}(d::ADP{T},
                          valuefn::ValueFunction,
                          samples::Vector,
                          state::Vector{T},
                          control::Vector{T}
                          )
    new_state = Vector{T}(d.state_dim)
    v = zero(T)

    v = d.reward(state, control)
    v /= (coeff = d.beta / length(samples))

    for shock in samples
       new_state[:] = d.transition(state, control, shock)
        v += valuefn[ new_state... ]
    end

    v *= coeff

    return v
end

# n.b. these return beta*∇E[v(...)] since this is more useful
# in building ∇reward(...) + beta*∇E[v(...)]
function bellman_gradient!{T}(d::ADP{T},
                              valuefn::ValueFunction,
                              samples::Vector,
                              state::Vector{T},
                              control::Vector{T},
                              g::Vector{T}
                              )
    new_state = Vector{T}(d.state_dim)
    jac       = Matrix{T}(d.state_dim, d.control_dim)
    _g        = Vector{T}(d.state_dim)

    ForwardDiff.gradient!(g, u->d.reward(state, u), control)
    g[:] /= (coeff = d.beta / length(samples))
    i = 1
    for ε in samples
        new_state[:] = d.transition(state, control, ε)
        ForwardDiff.jacobian!(jac, u->d.transition(state, u, ε), control)
        g[:] += jac'*Interpolations.gradient!(_g, valuefn, new_state...)
    end
    g[:] *= coeff
    return g
end

function bellman_gradient{T}(d::ADP{T},
                             valuefn::ValueFunction,
                             samples::Vector,
                             state::Vector{T},
                             control::Vector{T}
                             )
    g  = Vector{T}(d.control_dim)
    return bellman_gradient!(d, valuefn, samples, state, control, g)
end

function optimize_bellman{T}(d::ADP{T},
                             valuefn::ValueFunction,
                             samples::Vector,
                             state::Vector{T}
                             )

    n, m = d.control_dim, num_const(d)
    l, u = d.control_bounds
    lb = fill(-Inf, m)
    ub = fill(+Inf, m)

    model   = MathProgBase.model(d.solver)
    problem = BellmanIteration(d, valuefn, samples, state)

    MathProgBase.loadnonlinearproblem!(model, n, m, l, u, lb, ub, :Max, problem)

    # solve the model
    MathProgBase.setwarmstart!(model, d.initial(state))
    MathProgBase.optimize!(model)

    MathProgBase.status(model) == :Optimal || warn("Solver returned $(MathProgBase.status(model))")
    val     = MathProgBase.getobjval(model)
    control = MathProgBase.getsolution(model)

    return val, control
end

# this function is the most equivalent in definition to the mathematical bellman operator
function approximate_bellman_operator{T}(d::ADP{T},
                                         valuefn::ValueFunction,
                                         samples::Vector;
                                         verbose = false
                                         )
    length_per_dim = map(length, d.grid)
    vals = Array{T}(length_per_dim...)
    args = Array{Vector{T}}(length_per_dim...)
    for (i, state) in enumerate(product(d.grid...))
        verbose && @show i, state
        vals[i], args[i] = optimize_bellman(d, valuefn, samples, collect(state))
    end

    new_valuefn = scale(interpolate!(vals, BSpline(Linear()), OnGrid()), d.grid...)
    return new_valuefn
end

# method to supply the initial guess for the bellman,
# use the reward function with some state
function approximate_bellman_operator{T}(d::ADP{T}, samples::Vector)
    length_per_dim = map(length, d.grid)
    vals           = Array{T}(length_per_dim...)

    for (i,x) in enumerate(product(d.grid...))
        @inbounds vals[i] = zero(T)
    end

    new_valuefn = scale(interpolate!(vals, BSpline(Linear()), OnGrid()), d.grid...)

    return new_valuefn
end

#                                iter,state,time,allocd,total_time,gc_dff
typealias IterationState{T} Tuple{Int, T, Float64, Int, Float64, Base.GC_Diff}
typealias History{T}        Vector{IterationState{T}}

function pretty_integer_count(t::Int)
    @assert t >= 0 "What is this"
    if    t < 10^2
        @sprintf "%d" t
    elseif t < 10^3
        @sprintf "%.1f×10²" t / 10^2
    elseif t < 10^4
        @sprintf "%.1f×10³" t / 10^3
    elseif t < 10^5
        @sprintf "%.1f×10⁴" t / 10^4
    elseif t < 10^6
        @sprintf "%.1f×10⁵" t / 10^5
    elseif t < 10^7
        @sprintf "%.1f×10⁶" t / 10^6
    elseif t < 10^8
        @sprintf "%.1f×10⁷" t / 10^7
    elseif t < 10^9
        @sprintf "%.1f×10⁸" t / 10^8
    elseif t < 10^10
        @sprintf "%.1f×10⁹" t / 10^9
    else
        @sprintf "%s"       t
    end
end

function verbose(print_level::Int, hist::IterationState)
    print_level == 0 && return

    i, state, elapsedtime, allocd, gctime, diff = hist
    iter = pretty_integer_count(i)

    if print_level >= 1
        n = 100
        @printf "%s\n" Base.cpad(" Iteration $iter ", n, '=')

        Base.time_print(elapsedtime*1e9, allocd, gctime, Base.gc_alloc_count(diff))

        if print_level >= 2
            @printf "%s\n" Base.cpad(" Current State ", n, '-')
            println(state)
        end

        @printf "%s\n\n" Base.cpad("-", n, '-')
    end
end


function fixed_iter!{T}(f::Function,
                        t::Integer,
                        init::T,
                        print_level::Int = 0,
                        trace::Int = 0,
                        history::History{T} = History{T}())
    state = init::T
    for i = 1:t
        timed = @timed f(state)

        if trace !== 0 && i % trace == 0
            push!(history, (i, timed...))
        end

        verbose(print_level, (i, timed...))

        state = first(timed)
    end

    return state
end

supnorm(u::ValueFunction, v::ValueFunction) = maximum(abs(u - v))

function iterate_bellman_operator{T}(d::ADP{T}, samples::Vector, t::Integer; print_level = 0)
    init  = approximate_bellman_operator(d, samples)
    hist  = History{ValueFunction}()
    final = fixed_iter!(t, init, print_level, 0, hist) do valuefn
        return value, argmax = approximate_bellman_operator(d, valuefn, samples)
    end

    return final
end


# h = History{Int}()
# push!(h, (10, x...))
# push!(h, (3*10^6, x...))

# verbose(1, h[1:1]); verbose(1, h[1:2])

# i = 10
# x = @sprintf "Iteration %s\n" rpad(string(i," "), 40, '=')

# function fixed_iter!{T}(f::Function,
#                         t::Integer,
#                         init::T,
#                         verb::Int = 0,
#                         trace::Int = 0,
#                         history::History{T} = History{T}())
#     state = init::T

#     for i = 1:t
#         timed = @timed f(state)

#         tr = form_history(i, timed)


#         if trace !== 0 || i % trace == 0
#             push!(history, tr)
#         end

#         verbose(verb, tr)

#         state = tr[2]
#     end

#     return state
# end

