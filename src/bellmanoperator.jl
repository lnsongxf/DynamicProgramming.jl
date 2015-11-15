typealias ValueFunction Interpolations.ScaledInterpolation

function bellman_value{T}(d::AbstractDynamicProgramming{T},
                          valuefn::ValueFunction,
                          shocks::Vector,
                          state::Vector{T},
                          control::Vector{T}
                          )
    new_state = Vector{T}(d.state_dim)
    v = zero(T)

    v = d.reward(state, control)
    v /= (coeff = d.beta / length(shocks))

    for shock in shocks
       new_state[:] = d.transition(state, control, shock)
        v += valuefn[ new_state... ]
    end

    v *= coeff

    return v
end

# n.b. these return beta*∇E[v(...)] since this is more useful
# in building ∇reward(...) + beta*∇E[v(...)]
function bellman_gradient!{T}(d::AbstractDynamicProgramming{T},
                              valuefn::ValueFunction,
                              shocks::Vector,
                              state::Vector{T},
                              control::Vector{T},
                              g::Vector{T}
                              )
    new_state = Vector{T}(d.state_dim)
    jac       = Matrix{T}(d.state_dim, d.control_dim)
    vfn_g     = Vector{T}(d.state_dim)

    ForwardDiff.gradient!(g, u->d.reward(state, u), control)
    g[:] /= (coeff = d.beta / length(shocks))
    i = 1
    for ε in shocks
        new_state[:] = d.transition(state, control, ε)

        ForwardDiff.jacobian!(jac, u->d.transition(state, u, ε), control)
        Interpolations.gradient!(vfn_g, valuefn, new_state...)

        g[:] += jac' * vfn_g
    end
    g[:] *= coeff
    return g
end

function bellman_gradient{T}(d::AbstractDynamicProgramming{T},
                             valuefn::ValueFunction,
                             shocks::Vector,
                             state::Vector{T},
                             control::Vector{T}
                             )
    g  = Vector{T}(d.control_dim)
    return bellman_gradient!(d, valuefn, shocks, state, control, g)
end

# this function is the most equivalent in definition to the mathematical bellman operator
function approximate_bellman_operator{T}(d::AbstractDynamicProgramming{T},
                                         valuefn::ValueFunction,
                                         shocks::Vector;
                                         verbose = false
                                         )
    length_per_dim = map(length, d.grid)
    vals = Array{T}(length_per_dim...)
    args = Array{Vector{T}}(length_per_dim...)
    for (i, state) in enumerate(product(d.grid...))
        verbose && @printf "now approximating state: %s\r" state
        vals[i], args[i] = optimize_bellman(d, valuefn, shocks, collect(state))
    end
    @printf ""

    new_valuefn = scale(interpolate!(vals, BSpline(Linear()), OnGrid()), d.grid...)
    return new_valuefn
end

# method to supply the initial guess for the bellman,
# use the reward function with some state
function approximate_bellman_operator{T}(d::AbstractDynamicProgramming{T}, shocks::Vector)
    length_per_dim = map(length, d.grid)
    vals           = Array{T}(length_per_dim...)

    for (i,x) in enumerate(product(d.grid...))
        @inbounds vals[i] = zero(T)
    end

    new_valuefn = scale(interpolate!(vals, d.interp, OnGrid()), d.grid...)

    return new_valuefn
end

#                                iter,state,time,allocd,total_time,gc_dff
typealias IterationState{T} Tuple{Int, T, Float64, Int, Float64, Base.GC_Diff}
typealias History{T}        Vector{IterationState{T}}

function pretty_integer(t::Int)
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

function iterate_bellman_operator{T}(d::AbstractDynamicProgramming{T}, shocks::Vector, t::Integer; verbose = false)
    init  = approximate_bellman_operator(d, shocks)

    old_state = init
    for i = 1:t
        new_state = approximate_bellman_operator(d, old_state, shocks)

        if verbose
            @printf "iteration: %s \tfrobenius norm: %0.4f\t\tapprox. sup norm: %0.4f\n" pretty_integer(i) vecnorm(old_state-new_state) maximum(abs(old_state-new_state))
        end

        old_state = new_state
    end

    return old_state
end
