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
                                         verbose = false,
                                         gridsearch = false
                                         )
    length_per_dim = map(length, d.grid)
    vals = Array{T}(length_per_dim...)
    args = Array{Vector{T}}(length_per_dim...)
    for (i, state) in enumerate(product(d.grid...))
        verbose && @printf "now approximating state: %s\r" state
        vals[i], args[i] = if !gridsearch
            optimize_bellman(d, valuefn, shocks, collect(state))
        else
            optimize_bellman_with_gridsearch(d, valuefn, shocks, collect(state))
        end
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

function iterate_bellman_operator{T}(d::AbstractDynamicProgramming{T}, shocks::Vector, t::Integer; verbose = false, history = nothing)
    init  = approximate_bellman_operator(d, shocks)


    old_state = init
    isa(history, Vector) && push!(history, old_state)

    for i = 1:t
        new_state = approximate_bellman_operator(d, old_state, shocks)

        if verbose
            @printf "bellman iteration: %s\tfrobenius norm: %0.2f\tapprox. sup norm: %0.4f\n" i vecnorm(old_state-new_state) maximum(abs(old_state-new_state))
        end

        isa(history, Vector) && push!(history, new_state)

        old_state = new_state
    end

    return old_state
end
