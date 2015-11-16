type BellmanIteration{D<:AbstractDynamicProgramming} <: MathProgBase.AbstractNLPEvaluator
    d::D
    valuefn::ValueFunction
    samples::Vector
    state::Vector       # current state vector
end

BellmanIteration{D<:AbstractDynamicProgramming}(d::D, valuefn, samples::Vector, state::Number) = BellmanIteration(d, valuefn, samples, collect(state))

MathProgBase.features_available(d::BellmanIteration) = [:Grad, :Jac]

function MathProgBase.initialize(d::BellmanIteration, requested_features::Vector{Symbol})
    for feat in requested_features
        !(feat in MathProgBase.features_available(d)) && error("Unsupported feature $feat")
    end
end

MathProgBase.eval_f(bell::BellmanIteration, u)         = bellman_value(bell.d, bell.valuefn, bell.samples, bell.state, u)
MathProgBase.eval_grad_f(bell::BellmanIteration, g, u) = bellman_gradient!(bell.d, bell.valuefn, bell.samples, bell.state, u, g)

MathProgBase.jac_structure{D<:UnconstrainedDynamicProgramming}(d::BellmanIteration{D})       = [],[]
MathProgBase.eval_jac_g{D<:UnconstrainedDynamicProgramming}(bell::BellmanIteration{D}, j, u) = nothing
MathProgBase.eval_g{D<:UnconstrainedDynamicProgramming}(bell::BellmanIteration{D}, g, u)     = nothing


MathProgBase.jac_structure{D<:ConstrainedDynamicProgramming}(bell::BellmanIteration{D})      = fill(1, bell.d.control_dim), fill(1, num_const(bell.d))
MathProgBase.eval_jac_g{D<:ConstrainedDynamicProgramming}(bell::BellmanIteration{D}, j, u)   = j[:] = ForwardDiff.jacobian(u->bell.d.constraint(bell.state, u), u)
MathProgBase.eval_g{D<:ConstrainedDynamicProgramming}(bell::BellmanIteration{D}, g, u)       = bell.d.constraint(bell.state, u)

function optimize_bellman{T}(d::AbstractDynamicProgramming{T},
                             valuefn::ValueFunction,
                             shocks::Vector,
                             state::Vector{T}
                             )
    l, u = d.control_bounds
    lb = fill(-Inf, num_const(d))
    ub = fill(+Inf, num_const(d))

    model   = MathProgBase.model(d.solver)
    problem = BellmanIteration(d, valuefn, shocks, state)

    MathProgBase.loadnonlinearproblem!(model, d.control_dim, num_const(d), l, u, lb, ub, :Max, problem)

    # solve the model
    MathProgBase.setwarmstart!(model, d.initial(state))
    MathProgBase.optimize!(model)

    MathProgBase.status(model) == :Optimal || warn("Solver returned $(MathProgBase.status(model))")
    val     = MathProgBase.getobjval(model)
    control = MathProgBase.getsolution(model)

    return val, control
end
