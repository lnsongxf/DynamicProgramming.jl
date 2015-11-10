type BellmanIteration <: MathProgBase.AbstractNLPEvaluator
    d::DynamicProgramming
    valuefn
    samples::Vector
    state::Vector                   # state vector
end

BellmanIteration(d::DynamicProgramming, valuefn, samples::Vector, state::Number) = BellmanIteration(d, valuefn, samples, collect(state))

MathProgBase.features_available(d::BellmanIteration) = [:Grad, :Jac]

function MathProgBase.initialize(d::BellmanIteration, requested_features::Vector{Symbol})
    for feat in requested_features
        !(feat in MathProgBase.features_available(d)) && error("Unsupported feature $feat")
    end
end

function MathProgBase.eval_f(bell::BellmanIteration, control)
    return bell.d.reward(bell.state, control) +
        bell.d.beta*expected_bellman_value(bell.d, bell.valuefn, bell.samples, bell.state, control)
end

function MathProgBase.eval_grad_f(bell::BellmanIteration, g, k)
    ForwardDiff.gradient!(g, k->bell.d.reward(bell.state, k), k)
    g[:] += bell.d.beta*expected_bellman_gradient(bell.d, bell.valuefn, bell.samples, bell.state, k)
    return g
end

MathProgBase.jac_structure(d::BellmanIteration) = [],[]

function MathProgBase.eval_jac_g(bell::BellmanIteration, J, )
    # return []
    ForwardDiff.jacobian!(j, k->bell.d.constraint(bell.state, k), k)
end

function MathProgBase.eval_g(bell::BellmanIteration, g, k)
    # return []
    bell.d.constraint(bell.state, control, g)
end
