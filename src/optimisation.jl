type BellmanIteration <: MathProgBase.AbstractNLPEvaluator
    d::AbstractDynamicProgramming
    valuefn
    samples::Vector
    state::Vector                   # state vector
end

BellmanIteration(d::AbstractDynamicProgramming, valuefn, samples::Vector, state::Number) = BellmanIteration(d, valuefn, samples, collect(state))

MathProgBase.features_available(d::BellmanIteration) = [:Grad, :Jac]

function MathProgBase.initialize(d::BellmanIteration, requested_features::Vector{Symbol})
    for feat in requested_features
        !(feat in MathProgBase.features_available(d)) && error("Unsupported feature $feat")
    end
end

MathProgBase.eval_f(bell::BellmanIteration, u)         = bellman_value(bell.d, bell.valuefn, bell.samples, bell.state, u)
MathProgBase.eval_grad_f(bell::BellmanIteration, g, u) = bellman_gradient!(bell.d, bell.valuefn, bell.samples, bell.state, u, g)
MathProgBase.jac_structure(d::BellmanIteration)        = [],[]
MathProgBase.eval_jac_g(bell::BellmanIteration, J, u)  = ()
MathProgBase.eval_g(bell::BellmanIteration, g, k)      = ()
