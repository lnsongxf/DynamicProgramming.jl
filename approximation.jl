typealias GridSpace{T<:AbstractFloat} Tuple{Vararg{Range{T}}}

grid_points(grid::GridSpace) = product()

function evaluate_points!{T}(f::Function, vals::AbstractArray{T}, grid::GridSpace{T})
    size(vals) == ([length(d) for d in grid]...) || throw(DimensionMismatch())

    for (i,x) in enumerate(product(grid...))
        @inbounds vals[i] = f(collect(x))
    end

    return vals
end

function evaluate_points{T}(f::Function, grid::GridSpace{T})
    points_per_dimension = [length(d) for d in grid]
    vals = Array{T}(points_per_dimension...)

    return evaluate_points!(f, vals, grid)
end


