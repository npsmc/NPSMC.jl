struct LocallyConstant <: AbstractRegression

end

""" 
    Apply the analog method on catalog of historical data 
    to generate forecasts. 
"""
function predict( LocallyConstant, xf :: Array{Float64,2}, weight)

    # weighted mean and covariance
    xf_mean = sum(xf .* weights',dims=2)
    Exf   = xf .- xf_mean

    Symmetric(1.0 ./(1.0 .- sum(weights.^2)) .* (Exf .* weights') * Exf')

end
