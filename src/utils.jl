using Statistics, LinearAlgebra

" Compute the Root Mean Square Error between 2 n-dimensional vectors. "
function RMSE(a, b)
 
    sqrt.(mean((a .- b).^2))

end

" Normalize the entries of a multidimensional array sum to 1. "
function normalise!(M)

    c = sum(M)
    # Set any zeros to one before dividing
    c += c == 0
    M ./= c

end

""" 
Ensure the matrix is stochastic, i.e., 
the sum over the last dimension is 1.
"""
function mk_stochastic!(T)

    if last(size(T)) == 1
        normalise!(T)
    else
        n = ndims(T)
        # Copy the normaliser plane for each i.
        normaliser = sum(T,dims=n)
        # Set zeros to 1 before dividing
        # This is valid since normaliser(i) = 0 iff T(i) = 0
        normaliser .+= normaliser .== 0
        T ./= normaliser
    end

end

""" 
Sampling from a non-uniform distribution. 
"""
function sample_discrete(prob)

    # this speedup is due to Peter Acklam
    cumprob = cumsum(prob)
    R = rand()
    M = 0 :: Int64
    N = length(cumprob)
    for i in 1:N-1
        M += R > cumprob[i]
    end
    M
end


""" 
    inv_using_SVD(Mat, eigvalMax)

SVD decomposition of Matrix. 
"""
function inv_using_SVD(Mat, eigvalMax)
    
    F = svd(Mat; full = true)
    eigval = cumsum(F.S) ./ sum(F.S)
    # search the optimal number of eigen values
    i_cut = findfirst(eigval .>= eigvalMax)

    U_1 = @view F.U[1:i_cut,1:i_cut]
    U_3 = @view F.U[min(i_cut+1,end):end,1:i_cut]
    
    V_1 = @view F.Vt'[1:i_cut,1:i_cut]
    V_3 = @view F.Vt'[min(i_cut+1,end):end,1:i_cut]

    tmp1 = (V_1 ./ F.S[1:i_cut]') * U_1'
    tmp2 = (V_1 ./ F.S[1:i_cut]') * U_3'
    tmp3 = (V_3 ./ F.S[1:i_cut]') * U_1'
    tmp4 = (V_3 ./ F.S[1:i_cut]') * U_3'

    vcat(hcat(tmp1,tmp2),hcat(tmp3,tmp4))
    
end
    
