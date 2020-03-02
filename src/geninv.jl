""" 
    geninv(G)


Returns the Moore-Penrose inverse of the argument
Transpose if m < n 

"""
function geninv(G)
    m,n = size(G)
    transpose=false
    
    if m<n
        transpose = true
        A = G*G'
        n = m
    else 
        A = G'*G;
    end
    
    # Full rank Cholesky factorization of A
    
    dA  = Diagonal(A)
    tol = minimum(dA[dA .> 0]) * 1e-9
    L   = zero(A)
    r   = 0
    for k=1:n
        r = r+1; 
        if r == 1
           L[k:n,r] = A[k:n,k] 
        else
           L[k:n,r] = A[k:n,k] - L[k:n,1:(r-1)] * L[k,1:(r-1)]'
        end
        # Note: for r=1, the substracted vector is zero 
        if L[k,r] > tol
            L[k,r] = sqrt(L[k,r]); 
            if k<n
                L[(k+1):n,r] = L[(k+1):n,r]/L[k,r]
            end
        else 
           r=r-1;
        end 
    end
    L = L[:,1:r];
    # Computation of the generalized inverse of G
    M = inv(L'L)
    if transpose
        Y = G'*L*M*M*L'
    else
        Y = L*M*M*L'*G'
    end
    
    return Y

end
