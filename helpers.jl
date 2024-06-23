#=
Helper functions for the project
=#


#=
create Laplacian matrix based on out degrees
=#
function laplacian_out(W)
    for i in 1:N  # removing self-loops
        W[i,i] = 0
    end
    # create Laplacian from struct. connectome
    D = zeros(N,N)  # out-degree matrix
    for i in 1:N
        W[i,i] = 0
        D[i,i] = sum(W[i,:])
    end
    L = D - W 
    return L
end
