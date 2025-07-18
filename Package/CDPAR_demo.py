import numpy as np

from package.Tensor_V3 import _V, U_k_block, S_k_block
from package.utils import compute_FIT
from package.Causal_block import Causal_updated, opt_boundary

from package.Metric import shift_matrix


def CD_par(X_list, Rank, p, W = None, A_list = None, lambda_w = 0.5, lambda_a = 0.5, w_threshold = 0.3, a_threshold = 0.1, V = None, max_iter=5):

    weight = 6
    J, K = X_list[0].shape[1], len(X_list)

    # initial U and related variables
    Uk = [np.random.rand(X.shape[0], Rank) for X in X_list]
    H = np.random.rand(Rank, Rank)
    U_t = [U.copy() for U in Uk]
    U_h = [U.copy() for U in Uk]
    mu_tUk = [np.zeros_like(U) for U in Uk]
    mu_hUk = [np.zeros_like(U) for U in Uk]

    # initial S and related variables
    S_init =  np.random.rand(K, Rank) + weight
    Sk = [None] * K
    for k in range(K):
        Sk[k] = np.diag(S_init[k, :])
    S_t = [S.copy() for S in Sk]
    mu_tSk = [np.zeros_like(S) for S in Sk]
    V = np.random.rand(J, Rank)  # Random initialization of V

    # Initialize A and W
    if A_list is None:
        A_list = np.zeros((p, Rank, Rank))

    if W is None:
        W = np.eye(Rank)

    # Initalization or using the parafac2
    for t in range(5):
        "Initialization"
        # Update U_k block 
        Uk_new, U_t_new, U_h_new, mu_tUk_new, mu_hUk_new, H_new = U_k_block(X_list, Uk, Sk, V, H, U_t, U_h, mu_tUk, mu_hUk, W, A_list, p)
        # Update S_k block 
        Sk_new, S_t_new, mu_tSk_new  = S_k_block(X_list, Uk_new, Sk ,S_t, mu_tSk, V, W, A_list, p)
        # Update V block
        V_new = _V(X_list, Uk_new, Sk_new)

        Uk = Uk_new
        U_t = U_t_new
        U_h = U_h_new
        H = H_new
        mu_tUk = mu_tUk_new
        mu_hUk = mu_hUk_new

        Sk = Sk_new
        S_t = S_t_new
        mu_tSk = mu_tSk_new
        V = V_new

    # Thresholding or using the parafac2
    V[np.abs(V) < 0.9] = 0 
    V = V[:, ::-1]
    Vmask = V != 0 # keep the structure by previous knowledge
    V_weight = np.ones_like(V) * weight
    V = V + V_weight
    V[~Vmask] = 0

    for t in range(max_iter):
        A_list = A_list.reshape((p, Rank, Rank))
        "Tensor Block"
        # Update U_k block 
        Uk_new, U_t_new, U_h_new, mu_tUk_new, mu_hUk_new, H_new = U_k_block(X_list, Uk, Sk, V, H, U_t, U_h, mu_tUk, mu_hUk, W, A_list, p)

        # Update S_k block 
        Sk_new, S_t_new, mu_tSk_new  = S_k_block(X_list, Uk_new, Sk ,S_t, mu_tSk, V, W, A_list, p)

        # Update V block
        V_new = _V(X_list, Uk_new, Sk_new)

        Uk = Uk_new
        U_t = U_t_new
        U_h = U_h_new
        H = H_new
        mu_tUk = mu_tUk_new
        mu_hUk = mu_hUk_new

        Sk = Sk_new
        S_t = S_t_new
        mu_tSk = mu_tSk_new

        V = V_new
        
        X = []
        Y = []
        min_meeting = min([X.shape[0] for X in X_list])
        for k in range(K):
            pre = Uk[k] @ Sk[k]
            pre = pre[:min_meeting, :]
            lagged_features = []
            for i in range(p):
                Mi = shift_matrix(min_meeting, i + 1) 
                lagged_pre = Mi @ pre 
                lagged_features.append(lagged_pre)
            Y_k = np.hstack(lagged_features) 
            Y.append(Y_k)
            X.append(pre)
            
        X = np.array(X, dtype=np.float64)
        Y = np.array(Y, dtype=np.float64)
        d = X.shape[2]
        bnds = opt_boundary(X[0], Y[0], d)
        W_new, A_new = Fed_DBN(X, Y, bnds, lambda_w=lambda_w, lambda_a=lambda_a, w_threshold=w_threshold, a_threshold=a_threshold)
        W = W_new
        A_list = A_new

    return Uk, Sk, V, W, A_list
