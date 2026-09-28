import numpy as np

from package.Tensor_block import _V, U_k_block, S_k_block
from package.utils import compute_FIT
from package.Causal_block import Fed_DBN,opt_boundary

from package.Metric import shift_matrix

class Ragged:
    """Per-patient arrays with different row counts, shaped for Fed_DBN.

    Fed_DBN only reads X.shape -> (K, _, d) and X[k]; each local subproblem
    normalises by its own row count, so no patient has to be truncated.
    """
    def __init__(self, arrays):
        self.arrays = list(arrays)
        self.shape = (len(self.arrays), None, self.arrays[0].shape[1])

    def __getitem__(self, k):
        return self.arrays[k]


def causal_inputs_full(Uk, Sk, p):
    """X_k = U_k S_k and its p lags, each on the patient's own I_k rows."""
    X, Y = [], []
    for U, S in zip(Uk, Sk):
        pre = U @ S
        X.append(pre)
        Y.append(np.hstack([shift_matrix(pre.shape[0], i + 1) @ pre for i in range(p)]))
    return Ragged(X), Ragged(Y)


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
        
        X, Y = causal_inputs_full(Uk, Sk, p)
        bnds = opt_boundary(X[0], Y[0], X.shape[2])
        W_new, A_list_new = Fed_DBN(X, Y, bnds, lambda_w=lambda_w, lambda_a=lambda_a,
                            w_threshold=w_threshold, a_threshold=a_threshold)
        W = W_new
        A_list= A_list_new

    return Uk, Sk, V, W, A_list
