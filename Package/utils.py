import numpy as np

def V_mea(V, V_hat):
    """
    Computes the similarity measure:
    
        SIM(V, V_hat) = (1 / R) * sum_{i=1}^R [ max_{1 <= j <= R} ( v_i^T v_hat_j / (||v_i|| * ||v_hat_j||) ) ]
    
    where V and V_hat are each n x R matrices, and v_i (or v_hat_j) 
    refers to the i-th (or j-th) column.
    
    Parameters
    ----------
    V      : np.ndarray of shape (n, R)
        True (or ground-truth) phenotype matrix; each column is a vector in R^n.
    V_hat  : np.ndarray of shape (n, R)
        Estimated phenotype matrix to compare against V.
    
    Returns
    -------
    sim_value : float
        The computed similarity measure. 
    """
    # Number of columns (factors)
    _, R = V.shape
    
    # 1) Normalize 
    V_norms = np.linalg.norm(V, axis=0, keepdims=True)       
    V_hat_norms = np.linalg.norm(V_hat, axis=0, keepdims=True)  

    V_norms[V_norms == 0] = 1
    V_hat_norms[V_hat_norms == 0] = 1

    # Normalized versions of V and V_hat
    V_normalized = V / V_norms         # shape (n, R)
    V_hat_normalized = V_hat / V_hat_norms   # shape (n, R)

    # 2) Compute the similarity matrix:

    C = V_normalized.T @ V_hat_normalized  
    max_per_row = np.max(C, axis=1)  # shape (R,)
    sim_value = np.mean(max_per_row)
    
    return sim_value


def compute_Uk(Uk_list, Uk_ground):
    """
    Compute CPI based on ground-truth U’s:

        CPI_gt = 1 - sum_k ||Uk_list[k]^T Uk_list[k] - Uk_ground[k]^T Uk_ground[k]||_F^2
                       / sum_k ||Uk_ground[k]^T Uk_ground[k]||_F^2

    Args:
        Uk_list    : list of inferred U_k arrays, each shape (I_k, R)
        Uk_ground  : list of ground-truth U_k arrays, same shapes as Uk_list

    Returns:
        CPI_gt (float)
    """
    error = 0.0
    denom = 0.0

    # loop over each slice k
    for U_est, U_gt in zip(Uk_list, Uk_ground):
        C_est = U_est.T @ U_est     # shape (R,R)
        C_gt  = U_gt.T  @ U_gt      # shape (R,R)

        error += np.linalg.norm(C_est - C_gt, 'fro')**2
        denom += np.linalg.norm(C_gt,    'fro')**2

    # guard against divide-by-zero
    if denom == 0:
        return 0.0

    CPI_gt = 1.0 - error/denom
    return CPI_gt

def compute_FIT(X_list, U_list, S_list, V):
    total_error = 0.0
    total_norm_X = 0.0
    for k in range(len(X_list)):
        X_k = X_list[k]
        U_k = U_list[k]
        S_k = S_list[k]
        recon = U_k @ S_k @ V.T
        total_error += np.linalg.norm(X_k - recon, 'fro') ** 2
        total_norm_X += np.linalg.norm(X_k, 'fro') ** 2
    FIT = 1 - (total_error / total_norm_X)
    return FIT
