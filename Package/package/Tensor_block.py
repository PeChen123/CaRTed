import numpy as np
from scipy.linalg import svd, inv, norm
from scipy import linalg
from scipy.linalg import khatri_rao as ka

from package.Metric import compute_loss, shift_matrix

def U_k_block(X_list, Uk, Sk, V, H, U_t, U_h, mu_t_Uk, mu_h_Uk, W, A_list, p, max_iter=20):

    def _Uk(Xk, Sk, V, U_tilde_k, U_hat_k, mu_tilde_k, mu_hat_k, rho_k):
        """Update U_k using the closed-form solution."""
        left_term = Xk @ V @ Sk.T + (rho_k / 2) * (U_tilde_k + U_hat_k - mu_tilde_k - mu_hat_k)

        right_term = Sk @ V.T @ V @ Sk.T + rho_k * np.eye(Sk.shape[0])

        Uk_new = left_term @ np.linalg.inv(right_term)
        
        return Uk_new
    
    #Update tilde{U}_k
    def _t_Uk(Uk, Sk, W, A_list, mu_tilde_k, rho_k, I_k, p):
        """Update tilde{U}_k with SVAR term and shift matrices."""
        n_rows = Uk.shape[0]
        n_cols = Sk.shape[1]
        
        # Compute Phi matrix using Kronecker products
        I_minus_W = np.eye(n_cols) - W
        Phi = np.kron(I_minus_W.T @ Sk.T, np.eye(n_rows))
        
        # Subtract terms for each lag
        for i in range(p):
            Mi = shift_matrix(I_k, i + 1)  # i+1 since A_list is 0-indexed
            Phi -= np.kron(A_list[i].T @ Sk.T, Mi)
        
        # Compute v_k vector
        v_k = (Uk + mu_tilde_k).flatten()  # vec(U_k + mu_tilde_k)
        
        # Compute the matrix to invert
        PhiT_Phi = Phi.T @ Phi
        mat_to_inv = (1/I_k) * PhiT_Phi + rho_k * np.eye(Phi.shape[1])
        
        # Solve for u_k vector
        u_k = linalg.solve(mat_to_inv, rho_k * v_k)
        # Reshape back to matrix form
        U_tilde_k_new = u_k.reshape(n_rows, n_cols)
        
        return U_tilde_k_new


    def _h_Uk(Uk_list, H,  mu_hat_k, rho_k, K):
        """Update hat{U}_k and H using SVD."""
        Qk_list = []

        # Update Qk and H iteratively
        for _ in range(1):
            Qk_list = []
            for k in range(K):
                U_svd,_, Vt_svd = svd((Uk_list[k] + mu_hat_k[k]) @ H.T, full_matrices=False)
                Qk = U_svd @ Vt_svd
                Qk_list.append(Qk)
            H = (1 / sum(rho_k)) * sum(rho_k * Qk.T @ (Uk + mu) for rho_k, Qk, Uk, mu in zip(rho_k, Qk_list, Uk_list, mu_hat_k))
        
        hat_Uk_list = [Qk @ H for Qk in Qk_list]

        return hat_Uk_list,H
    
    def _dual(Uk, U_t, U_h, mu_tilde_Uk, mu_hat_Uk):

        """Update dual variables mu."""
        mu_tilde_Uk_new = mu_tilde_Uk + Uk - U_t
        mu_hat_Uk_new = mu_hat_Uk + Uk - U_h

        return mu_tilde_Uk_new, mu_hat_Uk_new
    
    K = len(X_list)
    rho_k = np.zeros(K) 
    R = V.shape[1]        

    for k in range(K):
        rho_k[k] = 1/R * np.trace(Sk[k] @ V.T @ V @ Sk[k])


    for t in range(max_iter):
        # Initialize variables
        Uk_new = [U.copy() for U in Uk]
        U_t_new = [U.copy() for U in U_t]
        U_h_new = [U.copy() for U in U_h]
        H_new = H.copy()
        mu_h_Uk_new = [mu.copy() for mu in mu_h_Uk]
        mu_t_Uk_new = [mu.copy() for mu in mu_t_Uk]
        
        U_h_new, H_new= _h_Uk(Uk, H, mu_h_Uk, rho_k, K) 

        for k in range(K):
            U_t_new[k] = _t_Uk(Uk[k], Sk[k], W, A_list, mu_t_Uk[k], rho_k[k], X_list[k].shape[0], p)
            Uk_new[k] = _Uk(X_list[k], Sk[k], V, U_t_new[k], U_h_new[k], mu_t_Uk[k], mu_h_Uk[k], rho_k[k])

        for k in range(K):
           mu_t_Uk_new[k], mu_h_Uk_new[k] = _dual(Uk_new[k], U_t_new[k], U_h_new[k], mu_t_Uk[k], mu_h_Uk[k])

        # Update variables
        Uk = Uk_new
        U_t = U_t_new
        U_h = U_h_new
        H = H_new
        mu_h_Uk = mu_h_Uk_new
        mu_t_Uk = mu_t_Uk_new

        sum1 = 0 
        for k in range(K):
            sum1 += norm(Uk[k] - U_t[k],'fro') / norm(Uk[k],'fro')
            sum1 = sum1/K

        # print(f"Iteration {t+1}, convergence measure: {sum1}")
        
        if sum1 < 1e-5:
            # print(f"Convergence achieved at iteration {t+1}, convergence measure: {sum1}")
            break

    return Uk, U_t, U_h, mu_h_Uk, mu_t_Uk, H

def S_k_block(X_list, Uk, Sk, Sk_t, mu_t_Sk, V, W, A_list, p, max_iter=10):
    """Update S_k using the closed-form solution."""

    def _Sk(Xk, Uk, V, S_tilde_k, mu_Sk, rho_k):
        """Update S_k using the closed-form solution."""

        left_term = np.diag(Uk.T @ Xk @ V) + (rho_k / 2) * np.diag(S_tilde_k - mu_Sk)

        right_term = np.multiply((V.T @ V), (Uk.T @ Uk)) + (rho_k / 2) * np.eye(Uk.shape[1])
        Sk_new = np.linalg.inv(right_term) @ left_term

        return np.diag(Sk_new)

    # Update tilde{S}_k

    def _t_Sk(Uk, Sk, W, A_list, mu_tilde_Sk, rho_k, I_k, p):
        """Update tilde{S}_k with SVAR term and shift matrices."""

        Tk = ka(np.eye(Sk.shape[1]).T,Uk) - ka(W.T, Uk)

        for i in range(1, p + 1):
            Mi = shift_matrix(I_k, i)
            Tk -= ka(A_list[i - 1].T, Mi@Uk)

        left_term = rho_k * np.diag((Sk + mu_tilde_Sk))
        right_term = (1 / I_k) * (Tk.T @ Tk) + rho_k * np.eye(Sk.shape[1])
        S_tilde_k_new = np.linalg.inv(right_term) @ left_term

        return np.diag(S_tilde_k_new)
    
    # Update dual variables

    def _dual(Sk, S_tilde_k, mu_tilde_Sk):
        """Update dual variables mu."""

        mu_tilde_Sk_new = mu_tilde_Sk + Sk - S_tilde_k

        return mu_tilde_Sk_new
    
    K = len(X_list)
    rho_k = np.zeros(K) 
    R = V.shape[1]      

    for k in range(K):
        rho_k[k] = (1 / R) * np.trace(np.multiply(V.T @ V , Uk[k].T @ Uk[k]))

    
    for t in range(max_iter):
        Sk_new = [S.copy() for S in Sk]
        S_t_new = [S.copy() for S in Sk_t]

        for k in range(K):
            Sk_new[k] = _Sk(X_list[k], Uk[k], V, Sk_t[k], mu_t_Sk[k], rho_k[k])
            S_t_new[k] = _t_Sk(Uk[k], Sk_new[k], W, A_list, mu_t_Sk[k], rho_k[k], X_list[k].shape[0], p)
            mu_t_Sk[k] = _dual(Sk_new[k], S_t_new[k], mu_t_Sk[k])

        Sk = Sk_new
        Sk_t = S_t_new

        sum1 = 0 
        for k in range(K):
            sum1 += norm(Sk[k] - Sk_t[k],'fro') / norm(Sk[k],'fro')
            sum1 = sum1/K

        # print(f"Iteration {t+1}, convergence measure: {sum1}")
        
        if sum1 < 1e-5:
            # print(f"Convergence achieved at iteration {t+1}, convergence measure: {sum1}")
            break

    return Sk, Sk_t, mu_t_Sk


def _V(X_list, Uk_list, Sk_list):
    """Update V using the closed-form solution."""
    K = len(X_list)
    d = X_list[0].shape[1]  # Columns of X_k.T (assuming X_k is n x d)
    r = Uk_list[0].shape[1]  # Columns of U_k (assuming U_k is n x r)

    V_numerator = np.zeros((d, r))
    ratio = np.zeros((r, r))
    
    for k in range(K):
        Xk = X_list[k]
        Uk = Uk_list[k]
        Sk = Sk_list[k]
        
        # Accumulate terms for V and ratio
        V_numerator += Xk.T @ Uk @ Sk
        ratio += Sk.T @ Uk.T @ Uk @ Sk

    V = V_numerator @ np.linalg.inv(ratio)
    
    return V
