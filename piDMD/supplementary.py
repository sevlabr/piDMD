import numpy as np
import numpy.linalg as LA


def rq(A_inp, return_permutations=False):
    """
    Performs RQ decomposition.
    """
    # n_row = A.shape[0]
    if return_permutations:
        raise NotImplementedError(
            "This fuctionality is unavailable because it's not needed anywhere yet."
        )
    else:
        Q, R = LA.qr(np.flipud(A_inp).conj().T)
    
    R = np.rot90(R.conj().T, k=2)
    Q = np.flipud(Q.conj().T)
    
    n_row, m_col = A_inp.shape
    if n_row > m_col:
        R = np.hstack((np.zeros((n_row, n_row - m_col)), R))
        Q = np.vstack((np.zeros((n_row - m_col, m_col)), Q))
        
    return R, Q
    
def cell2mat(cell, dtype=np.int64):
    """
    Expected performance in all cases is not guaranteed.
    """
    mat = []
    for item in cell:
        if len(item.shape) == 1:
            for val in item:
                mat.append(val)
        elif len(item.shape) == 2:
            for row in item:
                for val in row:
                    mat.append(val)
        
    return np.array(mat, dtype=dtype)
    
def tls(J, K):
    """
    Finds solution x to J @ x = K
    in the total least squares sense.
    """
    # add new dimension if needed for stacking
    if len(J.shape) == 1:
        J = J[:, np.newaxis]
    
    n_col = J.shape[1]
    if J.shape[0] != K.shape[0]:
        raise ValueError("Matrices are not conformant.")
    
    # add new dimension if needed for stacking
    if len(J.shape) > len(K.shape):
        K = K[:, np.newaxis]
    
    R1 = np.hstack((J, K))
    _, _, V = LA.svd(R1, full_matrices=True)
    # to be consistent with MatLab's SVD
    V = V.conj().T
    R, _ = rq(V[:, n_col:])
    Gamma = R[n_col:, :]
    Z = R[:n_col, :]
    Xhat = -Z @ LA.pinv(Gamma)
    
    return Xhat