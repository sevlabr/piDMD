import numpy as np
import numpy.linalg as LA
from numpy.matlib import repmat
from numpy.fft import fft, ifft, fft2
from scipy.sparse import csr_matrix, spdiags, hstack, vstack
from scipy.linalg import block_diag
from .supplementary import rq, cell2mat, tls


def piDMD(
    X: np.ndarray,
    Y: np.ndarray,
    method: str,
    rank: int = None,
    diag: np.ndarray = None,
    block_sizes: np.ndarray = None,
    complex: bool = False
):
    """
    Inputs:
      X - data snapshots x{1} ... x{m-1}
      Y - data snapshots x{2} ... x{m}
      
      The piDMD function finds matrix A such that it satisfies
      argmin{|Y - AX|} on a specific matrix manifold associated
      with the considered physics.
    
      Implemented methods:
    
      --- method ---------- physics ---------- matrix manifold ---
          "exact"            modal                low rank
          
        "orthogonal"      conservative       orthogonal (unitary)
        
      "uppertriangular"      causal            upper triangular
      "lowertriangular"    anti-causal         lower triangular
      
      "diagonal" /
      "diagonaltls" /
      "diagonalpinv"         local            diagonal or banded
      (the last one is
      recommended)
      
         "symmetric" /      Hermitian /            symmetric
       "skewsymmetric"    anti-Hermitian        skew-symmetric
       
        "toeplitz" /     shift-invariant /        Toeplitz /
          "hankel"     anti-shift-invariant         Hankel
                              (1D)
      
       "circulant" /     shift-invariant          circulant
      "circulantTLS"       and periodic
                              (1D)
                              
    "circulantunitary" /    shift-invariant       circulant
    "circulantsymmetric" /    and periodic           and
    "circulantskewsymmetric"   (1D)                unitary /
                              and                 symmetric /
                           conservative /       skew-symmetric
                             Hermitian /
                          anti-Hermitian
      
           "BCCB" /      shift-invariant       block-circulant-
          "BCCBtls"           and              circulant-block
                            periodic               (BCCB)
                              (2D)
    
       "BCCBunitary" /   shift-invariant            BCCB
      "BCCBsymmetric" /       and                    and
      "BCCBskewsymmetric"   periodic               unitary /
                              (2D)                symmetric /
                              and               skew-symmetric
                          conservative /
                           Hermitian /
                          anti-Hermitian
                          
            "BC" /          ----------          block-circulant
          "BCtri" /      no info provided       (BC, blocks are
          "BCtls"           ----------           circulant and
                                                 values are not)
                                                     and
                                                 tridiagonal
      
      "symtridiagonal"       local         symmetric and tridiagonal
                              and
                           Hermitian
                         (self-adjoint)
      
      rank, diag, block_sizes and complex are additional parameters.
      Check the code below for details.
    
    Outputs:
      Returns A in a form of a function that takes vectors v as inputs
      and outputs the vector matrix product A@v where A is the learned
      model. If the model (matrix A) is desired explicitly then
      it can be formed by A(np.eye(n)) where n is the state dimension.
      (This functionality is not always available. Check Notes below
      for details.)
    
      eVals and eVecs are the eigenvalues and eigenvectors of the model
      A respectively. This method exploits the structure of the matrix
      manifold to efficiently compute the model's eigendecomposition.
      As such, this technique is usually far more efficient than forming
      the model explicitly and computing the eigendecomposition.
      (This functionality is not always available. Check Notes below
      for details.)
      
    Notes:
      - Maybe check why sometimes SVD here gives Ux with smaller dimension
        (this probably doesn't change anything).
      - "orthogonal" works not better than exact DMD in case of small number
        of samples m (in current implementation; but MatLab version also has
        this problem). This means that reconstrucions have bad quality. But
        eigenvalues are still on the circle.
      - "---triangular" returns full A in a form of matrix and simple eigen-
        decomposition of it. Also same problems with reconstruction quality
        as "orth." Probably problems are connected with numerical errors due to
        ill-conditioned/singular matrices.
      - "---triangular" eigenvalues looks like downscaled versions of real
        eigenvalues.
      - "diagonal---" methods give downscaled eigenvalues. Matrix A reconstruction
        works fine. Outperforms exact DMD in case of high feature space and low
        number of samples.
      - "diagonaltls" works slower but may give better results in case of
        ill-conditioned/singular matrices.
      - "symmetric" and "skewsymmetric" give almost the same results as exact
        DMD. Advantage is that piDMD gives exactly symmetric/skew-symmetric
        matrices and enforces eigenvalues to lie strictly on Re/Im axis.
      - "toeplitz"/"hankel" returns matrix A itself. eVals and eVecs are
        computed directly. Outperforms exact DMD. Eigenvalues are downscaled
        in presence of noise.
      - "circulant---": Outperforms exact DMD. Eigenvalues are downscaled
        in presence of noise.
      - "BCCB---": check comments in the code to see how to properly use
        parameters (rank, block_sizes) and other info. Outperforms exact DMD
        in most cases. Eigenvalues are downscaled in presence of noise as in
        all other piDMD cases.
      - "BC---": same notes as for "BCCB---" case.
      - "symtridiagonal" methods give downscaled eigenvalues. Returns matrix A
        itself in a sparse format. Outperforms exact DMD in case of high
        feature space and low number of samples.
    """
    nx, nt = X.shape
    
    # choose output rank
    if rank is not None:
        r = rank
    else:
        r = min(nx, nt)
    
    if method == "exact":
        Ux, Sx, Vx = LA.svd(X, full_matrices=False)
        # to be consistent with MatLab's SVD
        Vx = Vx.conj().T
        if r <= len(Sx):
            Ux = Ux[:, :r]
            Sx = Sx[:r]
            Vx = Vx[:, :r]
        
        Sx = np.diag(Sx)
        Atilde = (Ux.conj().T @ Y) @ Vx @ LA.pinv(Sx)
        
        A = lambda v: Ux @ (Atilde @ (Ux.conj().T @ v))
        
        eVals, eVecs = LA.eig(Atilde)
        eVecs = Y @ Vx @ LA.pinv(Sx) @ eVecs / eVals
        
    elif method == "orthogonal":
        Ux, _, _ = LA.svd(X, full_matrices=False)
        if r <= Ux.shape[1]:
            Ux = Ux[:, :r]
        # project X and Y onto principal components
        Yproj = Ux.conj().T @ Y
        Xproj = Ux.conj().T @ X
        Uyx, _, Vyx = LA.svd(Yproj @ Xproj.conj().T, full_matrices=False)
        Aproj = Uyx @ Vyx
        A = lambda x: Ux @ (Aproj @ (Ux.conj().T @ x))
        
        eVals, eVecs = LA.eig(Aproj)
        eVecs = Ux @ eVecs
        
    elif method == "uppertriangular":
        # Q*Q' = I
        R, Q = rq(X)
        Ut = np.triu(Y @ Q.conj().T)
        A = Ut @ LA.pinv(R) # in MatLab B/A = B @ inv(A) = (A'\B')'
        
        # this part is missing in MatLab implementation;
        # maybe need to scale vals and vecs somehow
        eVals, eVecs = LA.eig(A)
        
    elif method == "lowertriangular":
        X = np.flipud(X)
        Y = np.flipud(Y)
        A, _, _ = piDMD(X, Y, "uppertriangular")
        A = np.rot90(A, k=2)
        
        # this part is missing in MatLab implementation;
        # maybe need to scale vals and vecs somehow
        eVals, eVecs = LA.eig(A)
        
    # Codes allow to use matrices of variable banded width
    # (width of non-zero diagonal). The diag input, a [n x 2]
    # matrix called d, specifies the upper and lower bounds
    # of the indices of the non-zero elements. The first
    # column corresponds to the width of the band below the
    # diagonal and the second column is the width of the band
    # above. For example, a diagonal matrix would have
    # d = np.ones((nx, 2))
    # and a tridiagonal matrix would have
    # d = 2 * np.ones((nx, 2)).
    # If you only specify d as a scalar then the algorithm
    # converts the input to obtain a banded  diagonal matrix of
    # width d.
    elif method.startswith("diagonal"):
        if diag is not None:
            # arrange d into an nx-by-2 matrix
            d = diag
            if d.size == 1:
                d *= np.ones((nx, 2))
            elif d.size == nx:
                d = repmat(d, 1, 2)
            elif d.shape[0] != nx or d.shape[1] != 2:
                raise ValueError("diag number is not in an allowable format.")
        else:
            # default is for a diagonal matrix
            d = np.ones((nx, 2))
        
        # allocate cells to build sparse matrix
        # TODO: maybe change to scipy.sparse for better performance
        Icell, Jcell, Rcell = [0]*nx, [0]*nx, [0]*nx
        for j in range(nx):
            l1 = int(max(j - (d[j, 0] - 2), 1)) - 1
            l2 = int(min(j + d[j, 1], nx)) - 1
            # preparing to solve min||Cx-b|| along each row
            C = X[l1:l2+1, :]
            b = Y[j, :]
            if method == "diagonal":
                sol = b @ LA.inv(C) # in MatLab B/A = B @ inv(A) = (A'\B')'
            elif method == "diagonalpinv":
                sol = b @ LA.pinv(C)
            elif method == 'diagonaltls':
                sol = tls(C.T, b.T).T
            
            # val - 1 because this arrays will be used for indexing
            # (TODO: probably this doesn't work anyway so consider better indexing correction)
            Icell[j] = (j + 1 - 1) * np.ones((1, 1 + l2 - l1))
            Jcell[j] = np.array([val - 1 for val in range(l1 + 1, l2 + 2)])
            Rcell[j] = sol
        
        # TODO: cell2mat function is silly
        Imat = cell2mat(Icell, dtype=np.int64)
        Jmat = cell2mat(Jcell, dtype=np.int64)
        if complex:
            Rmat = cell2mat(Rcell, dtype=np.complex128)
        else:
            Rmat = cell2mat(Rcell, dtype=np.float64)
        
        # https://stackoverflow.com/questions/40890960/numpy-scipy-equivalent-of-matlabs-sparse-function
        Asparse = csr_matrix(
            (Rmat, (Imat, Jmat)),
            shape=(nx, nx),
            dtype=Rmat.dtype
        ) # .eliminate_zeros() # probably use m.todense() or don't use eliminate_zeros()
        A = lambda v: Asparse @ v
        
        eVals, eVecs = LA.eig(Asparse.toarray())
        
    elif method == "symmetric" or method == "skewsymmetric":
        Ux, S, V = LA.svd(X, full_matrices=False)
        # to be consistent with MatLab svd output
        V = V.conj().T
        C = Ux.conj().T @ Y @ V
        if rank is None:
            r = LA.matrix_rank(X)
        Ux = Ux[:, :r]
        
        if method == "symmetric":
            Yf = np.zeros((r, r))
            for i in range(r):
                Yf[i, i] = C[i, i].real / S[i]
                for j in range(i+1, r):
                    Yf[i, j] = (S[i] * C[j, i].conjugate() + S[j] * C[i, j])\
                               / (S[i] * S[i] + S[j] * S[j])
            Yf = Yf + Yf.conj().T - np.diag(np.diag(Yf.real))
        
        elif method == "skewsymmetric":
            Yf = np.zeros((r, r), dtype=np.complex128)
            for i in range(r):
                Yf[i, i] = (1.0j * C[i, i].imag) / S[i]
                for j in range(i+1, r):
                    Yf[i, j] = (-S[i] * C[j, i].conjugate() +  S[j] * C[i, j])\
                               / (S[i] * S[i] + S[j] * S[j])
            Yf = Yf - Yf.conj().T - 1.0j * np.diag(np.diag(Yf.real))
            
        A = lambda v: Ux @ Yf @ (Ux.conj().T @ v)
        eVals, eVecs = LA.eig(Yf)
        eVecs = Ux @ eVecs
        
    elif method == "toeplitz" or method == "hankel":
        if method == "toeplitz":
            J = np.eye(nx)
        elif method == "hankel":
            J = np.fliplr(np.eye(nx))
            
        # define the left matrix
        Am = fft(
            np.hstack([np.eye(nx), np.zeros((nx, nx))]).T,
            axis=0
        ).conj().T / np.sqrt(2 * nx)
        
        # define the right matrix
        B = fft(
            np.hstack([(J @ X).conj().T, np.zeros((nt, nx))]).T,
            axis=0
        ).conj().T / np.sqrt(2 * nx)
        
        BtB = B.conj().T @ B
        # fast computation of A @ A.conj().T
        AAt = ifft(
            fft(
                np.vstack(
                    [np.hstack([np.eye(nx), np.zeros((nx, nx))]),
                     np.zeros((nx, 2 * nx))]
                ),
                axis=0
            ).T,
            axis=0
        ).T
        
        # construct the RHS of the linear system
        y = np.diag(Am.conj().T @ Y.conj() @ B).conj().T
        # construct the matrix for the linear system
        L = (AAt * BtB.T).conj().T
        # solve the linear system
        d = np.append(y[:-1] @ LA.inv(L[:-1, :-1]), 0)
        # convert the eigenvalues into the circulant matrix
        newA = ifft(fft(np.diag(d), axis=0).T, axis=0).T
        # extract the Toeplitz matrix from the circulant matrix
        A = newA[:nx, :nx] @ J
        
        # this part is missing in MatLab implementation
        eVals, eVecs = LA.eig(A)
        
    elif method.startswith("circulant"):
        fX = fft(X, axis=0)
        fY = fft(Y.conj(), axis=0)
        
        # solve in the total least squares sense
        if method.endswith("TLS"):
            d = np.zeros(nx, dtype=np.complex128)
            for j in range(nx):
                tls_res = tls(fX[j, :].conj().T, fY[j, :].conj().T)
                d[j] = tls_res[0, 0]
                
        # solve the other cases
        elif not method.endswith("TLS"):
            fX_col_norm2_sq = LA.norm(fX, 2, 1)**2
            d = np.diag(fX @ fY.conj().T) / fX_col_norm2_sq
            
            if method.endswith("unitary"):
                d = np.exp(1.0j * np.angle(d))
                
            elif method.endswith("skewsymmetric"):
                d = 1.0j * d.imag
                
            elif method.endswith("symmetric"):
                d = d.real
                
        # these are the eigenvalues
        eVals = d
        # these are the eigenvectors
        eVecs = fft(np.eye(nx), axis=0)
        
        # rank constraint
        if rank is not None:
            # identify least important eigenvalues
            res = np.diag(np.abs(fX @ fY.conj().T)) / LA.norm(fX.conj().T, axis=0)
            # remove least important eigenvalues
            idx = np.argpartition(res, kth=nx-rank)[:nx-rank]
            d[idx] = 0
            eVals[idx] = 0
            eVecs[:, idx] = 0
        
        # reconstruct the operator in terms of FFTs
        A = lambda v: fft((d * ifft(v, axis=0)).T, axis=0)
        
        # for tests
        # eVals_, eVecs_ = LA.eig(A(np.eye(nx)))
        
    # BCCB: Block Circulant - Circulant Block
    elif method == "BCCB" or method == "BCCBtls" or method == "BCCBskewsymmetric"\
         or method == "BCCBsymmetric" or method == "BCCBunitary":
        if block_sizes is None:
            raise ValueError("Need to specify size of blocks.")
        # np.prod(block_sizes) = X.size / X.shape[1];
        # equals number of unique values in block before
        # dimensionality reduction by "rank" parameter
        # (if specified);
        # in case of symm/skew-symm/unitary the number
        # of unique values changes in order to satisfy
        # constraints;
        # block_sizes define matrix structure as follows:
        # the 1st parameter defines the size of a block and
        # the 2nd one defines the number of blocks in a matrix
        # row;
        # values are circulant within each block and blocks
        # themselves are circulant within the output matrix
        p = np.prod(block_sizes)
        
        # equivalent to applying the block-DFT matrix F
        # defined by
        # F = np.kron(fft(np.eye(M)), fft(np.eye(N)))
        # to the matrix X;
        # where fft(np.eye(M)) is a dft matrix of size M
        aF = lambda x: np.reshape(
                fft2(
                    np.reshape(x, np.append(block_sizes, x.shape[1]), order="F").T
                ).T,
                np.array([p, x.shape[1]]),
                order="F"
            ) / np.sqrt(p)
        
        aFt = lambda x: aF(x.conj()).conj()
        
        fX = aF(X.conj())
        fY = aF(Y.conj())
        d = np.zeros((p, 1), dtype=np.complex128)
        
        if method == "BCCB":
            for j in range(p):
                denom = LA.norm(fX[j, :].conj().T)**2
                d[j] = (fX[j, :] @ fY[j, :].conj().T).conj() / denom
                
        elif method == "BCCBtls":
            for j in range(p):
                d[j] = tls(fX[j, :].conj().T, fY[j, :].conj().T).conj().T
                
        elif method == "BCCBskewsymmetric":
            for j in range(p):
                fXj_inv = LA.pinv(np.reshape(fX[j, :], (-1, 1))).ravel()
                d[j] = 1.0j * (fY[j, :] @ fXj_inv).imag
                
        elif method == "BCCBsymmetric":
            for j in range(p):
                fXj_inv = LA.pinv(np.reshape(fX[j, :], (-1, 1))).ravel()
                d[j] = (fY[j, :] @ fXj_inv).real
        
        elif method == "BCCBunitary":
            for j in range(p):
                fXj_inv = LA.pinv(np.reshape(fX[j, :], (-1, 1))).ravel()
                d[j] = np.exp(1.0j * np.angle(fY[j, :] @ fXj_inv))
                
        # rank reduces number of unique values from which
        # matrix will be constructed from np.prod(block_sizes)
        # to rank
        # (not always interpretable like this in case
        # value of rank is close to np.prod(block_sizes));
        # but mainly it defines the number of zeroed eVals
        # as nx-rank, so the number of non-zero eVals is
        # equal to rank;
        # see below for details of how importance
        # of eVals is defined
        if rank is not None:
            res = np.diag(np.abs(fX @ fY.conj().T)) / LA.norm(fX.conj().T, axis=0)
            idx = np.argpartition(res, kth=nx-rank)[:nx-rank]
            d[idx] = 0
            
        # returns a function handle that applies A
        A = lambda x: aF(d.conj() * aFt(x))
        # eigenvalues are given by d
        eVals = d
        # use "_, eVecs = LA.eig(A(np.eye(nx)))" if needed; (nx = rank if needed)
        eVecs = None
        
    elif method == "BC" or method == "BCtri" or method == "BCtls":
        # blocks are circulant within matrix but values in blocks are not
        # for info about block_sizes check "BCCB" code section
        if block_sizes is None:
            raise ValueError("Need to specify size of blocks.")
        p = np.prod(block_sizes)
        N, M = block_sizes[0], block_sizes[1]
        
        # equivalent to applying the block-DFT matrix F
        # defined by
        # F = np.kron(fft(np.eye(M)), np.eye(N))
        # to the matrix X;
        # where fft(np.eye(M)) is a dft matrix of size M
        aF = lambda x: np.reshape(
                fft(
                    np.reshape(x, np.append(block_sizes, x.shape[1]), order="F"), axis=1
                ),
                np.array([p, x.shape[1]]),
                order="F"
            ) / np.sqrt(M)
            
        aFt = lambda x: aF(x.conj()).conj()
        
        fX = aF(X)
        fY = aF(Y)
        d = [0]*M
        
        for j in range(M):
            ls = j * N + [idx for idx in range(N)]
            if method == "BC":
                d[j] = fY[ls, :] @ LA.pinv(fX[ls, :])
                
            # doesn't work in MatLab because d[j] is a function
            elif method == "BCtri":
                d[j], _, _ = piDMD(
                    fX[ls, :], fY[ls, :],
                    "diagonalpinv", diag=2*np.ones((fX[ls, :].shape[0], 2)), complex=True
                )
                d[j] = d[j](np.eye(fX[ls, :].shape[0]))
                
            elif method == "BCtls":
                d[j] = tls(fX[ls, :].conj().T, fY[ls, :].conj().T).conj().T
                
        BD = block_diag(*d)
        A = lambda v: aFt(BD @ aF(v))
        
        # this part is missing in MatLab implementation
        eVals, eVecs = LA.eig(A(np.eye(nx)))
        
    elif method == "symtridiagonal":
        # compute the entries of the first block T1e
        X_col_norm2 = LA.norm(X, 2, 1)
        # element-wise multiplication
        T1e = X_col_norm2 * X_col_norm2
        # form the leading block
        T1 = spdiags(T1e, 0, nx, nx)
        # compute the entries of the second block
        T2e = np.diag(X[1:, :] @ X[:-1, :].conj().T) # probably .conj() is not needed
        # form the second and third blocks
        T2 = spdiags(np.vstack((T2e, T2e)), (-1, 0), nx, nx-1)
        # compute the entries of the final block
        T3e = np.insert(
            np.diag(X[2:, :] @ X[:-2, :].conj().T), # probably .conj() is not needed
            0, 0
        )
        # form the final block
        T3 = spdiags(T1e[:-1] + T1e[1:], 0, nx-1, nx-1)\
             + spdiags(T3e, 1, nx-1, nx-1)\
             + spdiags(T3e, 1, nx-1, nx-1).conj().T
        # form the block tridiagonal matrix
        T = vstack((
            hstack((T1, T2)),
            hstack((T2.conj().T, T3))
        ))
        # compute the RHS vector
        # (probably .conj() is not needed)
        d_up = np.diag(X @ Y.conj().T)
        d_down = np.diag(X[:-1, :] @ Y[1:, :].conj().T)\
                 + np.diag(X[1:, :] @ Y[:-1, :].conj().T)
        d = np.hstack((d_up, d_down)) # [:, np.newaxis] to make column from row
        # take real parts then solve linear system
        c = LA.inv(T.toarray().real) @ d.real
        # form the solution matrix
        A = spdiags(c[:nx], 0, nx, nx)\
            + spdiags(
                np.insert(c[nx:], 0, 0),
                1, nx, nx
            )\
            + spdiags(
                np.append(c[nx:], 0),
                -1, nx, nx
            )
        
        # this part is missing in MatLab implementation
        eVals, eVecs = LA.eig(A.toarray())
    
    else:
        A, eVals, eVecs = None, None, None
        raise NotImplementedError(f"The selected method: {method} doesn't exist!")
    
    return A, eVals, eVecs