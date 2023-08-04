# Matrix generators for some of the tests
# (BCCB, BC, Toeplitz, Hankel and circulant families)

import numpy as np
from scipy.linalg import circulant, toeplitz, hankel


# generates Block-Circulant-Circulant-Block (+symm/skew-symm) for simple tests
def gen_BCCB(method=None):
    """
    Very simple BCCBs (20x20 and 4 blocks 5x5).
    Can generate BCCB, BCCB skew-symm and BCCB symm.
    Can not generate BCCBunitary.
    """
    blocksA = []
    for i in range(4):
        rowA_i = np.random.randn(5)
        blocksA.append(circulant(rowA_i))
        
    trueA = np.zeros((20, 20))
    
    trueA[:5, :5] = blocksA[0]
    trueA[:5, 5:10] = blocksA[1]
    trueA[:5, 10:15] = blocksA[2]
    trueA[:5, 15:20] = blocksA[3]
    
    trueA[5:10, :5] = blocksA[3]
    trueA[5:10, 5:10] = blocksA[0]
    trueA[5:10, 10:15] = blocksA[1]
    trueA[5:10, 15:20] = blocksA[2]
    
    trueA[10:15, :5] = blocksA[2]
    trueA[10:15, 5:10] = blocksA[3]
    trueA[10:15, 10:15] = blocksA[0]
    trueA[10:15, 15:20] = blocksA[1]
    
    trueA[15:20, :5] = blocksA[1]
    trueA[15:20, 5:10] = blocksA[2]
    trueA[15:20, 10:15] = blocksA[3]
    trueA[15:20, 15:20] = blocksA[0]
    
    if method == "skewsymmetric":
        trueA = (trueA - trueA.T) / 2
    elif method == "symmetric":
        trueA = (trueA + trueA.T) / 2
    
    return trueA

# generates Block-Circulant (+tridiagonal) for simple tests
def gen_BC(method="simple"):
    """
    Very simple BCs (18x18 and 3 blocks 6x6).
    Can generate BC and BC tridiagonal.
    """
    blocksA = []
    for i in range(3):
        if method == "simple":
            blocksA.append(np.random.randn(6, 6))
        elif method == "tridiagonal":
            diag_1 = np.random.randn(5)
            diag_2 = np.random.randn(6)
            diag_3 = np.random.randn(5)
            blockA = np.diag(diag_1, k=-1) + np.diag(diag_2, k=0) + np.diag(diag_3, k=1)
            blocksA.append(blockA)
    
    trueA = np.zeros((18, 18))
    
    trueA[:6, :6] = blocksA[0]
    trueA[:6, 6:12] = blocksA[1]
    trueA[:6, 12:18] = blocksA[2]
    
    trueA[6:12, :6] = blocksA[2]
    trueA[6:12, 6:12] = blocksA[0]
    trueA[6:12, 12:18] = blocksA[1]
    
    trueA[12:18, :6] = blocksA[1]
    trueA[12:18, 6:12] = blocksA[2]
    trueA[12:18, 12:18] = blocksA[0]
    
    return trueA
    
def gen_toeplitz(n):
    """
    Returns n-by-n random Toeplitz matrix.
    """
    return toeplitz(
        c=np.random.randn(n),
        r=np.random.randn(n),
    )

def gen_hankel(n):
    """
    Returns n-by-n random Hankel matrix.
    """
    return hankel(
        c=np.random.randn(n),
        r=np.random.randn(n),
    )
    
def gen_circulant(n, method="simple"):
    """
    Returns random circulant n-by-n matrix.
    Can generate circulant symmetric and
    circulant skew-symmetric matrices.
    Can not generate unitary circulant
    matrices.
    """
    matrix_row = np.random.randn(n)
    mtx = circulant(matrix_row)
    if method == "simple":
        res = mtx
    elif method == "symmetric":
        res = (mtx + mtx.T) / 2
    elif method == "skewsymmetric":
        res = (mtx - mtx.T) / 2
        
    return res