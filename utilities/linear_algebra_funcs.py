from scipy.linalg import get_blas_funcs
import numpy as np

def dgemm(A,B):
    """
    Compute Matrix-Matrix multiplication from the BLAS routine DGEMM
    """
    #print len(A),len(B)
    if type(A)==list :
        A=np.asarray(A,order='F')
    if type(B)==list:
        B=np.asarray(B,order='F')

    matdot=get_blas_funcs('gemm',dtype=A.dtype)

    return matdot(alpha=1.0, a=A.T, b=B, trans_b=True,trans_a=False)

def norm2(q):
    """
    Compute the euclidean norm of an array ``q`` by calling the BLAS routine
    """
    q = np.asarray(q)
    nrm2 = get_blas_funcs('nrm2', dtype=q.dtype)
    return nrm2(q)

def scalprod(a,b):
    dot=get_blas_funcs('dot', (a,b))
    return dot(a,b)
