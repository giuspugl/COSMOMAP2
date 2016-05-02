import numpy as np
from interfaces import *
from utilities import *

def test_SPD_properties_block_diagonal_preconditioner():
    """
    to converge the  conjugate gradient the preconditioner has to be
    Symmetric Positive Definite.
    This test the BlockDiagonalPreconditioner Linear operator implemented as
    the explicit inverse of the matrix ``P.T*N*P``.
    """
    # input  from file, nt and np fixed
    nb=6
    blocksize=2*[500,400,124]
    nt=sum(blocksize)
    npix=64
    runcase={'I':1,'QU':2,'IQU':3}
    for pol in runcase.values():
        d,pairs,phi,t,diag=system_setup(nt,npix,nb)
        #construct the block diagonal operator
        N=BlockLO(blocksize,diag,offdiag=False)
        P=SparseLO(npix,nt,pairs,phi,pol=pol,w=N.diag)
        npix=P.ncols
        randarray=np.random.rand(pol*npix)
        A=P.T*N*P
        assert  np.allclose(A*randarray, A.T *randarray)
        assert scalprod(randarray,A*randarray)>0.

        Mbd=BlockDiagonalPreconditionerLO(P,npix,pol)
        assert  np.allclose(Mbd*randarray, Mbd.T *randarray)
        assert scalprod(randarray,Mbd*randarray)>0.

filter_warnings("ignore")

test_SPD_properties_block_diagonal_preconditioner()
