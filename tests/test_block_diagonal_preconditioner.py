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
    pol=3
    runcase={'I':1,'QU':2,'IQU':3}
    for pol in runcase.values():
        d,weight,phi,pairs,hp_pixs,ground,ces_size=read_from_data('data/20120718_093931.hdf5',pol=pol,npairs=4)
        nt,npix,nb=len(d),len(hp_pixs),len(weight)
        #construct the block diagonal operator
        blocksize=nt/nb
        N=BlockLO(blocksize,weight,offdiag=False)
        P=SparseLO(npix,nt,pairs,phi,pol=pol,w=N.diag)
        randarray=np.random.rand(pol*npix)
        print P.T*d
        A=P.T*N*P
        assert  np.allclose(A*randarray, A.T *randarray)
        assert scalprod(randarray,A*randarray)>0.

        Mbd=BlockDiagonalPreconditionerLO(P,npix,pol)
        assert  np.allclose(Mbd*randarray, Mbd.T *randarray)
        assert scalprod(randarray,Mbd*randarray)>0.

filter_warnings("ignore")

#test_SPD_properties_block_diagonal_preconditioner()
