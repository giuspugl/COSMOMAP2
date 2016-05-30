import scipy.linalg as la
from interfaces import *
from utilities import *
import numpy as np
import time
import matplotlib.pyplot as plt

def test_block_diagonal_operator():
    """
    test the :class:`BlockDiagonalLO`.
    """
    runcase={'I':1,'QU':2,'IQU':3}
    a=[64,128,256]
    nb=1
    for nt in [2**14,2**15]:
        for pol in runcase.values():
            blocksize=nt/nb
            for i in a:
                npix=int(i)
                d,pairs,phi,t,diag=system_setup(nt,npix,nb)
                processd =ProcessTimeSamples(pairs,npix,pol=pol ,phi=phi)
                npix=processd.get_new_pixel[0]
                P=SparseLO(npix,nt,pairs,pol=pol,angle_processed=processd)
                x=np.ones(pol*npix)
                Mbd=BlockDiagonalPreconditionerLO(processd,npix,pol=pol)
                invMbd=BlockDiagonalLO(processd,npix,pol=pol)
                #invMbd and P.T*P  operate in the same on to a pixel vector
                y=invMbd*x
                y2=P.T*P*x

                assert np.allclose(y,y2)

                # invMbd*Mbd = Identity

                v=Mbd*invMbd*x
                assert np.allclose(v,x)


def test_SPD_properties_block_diagonal_preconditioner():
    """
    to converge there has to be Symmetric Positive Definite.
    This test the BlockDiagonalPreconditioner Linear operator implemented as
    the explicit inverse of the matrix ``P.T*N*P``.
    """
    nb=6
    blocksize=2*[500,400,124]
    nt=sum(blocksize)
    npix=64
    runcase={'I':1,'QU':2,'IQU':3}
    for pol in runcase.values():
        d,pairs,phi,t,diag=system_setup(nt,npix,nb)
        N=BlockLO(blocksize,diag,offdiag=False)
        processd =ProcessTimeSamples(pairs,npix,pol=pol ,phi=phi,w=N.diag)
        npix=processd.get_new_pixel[0]
        P=SparseLO(npix,nt,pairs,pol=pol,angle_processed=processd)
        #construct the block diagonal operator
        randarray=np.random.rand(pol*npix)
        A=P.T*N*P
        assert  np.allclose(A*randarray, A.T *randarray)
        assert scalprod(randarray,A*randarray)>0.

        Mbd=BlockDiagonalPreconditionerLO(processd,npix,pol)
        assert  np.allclose(Mbd*randarray, Mbd.T *randarray)
        assert scalprod(randarray,Mbd*randarray)>0.


filter_warnings("ignore")
