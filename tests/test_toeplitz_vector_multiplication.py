from interfaces import *
from utilities import *
import numpy as np

filter_warnings("ignore")
def test_toeplitz_vector_product():
    """
    test matrix vector multiplication of A^T N^{-1} A
    """
    nb=6
    blocksize=2*[500,400,124]
    nt=sum(blocksize)
    print blocksize
    npix=64
    runcase={'I':1,'QU':2,'IQU':3}
    for pol in runcase.values():
        d,pairs,phi,t,diag=system_setup(nt,npix,nb)
        N=BlockLO(blocksize,diag,offdiag=False )
        P=SparseLO(npix,nt,pairs,phi=phi,pol=pol, w=N.diag)
        x=np.ones(pol*npix)

        y=P*x
        w=N*y
        z=P.T*w
        z2=P.T*N*P*x
        assert np.allclose(z2,z)


def test_toeplitzband_vector_product():
    """
    test matrix vector multiplication of A^T N^{-1} A
    with N containing offdiagonal terms
    """
    nb=6
    blocksize=2*[500,400,124]
    nt=sum(blocksize)
    print blocksize
    npix=64
    runcase={'I':1}
    runcase={'I':1,'QU':2,'IQU':3}
    for pol in runcase.values():
        d,pairs,phi,t,diag=system_setup(nt,npix,nb)

        N=BlockLO(blocksize,diag,offdiag=True)
        P=SparseLO(npix,nt,pairs,phi=phi,pol=pol,w=N.diag)
        x=np.ones(pol*npix)

        y=P*x
        w=N*y
        z=P.T*w
        z2=P.T*N*P*x
        assert np.allclose(z2,z)

def test_different_block_size():
    """
    test noise matrix initialization with different block sizes
    """
    nb=6
    blocksize=2*[500,400,124]
    nt=sum(blocksize)
    print blocksize
    npix=64
    runcase={'I':1,'QU':2,'IQU':3}
    for pol in runcase.values():
        d,pairs,phi,t,diag=system_setup(nt,npix,nb)
        N=BlockLO(blocksize,diag)
        P=SparseLO(npix,nt,pairs,phi=phi,pol=pol, w=N.diag)
        x=np.ones(pol*npix)
        PtNP=BlockDiagonalLO(P,npix,pol=pol)
        z=PtNP*x
        z2=P.T*N*P*x
        assert np.allclose(z2,z)

#test_different_block_size()
#test_toeplitz_vector_product()
#test_toeplitzband_vector_product()
