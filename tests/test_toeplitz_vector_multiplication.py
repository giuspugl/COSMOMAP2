from interfaces import *
from utilities import *
import numpy as np

def test_toeplitz_vector_product():
    """
    test matrix vector multiplication of A^T N^{-1} A
    """
    for nt in np.arange(20,100,20):
        for npix in np.arange(5,nt,10):
            x=np.arange(npix)

            pairs=pairs_gen(nt,npix)

            P=SparseLO(npix,nt,pairs)
            #construct the block diagonal operator
            for nb in np.arange(2,6,2):
                t=np.random.random(nb)

                blocksize=nt/nb
                N=BlockLO(blocksize,t,offdiag=False )

                y=P*x
                w=N*y
                z=P.T*w

                z2=P.T*N*P*x

                assert np.allclose(z2,z)

def test_toeplitz_vector_product_pol():
    """
    test matrix vector multiplication of A^T N^{-1} A
    POLARIZATION CASE
    """
    for nt in np.arange(20,100,20):
        for npix in np.arange(5,nt,10):

            x=np.arange(3*npix)

            pairs=pairs_gen(nt,npix)
            phi=angles_gen(rd.uniform(0,np.pi),nt)
            P=SparseLO(npix,nt,pairs,phi,pol=3)
            #construct the block diagonal operator
            for nb in np.arange(2,6,2):
                t=np.random.random(nb)

                blocksize=nt/nb
                N=BlockLO(blocksize,t,offdiag=False )

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
    for nt in np.arange(20,100,20):
        for npix in np.arange(5,nt,10):
            x=np.arange(npix)

            pairs=pairs_gen(nt,npix)

            P=SparseLO(npix,nt,pairs)
            #construct the block diagonal operator
            for nb in np.arange(2,6,2):
                bandsize=nb
                t, diag=noise_val(nb,bandsize)

                blocksize=nt/nb
                N=BlockLO(blocksize,t,offdiag=True)

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
#test_toeplitz_vector_product_pol()
#test_toeplitzband_vector_product()
