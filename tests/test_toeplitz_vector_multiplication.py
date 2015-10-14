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

            pairs=pairs_gen(nt,npix,pol=3)
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


test_toeplitz_vector_product()
test_toeplitz_vector_product_pol()
test_toeplitzband_vector_product()
