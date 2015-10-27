import numpy as np
from interfaces import *
from utilities import *
#from utilities.IOfiles import read_from_hdf5
#from utilities.linear_algebra_funcs import *
#from utilities.utilities_functions import *

def test_block_diagonal_preconditioner():
    """
    Build the block diagonal preconditioner and check
    its action when applied onto a vector

    """
    for nt in np.arange(20,100,20):
        d=np.arange(nt)
        for npix in np.arange(5,nt,10):
            x0=np.ones(npix)
            pairs=pairs_gen(nt,npix)
            P=SparseLO(npix,nt,pairs)
            #construct the block diagonal operator
            for nb in np.arange(2,6,2):

                t,diag=noise_val(nb)
                blocksize=nt/nb
                N=BlockLO(blocksize,diag,offdiag=False)

                #construct the block diagonal operator

                b=P.T*N*d

                M_bd=InverseLO(P.T*N*P,method=spla.cg)

                vec=M_bd*b
                assert checking_output(M_bd.converged)
                y,info=spla.cg(P.T*N*P,b)
                assert checking_output(info)
                assert np.allclose(vec,y)



def test_block_diagonal_preconditioner_pol():
    """
    Build the block diagonal preconditioner and check
    its action when applied onto a POLARIZATION vector

    """
    for nt in np.arange(20,100,20):
        d=np.arange(nt)
        for npix in np.arange(5,nt,10):
            x0=np.ones(3*npix)
            pairs=pairs_gen(nt,npix,pol=3)
            phi=angles_gen(rd.uniform(0,np.pi),nt)
            P=SparseLO(npix,nt,pairs,phi,pol=3)

            #construct the block diagonal operator
            for nb in np.arange(2,6,2):

                t,diag=noise_val(nb,bandwidth=1)
                blocksize=nt/nb
                N=BlockLO(blocksize,diag,offdiag=False)
                #construct the block diagonal operator
                b=P.T*N*d

                M_bd=InverseLO(P.T*N*P,method=spla.cg)

                vec=M_bd*b
                assert checking_output(M_bd.converged)
                y,info=spla.cg(P.T*N*P,b)
                assert checking_output(info)
                assert np.allclose(vec,y)

def test_block_diag_precond_action():
    """
    check the action of the block diag preconditioner

    Given an initial guess vector ``x0`` we compute ``b= A x0``,
    then

    ``x1= CG(A, b, Precond=M_bd).``
    we check whether ``x0 = x1``.

    The input come from an hdf5 file in data/ directory.

    """
    # input  from file, nt and np fixed
    nt,npix,nb,pol=100,15,2,3
    pairs,phi,t,diag,d=read_from_hdf5('data/testcase_block_diag_3.hdf5')

    #construct the block diagonal operator
    x0=np.ones(pol*npix)
    blocksize=nt/nb
    N=BlockLO(blocksize,diag,offdiag=False)
    P=SparseLO(npix,nt,pairs,phi,pol=pol)

    A=P.T*N*P
    b=A*x0
    M_bd=InverseLO(A,method=spla.cg)
    vec=M_bd*b
    assert checking_output(M_bd.converged)
    print vec,x0
    assert  np.allclose(vec,x0,atol=1.e-4)



test_block_diagonal_preconditioner()
test_block_diagonal_preconditioner_pol()
test_block_diag_precond_action()
