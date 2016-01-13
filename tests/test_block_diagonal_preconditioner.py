import numpy as np
from interfaces import *
from utilities import *

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

def test_block_diagonal_precond_action_as_inverse_operator():
    """
    check the action of the block diag preconditioner

    Given an initial guess vector ``x0`` we compute ``b= A x0``,
    then

    ``x1= CG(A, b, Precond=M_bd).``
    we check whether ``x0 = x1``.

    The input comes from an hdf5 file in data/ directory.

    """
    # input  from file, nt and np fixed
    npix,pol=15,3
    d,pairs,phi,weight=read_from_hdf5('data/testcase_block_diag_4.hdf5')
    nt,nb=len(d),len(weight)


    #construct the block diagonal operator
    x0=np.ones(pol*npix)
    blocksize=nt/nb
    N=BlockLO(blocksize,weight,offdiag=False)
    P=SparseLO(npix,nt,pairs,phi,pol=pol)

    A=P.T*N*P
    b=A*x0
    M_bd=InverseLO(A,method=spla.cg)
    vec=M_bd*b
    assert checking_output(M_bd.converged)
    #print vec,x0
    assert  np.allclose(vec,x0,atol=1.e-3)

def test_SPD_properties_block_diagonal_preconditioner():
    """
    to converge the  conjugate gradient the preconditioner has to be
    Symmetric Positive Definite.
    """
    # input  from file, nt and np fixed
    npix,pol=15,3
    d,pairs,phi,weight=read_from_hdf5('data/testcase_block_diag_4.hdf5')
    nt,nb=len(d),len(weight)

    #construct the block diagonal operator

    blocksize=nt/nb
    N=BlockLO(blocksize,weight,offdiag=False)
    P=SparseLO(npix,nt,pairs,phi,pol=pol,w=N.diag)
    randarray=np.random.rand(pol*npix)
    A=P.T*N*P
    assert  np.allclose(A*randarray, A.T *randarray)
    assert scalprod(randarray,A*randarray)>0.

    Mbd=BlockDiagonalPreconditionerLO(P.counts,P.mask,npix,pol,P.sin2,P.cos2,P.sincos)
    assert  np.allclose(Mbd*randarray, Mbd.T *randarray)
    assert scalprod(randarray,Mbd*randarray)>0.



test_block_diagonal_preconditioner()
test_block_diagonal_preconditioner_pol()
test_block_diagonal_precond_action_as_inverse_operator()
test_SPD_properties_block_diagonal_preconditioner()
