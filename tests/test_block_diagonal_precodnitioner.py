from interfaces import *
from utilities import *
import numpy as np

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
    Build the block diagonal preconditioner and check
    its action when applied onto a vector

    """
    for nt in np.arange(10,11,20):
        for npix in np.arange(5,nt,10):
            x0=np.random.random(npix)
            pairs=pairs_gen(nt,npix)
            #construct the block diagonal operator
            nb=2
            t,diag=noise_val(nb)
            blocksize=nt/nb
            N=BlockLO(blocksize,diag,offdiag=False)
            P=SparseLO(npix,nt,pairs)
            A=P.T*N*P
            b=A*x0
            print b

            M_bd=InverseLO(A,method=spla.cg)
            vec=M_bd*b
            #vec,info=spla.cg(A,b, M=M_bd)

            #assert checking_output(M_bd.converged)
            print vec,x0
            assert  np.allclose(vec,x0)



test_block_diagonal_preconditioner()
test_block_diagonal_preconditioner_pol()
test_block_diag_precond_action()
