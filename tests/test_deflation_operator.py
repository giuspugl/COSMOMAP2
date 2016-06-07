import scipy.linalg as la
from interfaces import *
from utilities import *
import numpy as np

def test_deflation_operator():
    """
    Build and test the deflation subspace matrix Z checking
    whether its columns are linearly independent <=> determ(Z.T*Z)!=0
    being Z.T*Z a r x r matrix, r =rank(Z)
    """

    nt,nb=1000,2
    blocksize=nt/nb
    runcase={'I':1,'QU':2,'IQU':3}
    for pol in runcase.values():
        npix=20
        d,pairs,phi,t,diag=system_setup(nt,npix,nb)
        N=BlockLO(blocksize,t,offdiag=True)
        processd =  ProcessTimeSamples(pairs,npix,pol=pol ,phi=phi)
        npix=   processd.get_new_pixel[0]
        P   =   SparseLO(npix,nt,pairs,pol=pol,angle_processed=processd)
        x0  =   np.zeros(pol*npix)
        Mbd =   BlockDiagonalPreconditionerLO(processd ,npix,pol=pol)
        b=P.T*N*d
        A=P.T*N*P
        # Build deflation supspace
        tol=1.e-4
        B=BlockDiagonalLO(processd,npix,pol=pol)
        eigv ,Z=spla.eigsh(A,M=B,Minv=Mbd,k=3,which='SM',ncv=15,maxiter=40,tol=tol)
        r=Z.shape[1]
        rank= np.linalg.matrix_rank(Z)
        #rank(Z) must be equal to the size of the deflation subspace
        assert rank==r
        prod=dgemm(Z,Z.T)
        determ=la.det(prod)
        determ=la.det(prod)
        # the selected eigenvectors (columns of Z) has to  be linearly
        # independent <=> determ(Z.T*Z)!=0 being a r x r matrix
        assert determ!=0
        v=np.ones(r)
        Zd=DeflationLO(Z)
        y2=Z.dot(v)
        y= Zd*v
        x=np.ones(pol*npix)
        Z=np.matrix(Z)
        s2=Z.H.dot(x)
        s=Zd.H*x

        assert np.allclose(y2,y) and np.allclose(s2,s)
filter_warnings("ignore")
test_deflation_operator()
