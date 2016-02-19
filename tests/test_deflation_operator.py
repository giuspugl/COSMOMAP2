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

    nt,npix,nb=400,200,2
    blocksize=nt/nb
    d,pairs,phi,t,diag=system_setup(nt,npix,nb)
    for pol in [1,2,3]:
        x0=np.zeros(pol*npix)
        N=BlockLO(blocksize,t,offdiag=True)
        #P=SparseLO(npix,nt,pairs,phi,pol)
        #diagN=BlockLO(blocksize,diag,offdiag=False)
        #N=BlockLO(nt/nb,diag)
        P=SparseLO(npix,nt,pairs,phi,pol=pol,w=diag )
        #M=InverseLO(P.T*diagN*P,method=spla.cg)
        Mbd=BlockDiagonalPreconditionerLO(P,npix,pol=pol)

        #print nb,nt,npix
        b=P.T*N*d
        A=P.T*N*P
        # Build deflation supspace
        h=[]
        w=[]
        for i in range(4,7):
            tol=10**(-i)
            """
            w,h=arnoldi(M*A,b,x0=x0,tol=tol,maxiter=1,inner_m=pol*npix)
            m=len(w)
            H=build_hess(h,m)
            z,y=la.eig(H,check_finite=False)
            #plot_histogram_eigenvalues(z)
            total_energy=np.sqrt(sum(abs(z)**2))
            #eps= .2 * total_energy
            eps=.01*abs(max(z))
            Z,r= build_Z(z,y, w, eps)
            """

            eigv,Z=spla.eigs(Mbd*A,k=10,which='SM',ncv=22,tol=tol)
            #print Z.shape
            r=Z.shape[1]
            rank= np.linalg.matrix_rank(Z)
            #print rank
            #rank(Z) must be equal to the size of the deflation subspace
            #assert rank==r
            print rank==r

            prod=dgemm(Z,Z.T)

            determ=la.det(prod)
            determ=la.det(prod)
            # the selected eigenvectors (columns of Z) has to
            # be linearly independent <=> determ(Z.T*Z)!=0 being a r x r matrix
            assert determ!=0
            v=np.ones(r)
            Zd=DeflationLO(Z)

            y2=Z.dot(v)
            y= Zd*v
            #print y2.dtype,y.dtype
            x=np.ones(pol*npix)
            Z=np.matrix(Z)
            s2=Z.H.dot(x)
            s=Zd.H*x
            #print s2,s

            if (Z.dtype=='float64' ):
                assert np.allclose(y2,y)
                assert np.allclose(s2,s)
            elif (Z.dtype=='complex128' ):
                assert np.allclose(y2.real,y.real)
                        #and np.allclose(y2.imag,y.imag)
                assert np.allclose(s2.real,s.real)
                        #and np.allclose(s2.imag,s.imag)


test_deflation_operator()
