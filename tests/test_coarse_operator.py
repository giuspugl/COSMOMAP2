import scipy.linalg as la
from interfaces import *
from utilities import *
import numpy as np

def test_coarse_operator():
    """
    Build and test the :class:`CoarseLO`.
    """
    filter_warnings("ignore")

    nt,npix,nb= 400,20,2
    blocksize=nt/nb
    d,pairs,phi,t,diag=system_setup(nt,npix,nb)
    for pol in [1,2,3]:

        x0=np.zeros(pol*npix)
        P=SparseLO(npix,nt,pairs,phi,pol,w=diag)
        N=BlockLO(blocksize,t,offdiag=True)
        Mbd=BlockDiagonalPreconditionerLO(P,npix,pol=pol)
        #M=InverseLO(P.T*diagN*P,method=spla.cg)
        #print nb,nt,npix
        b=P.T*N*d
        A=P.T*N*P

        tol=1.e-5
        """
        # Build deflation supspace
        h=[]
        w=[]
        w,h=arnoldi(M*A,b,x0=x0,tol=tol,maxiter=1,inner_m=pol*npix)
        m=len(w)
        H=build_hess(h,m)

        z,y=la.eig(H,check_finite=False)
        total_energy=np.sqrt(sum(abs(z)**2))
        eps= .2 * total_energy

    #       eps=.1*abs(max(z))
        Z,r= build_Z(z,y, w, eps)
        """

        eigv,Z=spla.eigs(Mbd*A,k=3,which='SR',ncv=8,tol=tol)
        #print eigv,Z
        r=Z.shape[1]
        Zd=DeflationLO(Z)
        # Build Coarse operator
        Az=Z*0.
        for i in xrange(r):
            Az[:,i]=A*Z[:,i]
        invE=CoarseLO(Z,Az,r)
        E=dgemm(Z,Az.T)

        v=np.ones(r)
        y=invE*v
        v2=np.dot(E,invE*v)
        y2=la.solve(E,v)
        #print y,"\n",y2
        if (y2.dtype=='float64' ):
            #assert np.allclose(v,v2)
            assert  np.allclose(y2,y)
        elif (y2.dtype=='complex128' ):
            print "complex",y2.real,y.real,y2.imag,y.imag
            #assert  np.allclose(y2,y)
            assert np.allclose(y2.real,y.real) #and np.allclose(y2.imag,y.imag)
            #assert np.allclose(v,v2)

test_coarse_operator()
