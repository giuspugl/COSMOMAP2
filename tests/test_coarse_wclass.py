import nose
from nose.tools import with_setup
import unittest
from interfaces import *
from utilities import *
import scipy.linalg as la

filter_warnings('ignore')

class TestCoarseOperator_I():
    @classmethod
    def setup_class(self):
        self.nt,self.npix,self.nb= 400,50,2
        self.data=system_setup(self.nt,self.npix,self.nb)
    @classmethod
    def teardown_class(self):
        self.data=None
        #print(__name__, ': TestClass.teardown_class() -------')

    def setup(self):
        #print(__name__, ': TestClass.setup()  - - - - - - - -')
        blocksize=self.nt/self.nb
        pol=1
        d,pairs,phi,t,diag=self.data
        N   =   BlockLO(blocksize,t,offdiag=True)
        processd =  ProcessTimeSamples(pairs,self.npix,pol=pol ,phi=phi)
        self.npix=   processd.get_new_pixel[0]
        P   =   SparseLO(self.npix,self.nt,pairs,pol=pol,angle_processed=processd)
        Mbd =   BlockDiagonalPreconditionerLO(processd ,self.npix,pol=pol)
        B   =   BlockDiagonalLO(processd,self.npix,pol=pol)

        A   =   P.T*N*P
        tol=1.e-5
        eigv ,Z=spla.eigsh(A,M=B,Minv=Mbd,k=5,which='SM',ncv=15,tol=tol)
        r=Z.shape[1]
        Zd=DeflationLO(Z)
        # Build Coarse operator
        Az=Z*0.
        for i in xrange(r):
            Az[:,i]=A*Z[:,i]
        self.invE=CoarseLO(Z,Az,r,apply='eig')
        self.E=dgemm(Z,Az.T)
        self.r=r
    def teardown(self):
        pass
    def test_multiplication(self):
        #print(__name__, ': TestClass.test_method_1()')
        v=np.ones(self.r)
        y=self.invE*v
        v2=np.dot(self.E,self.invE*v)
        y2=la.solve(self.E,v)
        assert np.allclose(v,v2) and np.allclose(y2,y)

    def test_singular_matrix(self):
        #print(__name__, ': TestClass.test_method_2()')
        invE=self.invE.to_array()
        evals=la.eigvalsh(invE)
        k= max(evals)/min(evals)
        assert ( abs(k)<=1.e3)


class TestCoarseOperator_QU():
    @classmethod
    def setup_class(self):
        self.nt,self.npix,self.nb= 400,20,2
        self.data=system_setup(self.nt,self.npix,self.nb)
    @classmethod
    def teardown_class(self):
        self.data=None
        #print(__name__, ': TestClass.teardown_class() -------')

    def setup(self):
        #print(__name__, ': TestClass.setup()  - - - - - - - -')
        blocksize=self.nt/self.nb
        pol=2
        d,pairs,phi,t,diag=self.data
        N   =   BlockLO(blocksize,t,offdiag=True)
        processd =  ProcessTimeSamples(pairs,self.npix,pol=pol ,phi=phi)
        self.npix=   processd.get_new_pixel[0]
        P   =   SparseLO(self.npix,self.nt,pairs,pol=pol,angle_processed=processd)
        Mbd =   BlockDiagonalPreconditionerLO(processd ,self.npix,pol=pol)
        B   =   BlockDiagonalLO(processd,self.npix,pol=pol)

        A   =   P.T*N*P
        tol=1.e-5
        eigv ,Z=spla.eigsh(A,M=B,Minv=Mbd,k=5,which='SM',ncv=15,tol=tol)
        r=Z.shape[1]
        Zd=DeflationLO(Z)
        # Build Coarse operator
        Az=Z*0.
        for i in xrange(r):
            Az[:,i]=A*Z[:,i]
        self.invE=CoarseLO(Z,Az,r,apply='eig')
        self.E=dgemm(Z,Az.T)
        self.r=r
    def teardown(self):
        pass
    def test_multiplication(self):
        #print(__name__, ': TestClass.test_method_1()')
        v=np.ones(self.r)
        y=self.invE*v
        v2=np.dot(self.E,self.invE*v)
        y2=la.solve(self.E,v)
        assert np.allclose(v,v2) and np.allclose(y2,y)

    def test_singular_matrix(self):
        #print(__name__, ': TestClass.test_method_2()')
        invE=self.invE.to_array()
        evals=la.eigvalsh(invE)
        k= max(evals)/min(evals)
        assert ( abs(k)<=1.e3)

class TestCoarseOperator_IQU():
    @classmethod
    def setup_class(self):
        self.nt,self.npix,self.nb= 400,20,2
        self.data=system_setup(self.nt,self.npix,self.nb)
    @classmethod
    def teardown_class(self):
        self.data=None
        #print(__name__, ': TestClass.teardown_class() -------')

    def setup(self):
        #print(__name__, ': TestClass.setup()  - - - - - - - -')
        blocksize=self.nt/self.nb
        pol=3
        d,pairs,phi,t,diag=self.data
        N   =   BlockLO(blocksize,t,offdiag=True)
        processd =  ProcessTimeSamples(pairs,self.npix,pol=pol ,phi=phi)
        self.npix=   processd.get_new_pixel[0]
        P   =   SparseLO(self.npix,self.nt,pairs,pol=pol,angle_processed=processd)
        Mbd =   BlockDiagonalPreconditionerLO(processd ,self.npix,pol=pol)
        B   =   BlockDiagonalLO(processd,self.npix,pol=pol)
        A   =   P.T*N*P
        tol=1.e-5
        eigv ,Z=spla.eigsh(A,M=B,Minv=Mbd,k=5,which='SM',ncv=15,tol=tol)
        r=Z.shape[1]
        Zd=DeflationLO(Z)
        # Build Coarse operator
        Az=Z*0.
        for i in xrange(r):
            Az[:,i]=A*Z[:,i]
        self.invE=CoarseLO(Z,Az,r,apply='eig')
        self.E=dgemm(Z,Az.T)
        self.r=r
    def teardown(self):
        pass
    def test_multiplication(self):
        #print(__name__, ': TestClass.test_method_1()')
        v=np.ones(self.r)
        y=self.invE*v
        v2=np.dot(self.E,self.invE*v)
        y2=la.solve(self.E,v)
        assert np.allclose(v,v2) and np.allclose(y2,y)

    def test_singular_matrix(self):
        #print(__name__, ': TestClass.test_method_2()')
        invE=self.invE.to_array()
        evals=la.eigvalsh(invE)
        k= max(evals)/min(evals)
        assert ( abs(k)<=1.e3)
filter_warnings("ignore")
