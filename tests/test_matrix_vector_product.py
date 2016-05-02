from interfaces import *
from utilities import *
import numpy as np
import matplotlib.pyplot as plt
filter_warnings("ignore")
def test_matrix_vector_product():
    """
    test the matrix vector multiplication of A and A^T.
    """

    nt,npix=80,50
    pairs=pairs_gen(nt,npix)

    #print "obs_pixs\t",pairs
    P=SparseLO(npix,nt,pairs)
    npix=P.ncols

    x=np.ones(npix)
    y=P.T*P*x
    assert np.allclose(y,P.counts)


def test_explicit_implementation_blockdiagonal_preconditioner():
    """
    test  the inverse matrix ``(A^T*A)^-1``. Whose action when applied on a vector
    is explicitly implemented through the derived  :class:`BlockDiagonalPreconditionerLO`.

    """
    #import time
    from scipy.linalg import inv
    runcase={'IQU':3,'I':1,'QU':2}
    nt=10000
    for pol in runcase.values():
        #print pol
        phi=angles_gen(2.,nt)
        a=[10,50,100,300,600]
        for i in a:
            npix=int(i)
            pairs=pairs_gen(nt,npix)
            P=SparseLO(npix,nt,pairs,phi,pol=pol)
            npix=P.ncols
            x=np.ones(npix*pol)
            v=P.T*P*x
            v2=v*0.
            Mbd=BlockDiagonalPreconditionerLO(P,npix,pol=pol)
            if pol==1:
                v2=v/P.counts
            elif pol==3:
                for j,s2,c2,cs,s,c,hits in zip(np.arange(npix),Mbd.sin2,Mbd.cos2, Mbd.sincos,\
                                                Mbd.sin,Mbd.cos,Mbd.counts) :
                    matr=np.array([[hits,c,s],[c,c2,cs],[s,cs,s2]])
                    ainv=inv(matr)
                    v2[pol*j:pol*j+pol]=np.dot(ainv,v[pol*j:pol*j+pol])
            elif pol==2:
                for j,s2,c2,cs in zip(np.arange(npix),Mbd.sin2,Mbd.cos2, Mbd.sincos):
                    matr =np.array([[c2,cs],[cs,s2]])
                    ainv=inv(matr)
                    v2[pol*j:pol*j+pol]=np.dot(ainv,v[pol*j:pol*j+pol])

            v3=Mbd*v
            assert np.allclose(v2,v3)

def test_preconditioner_times_matrix_gives_identity():
    """
    test  the action of  ``v=(A^T*A)^-1 * A^T*A * x`` onto  a vector x.
    check whther x==v.

    """
    runcase={'IQU':3,'I':1,'QU':2}
    for pol in runcase.values():
        for nt in xrange(20000,20001):
            phi=angles_gen(2.,nt)
            a=[10,50,100]
            for i in a:
                npix=int(i)
                pairs=pairs_gen(nt,npix)
                P=SparseLO(npix,nt,pairs,phi,pol=pol)
                npix=P.ncols
                Mbd=BlockDiagonalPreconditionerLO(P,npix,pol=pol)
                if pol==1:
                    offset=0
                    x=np.ones(npix)
                elif pol==3:
                    x=np.tile([0,0,1],(npix))
                    offset=2
                elif pol==2:
                    x=np.tile([0,1],(npix))
                    offset=1

                v=Mbd*P.T*P*x
                #pixel_to_check=[ pol*i+offset for i in P.obspix]
                #assert np.allclose(v[pixel_to_check],x[pixel_to_check])
                assert np.allclose(v,x)


#test_matrix_vector_product()
#test_preconditioner_times_matrix_gives_identity()
#test_explicit_implementation_blockdiagonal_preconditioner()
