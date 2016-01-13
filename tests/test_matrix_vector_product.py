from interfaces import *
from utilities import *
import numpy as np

def test_matrix_vector_product():
    """
    test the matrix vector multiplication of A and A^T.
    """
    pol=1

    for nt in xrange(6,7):
        phi=angles_gen(2.,nt)

        for npix in xrange(5,nt ):
            x=np.ones(pol*npix)
            pairs=pairs_gen(nt,npix,pol=pol)
            #print "obs_pixs\t",pairs
            P=SparseLO(npix,nt,pairs,phi,pol=pol)

            y=P*x
            #print "A*x\n",y


            y2=[x[j] for j in pairs]
            assert np.allclose(y,y2)
def test_explicit_implementation_blockdiagonal_preconditioner():
    """
    test  the inverse matrix ``(A^T*A)^-1``. Whose action when applied on a vector
    is explicitly implemented through the derived  :class:`BlockDiagonalPreconditionerLO`.

    """

    for pol in [1,3]:

        for nt in xrange(6,7):
            phi=angles_gen(2.,nt)

            for npix in xrange(5,nt ):
                x=np.ones(pol*npix)
                pairs=pairs_gen(nt,npix,pol=pol)
                #print "obs_pixs\t",pairs
                P=SparseLO(npix,nt,pairs,phi,pol=pol)

                v=P.T*P*x
                #print "nhits\n",P.counts
                #print " A^T A*x\n", v
                #print P.mask
                v2=v*0.
                if pol==1:
                    M_bd=BlockDiagonalPreconditionerLO(P.counts,P.mask,npix,pol)
                    v2[P.mask]=v[P.mask]/P.counts[P.mask]
                elif pol==3:
                    M_bd=BlockDiagonalPreconditionerLO(P.counts,P.mask,npix,pol,P.sin2,P.cos2,P.sincos)
                    for j in xrange(len(P.mask)):
                        i=P.mask[j]
                        v2[3*i]=v[3*i]/P.counts[i]
                        qtmp=P.sin2[i]*v[i*3+1]-P.sincos[i]*v[i*3+2]
                        utmp=P.cos2[i]*v[i*3+2]-P.sincos[i]*v[i*3+1]
                        v2[i*3+1],v2[i*3+2]=qtmp,utmp

                #print "(A^T A)-1 A^T A*x\n",v2

                v3=M_bd*P.T*P*x
                #print "Mbd*v\n",v3
                assert np.allclose(v2,v3)




test_matrix_vector_product()
test_explicit_implementation_blockdiagonal_preconditioner()
