from interfaces import *
from utilities import *
import numpy as np
import matplotlib.pyplot as plt

def test_matrix_vector_product():
    """
    test the matrix vector multiplication of A and A^T.
    """
    pol=1
    for nt in xrange(6,10):
        phi=angles_gen(2.,nt)
        for npix in xrange(5,nt ):
            x=np.ones(pol*npix)
            pairs=pairs_gen(nt,npix,pol=pol)
            #print "obs_pixs\t",pairs
            P=SparseLO(npix,nt,pairs,phi,pol=pol)

            y=P*x
            #print len(P.pairs),len(P.mask)

            #print "A*x\n",y
            y2=[x[j] for j in pairs]
            assert np.allclose(y,y2)

def test_explicit_implementation_blockdiagonal_preconditioner():
    """
    test  the inverse matrix ``(A^T*A)^-1``. Whose action when applied on a vector
    is explicitly implemented through the derived  :class:`BlockDiagonalPreconditionerLO`.

    """
    #import time
    from scipy.linalg import inv
    runcase={'IQU':3,'I':1,'QU':2}
    nt=6000
    for pol in runcase.values():

        phi=angles_gen(2.,nt)
        a=[10,50,100,300,600]
        for i in a:
            npix=int(i)
            pairs=pairs_gen(nt,npix,pol=pol)
            P=SparseLO(npix,nt,pairs,phi,pol=pol)
            x=np.ones(npix*pol)
            v=P.T*P*x
            v2=v*0.
            Mbd=BlockDiagonalPreconditionerLO(P,npix,pol=pol)
            if pol==1:
                v2[P.mask]=v[P.mask]/P.counts[P.mask]
            elif pol==3:
                for j,s2,c2,cs,s,c,hits in zip(P.mask,Mbd.sin2,Mbd.cos2, Mbd.sincos,\
                                                Mbd.sin,Mbd.cos,Mbd.counts) :
                    matr=np.array([[hits,c,s],[c,c2,cs],[s,cs,s2]])
                    ainv=inv(matr)
                    v2[pol*j:pol*j+pol]=np.dot(ainv,v[pol*j:pol*j+pol])
            elif pol==2:

                for j,s2,c2,cs in zip(P.mask,Mbd.sin2,Mbd.cos2, Mbd.sincos):
                    ainv=np.array([[s2,-cs],[-cs,c2]])
                    v2[pol*j:pol*j+pol]=np.dot(ainv,v[pol*j:pol*j+pol])

            v3=Mbd*v
            assert np.allclose(v2,v3)
            """
                scipytime.append(end-start)
                #print P.mask
                start=time.clock()
                end=time.clock()
                mytime.append(end-start )

                #print scipytime," \t ",end-start,"\n"

            n_pixels=a
            #print len(a),len(scipytime),len(mytime)
            plt.plot(n_pixels,scipytime,label='scipy time')
            plt.plot(n_pixels,mytime,label='my implementation')
            plt.yscale('log')
            plt.xscale('log')
            #plt.plot(n_pixels,mytime/scipytime,label='speedup')
            plt.legend(loc='lower right', numpoints = 1,prop={'size':9} )
            plt.ylabel('Time [sec]')
            plt.xlabel('#pixels')
            plt.show()

            plt.plot(n_pixels,np.array(mytime)/np.array(scipytime),label='speedup')
            plt.legend(loc='lower right', numpoints = 1,prop={'size':9} )
            plt.xlabel('#pixels')

            plt.show()
            """


def test_preconditioner_times_matrix_gives_identity():
    """
    test  the action of  ``v=(A^T*A)^-1 * A^T*A * x`` onto  a vector x.
    check whther x==v.

    """
    runcase={'IQU':3,'I':1,'QU':2}
    for pol in runcase.values():
        for nt in xrange(2000,2001):
            phi=angles_gen(2.,nt)
            a=[10,50,100]
            for i in a:
                npix=int(i)
                pairs=pairs_gen(nt,npix,pol=pol)
                P=SparseLO(npix,nt,pairs,phi,pol=pol)
                Mbd=BlockDiagonalPreconditionerLO(P,npix,pol=pol)
                if pol==1:
                    offset=0
                    x=np.ones(npix)
                elif pol==3:
                    x=np.array([0,0,1]*(npix))
                    offset=2
                    #Mbd=BlockDiagonalPreconditionerLO(P.counts,P.mask,npix,pol=pol,\
                    #sin2=P.sin2,cos2=P.cos2,sincos=P.sincos,cos=P.cosine,sin=P.sine)
                    A=Mbd*P.T*P
                    #show_matrix_form(A)
                    #Mbd=BlockDiagonalPreconditionerLO(P.counts,P.mask,npix,pol,\
                    #                                    P.sin2,P.cos2,P.sincos, P.cosine,P.sine)
                elif pol==2:

                    #Mbd=BlockDiagonalPreconditionerLO(None,P.mask,npix,pol,P.sin2,P.cos2,P.sincos)
                    x=np.array([0,1]*(npix))
                    offset=1

                v=Mbd*P.T*P*x
                pixel_to_check=[ pol*i+offset for i in P.mask]

                assert np.allclose(v[pixel_to_check],x[pixel_to_check])


test_matrix_vector_product()
test_preconditioner_times_matrix_gives_identity()
test_explicit_implementation_blockdiagonal_preconditioner()
