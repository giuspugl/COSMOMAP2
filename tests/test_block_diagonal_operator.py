import scipy.linalg as la
from interfaces import *
from utilities import *
import numpy as np
import time
import matplotlib.pyplot as plt

def test_block_diagonal_operator():
    """
    Build and test the :class:`CoarseLO`.
    """
    #filter_warnings("ignore")
    runcase={'I':1,'QU':2,'IQU':3}
    a=[10,50,100,300]
    nb=1
    for nt in [5000,50000]:
        pixels,t_expl,t_mult,speedup=[],[],[],[]
        for pol in runcase.values():
            blocksize=nt/nb
            for i in a:
                npix=int(i)
                x=np.ones(pol*npix)
                d,pairs,phi,t,diag=system_setup(nt,npix,nb)
                P=SparseLO(npix,nt,pairs,phi,pol=pol)
                Mbd=BlockDiagonalPreconditionerLO(P,npix,pol=pol)
                invMbd=BlockDiagonalLO(P,npix,pol=pol)
                #show_matrix_form(Mbd*invMbd)
                s=time.clock()
                y=invMbd*x
                e=time.clock()
                t=e-s
                s=time.clock()
                y2=P.T*P*x
                e=time.clock()

                #print nt,pol*npix,t,e-s
                pixels.append(pol*npix)
                t_expl.append(t)
                t_mult.append(e-s)
                speedup.append((e-s)/t)
                #print pol,i
                assert np.allclose(y,y2)
                v=Mbd*invMbd*x
                assert np.allclose(v,x)
        #plt.xscale('log')
#        plt.yscale('log')
#        plt.plot(pixels,speedup,'o')
            #plt.plot(pixels,t_expl,'ro')
            #plt.plot(pixels,t_mult,'go')
#    plt.show()



test_block_diagonal_operator()
