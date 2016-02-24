import scipy.linalg as la
from interfaces import *
from utilities import *
import numpy as np
import time
import matplotlib.pyplot as plt

def test_block_diagonal_operator():
    """
    test the :class:`BlockDiagonalLO`.
    """
    runcase={'I':1,'QU':2,'IQU':3}
    a=[64,128,256,512]
    nb=1
    for nt in [2**12,2**16]:
        for pol in runcase.values():
            blocksize=nt/nb
            for i in a:
                npix=int(i)
                x=np.ones(pol*npix)
                d,pairs,phi,t,diag=system_setup(nt,npix,nb)
                P=SparseLO(npix,nt,pairs,phi,pol=pol)
                Mbd=BlockDiagonalPreconditionerLO(P,npix,pol=pol)
                invMbd=BlockDiagonalLO(P,npix,pol=pol)
                #invMbd and P.T*P  operate in the same on to a pixel vector
                y=invMbd*x
                y2=P.T*P*x
                assert np.allclose(y,y2)

                # invMbd*Mbd = Identity
                v=Mbd*invMbd*x
                assert np.allclose(v,x)

test_block_diagonal_operator()
