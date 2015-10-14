from interfaces import *
from utilities import *
import numpy as np

def test_matrix_vector_product():
    """
    test the matrix vector multiplication of A and A^T.
    """

    for nt in xrange(5,30):
        for npix in xrange(4,nt ):
            x=np.arange(npix)
            pairs=[]
            for i in xrange(nt):
                pairs.append((i,rd.randint(0 ,npix-1)))

            P=SparseLO(npix,nt,pairs)

            y=P*x
            y2=[x[i[1]] for i in pairs]

            assert np.allclose(y,y2)




test_matrix_vector_product()
