import numpy as np
import time
import matplotlib.pyplot as plt
from interfaces import *
filter_warnings("ignore")
def test_subscan_filtering():

    """
    Test the action of the :class:`FilterLO`.
    """
    runcase={'I':1,'QU':2,'IQU':3}
    for pol in runcase.values():
        d,t,phi,pixs,hp_pixs,ground,ces_size,nbolopairs,subscan_nsample=read_from_data_with_subscan_resize(
                                                                'data/20131011_092136.hdf5',npairs=5,pol=pol)
        nt,npix,nb=len(d),len(hp_pixs),len(t)
        print nbolopairs,ces_size
        processd =  ProcessTimeSamples(pixs,npix,pol=pol ,obspix=hp_pixs,phi=phi)
        npix=   processd.get_new_pixel[0]
        P   =   SparseLO(npix,nt,pixs,pol=pol,angle_processed=processd)
        F   =   FilterLO(nt,subscan_nsample,ces_size,nbolopairs,P.pairs)

        s=time.clock()
        filterdata=F*d
        e=time.clock()
        time1=e-s


        assert not np.allclose(filterdata,d)
#test_subscan_filtering()
