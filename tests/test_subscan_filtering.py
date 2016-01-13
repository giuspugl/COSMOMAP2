import numpy as np
from interfaces import *

def test_subscan_filtering():

    """
    Test the action of the :class:`FilterLO`.
    """

    pol=3

    d,t,phi,pixs,hp_pixs,ground,ces_size,subscan_nsample=read_from_data_with_subscan_resize(
                                                    'data/20131011_092136.hdf5',pol=pol)

    nt,npix,nb=len(d),len(hp_pixs),len(t)
    #print nt,npix,nb,len(hp_pixs[pixs])

    F=FilterLO(nt,subscan_nsample)
    subscan_size=sum(subscan_nsample)

    filterdata=F*d


    offset=0
    while offset< nt :
        for i in subscan_nsample:
            start=offset
            end=start + i
            offset+=i
            assert (d[start:end] != filterdata[start:end]).all()

test_subscan_filtering()
