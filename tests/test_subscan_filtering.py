import numpy as np
import time
import matplotlib.pyplot as plt
from interfaces import *
filter_warnings("ignore")
def test_subscan_filtering():

    """
    Test the action of the :class:`FilterLO`.
    """
    #runcase={'QU':2}
    runcase={'I':1,'QU':2,'IQU':3}
    for pol in runcase.values():
        d,t,phi,pixs,hp_pixs,ground,ces_size,subscan_nsample=read_from_data_with_subscan_resize(
                                                                'data/20131011_092136.hdf5',pol=pol)
        nt,npix,nb=len(d),len(hp_pixs),len(t)
        P=SparseLO(npix,nt,pixs,phi,pol=pol)
        F=FilterLO(nt,subscan_nsample,P.pairs)
        subscan_size=sum(subscan_nsample)

        s=time.clock()
        filterdata=F*d
        #plt.subplot(1,2,1)
        #plt.plot(d)
        #plt.show()
        #plt.plot(filterdata)
        #plt.show()
        e=time.clock()
        time1=e-s
        filterdata2=filterdata*0.
        offset=0

        s=time.clock()
        listmean=[]
        vals=[]
        while offset< nt :
            for i in subscan_nsample:
                start=offset
                end=start + i
                offset+=i
                mean=0.
                counter=0
                for j in range(start,end):
                    if P.pairs[j]==-1: continue
                    mean+=d[j]
                    counter+=1
                #vals.append(.5*(end+start) )
                if (counter ==0):
                #    listmean.append(0)
                    continue
                else:
                    mean/=counter
                #    listmean.append(mean)
                    filterdata2[start:end]=d[start:end]-mean
        e=time.clock()
        #print "speedup",(e-s)/time1
        #plt.plot(vals,listmean)
        #plt.xlim([start,end])
        #plt.subplot(1,2,2)
        #plt.plot(filterdata2)
        #plt.show()
        #plt.show()
        assert np.allclose(filterdata,filterdata2)
#test_subscan_filtering()
