import numpy as np
from interfaces import *
from utilities import *
import time
from numpy.polynomial.legendre import Legendre  as poly
import pylab as pl
from scipy.special import legendre


def test_ground_filter():
    pol=1
    filelist=['data/20120718_093931.hdf5']
    d,t,phi,pixs,hp_pixs,ground,subscan_nsample,tstart,samples_per_bolopair,bolos_per_ces=\
                read_multiple_ces(filelist,pol, npairs=10,filtersubscan=True)

    nt,npix,nb=len(d),len(hp_pixs),len(t)

    CESs= ProcessTimeSamples(pixs,npix,hp_pixs,pol=pol,phi=phi)
    npix,hp_pixs=CESs.get_new_pixel
    nbins_ground=int(max(ground))+1
    flag=np.ma.masked_equal(pixs,-1)
    ground[flag.mask]=-1
    negs=np.ma.masked_less(ground,0)
    m=np.logical_and(flag.mask, negs.mask )
    ground[negs.mask]=-1
    print len(ground[negs.mask])-len(ground[flag.mask]),"missed negative ground bins"
    unflag=np.logical_not(negs.mask)
    groundbins=[]
    for i in ground[unflag] :
        counts[ground[i]]+=1.
        if not groundbins.__contains__(i):
            groundbins.append(i)
        else : continue


    print groundbins

    P=SparseLO(npix,nt,pixs)
    G=SparseLO(nbins_ground,nt,ground)
    counts=np.zeros(nbins_ground)
    c=0
    for i in negs.data:
        if ground[i]==-1:
            continue

    G.counts=counts
    invGtG=BlockDiagonalPreconditionerLO(G,nbins_ground)
    I= lp.IdentityOperator(nt)
    F=I -G*invGtG*G.T

    print F*d


def test_polynomials():
    nt=50
    npix=5
    nb=1
    d,pairs,phi,t,diag=system_setup(nt,npix,nb)

    #x=np.arange(nt)
    polyorder=3
    #pl.plot(d,label='before filtering ')
    legendres=get_legendre_polynomials(polyorder,nt)
    print "order","< p,d > "
    for i in range(polyorder):
        #pl.plot(legendres[:,i],label='leg not norm')
        legendres[:,i]/=norm2(legendres[:,i])
        #pl.plot(legendres[:,i],label='leg norm')

        #pl.plot(q[:,i],label='qs')

        #print np.allclose(legendres[:,i],q[:,i]),legendres[:,i],q[:,i]
        #print scalprod(legendres[:,i],legendres[:,i+1]),norm2(legendres[:,i])
        projection = scalprod(legendres[:,i],d)*legendres[:,i]
        print i,projection[0],scalprod(q[:,i],d)*q[:,i][0]
        #d-=projection
        pl.plot(d-projection,label='filter w/ '+str(i)+'th order Legendre polynomial')
        pl.plot(projection,label='filter vector '+str(i)+'th order Legendre')
    print np.mean(d)
    plt.legend(loc='best')

    pl.show()
def test_poly_filtering():
    pol=1
    filelist=['data/20120718_093931.hdf5']
    d,t,phi,pixs,hp_pixs,ground,subscan_nsample,tstart,samples_per_bolopair,bolos_per_ces=\
                read_multiple_ces(filelist,pol, npairs=None,filtersubscan=True)

    nt,npix,nb=len(d),len(hp_pixs),len(t)

    CESs= ProcessTimeSamples(pixs,npix,hp_pixs,pol=pol,phi=phi)
    npix,hp_pixs=CESs.get_new_pixel
    print nt,npix,nb
    pl.plot(d,label='before filtering ')
    P=SparseLO(npix,nt,pixs,angle_processed=CESs,pol=pol)

    F=FilterLO(nt,[subscan_nsample,tstart],samples_per_bolopair,bolos_per_ces,P.pairs,poly_order=0)
    F1=FilterLO(nt,[subscan_nsample,tstart],samples_per_bolopair,bolos_per_ces,P.pairs,poly_order=3)
    s=time.clock()
    fd1=F1*d
    e=time.clock()
    print "poly",e-s
    s=time.clock()
    fd= F*d
    e=time.clock()
    print "mean",e-s

    pl.plot(fd,label='after filtering ')
    pl.plot(fd1,label='after filtering 1')
    plt.legend(loc='best')#

    pl.show()
test_poly_filtering()
#test_polynomials()
#test_ground_filter()
