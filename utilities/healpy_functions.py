import healpy as hp
import numpy as np
import matplotlib.pyplot as plt

def  obspix2mask(obspix,pixs,nside,fname,write=False):
    """
    From the observed pixels to a binary mask, (``mask[obspix]=1 , 0 elsewhere``)

    **Parameters**

    - ``osbpix``:{array}
        pixels observed during the scanning of the telescope.
        Already ordered in the  HEALPIX pixelization.
    - ``nside``: {int}
        Healpix parameter to define the pixelization grid of the map
    - ``fname``:{str}
        path to the fits file to store/read the map
    - ``write``:{bool}
        if ``True`` it writes onto the file, it reads from it otherwise

    **Returns**

    - mask :{array}


    """
    mask=np.zeros(hp.nside2npix(nside))
    mask[obspix[pixs]]=1.
    if write:
        hp.write_map(fname,mask)

    return mask

def reorganize_map(mapin,obspix,npix,nside,pol,fname,write=False):
    """
    From the solution map of the preconditioner to a Healpix map.
    It specially split the input array ``mapin`` which is a IQU
    for a polarization analysis in to 3 arrays ``i,q,u``.

    **Parameters**

    - ``mapin``:{array}
        solution array map (``size=npix*pol``);
    - ``obspix``:{array}
        array containing the observed pixels in the Healpix ordering;
    - ``npix``:{int}
    - ``nside``: {int}
        the same as in ``obspix2mask``;
    - ``pol``:{int}
    - ``fname``:{str}
    - ``write``:{bool}

    **Returns**

    - healpix_map:{list of arrays}
         pixelized map  with Healpix.

    """

    healpix_npix=hp.nside2npix(nside)


    if pol==3:
        healpix_map=np.zeros(healpix_npix*pol).reshape((healpix_npix,pol))
        i=mapin[np.arange(0,npix*3,3)]
        q,u=mapin[np.arange(1,npix*3,3)],mapin[np.arange(2,npix*3,3)]

        m=np.where(q!=0.)[0]
        healpix_map[obspix,0]=i
        healpix_map[obspix,1]=q
        healpix_map[obspix,2]=u
        hp_list=[healpix_map[:,0],healpix_map[:,1],healpix_map[:,2]]

    elif pol==1:
        healpix_map=np.zeros(healpix_npix)

        healpix_map[obspix]=mapin
        hp_list=[healpix_map]
    if write:
        hp.write_map(fname,hp_list)

    return hp_list

def compare_maps(outm,inm,pol,coords,mask):
    #inm*=mask
    if pol==1:
        maxval=max(inm)
        minval=min(inm)

        hp.gnomview(inm,rot=coords,xsize=600,min=minval,max=maxval,title='I input map',sub=131)
        hp.graticule(dpar=5,dmer=5,local=True)
        hp.gnomview(outm[0],rot=coords,xsize=600,min=minval,max=maxval,title='I output',sub=132)
        hp.graticule(dpar=5,dmer=5,local=True)
        hp.gnomview(inm-outm[0],rot=coords,xsize=600,min=minval,max=maxval,title=' I diff',sub=133)
        hp.graticule(dpar=5,dmer=5,local=True)

    elif pol==3:
        maxval=max(inm[1])
        minval=min(inm[1])
        hp.gnomview(inm[1],rot=coords,xsize=600,min=minval,max=maxval,title='Q input map',sub=231)
        hp.graticule(dpar=5,dmer=5,local=True)
        hp.gnomview(outm[1],rot=coords,xsize=600,min=minval,max=maxval,title='Q output map',sub=232)
        hp.graticule(dpar=5,dmer=5,local=True)
        hp.gnomview(inm[1]-outm[1],rot=coords,xsize=600,min=minval,max=maxval,title='Q diff',sub=233)
        hp.graticule(dpar=5,dmer=5,local=True)
        maxval=max(inm[2])
        minval=min(inm[2])
        hp.gnomview(inm[2],rot=coords,xsize=600,min=minval,max=maxval,title='U input map',sub=234)
        hp.graticule(dpar=5,dmer=5,local=True)
        hp.gnomview(outm[2],rot=coords,xsize=600,min=minval,max=maxval,title='U output map',sub=235)
        hp.graticule(dpar=5,dmer=5,local=True)
        #hp.gnomview(inm[2]-outm[2],rot=coords,xsize=600,min=minval,max=maxval,title='U diff',sub=236)
        hp.gnomview(inm[2]-outm[2],rot=coords,xsize=600,title='U diff',sub=236)

        hp.graticule(dpar=5,dmer=5,local=True)

    plt.show()
    pass
