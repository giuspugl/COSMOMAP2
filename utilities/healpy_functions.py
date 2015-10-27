import healpy as hp
import numpy as np


def  obspix2mask(obspix,nside,fname,write=False):
    """
    From the observed pixels to a binary mask, (``mask[obspix]=1 , 0 elsewhere``)

    **Parameters**

    - ``osbpix``:{array}
        pixels observed in the scanning of the telescope.
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
    if write:
        mask=np.zeros(hp.nside2npix(nside))
        mask[obspix]=1.
        hp.write_map(fname,mask)
    else:
        mask=hp.read_map(fname)
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

    healpix_map=np.zeros(healpix_npix*pol).reshape((healpix_npix,pol))
    print len(healpix_map)
    if pol==3:
        i=mapin[np.arange(0,npix,3)]
        q,u=mapin[np.arange(1,npix,3)],mapin[np.arange(2,npix,3)]
        healpix_map[obspix,0]=i
        healpix_map[obspix,1]=q
        healpix_map[obspix,2]=u

    elif pol==1:
        healpix_map[obspix,0]=mapin
    if write:
        hp.write_map(fname,healpix_map)
    return healpix_map

def compare_maps(outm,inm,pol,coords):
    if pol==1:
        hp.gnomview(inm,rot=coords,xsize=600,min=-239,max=139,title='I input map')
        hp.graticule(dpar=5,dmer=5,local=True)
        hp.gnomview(hp_map[:,0],rot=coords,xsize=600,min=-239,max=139,title='I output')
        hp.graticule(dpar=5,dmer=5,local=True)
        hp.gnomview(inm-hp_map[:,0],rot=coords,xsize=600,min=-239,max=139,title=' I diff')
        hp.graticule(dpar=5,dmer=5,local=True)

    elif pol==3:
        hp.gnomview(inm[:,1],rot=coords,xsize=600,min=-239,max=139,title='Q input map')
        hp.graticule(dpar=5,dmer=5,local=True)
        hp.gnomview(hp_map[:,1],rot=coords,xsize=600,min=-239,max=139,title='Q output map')
        hp.graticule(dpar=5,dmer=5,local=True)
        hp.gnomview(inm[:,1]-hp_map[:,1],rot=coords,xsize=600,min=-239,max=139,title='Q diff')
        hp.graticule(dpar=5,dmer=5,local=True)

        hp.gnomview(inm[:,2],rot=coords,xsize=600,min=-239,max=139,title='U input map')
        hp.graticule(dpar=5,dmer=5,local=True)
        hp.gnomview(hp_map[:,2],rot=coords,xsize=600,min=-239,max=139,title='U output map')
        hp.graticule(dpar=5,dmer=5,local=True)
        hp.gnomview(inm[:,2]-hp_map[:,2],rot=coords,xsize=600,min=-239,max=139,title='U diff')
        hp.graticule(dpar=5,dmer=5,local=True)

    plt.show()
    return 
