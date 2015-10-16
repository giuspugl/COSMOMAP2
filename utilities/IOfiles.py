import numpy as np
import h5py as h5

def read_from_hdf5(filename):
    f=h5.File(filename,"r")
    f.close()
    return


def write_to_hdf5(filename,pairs,noise_values,d,phi=None):
    obs_pixels=[i[1] for i in pairs]
    f=h5.File(filename,"w")
    group=f.create_group("bolo_pair")

    pixs=group.create_dataset('pixel',np.shape(obs_pixels), dtype=h5.h5t.STD_I32BE)
    weight=group.create_dataset('weight',np.shape(noise_values),dtype=h5.h5t.IEEE_F64BE)
    det=group.create_dataset('sum',np.shape(d),dtype=h5.h5t.IEEE_F64BE)
    if phi!=None:
        polang=group.create_dataset('pol_angle',np.shape(phi), dtype=h5.h5t.IEEE_F64BE)
        polang[...]=phi

    pixs[...]=obs_pixels
    weight[...]=noise_values
    det[...]=d


    f.close()
def read_from_hdf5(filename):
    f=h5.File(filename,"r")

    obs_pix=f['/bolo_pair/pixel'][:]
    polang=f['/bolo_pair/pol_angle'][:]
    weight=[i for i in f['/bolo_pair/weight'][:]]
    diag=[i[0] for i in weight]
    det=f['bolo_pair/sum'][:]
    f.close()

    nt=len(obs_pix)
    pairs=[(i,obs_pix[i]) for i in xrange(nt)]

    return pairs,polang,weight,diag,det
