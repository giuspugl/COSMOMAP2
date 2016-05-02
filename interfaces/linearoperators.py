import math as m
import numpy as np
import linop.linop as lp
import blkop as blk
import random as rd
from scipy import weave
from scipy.weave import inline
import scipy.sparse.linalg as spla
from utilities import *

class FilterLO(lp.LinearOperator):
    """
    When applied to :math:`n_t` vector, this  operator filters out
    its components by removing a constant (its mean value) within a *subscan*
    interval.

    **Parameters**

    - ``size``: {int}
        the size of the input array;
    - ``subscan_nsample``: {array}
        contains the size of each chunk of the samples which has to be processed.
        :math:`\sum_i subscan_{i} = size`.

    """
    def mult(self,d):
        vec_out=d*0.
        offset=0
        while offset<self.n:
            for i in self.chunks:
                start=offset
                end=start + i
                offset+=i
                
                code = r"""
        	    int j;
                for (j=start;j<end;++j){
                    if (pixs(j) == -1) continue;
                    x(i)+= v(pixs(i));
                    }
                """
                res = inline(code,['pixs','v','x','Nrows'],verbose=1,
        		      extra_compile_args=['-march=native  -O3  -fopenmp ' ],
        		            support_code = r"""
        	                   #include <stdio.h>
                               #include <omp.h>
        	                   #include <math.h>""",
                               libraries=['gomp'],type_converters=weave.converters.blitz)

                vec_out[start:end ]=d[start:end] - np.mean(d[start:end])

        return vec_out

    def __init__(self,size,subscan_nsample):
        self.n=size
        self.chunks=subscan_nsample
        super(FilterLO, self).__init__(nargin=size,nargout=size, matvec=self.mult,
                                                symmetric=False )

class SparseLO(lp.LinearOperator):
    """
    Derived class from the one from the  :class:`LinearOperator` in :mod:`linop`.
    It constitutes an interface for dealing with the projection operator
    (pointing matrix).

    Since this can be represented as a sparse matrix, it is initialized \
    by an array of observed pixels which resembles the  ``(i,j)`` positions \
    of the non-null elements of  the matrix,``obs_pixs``.

    .. note::

        During its initialization,  a private member function :func:`initializeweights`
         is called to precompute arrays needed for the explicit implementation of
        :class:`interfaces.BlockDiagonalPreconditionerLO`.
        Moreover it masks all the unobserved or pathological pixels which won't
        be taken into account, via the functions   :func:`repixelization`  and
        :func:`flagging_samples`.

    **Parameters**

    - ``n`` : {int}
        number of columns;
    - ``m`` : {int}
        number of rows;
    - ``pix_samples`` : {array}
        list of pixels observed in the time domain,
        (or the non-null elements in a row of :math:`A_{i,j}`);
    - ``pol`` : {int,[*default* `pol=1`]}
        process an intensity only (``pol=1``), polarization only ``pol=2``
        and intensity+polarization map (``pol=3``);
    - `phi`: {array, [*default* `None`]}
        array with polarization angles (needed if ``pol=3,2``).
    - ``w``: {array, [*default* `None`]}
        array with noise weights , :math:`w_t= N^{-1} _{tt}`, computed by
        :func:`BlockLO.build_blocks`.   If it is  not set :func:`SparseLO.initializeweights`
        assumes it to be a :func:`numpy.ones` array.
    - ``pixel_schema``:{array}
        Map from the internal pixelization to an external one, i.e. HEALPIX, it has to be modified when
        pathological pixels are not taken into account.
        Default is :func:`numpy.arange(npix)`, i.e. identity map;
    - ``threshold_cond``: {float}
        set the condition number threshold to mask bad conditioned pixels (it's used in polarization cases).
        Default is set to 1.e3.
    """

    def mult(self,v):
        """
        Performs the product of a sparse matrix :math:`Av`,\
         with :math:`v` a  :mod:`numpy`  array (:math:`dim(v)=n_{pix}`)  .

        It extracts the components of :math:`v` corresponding  to the non-null \
        elements of the operator.

        """
        x=np.zeros(self.nrows)
        Nrows=self.nrows
        pixs=self.pairs
        code = r"""
	    int i;
        for (i=0;i<Nrows;++i){
            if (pixs(i) == -1) continue;
            x(i)+= v(pixs(i));
            }
        """
        res = inline(code,['pixs','v','x','Nrows'],verbose=1,
		      extra_compile_args=['-march=native  -O3  -fopenmp ' ],
		            support_code = r"""
	                   #include <stdio.h>
                       #include <omp.h>
	                   #include <math.h>""",
                       libraries=['gomp'],type_converters=weave.converters.blitz)
        return x
    def rmult(self,v):
        """
        Performs the product for the transpose operator :math:`A^T`.

        """
        x=np.zeros(self.ncols)
        Nrows=self.nrows
        pixs=self.pairs
        code = r"""
	       int i ;
           for ( i=0;i<Nrows;++i){
            if (pixs(i) == -1) continue;
            x(pixs(i))+=v(i);
            }
        """
        inline(code,['pixs','v','x','Nrows'],verbose=1,
		      extra_compile_args=['-march=native  -O3  -fopenmp ' ],
		            support_code = r"""
	                   #include <stdio.h>
                       #include <omp.h>
	                   #include <math.h>""",
                       libraries=['gomp'],type_converters=weave.converters.blitz)

        return x
    def mult_qu(self,v):
        """
        Performs :math:`A * v` with :math:`v` being a *polarization* vector.
        The output array will encode a linear combination of the two Stokes
        parameters,  (whose components are stored contiguously).

        .. math::
            d_t=  Q_p \cos(2\phi_t)+ U_p \sin(2\phi_t).
        """
        x=np.zeros(self.nrows)
        Nrows=self.nrows
        pixs=self.pairs
        cos,sin=self.cos,self.sin
        code = """
	       int i ;
           for ( i=0;i<Nrows;++i){
            if (pixs(i) == -1) continue;
            x(i)+=v(2*pixs(i)) *cos(i) + v(2*pixs(i)+1) *sin(i);
            }
        """
        inline(code,['pixs','v','x','Nrows','cos','sin'],verbose=1,
		      extra_compile_args=['-march=native  -O3  -fopenmp ' ],
		      support_code = r"""
	               #include <stdio.h>
                   #include <omp.h>
	               #include <math.h>""",
              libraries=['gomp'],type_converters=weave.converters.blitz)
        return x
    def rmult_qu(self,v):
        """
        Performs :math:`A^T * v`. The output vector will be a QU-map-like array.
        """
        vec_out= np.zeros(self.ncols*self.pol)
        Nrows=self.nrows
        pixs=self.pairs
        cos,sin=self.cos,self.sin
        code = """
	       int i;
           for ( i=0;i<Nrows;++i){
            if (pixs(i) == -1) continue;
            vec_out(2*pixs(i)) += v(i)*cos(i);
            vec_out(2*pixs(i)+1) += v(i)*sin(i);
            }
        """
        inline(code,['pixs','v','vec_out','Nrows','cos','sin'],verbose=1,
		      extra_compile_args=['-march=native  -O3  -fopenmp ' ],
		      support_code = r"""
	               #include <stdio.h>
                   #include <omp.h>
	               #include <math.h>""",
              libraries=['gomp'],type_converters=weave.converters.blitz)
        return vec_out
    def mult_iqu(self,v):
        """
        Performs the product of a sparse matrix :math:`Av`,\
        with ``v`` a  :mod:`numpy` array containing the
        three Stokes parameters [IQU] .

        .. note::
            Compared to the operation ``mult`` this routine returns a
            :math:`n_t`-size vector defined as:

            .. math::
                d_t= I_p + Q_p \cos(2\phi_t)+ U_p \sin(2\phi_t).

            with :math:`p` is the pixel observed at time :math:`t` with polarization angle
            :math:`\phi_t`.
        """
        x=np.zeros(self.nrows)
        Nrows=self.nrows
        pixs=self.pairs
        cos,sin=self.cos,self.sin
        code = r"""
	       int i ;
           for ( i=0;i<Nrows;++i){
            if (pixs(i) == -1) continue;
            x(i) +=  v(3*pixs(i)) + v(3*pixs(i)+1) *cos(i) + v(3*pixs(i)+2) *sin(i);
            }
        """
        inline(code,['pixs','v','x','Nrows','cos','sin'],verbose=1,
		      extra_compile_args=['-march=native  -O3  -fopenmp ' ],
		      support_code = r"""
	               #include <stdio.h>
                   #include <omp.h>
	               #include <math.h>""",
              libraries=['gomp'],type_converters=weave.converters.blitz)
        return x
    def rmult_iqu(self,v):
        """
        Performs the product for the transpose operator :math:`A^T` to get a IQU map-like vector.
        Since this vector resembles the pixel of 3 maps it has 3 times the size ``Npix``.
        IQU values referring to the same pixel are  contiguously stored in the memory.

        """
        x=np.zeros(self.ncols*self.pol)
        N=self.nrows
        pixs=self.pairs
        cos,sin=self.cos,self.sin
        code = """
	       int i;
           for ( i=0;i<N;++i){
            if (pixs(i) == -1) continue;
            x(3*pixs(i))   += v(i);
            x(3*pixs(i)+1) += v(i)*cos(i);
            x(3*pixs(i)+2) += v(i)*sin(i);
            }
        """
        inline(code,['pixs','v','x','N','cos','sin'],verbose=1,
		      extra_compile_args=['-march=native  -O3  -fopenmp ' ],
		      support_code = r"""
	               #include <stdio.h>
                   #include <omp.h>
	               #include <math.h>""",
              libraries=['gomp'],type_converters=weave.converters.blitz)

        return x
    def repixelization(self):
        """
        Performs pixel reordering by excluding all the unbserved or
        pathological pixels.
        """
        n_new_pix=0
        n_removed_pix=0
        self.old2new=np.zeros(self.ncols,dtype=int)
        if self.pol==1:
            #print "no repixelization",self.counts
            for jpix in xrange(self.ncols):
                if jpix in self.mask:
                    self.old2new[jpix]=n_new_pix
                    self.counts[n_new_pix]=self.counts[jpix]
                    self.obspix[n_new_pix]=self.obspix[jpix]
                    n_new_pix+=1
                else:
                    self.old2new[jpix]=-1
                    n_removed_pix+=1
            #resize array
            self.counts=np.delete(self.counts,xrange(n_new_pix,self.ncols))
            #print "repixelization",self.counts

        else:
            #print "no repixelization",self.cos2

            for jpix in xrange(self.ncols):
                if jpix in self.mask:
                    self.old2new[jpix]=n_new_pix
                    self.obspix[n_new_pix]=self.obspix[jpix]
                    self.cos2[n_new_pix]=self.cos2[jpix]
                    self.sin2[n_new_pix]=self.sin2[jpix]
                    self.sincos[n_new_pix]=self.sincos[jpix]
                    if self.pol==3:
                        self.counts[n_new_pix]=self.counts[jpix]
                        self.sine[n_new_pix]=self.sine[jpix]
                        self.cosine[n_new_pix]=self.cosine[jpix]
                    n_new_pix+=1
                else:
                    self.old2new[jpix]=-1
                    n_removed_pix+=1
            #resize
            self.cos2=np.delete(self.cos2,xrange(n_new_pix,self.ncols))
            self.sin2=np.delete(self.sin2,xrange(n_new_pix,self.ncols))
            self.sincos=np.delete(self.sincos,xrange(n_new_pix,self.ncols))
            if self.pol==3:
                self.counts=np.delete(self.counts,xrange(n_new_pix,self.ncols))
                self.sine=np.delete(self.sine,xrange(n_new_pix,self.ncols))
                self.cosine=np.delete(self.cosine,xrange(n_new_pix,self.ncols))
            #print "repixelization",self.cos2
        c=bash_colors()
        print c.header("___"*30)
        print c.blue("Found %d pathological pixels\nRepixelizing  w/ %d pixels."%(n_removed_pix,n_new_pix))
        print c.header("___"*30)
        #print "map old2new",self.old2new
        #resizing all the arrays
        self.obspix=np.delete(self.obspix,xrange(n_new_pix,self.ncols))
    def flagging_samples(self):
        """
        Flags the time samples related to bad pixels to -1.
        """
        N=self.nrows
        pixs=self.pairs
        o2n=self.old2new
        code = """
	       int i,pixel;
           for ( i=0;i<N;++i){
            pixel=pixs(i);
            if (pixel == -1) continue;
            pixs(i)=o2n(pixel);
            }
        """
        inline(code,['pixs','o2n','N'],verbose=1,
		      extra_compile_args=['-march=native  -O3  -fopenmp ' ],
		      support_code = r"""
	               #include <stdio.h>
                   #include <omp.h>
	               #include <math.h>""",
              libraries=['gomp'],type_converters=weave.converters.blitz)
    def initializeweights(self,phi,w):
        """
        Pre-compute the quantitities needed for the implementation of :math:`(A^T A)`
        and to masks bad pixels.

        **Parameters**

        - ``counts`` :
            how many times a given pixel is observed in the timestream;
        - ``mask``:
            mask  either unobserved  (``counts=0``)  or   bad constrained pixels
            (see the ``pol=3,2`` following cases) ;
        - *If* ``pol=2``:
            the matrix :math:`(A^T A)`  is  symmetric and block-diagonal, each block
            can be written as :

            .. csv-table::

                ":math:`\sum_t cos^2 2 \phi_t`", ":math:`\sum_t sin 2\phi_t cos 2 \phi_t`"
                ":math:`\sum_t sin2 \phi_t cos 2 \phi_t`",   ":math:`\sum_t sin^2 2 \phi_t`"

            the determinant, the trace are therefore needed to compute the  eigenvalues
            of each block via the formula:

            .. math::
                \lambda_{min,max}= Tr(M)/2 \pm \sqrt{Tr^2(M)/4 - det(M)}

            being :math:`M` a ``2x2`` matrix.
            The eigenvalues are needed to define the mask of bad constrained pixels whose
            condition number is :math:`\gg 1`.

        - [*If*  ``pol=3``]:
            each block of the matrix :math:`(A^T A)`  is a ``3 x 3`` matrix:

            .. csv-table::

                ":math:`n_{hits}`", ":math:`\sum_t cos 2 \phi_t`", ":math:`\sum_t sin 2 \phi_t`"
                ":math:`\sum_t cos 2 \phi_t`", ":math:`\sum_t cos^2 2 \phi_t`", ":math:`\sum_t sin 2\phi_t cos 2 \phi_t`"
                ":math:`\sum_t sin 2 \phi_t`",  ":math:`\sum_t sin2 \phi_t cos 2 \phi_t`",   ":math:`\sum_t sin^2 2 \phi_t`"

            We then define the mask of bad constrained pixels by both  considering
            the condition number similarly as in the ``pol=2`` case and the pixels
            whose count is :math:`\geq 3`.

        """
        if self.pol==1:
            self.counts=np.zeros(self.ncols)
            for j,weight in zip(self.pairs,w):
                self.counts[j]+=weight
            self.mask=np.where(self.counts >0)[0]

        if self.pol==2:
            self.cos=np.cos(2.*phi)
            self.sin=np.sin(2.*phi)
            self.cos2=np.zeros(self.ncols)
            self.sin2=np.zeros(self.ncols)
            self.sincos=np.zeros(self.ncols)
            for j,weight,cos,sin in zip(self.pairs,w,self.cos,self.sin):
                self.cos2[j]    +=  weight*cos*cos
                self.sin2[j]    +=  weight*sin*sin
                self.sincos[j]  +=  weight*cos*sin
            det=(self.cos2*self.sin2)-(self.sincos*self.sincos)
            tr=self.cos2+self.sin2
            sqrt=np.sqrt(tr*tr/4. -det)
            lambda_max=tr/2. + sqrt
            lambda_min=tr/2. - sqrt
            cond_num=np.abs(lambda_max/lambda_min)
            mask=np.where(cond_num<=self.threshold)[0]
            #self.det=det
            self.mask=mask

        if self.pol==3:
            self.cos=np.cos(2.*phi)
            self.sin=np.sin(2.*phi)
            self.counts=np.zeros(self.ncols)
            self.cosine=np.zeros(self.ncols)
            self.sine=np.zeros(self.ncols)
            self.cos2=np.zeros(self.ncols)
            self.sin2=np.zeros(self.ncols)
            self.sincos=np.zeros(self.ncols)
            for pix,weight,cos,sin in zip(self.pairs,w,self.cos,self.sin):
                self.counts[pix]    +=  weight
                self.cosine[pix]    +=  weight*cos
                self.sine[pix]      +=  weight*sin
                self.cos2[pix]      +=  weight*cos*cos
                self.sin2[pix]      +=  weight*sin*sin
                self.sincos[pix]    +=  weight*sin*cos

            det_block=(self.cos2*self.sin2)-(self.sincos*self.sincos)
            Tr_block=self.cos2+self.sin2
            sqrt=np.sqrt(Tr_block*Tr_block/4. -det_block)
            lambda_max=Tr_block/2. + sqrt
            lambda_min=Tr_block/2. - sqrt
            cond_num=np.abs(lambda_max/lambda_min)
            mask1=np.where(self.counts>2)[0]
            mask=np.where(cond_num<=self.threshold )[0]
            self.mask=np.intersect1d(mask1,mask)

    def __init__(self,n,m,pix_samples,phi=None,pol=1,w=None,pixel_schema=None,threshold_cond=1.e3):
        self.ncols=n

        self.pol=pol
        self.nrows=m
        if w is None:
            w=np.ones(m)
        if pixel_schema  is None:
            pixel_schema =np.arange(self.ncols)


        self.pairs=pix_samples
        self.obspix=pixel_schema
        self.threshold=threshold_cond
        self.initializeweights(phi,w)
        self.repixelization()
        self.flagging_samples()
        self.ncols=len(self.obspix)
        n=self.ncols
        if pol==3:
            self.__runcase='IQU'
            super(SparseLO, self).__init__(nargin=self.pol*self.ncols,nargout=self.nrows, matvec=self.mult_iqu,
                                            symmetric=False, rmatvec=self.rmult_iqu )
        elif pol==1:
            self.__runcase='I'
            super(SparseLO, self).__init__(nargin=self.pol*self.ncols,nargout=self.nrows, matvec=self.mult,
                                                symmetric=False, rmatvec=self.rmult )
        elif pol==2:
            self.__runcase='QU'
            super(SparseLO, self).__init__(nargin=self.pol*self.ncols,nargout=self.nrows, matvec=self.mult_qu,
                                                symmetric=False, rmatvec=self.rmult_qu )
        else:
            raise RuntimeError("No valid polarization key set!\t=>\tpol=%d \n \
                                    Possible values are pol=%d(I),%d(QU), %d(IQU)."%(pol,1,2,3))

    @property
    def maptype(self):
        """
        Return a string depending on the map you are processing
        """
        return self.__runcase


class ToeplitzLO(lp.LinearOperator):
    """
    Derived Class from a LinearOperator. It exploit the symmetries of an ``dim x dim``
    Toeplitz matrix.
    This particular kind of matrices satisfy the following relation:

    .. math::

        A_{i,j}=A_{i+1,j+1}=a_{i-j}

    Therefore, it is enough to initialize ``A`` by mean of an array ``a`` of ``size = dim``.

    **Parameters**

    - ``a`` : {array, list}
        the array which resembles all the elements of the Toeplitz matrix;
    - ``size`` : {int}
        size of the block.

    """


    def mult(self,v):
        """
        Performs the product of a Toeplitz matrix with a vector ``x``.

        """
        val=self.array[0]
        y=val*v
        for i in xrange(1,len(self.array)):
            val=self.array[i]
            temp=val*v
            y[:-i]+=temp[i:]
            y[i:]+=temp[:-i]

        return y


    def __init__(self,a,size):

        super(ToeplitzLO, self).__init__(nargin=size,nargout=size, matvec=self.mult,
                                                symmetric=True )
        self.array=a

class BlockLO(blk.BlockDiagonalLinearOperator):
    """
    Derived class from  :mod:`blkop.BlockDiagonalLinearOperator`.
    It basically relies on the definition of a block diagonal operator,
    composed by ``nblocks`` diagonal operators.
    If it does not have any  off-diagonal terms (*default case* ), each block is a multiple  of
    the identity characterized by the  values listed in ``t`` and therefore is
    initialized by the :func:`BlockLO.build_blocks` as a :class:`linop.DiagonalOperator`.

    **Parameters**

    - ``blocksize`` : {int or list }
        size of each diagonal block, it is : :math:`blocksize= n/nblocks`.
    - ``t`` : {array}
        noise values for each block
    - ``offdiag`` : {bool, default ``False``}
        strictly  related to the way  the array ``t`` is passed (see notes ).

        .. note::

            - True : ``t`` is a list of array,
                    ``shape(t)= [nblocks,bandsize]``, to have a Toeplitz band diagonal operator,
                    :math:`bandsize != blocksize`
            - False : ``t`` is an array, ``shape(t)=[nblocks]``.
                    each block is identified by a scalar value in the diagonal.

    """
    def build_blocks(self):
        """
        Build each block of the operator either with or
        without off diagonal terms.
        Each block is initialized as a Toeplitz (either **band** or **diagonal**)
        linear operator.

        .. see also::

        ``self.diag``: {numpy array}
            the array resuming the :math:`diag(N^{-1})`.


        """

        tmplist=[]
        self.blocklist=[]
        if self.isoffdiag:
            tmplist.append(self.covnoise[0])
            self.blocklist = [ToeplitzLO(i,self.blocksize) for i in self.covnoise]

        if not self.isoffdiag:
            d=np.ones(self.blocksize)
            for i in self.covnoise:
                d.fill(i)
                self.blocklist.append(lp.DiagonalOperator(d))
                tmplist.append(d)
                d=np.empty(self.blocksize)
        self.diag=np.concatenate(tmplist)

    def build_unbalanced_blocks(self):
        """
        Build the  list of Diagonal blocks of :class:`blk.BlockDiagonalLinearOperator`
        by assuming that the blocks have different size. Of course it is required that:

        .. math::
            \sum _{i=1} ^{nblocks} size(block_i) = N_t

        """
        self.blocklist=[]
        tmplist=[]
        for size,weight in zip(self.blocksize,self.covnoise):
            d=np.ones(size)
            d.fill(weight)
            self.blocklist.append(lp.DiagonalOperator(d))
            tmplist.append(d)
            d=np.empty(size)
        self.diag=np.concatenate(tmplist)


    def __init__(self,blocksize,t,offdiag=False):
        self.__isoffdiag = offdiag
        self.blocksize=blocksize
        self.covnoise=t
        if type(blocksize) is list:
            self.build_unbalanced_blocks()
        else:
            self.build_blocks()

        super(BlockLO, self).__init__(self.blocklist)

    @property
    def isoffdiag(self):
        """
        Property saying whether or not the operator has
        off-diagonal terms.
        """
        return self.__isoffdiag


class BlockDiagonalLO(lp.LinearOperator):
    """
    Explicit implementation of :math:`A \, diag(N^{-1}) A^T`, in order to save time
    in the application of the two matrices onto a vector (in this way the leading dimension  will be :math:`n_{pix}`
    instead of  :math:`n_t`).

    .. note::
        it is initialized as the  :class:`BlockDiagonalPreconditionerLO` since it involves
        computation with  the same matrices.
    """

    def __init__(self, A,n,pol=1):
        self.size=pol*n
        self.pol=pol
        super(BlockDiagonalLO, self).__init__(nargin=self.size,nargout=self.size,\
                                                matvec=self.mult, symmetric=True)
        self.pixels=np.arange(n)
        if pol==1 :
            self.counts=A.counts
        elif pol>1:
            self.sin2=A.sin2
            self.sincos=A.sincos
            self.cos2=A.cos2
            if pol==3:
                self.counts=A.counts
                self.cos=A.cosine
                self.sin=A.sine
        print "initialized"

    def mult(self,x):
        """
        Multiplication of  :math:`A \, diag(N^{-1}) A^T` on to a vector math:`x`
        ( :math:`n_{pix}` array).
        """
        y=x*0.
        #print len(self.counts),len(x)
        if self.pol==1:
            #y[self.mask]=x[self.mask]*self.counts[self.mask]
            y=x*self.counts
        elif self.pol==3:
            for pix,s2,c2,cs,c,s,hits in zip(self.pixels,self.sin2,self.cos2,self.sincos,\
                                                self.cos,self.sin,self.counts):
                y[3*pix]  = hits*x[3*pix] + c *x[3*pix+1] +  s*x[3*pix+2]
                y[3*pix+1]=  c * x[3*pix] + c2*x[3*pix+1] + cs*x[3*pix+2]
                y[3*pix+2]=  s * x[3*pix] + cs*x[3*pix+1] + s2*x[3*pix+2]
        elif self.pol==2:
            for pix,s2,c2,cs in zip( self.pixels,self.sin2,self.cos2,self.sincos):
                y[pix*2]  =  c2  *x[2*pix] + cs *x[pix*2+1]
                y[pix*2+1]=  cs  *x[2*pix] + s2 *x[pix*2+1]
        return y


class BlockDiagonalPreconditionerLO(lp.LinearOperator):
    """
    Standard preconditioner defined as:

    .. math::

        M_{BD}=( A \, diag(N^{-1}) A^T)^{-1}

    where :math:`A` is a :class:`SparseLO` operator.
    Such inverse operator  could be easily computed given the structure of the
    matrix :math:`A`. It could be  sparse in the case of Intensity only analysis (`pol=1`),
    block-sparse if polarization is included (`pol=3,2`).


    **Parameters**

    - ``n``:{int}
        the size of the problem, ``npix``;
    - ``A``:{:class:SparseLO}
        the linear operator related to the pointing matrix. Its members (`counts`, `masks`,
        `sine`, `cosine`, etc... ) are  needed to explicitly compute the inverse of the
        :math:`n_{pix}` blocks of :math:`M_{BD}`.
    - ``pol``:{int}
        the size of each block of the matrix.
    """

    def mult(self,x):
        """
        Action of :math:`y=( A \, diag(N^{-1}) A^T)^{-1} x`,
        where :math:`x` is   an :math:`n_{pix}` array.
        """
        y=x*0.

        if self.pol==1:
            #y[self.mask]=x[self.mask]/self.counts[self.mask]
            y=x/self.counts
        elif self.pol==3:
            determ=self.counts*(self.cos2*self.sin2 - self.sincos*self.sincos)\
                - self.cos*self.cos*self.sin2 - self.sin*self.sin*self.cos2\
                +2.*self.cos*self.sin*self.sincos

            for pix,det,s2,c2,cs,c,s,hits in zip(self.pixels,determ,self.sin2,self.cos2,self.sincos,\
                                self.cos,self.sin,self.counts):
                y[3*pix]  =((c2*s2-cs*cs)*x[3*pix]+ (s*cs-c*s2)  *x[3*pix+1]  +( c*cs-s*c2)  *x[3*pix+2])/det
                y[3*pix+1]=((s*cs-c*s2)  *x[3*pix]+ (hits*s2-s*s)*x[3*pix+1]  +( s*c-hits*cs)*x[3*pix+2])/det
                y[3*pix+2]=((c*cs -s*c2) *x[3*pix]+(-hits*cs+c*s)*x[3*pix+1]  +(hits*c2-c*c) *x[3*pix+2])/det

        elif self.pol==2:
            det=(self.cos2*self.sin2)-(self.sincos*self.sincos)
            for pix,s2,c2,cs,determ in zip( self.pixels,self.sin2,self.cos2,self.sincos,det):
                y[pix*2]  =  (s2  *x[2*pix] - cs *x[pix*2+1])/determ
                y[pix*2+1]=  (-cs  *x[2*pix] + c2 *x[pix*2+1])/determ

        return y

    def __init__(self,A,n,pol=1):

        self.size=pol*n
        self.pixels=np.arange(n)
        self.pol=pol
        if pol==1 :
            self.counts=A.counts
        elif pol==2:
            #self.det=A.det[A.mask]
            self.sin2=A.sin2
            self.cos2=A.cos2
            self.sincos=A.sincos
        elif pol==3:
            self.counts=A.counts
            self.sin2=A.sin2
            self.cos2=A.cos2
            self.sincos=A.sincos
            self.cos=A.cosine
            self.sin=A.sine

        super(BlockDiagonalPreconditionerLO,self).__init__(nargin=self.size,nargout=self.size,\
                                                            matvec=self.mult, symmetric=True)

class InverseLO(lp.LinearOperator):
    """
    Construct the inverse operator of a matrix :math:`A`, as a linear operator.

    **Parameters**

    - ``A`` : {linear operator}
        the linear operator of the linear system to invert;
    - ``method`` : {function }
        the method to compute ``A^-1`` (see below);
    - ``P`` : {linear operator } (optional)
        the preconditioner for the computation of the inverse operator.

    """
    def mult(self,x):
        """
        It returns  :math:`y=A^{-1}x` by solving the linear system :math:`Ay=x`
        with a certain :mod:`scipy` routine (e.g. :func:`scipy.sparse.linalg.cg`)
        defined above as ``method``.
        """

        y,info = self.method(self.A,x,M=self.preconditioner)
        self.isconverged(info)
        return y

    def isconverged(self,info):
        """
        It returns a Boolean value  depending on the
        exit status of the solver.

        **Parameters**

        - ``info`` : {int}
            output of the solver method (usually :func:`scipy.sparse.cg`).



        """
        self.__converged=info
        if info ==0:
            return True
        else :
            return False


    def __init__(self,A,method=None,preconditioner=None):
        super(InverseLO, self).__init__(nargin=A.shape[0],nargout=A.shape[1], matvec=self.mult,
                                                symmetric=True )
        self.A=A
        self.__method=method
        self.__preconditioner=preconditioner
        self.__converged=None

    @property
    def method(self):
        """
        The method to compute the inverse of A. \
        It can be any :mod:`scipy.sparse.linalg` solver, namely :func:`scipy.sparse.linalg.cg`,
        :func:`scipy.sparse.linalg.bicg`, etc.

        """
        return self.__method

    @property
    def converged(self):
        """
        provides convergence information:

        - 0 : successful exit;
        - >0 : convergence to tolerance not achieved, number of iterations;
        - <0 : illegal input or breakdown.

        """
        return self.__converged

    @property
    def preconditioner(self):
        """
        Preconditioner for the solver.
        """
        return self.__preconditioner


from scipy.linalg import solve,lu,eigh

class CoarseLO(lp.LinearOperator):
    """
    This class contains all the operation involving the coarse operator :math:`E`.
    In this implementation :math:`E` is always applied to a vector wiht
    its inverse : :math:`E^{-1}`.
    When initialized it performs either an LU or an eigenvalue  decomposition
    to accelerate the performances of the inversion.

    **Parameters**

    - ``Z`` : {np.matrix}
            deflation matrix;
    - ``A`` : {SparseLO}
            to  compute vectors :math:`AZ_i`;
    - ``r`` :  {int}
            :math:`rank(Z)`, dimension of the deflation subspace.
    -``apply``:{str}
            -``LU``: performs LU decomposition,
            -``eig``: compute the eigenvalues and eigenvectors of ``E``.
    """

    def mult(self,v):
        """
        Perform the multiplication of the inverse coarse operator :math:`x=E^{-1} v`.
        It exploits the LU decomposition of :math:`E` to solve the system :math:`Ex=v`.
        It first solves :math:`y=L^{-1} v` and then :math:`x=U^{-1}y`.
        """
        y=solve(self.L,v,lower=True,overwrite_b=False )
        x=solve(self.U,y,overwrite_b=True)
        return x

    def mult_eig(self,v):
        """
        Matrix vector multiplication with :math:`E^{-1}` computed via
        :func:`setting_inverse_w_eigenvalues`.
        """
        return self.invE.dot(v)

    def setting_inverse_w_eigenvalues(self,E):
        """
        This routine computes the inverse of ``E`` by a decomposition through an  eigenvalue
        decomposition. It further checks whether it has some degenerate eigenvalues,
        i.e. 0 to numerical precision (``1.e-15``), and eventually excludes these eigenvalues
        from the anaysis.
        """

        eigenvals,W=eigh(E)
        lambda_max=max(eigenvals)
        diags=eigenvals*0.
        #print abs(eigenvals/lambda_max)
        threshold_to_degen=1.e-5
        nondegenerate=np.where(abs(eigenvals/lambda_max)>threshold_to_degen)[0]
        degenerate=np.where(abs(eigenvals/lambda_max)<threshold_to_degen)[0]
        c=bash_colors()
        if len(degenerate)!=0:
            print c.header("==="*30)
            print c.warning("\t DISCARDING %d OUT OF %d EIGENVALUES\t"%(len(degenerate),len(eigenvals)))
            print eigenvals[degenerate]
            print c.header("==="*30)
        else:
            print c.header("==="*30)
            print c.header("\t Matrix E is not singular, all its eigenvalues have been taken into account\t")
            print c.header("==="*30)

        for i in nondegenerate:
                diags[i]=1./eigenvals[i]
        D=np.diag(diags)
        tmp=dgemm(D,W)
        self.invE=dgemm(W.T,tmp)

    def __init__(self,Z,Az,r,apply='LU'):
        M=dgemm(Z,Az.T)
        if apply=='eig':
            self.setting_inverse_w_eigenvalues(M)
            super(CoarseLO,self).__init__(nargin=r,nargout=r,matvec=self.mult_eig,
                                            symmetric=True)
        elif apply =='LU':
            self.L,self.U=lu(M,permute_l=True,overwrite_a=True,check_finite=False)
            super(CoarseLO,self).__init__(nargin=r,nargout=r,matvec=self.mult,
                                            symmetric=True)

class DeflationLO(lp.LinearOperator):
    """
    This class builds the Deflation operator (and its transpose)
    from the columns of the matrix ``Z``.

    **Parameters**

    - ``z`` : {np.matrix}
            the deflation matrix. Its columns are read as arrays in a list ``self.z``.

    """

    def mult(self,x):
        """
        Performs the matrix vector multiplication   :math:`Z x`
        with  :math:`dim(x) = rank(Z)`.

        """
        y=np.zeros(self.nrows)
        for i in xrange(self.ncols):
            y+=(self.z[i]*x[i])   #.astype(x.dtype)
        return y
    def rmult(self,x):
        """
        Performs the product onto a ``N_pix`` vector with :math:`Z^T`.

        """
        return np.array( [scalprod(i,x) for i in self.z] )

    def __init__(self, z):
        self.z=[]
        self.nrows,self.ncols=z.shape
        for j in xrange(self.ncols):
            self.z.append(z[:,j])
        super(DeflationLO,self).__init__(nargin=self.ncols, nargout=self.nrows,
                                                matvec=self.mult, symmetric=False,
                                                rmatvec=self.rmult)
