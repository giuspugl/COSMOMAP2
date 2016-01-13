import math as m
import numpy as np
import linop.linop as lp
import blkop as blk
import random as rd
import scipy.sparse.linalg as spla
from utilities import *

class FilterLO(lp.LinearOperator):
    """
    When applied to :math:`n_t` vector, this  operator filters out
    its components by removing a constant value.
    """
    def mult(self,d):
        vec_out=d*0.
        offset=0
        while offset<self.n:
            for i in self.chunks:
                start=offset
                end=start + i
                offset+=i
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
    It constitutes an interface for dealing with the projection operator.

    Since this can be represented as a sparse matrix, it is initialized \
    by passing an array of observed pixels which resembles the  ``(i,j)`` positions \
    of the non-null elements of  the matrix.

    .. note::

        During its initialization,  a private member function :func:`initializeweights` is called
        to precompute arrays which are  related to the scanning strategy and the angles, to the
        :class:`interfaces.BlockDiagonalPreconditionerLO`.


    **Parameters**

    - ``n`` : {int}
        number of columns;
    - ``m`` : {int}
        number of rows;
    - ``obs_pixs`` : {array}
         pixels id, ``j`` of the non-null elements of  the matrix, :math:`A_{i,j}`;
    - ``pol`` : {int,[*default* `pol=1`]}
        process an intensity only map (``[default] pol=1``), intensity/polarization
        map (``pol=3``);
    - `phi`: {array, [*default* `None`]}
        array with polarization angles (if ``pol=3``).
    - ``w``: {array, [*default* `None`]}
        array with noise weights , :math:`w_t= N^{-1} _{tt}`, computed by
        :func:`BlockLO.build_blocks`. They are used to compute  the
        BlockDiagonal Preconditioner by :func:`SparseLO.initializeweights`.

    """

    def mult(self,v):
        """
        Performs the product of a sparse matrix :math:`Av`,\
         with :math:`v` a  ``numpy`` npix  dense vector.

        It extracts the components of :math:`v` corresponding  to the non-null \
        elements of the operator.

        """
        x=np.zeros(self.nrows)

        for i,j in np.ndenumerate(self.pairs):
            x[i]+=v[j]

        return x

    def rmult(self,v):
        """
        Performs the product for the transpose operator :math:`A^T`.

        """
        x=np.zeros(self.ncols)
        for i,j in np.ndenumerate(self.pairs):
            x[j]+=v[i]

        return x


    def mult_iqu(self,v):
        """
        Performs the product of a sparse matrix :math:`Av`,\
        with ``v`` a  ``numpy`` dense vector containing the three Stokes [IQU].

        .. note::
            Compared to the operation ``mult`` this routine returns a TOD like vector,
            defined as:

            .. math::

                d_t= I_p + Q_p \cos(2\phi_t)+ U_p \sin(2\phi_t).


            with :math:`p` is the pixel observed at time :math:`t` with polarization angle
            :math:`\phi_t`.

        """
        x=np.zeros(self.nrows)
        i=0
        #for i,j in np.ndenumerate(self.pairs):
        for j in self.pairs:

            x[i]+=v[3*j]+v[3*j+1]*self.cos[i]+v[3*j+2]*self.sin[i]
            i+=1

        return x

    def rmult_iqu(self,v):
        """
        Performs the product for the transpose operator :math:`A^T` to get a IQU map-like vector.
        Since this vector resembles the pixel of 3 maps it has 3 times the size ``Npix``.
        Values of the same pixel are stored in the memory contiguously.

        """
        x=np.zeros(self.ncols*self.pol)
        i=0
        #for i,j in np.ndenumerate(self.pairs):
        for j in self.pairs:
            x[3*j]+=v[i]

            x[1+3*j]+=v[i]*self.cos[i]
            x[2+3*j]+=v[i]*self.sin[i]
            i+=1
        return x

    def initializeweights(self,phi,w):
        """

        Pre-compute the quantitities needed in the explicit
        implementation of :math:`(P^T P)`:

        **Parameters**

        - ``counts`` :
            how many times a given pixel is observed in the timestream;
        - ``mask``:
          mask from  either unobserved pixels (``counts=0``)  or   bad constrained
          (see the ``pol==3`` following case) ;
        - [*If*  ``pol==3``]:
            the matrix :math:`(A^T A)`  is  sparse-block, components are :

            .. csv-table::

                    ":math:`n_{hits}`", ":math:`0`", ":math:`0`"
                    ":math:`0`", ":math:`\sum_t cos^2 2 \phi_t`", ":math:`\sum_t sin 2\phi_t cos 2 \phi_t`"
                    ":math:`0`",  ":math:`\sum_t sin2 \phi_t cos 2 \phi_t`",   ":math:`\sum_t sin^2 2 \phi_t`"

            the determinant, the trace are therefore needed to compute the  eigenvalues
            of each block via the formula:

            .. math::

                \lambda_{min,max}= Tr(M)/2 \pm \sqrt{Tr^2(M)/4 - det(M)}

            being :math:`M` a ``2x2`` matrix.
            The eigenvalues are needed to define the mask of bad constrained pixels whose
            condition number is :math:`\gg 1`.

        """
        self.counts=np.zeros(self.ncols)
        if self.pol==1:
            i=0
            for j in self.pairs:
                self.counts[j]+=w[i]
                i+=1
            self.mask=np.where(self.counts !=0)[0]

        if self.pol==3:

            self.cos=np.cos(2*phi)
            self.sin=np.sin(2*phi)
            self.cos2=np.zeros(self.ncols)
            self.sin2=np.zeros(self.ncols)
            self.sincos=np.zeros(self.ncols)

            i=0
            for j in self.pairs:
                self.counts[j]+=w[i]

                self.cos2[j]+=w[i]*self.cos[i]*self.cos[i]
                self.sin2[j]+=w[i]*self.sin[i]*self.sin[i]
                self.sincos[j]+=w[i]*self.cos[i]*self.sin[i]
                i+=1

            det=(self.cos2*self.sin2)-(self.sincos*self.sincos)

            tr=self.cos2+self.sin2

            sqrt=np.sqrt(tr*tr/4. -det)
            lambda_max=tr/2. + sqrt
            lambda_min=tr/2. - sqrt

            cond_num=np.abs(lambda_max/lambda_min)
            mask=np.where(cond_num<=1.e3)[0]

            self.cos2[mask]/=det[mask]
            self.sin2[mask]/=det[mask]
            self.sincos[mask]/=det[mask]
            self.mask=mask


    def __init__(self,n,m,obs_pixs,phi=None,pol=1,w=None):
        if w==None:
            w=np.ones(m)

        self.ncols=n
        self.pol=pol
        self.nrows=m
        self.pairs=obs_pixs
        self.initializeweights(phi,w)
        if pol==3:
            self.__runcase='IQU'
            super(SparseLO, self).__init__(nargin=pol*n,nargout=m, matvec=self.mult_iqu,
                                                symmetric=False, rmatvec=self.rmult_iqu )
        elif pol==1:
            self.__runcase='I'
            super(SparseLO, self).__init__(nargin=n,nargout=m, matvec=self.mult,
                                                symmetric=False, rmatvec=self.rmult )
        else:
            raise RuntimeError("No valid polarization key set!\t=>\tpol=%d \n \
                                    Possible values are pol=%d(I),%d(IQU)."%(pol,1,3))

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

    - ``blocksize`` : {int}
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

        self.blocklist=[]
        if self.isoffdiag:
            self.blocklist = [ToeplitzLO(i,self.blocksize) for i in self.covnoise]

        if not self.isoffdiag:
            tmplist=[]
            d=np.ones(self.blocksize)
            for i in self.covnoise:
                d.fill(i)
                self.blocklist.append(lp.DiagonalOperator(d))
                tmplist.append(d)

                d=np.empty(self.blocksize)

            self.diag=np.concatenate(tmplist)

    def __init__(self,blocksize,t,offdiag=False):
        self.__isoffdiag = offdiag
        self.blocksize=blocksize
        self.covnoise=t
        self.build_blocks()

        super(BlockLO, self).__init__(self.blocklist)

    @property
    def isoffdiag(self):
        """
        Property saying whether or not the operator has
        off-diagonal terms.
        """
        return self.__isoffdiag

class BlockDiagonalPreconditionerLO(lp.LinearOperator):
    """
    Standard preconditioner defined as:

    .. math::

        M_{BD}=( A \, diag(N^{-1}) A^T)^{-1}

    where :math:`A` is a :class:`SparseLO` operator.
    Such inverse action could be easily computed given the structure of the
    matrix :math:`A`, (sparse if `pol=1`, block-sparse if `pol=3`).

    **Parameters**

    - ``counts``: {array}
        member of SparseLO,  given a pixel-id it returns the :math:`n_{hits}`
    - ``masks``:{array}
        masking the bad or unobserved pixels, member of :class:`SparseLO`;
    - ``n``:{int}
        the size of the problem, ``npix``;
    - ``sin2, cos2,sincos``: {arrays}
        are members of :class:`SparseLO` and they refer to the trigoniometric functions of :math:`\phi_t`.


    """

    def mult(self,x):
        """
        Action of :math:`y=( A \, diag(N^{-1}) A^T)^{-1} x`,
        where :math:`x` is   an :math:`n_{pix}` array.
        """
        y=x*0.

        if self.pol==1:
            y[self.mask]=x[self.mask]/self.counts[self.mask]
        elif self.pol==3:
            m=len(self.mask)
            for j in xrange(m):
                i=self.mask[j]
                y[3*i]=x[3*i]/self.counts[i]
                qtmp=self.sin2[i]*x[i*3+1]-self.sincos[i]*x[i*3+2]
                utmp=self.cos2[i]*x[i*3+2]-self.sincos[i]*x[i*3+1]
                y[i*3+1],y[i*3+2]=qtmp,utmp

        return y

    #def to_array(self):
    #    I=np.identity(self.size)
    #    return self.matmat(I)
    #@property
    #def T(self):
    #    return self
    def __init__(self,counts,mask,n,pol=1,sin2=None,cos2=None,sincos=None,noisediag=None):
        self.counts=counts
        self.size=pol*n
        self.mask=mask
        self.pol=pol

        if pol==3:
            self.sin2=sin2
            self.cos2=cos2
            self.sincos=sincos
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
        with a certain ``scipy`` routine  defined above as ``method``.

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
        It can be any ``scipy`` solver, namely ``cg, gmres``.

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


from scipy.linalg import solve,lu

class CoarseLO(lp.LinearOperator):
    """
    This class contains all the operation involving the coarse operator :math:`E`.
    In this implementation :math:`E` is always applied to a vector wiht
    its inverse : :math:`E^{-1}`.
    When initialized it performs an LU decomposition to accelerate the performances
    of the inversion.

    **Parameters**

    - ``Z`` : {np.matrix}
            deflation matrix;
    - ``A`` : {SparseLO}
            to  compute vectors :math:`AZ_i`;
    - ``r`` :  {int}
            :math:`rank(Z)`, dimension of the deflation subspace.
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

    def set_operator(self,Z,Az,r):
        M=dgemm(Z,Az.T)
        self.L,self.U=lu(M,permute_l=True,overwrite_a=True,check_finite=False)
    #def __init__(self,Z,A,r):
    def __init__(self,Z,Az,r):
        #Az=Z*0.
        #for i in xrange(r):
        #    Az[:,i]=A*Z[:,i]
        self.set_operator(Z,Az,r)
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
            y+=self.z[i]*x[i]
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
