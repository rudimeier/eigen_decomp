# read-in cifti (nifti2) format via Satra's nibabel repository:
# $git clone --branch enh/cifti2 https://github.com/satra/nibabel.git

from glob import glob
import os
import numpy as np
import numexpr as ne
ne.set_num_threads(ne.ncores) # inclusive HyperThreading cores
import nibabel as nb
import sys
import math
import time

sys.path.append(os.path.expanduser('~/devel/mapalign/mapalign'))
import embed

# global n, dimension of corr matrix , will be set when reading files ...
NN = 0
#NN = 11448 # fake this for testing: 8095...0.5 GB, 11448...1 GB, 16190...2 GB


print "python version: ", sys.version[0:5]
# (HYDRA) 2.7.9
print "numpy version: ", np.__version__
# (HYDRA) 1.9.1

def load_nii_subject(subject, dtype=None):
    template = 'rfMRI_REST?_??_Atlas_hp2000_clean.dtseries.nii'
    files = [val for val in sorted(glob(os.path.join(subject, template)))]
    files = files[:4]

    # read in data and create correlation matrix:
    # for left hemisphere; 'range(0,nsamples)' for all ??
    # data_range = range(0,32492) ??

    for x in xrange(0, 4):

        img = nb.load(files[x])
        # ntimepoints, nsamples = img.data.shape

        # the following should be a concatenation of all 4 scans from each subject:
        # brainModels[2] will include both left and right hemispheres
        # for only left hemisphere: brainModels[1]

        header = img.header.matrix.mims[1].brainModels[2].indexOffset

        # set global n, sometimes it's difficult to know about it
        global NN
        if NN == 0:
            NN = header
        else:
            # globally faked ... for testing
            header = NN

        single_t_series = img.data[:, :header].T

        mean_series = single_t_series.mean(axis=0)
        std_series = single_t_series.std(axis=0)

        if x == 0:
            # In first loop we initialize matrix K to be filled up and returned.
            n = single_t_series.shape[0]
            m_single = single_t_series.shape[1]
            # By default we are using the same dtype like input file (float32).
            init_dtype = single_t_series.dtype if dtype == None else dtype
            K = np.ndarray(shape=[n,4*m_single], dtype=init_dtype, order='F')

        K[:, x*m_single:(x+1)*m_single] = (
            (single_t_series - mean_series) / std_series)

        del img
        del single_t_series

    return K

def load_random_subject(n,m):
    return np.random.randn(n, m)

def my_cov(m):

    # Handles complex arrays too
    m = np.asarray(m)
    dtype = np.result_type(m, np.float64)
    X = np.array(m, ndmin=2, dtype=dtype)
    print "X", X.shape
    N = X.shape[1]

    fact = float(N - 1)
    if fact <= 0:
        warnings.warn("Degrees of freedom <= 0 for slice", RuntimeWarning)
        fact = 0.0

    X -= X.mean(axis=1, keepdims=True)
    print_time("mean:")
    return np.dot(X, X.T / fact)

# This is corrcoef from numpy 1.9.2 ... mem usage optimized
def corrcoef_upper(x):
    c = my_cov(x)
    print_time("my_cov:")
    print "c", c.shape, c.dtype
    d = np.diag(c)
    c = mat_to_upper(c)
    d = np.sqrt(d)
    d = np.multiply.outer(d, d)
    d = mat_to_upper(d)
    return ne.evaluate('c / d')

def correlation_matrix(subject):
    set_time()
    K = load_nii_subject(subject)
    #K = load_random_subject(NN,4800)
    # K : matrix of similarities / Kernel matrix / Gram matrix
    print_time("load:")
    print "input data shape:", K.shape
    K = corrcoef_upper(K)
    print_time("corrcoef:")
    return K

def fisher_r2z(R):
    return ne.evaluate('arctanh(R)')

def old_fisher_r2z(R):
    # convert 1.0's into largest smaller value than 1.0
    di = np.diag_indices(R.shape[1])
    epsilon = np.finfo(float).eps
    R[di] = 1.0 - epsilon
    # Fisher r to z transform 
    Z = np.arctanh(R)
    return Z 

def fisher_z2r(Z):
    X = ne.evaluate('exp(2*Z)')
    return ne.evaluate('(X - 1) / (X + 1)')

def old_fisher_z2r(Z):
    # Fisher z to r transform
    R = (np.exp(2*Z) - 1)/(np.exp(2*Z) +1)
    # set diagonals back to 1.0
    di = np.diag_indices(R.shape[1])
    R[di] = 1.0
    return R

def mat_to_upper(A):
    n = A.shape[0]
    size = (n - 1) * n / 2
    U = np.ndarray(shape=[size,], dtype=A.dtype)
    k = 0
    for i in range(0, n-1):
        len = n - 1 - i
        U[k:k+len] = A[i,i+1:n]
        k += len
    return U

def upper_to_mat(A):
    n = int(round( 0.5 + np.sqrt(0.25 + 2 * A.shape[0]) ))
    M = np.zeros(shape=[n,n], dtype=A.dtype)
    k = 0
    for i in range(0,n):
        len = n - 1 - i
        M[i,i+1:n] = A[k:k+len]
        M[i,i] = 1.0
        M[i,0:i] = M[0:i,i]
        k += len
    return M

last_time = 0

def set_time():
    global last_time
    last_time = time.time()

def print_time(s):
    global last_time
    now = time.time()
    print "time", s, (now - last_time)
    last_time = now

# here we go ...

subject_list = np.array(sys.argv)[1:] # e.g. /ptmp/sbayrak/hcp/100307
N = len(subject_list)

for i in range(0, N):
    subject = subject_list[i]
    print "do loop %d/%d, %s" % (i+1, N, subject)
    # This always returns dtype=np.float64, consider adding .astype(np.float32)
    K = correlation_matrix(subject)

    # For the next calculations we only use the upper triangular matrix!

    K = fisher_r2z(K)
    print_time("fisher_r2z:")
    if i == 0:
        SUM = K.copy()
    else:
        SUM = ne.evaluate('SUM + K')
    print_time("sum:")
    del K

print "loop done"
print

SUM = ne.evaluate('SUM / N')
print_time("final division:")
SUM = fisher_z2r(SUM)
print_time("final fisher_z2r:")
SUM = upper_to_mat(SUM)
print_time("final upper_to_mat:")

print "do embed for correlation matrix:", SUM.shape
embedding, result = embed.compute_diffusion_map(SUM, alpha=0, n_components=20,
    diffusion_time=0, skip_checks=True, overwrite=True)
print_time("final embedding:")

#save_output(subject, embedding)
np.savetxt("out_test", embedding, fmt='%5.5e', delimiter='\t', newline='\n')
print_time("final save:")

print result['lambdas']

