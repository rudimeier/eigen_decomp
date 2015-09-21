# read-in cifti (nifti2) format via Satra's nibabel repository:
# $git clone --branch enh/cifti2 https://github.com/satra/nibabel.git

from glob import glob
import os
import numpy as np
import nibabel as nb
import sys
import math

satra_path = sys.path.append('/u/sbayrak/devel/mapalign/mapalign')
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

def correlation_matrix(subject):
    K = load_nii_subject(subject)
    #K = load_random_subject(NN,4800)
    # K : matrix of similarities / Kernel matrix / Gram matrix
    K = np.corrcoef(K)
    return K

def fisher_r2z(R):
    return np.arctanh(R)

def old_fisher_r2z(R):
    # convert 1.0's into largest smaller value than 1.0
    di = np.diag_indices(R.shape[1])
    epsilon = np.finfo(float).eps
    R[di] = 1.0 - epsilon
    # Fisher r to z transform 
    Z = np.arctanh(R)
    return Z 

def fisher_z2r(Z):
    X=np.exp(2*Z)
    return (X - 1) / (X + 1)

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

# here we go ...

subject_list = np.array(sys.argv)[1:] # e.g. /ptmp/sbayrak/hcp/100307
N = len(subject_list)

for i in range(0, N):
    subject = subject_list[i]
    print i, "do corr"
    # this always returns dtype=np.float64, consider adding .astype(np.float32)
    K = correlation_matrix(subject)
    # for the next calculations we only use the upper triangular matrix
    K = mat_to_upper(K)

    print i, "do r2z"
    K = fisher_r2z(K)
    print i, "do sum"
    if i == 0:
        SUM = K.copy()
    else:
        SUM += K
    del K

print "loop done, do fisher_z2r"
SUM /= float(N)
SUM = fisher_z2r(SUM)
SUM = upper_to_mat(SUM)

print "do embed"
print "correlation matrix:", SUM.shape
embedding, result = embed.compute_diffusion_map(SUM, alpha=0, n_components=20,
    diffusion_time=0, skip_checks=True, overwrite=True)

#save_output(subject, embedding)
np.savetxt("out_test", embedding, fmt='%5.5e', delimiter='\t', newline='\n')

print result['lambdas']

