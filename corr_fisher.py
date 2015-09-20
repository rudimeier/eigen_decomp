# read-in cifti (nifti2) format via Satra's nibabel repository:
# $git clone --branch enh/cifti2 https://github.com/satra/nibabel.git

from glob import glob
import os
import numpy as np
import nibabel as nb
import sys
import math

print "python version: ", sys.version[0:5]
# (HYDRA) 2.7.9
print "numpy version: ", np.__version__
# (HYDRA) 1.9.1

def load_nii_subject(subject, dtype=None):
    template = 'rfMRI_REST?_??_Atlas_hp2000_clean.dtseries.nii'
    # template = ('%s/MNINonLinear/Results/rfMRI_REST?_??/rfMRI_REST?_??_Atlas_hp2000_clean.dtseries.nii' % subject)
    files = [val for val in sorted(glob(os.path.join(subject, template)))]
    filename = files[:4]

    # read in data and create correlation matrix:
    # for left hemisphere; 'range(0,nsamples)' for all ??
    # data_range = range(0,32492) ??

    for x in xrange(0, 4):

        img = nb.load(filename[x])
        # ntimepoints, nsamples = img.data.shape

        # the following should be a concatenation of all 4 scans from each subject:
        # brainModels[2] will include both left and right hemispheres
        # for only left hemisphere: brainModels[1]

        header = img.header.matrix.mims[1].brainModels[2].indexOffset
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
    #K = load_random_subject(4000,4800)
    # K : matrix of similarities / Kernel matrix / Gram matrix
    K = np.corrcoef(K)
    return K

def fisher_r2z(R):
    # convert 1.0's into largest smaller value than 1.0
    di = np.diag_indices(R.shape[1])
    epsilon = np.finfo(float).eps
    R[di] = 1.0 - epsilon
    # Fisher r to z transform 
    Z = np.arctanh(R)
    return Z 

def fisher_z2r(Z):
    # Fisher z to r transform
    R = (np.exp(2*Z) - 1)/(np.exp(2*Z) +1)
    # set diagonals back to 1.0
    di = np.diag_indices(R.shape[1])
    R[di] = 1.0
    return R

# here we go ...

subject_list = np.array(sys.argv)[1:] # e.g. /ptmp/sbayrak/hcp/100307
N = len(subject_list)

for i in range(0, N):
    subject = subject_list[i]
    K = correlation_matrix(subject)
    if i == 0:
        SUM = np.zeros(K.shape,K.dtype)
    SUM += fisher_r2z(K)
    del K

SUM /= float(N)
SUM = fisher_z2r(SUM)

# Just testing ... the diagonal of the average correlation matrix should be 1.0
di = np.diag_indices(SUM.shape[1])
print np.allclose(SUM[di], 1.0)
