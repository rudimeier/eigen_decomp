# read-in cifti (nifti2) format via Satra's nibabel repository:
# $git clone --branch enh/cifti2 https://github.com/satra/nibabel.git

from glob import glob
import os
import numpy as np
import nibabel as nb
import sys

print "python version: ", sys.version[0:5]
# (HYDRA) 2.7.9
print "numpy version: ", np.__version__
# (HYDRA) 1.9.1

# # set as a local input directory (MPI)
# data_path = '/a/documents/connectome/_all'
# # set a list for the subject ID's
# subject_list = ['100307', '100408', '101006', '101107', '101309']
# # set as local output directory (MPI)
# out_path = '/home/raid/bayrak/devel/eigen_decomp/hcp_prep_out'

subject = sys.argv[1] # e.g. /ptmp/mdani/hcp/100307
# set as local output directory (HYDRA)
out_path = '/ptmp/sbayrak/hcp_prep_out'

def correlation_matrix(subject):
    template = 'rfMRI_REST?_??_Atlas_hp2000_clean.dtseries.nii'
    # template = ('%s/MNINonLinear/Results/rfMRI_REST?_??/rfMRI_REST?_??_Atlas_hp2000_clean.dtseries.nii' % subject)
    files = [val for val in sorted(glob(os.path.join(subject, template)))]
    filename = files[:4]

    # read in data and create correlation matrix:
    # for left hemisphere; 'range(0,nsamples)' for all ??
    # data_range = range(0,32492) ??

    tmp_t_series = []
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

        tmp_t_series.extend(((single_t_series - mean_series) / std_series).T)

        del img
        del single_t_series

    # K : matrix of similarities / Kernel matrix / Gram matrix
    K = (np.corrcoef(np.array(tmp_t_series).T) + 1) / 2.
    del tmp_t_series

    return K

def save_output(subject, matrix):
    out_dir = os.path.join(out_path)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    filename = os.path.basename(subject) + '_hcp_prep_out_COMP.csv'
    print filename
    out_file = os.path.join(out_dir, filename)
    # %.e = Floating point exponential format (lowercase)
    np.savetxt(out_file, matrix, fmt='%5.5e', delimiter='\t', newline='\n')
    return out_file

## calculate correlation matrices for the subject directory save it
#K = correlation_matrix(subject)
#save_output(subject, K)

import load_nifti
template = 'rfMRI_REST?_??_Atlas_hp2000_clean.dtseries.nii'
K = load_nifti.load_nii(subject, template, 4)
L = np.corrcoef(K.T)
del K
save_output(subject, L)
