import numpy as np
import sys
import os
import scipy
import sklearn
import csv
from glob import glob
import nibabel as nb

print "aaaa scipy", scipy.__version__
print "bbbb numpy", np.__version__
print "ccccc sklearn", sklearn.__version__

sys.path.append(os.path.expanduser('~/devel/mapalign/mapalign'))
import embed 

infile = sys.argv[1]
# e.g. /ptmp/sbayrak/hcp/%SUBJ_ID% (HYDRA)

subject = os.path.basename(infile)[0:6]

out_path = '/ptmp/sbayrak/embed_out'
# set as local output directory (HYDRA)

# load .nii files and calculate correlation matrix (hcp_prep)
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

def save_output(subject, embed_matrix):
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    filename = subject + '_embed_out.csv'
    print filename
    out_file = os.path.join(out_path, filename)
    # %.e = Floating point exponential format (lowercase)
    np.savetxt(out_file, embed_matrix, fmt='%5.5e', delimiter='\t', newline='\n')
    return out_file

# calculate correlation matrices from the subject directory save it
L = correlation_matrix(infile)

print "correlation matrix:", L.shape

embedding, result = embed.compute_diffusion_map(L, alpha=0, n_components=20,
    diffusion_time=0, skip_checks=True, overwrite=True)

save_output(subject, embedding)

print result['lambdas']

