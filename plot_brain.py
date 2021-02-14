import os
import numpy as np
import matplotlib.pyplot as plt
from itertools import compress
from mayavi import mlab
import mne
import os.path as op
from surfer import Brain
from numpy import arange
from numpy.random import permutation
import nibabel as nib


def plot_brain_mne(raw):
    # fnirs_data_folder = mne.datasets.fnirs_motor.data_path()
    # fnirs_cw_amplitude_dir = os.path.join(fnirs_data_folder, 'Participant-1')
    # raw_intensity = mne.io.read_raw_nirx(fnirs_cw_amplitude_dir, verbose=True)
    # raw_intensity.load_data()

    subjects_dir = mne.datasets.sample.data_path() + '/subjects'

    fig = mne.viz.create_3d_figure(size=(800, 600), bgcolor='white')
    fig = mne.viz.plot_alignment(raw.info, show_axes=True,
                                 subject='fsaverage', coord_frame='mri',
                                 trans='fsaverage', surfaces=['brain'],
                                 subjects_dir=subjects_dir, fig=fig)
    fig2 = mne.viz.plot_alignment(raw.info, show_axes=True,
                                  subject='fsaverage', coord_frame='mri',
                                  trans='fsaverage', surfaces=['brain'],
                                  subjects_dir=subjects_dir, fig=fig)
    mne.viz.set_3d_view(figure=fig, azimuth=20, elevation=60, distance=0.4,
                        focalpoint=(0., -0.01, 0.02))
    mne.viz.set_3d_view(figure=fig2, azimuth=20, elevation=60, distance=0.4,
                        focalpoint=(0.5, 0.5, 0.5))

    print('fsf')


hemi = 'lh'
surf = 'inflated'
subjects_dir = os.environ["SUBJECTS_DIR"] = 'C:\\Users\\t-deangeckt\\mne_data\\MNE-sample-data/subjects'
subject = "fsaverage"

fname_fs = subjects_dir + "/fsaverage/bem/fsaverage-ico-5-src.fif"
src_fs = mne.read_source_spaces(fname_fs)
nv = 1000  # keep only n_v vertices per hemi
vertices = [src_fs[0]["vertno"][:nv], src_fs[1]["vertno"][:nv]]

n_subjects = 2
sources_l = np.zeros((nv, n_subjects))
sources_r = np.zeros((nv, n_subjects))

# subject 1
sources_l[:10, 0] = 5
sources_r[:10, 0] = 3

# subject 2
sources_l[100:110, 1] = 2
sources_r[100:110, 1] = 5

sources = [sources_l, sources_r]
colormaps = ["Reds", "Blues"]


def plot_sources(sources, sub_id=[0, 1], order=1):
    fmax = 5
    brain = Brain(subject, hemi="both", surf="inflated", views="dorsal")
    hemis = ["lh", "rh"]
    for sources_h, v, h in zip(sources[::order], vertices[::order],
                               hemis[::order]):
        for data, colormap in zip(sources_h.T[sub_id],
                                  np.array(colormaps)[sub_id]):
            brain.add_data(data, colormap=colormap, vertices=v,
                           verbose=False, colorbar=False,
                           smoothing_steps=10,
                           time_label=None, hemi=h, alpha=0.8,
                           min=0., mid=fmax / 5, max=fmax,
                           transparent=True)
    return brain


b = plot_sources(sources)
print('fs')
