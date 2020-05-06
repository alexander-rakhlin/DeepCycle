import pickle
import numpy as np
from const import double_division_tracks, curated_tracks, DATA_ROOT
from numpy import savez_compressed, load

from notebook_functions import umap_transform, plot_distributions,\
                timelapse_single_frame_sync, timelapse_double_frame_sync,\
                circular_tracking, normalize_intensities, project_onto_fluo_plane

import matplotlib.cm as cm
from matplotlib import pyplot as plt
plt.style.use('_classic_test')

log_const = 300


if __name__ == '__main__':
    with open(DATA_ROOT / 'descriptors.r34.sz48.pkl', 'rb') as f:
        y, descriptors, intensities, df_index = pickle.load(f)
    descriptors = np.log(descriptors + 1e-7)
    intensities = np.log(intensities + log_const)
    print(y.shape, descriptors.shape, intensities.shape)

    embeddings_preds, fit = umap_transform(descriptors, n_neighbors=300, min_dist=0.05, n_components=2,
                                           metric='correlation')

    fig = plt.figure(figsize=(15, 15))
    plt.scatter(embeddings_preds[:, 0], embeddings_preds[:, 1])
    plt.show()


    with open(DATA_ROOT / 'descriptors_all.r34.sz48.pkl', 'rb') as f:
        y_all, descriptors_all, intensities_all, df_index_all = pickle.load(f)
    descriptors_all = np.log(descriptors_all + 1e-7)
    intensities_all = np.log(intensities_all + log_const)
    print(y_all.shape, descriptors_all.shape, intensities_all.shape)

    n_batches = 20
    batch_size = int(np.ceil(len(descriptors_all) / n_batches))
    print(len(descriptors_all), batch_size)
    for i in range(n_batches):
        start = i * batch_size
        stop = min((i + 1) * batch_size, len(descriptors_all))
        print(f'start {start}, stop {stop}')
        embeddings_preds_all = fit.transform(descriptors_all[i * batch_size: (i + 1) * batch_size])

        # fig = plt.figure(figsize=(15, 15))
        # plt.scatter(embeddings_preds_all[:, 0], embeddings_preds_all[:, 1])
        # plt.show()

        savez_compressed(DATA_ROOT / f'_embeddings_preds_all_batch{i}.npz', embeddings_preds_all)
