import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from datagen import Iterator
from models import r34
import numpy as np
import pandas as pd
import pickle
from const import DATA_ROOT, PACKAGE_DIRECTORY, double_division_tracks, curated_tracks
from argparse import ArgumentParser


INTENSITY_COLS = ['GFP_20', 'Cy3_20']
CHANNEL_ROOTS = [DATA_ROOT / d for d in ['DAPI', 'BF']]
TARGET_COLUMN = 'sq20_cls2x2'
CROP_SZ = 48
LR = 1e-3

MODEL = r34
CLASSES = 4
CHANNELS = 2
BATCH_SIZE = 32
EPOCHS = 120
VERBOSE = False
INIT_WEIGHTS = None     # initial weights; you can start from pretrained model
MODEL_CHECKPOINT = str(PACKAGE_DIRECTORY / f'checkpoints/checkpoint.{MODEL.__name__}.sz{CROP_SZ}.'
                                                        f'{{epoch:02d}}-{{val_accuracy:.2f}}.hdf5')

FRAMES = range(0, 200, 1)
VAL_TRACKS = list(double_division_tracks)
TRAIN_TRACKS = [t for t in curated_tracks if t not in VAL_TRACKS]
CELL_DF = pd.read_csv(DATA_ROOT / 'statistics_mean_std.csv')


def filter_cells(cell_df, frames='all', tracks='all'):
    if frames != 'all':
        cell_df = cell_df.loc[cell_df['FRAME'].isin(frames)]
    if tracks != 'all':
        cell_df = cell_df.loc[cell_df['TRACK_ID'].isin(tracks)]
    return cell_df


def train_test_split(cell_df, frames, train_tracks, test_tracks):
    train_df = cell_df.loc[cell_df['FRAME'].isin(frames) & cell_df['TRACK_ID'].isin(train_tracks)]
    test_df = cell_df.loc[cell_df['FRAME'].isin(frames) & cell_df['TRACK_ID'].isin(test_tracks)]
    return train_df, test_df


def predict_batched(weights_path, df, descriptors_path):
    model = MODEL(channels=CHANNELS, include_top=True, classes=CLASSES, weights=weights_path)

    val_iterator = Iterator(CHANNEL_ROOTS, df, CROP_SZ,
                            shuffle=False, seed=None, infinite_loop=False, batch_size=BATCH_SIZE,
                            classes=CLASSES, target_column=TARGET_COLUMN,
                            intensity_cols=INTENSITY_COLS, output_intensities=True,
                            output_df_index=True, verbose=True, gen_id='val')
    y = []
    df_index = []
    intensities = []
    descriptors = []
    for x_, y_, intensities_, df_index_ in val_iterator:
        descriptors_ = model.predict_on_batch(x_)
        y.extend(y_)
        df_index.extend(df_index_)
        intensities.extend(intensities_)
        descriptors.extend(descriptors_)

    y = np.array(y)
    df_index = np.array(df_index)
    intensities = np.array(intensities)
    descriptors = np.array(descriptors)

    with open(descriptors_path, 'wb') as f:
        pickle.dump((y, descriptors, intensities, df_index), f)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--mode', type=str, default='encode_val', choices=['encode_val', 'encode_all'])
    parser.add_argument('--model', type=str, default='checkpoint.r34.sz48.03-0.73.hdf5')
    args = parser.parse_args()

    if args.mode == 'encode_val':
        _, pred_df = train_test_split(CELL_DF, frames=range(200), train_tracks=TRAIN_TRACKS, test_tracks=VAL_TRACKS)
        descriptors_path = DATA_ROOT / f'descriptors.{MODEL.__name__}.sz{CROP_SZ}.pkl'
    else:
        pred_df = filter_cells(CELL_DF, frames='all')
        descriptors_path = DATA_ROOT / f'descriptors_all.{MODEL.__name__}.sz{CROP_SZ}.pkl'
    predict_batched(PACKAGE_DIRECTORY / 'models' / args.model, pred_df, descriptors_path)
