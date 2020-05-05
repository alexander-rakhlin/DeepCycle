import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from datagen import Iterator
from models import r34
import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint, CSVLogger
from const import DATA_ROOT, PACKAGE_DIRECTORY, double_division_tracks, curated_tracks


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
CSV_LOGGER = CSVLogger(PACKAGE_DIRECTORY / f'logs/{MODEL.__name__}.sz{CROP_SZ}.log', append=True)

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


def train():
    train_df, val_df = train_test_split(CELL_DF, FRAMES, TRAIN_TRACKS, VAL_TRACKS)
    model = MODEL(channels=CHANNELS, lr=LR, include_top=True, classes=CLASSES, weights=INIT_WEIGHTS)
    train_iterator = Iterator(CHANNEL_ROOTS, train_df, CROP_SZ,
                              shuffle=True, seed=None, infinite_loop=True, batch_size=BATCH_SIZE,
                              classes=CLASSES, target_column=TARGET_COLUMN,
                              intensity_cols=INTENSITY_COLS, output_intensities=False,
                              output_df_index=False, verbose=VERBOSE, gen_id='train')
    val_iterator = Iterator(CHANNEL_ROOTS, val_df, CROP_SZ,
                            shuffle=False, seed=None, infinite_loop=False, batch_size=BATCH_SIZE,
                            classes=CLASSES, target_column=TARGET_COLUMN,
                            intensity_cols=INTENSITY_COLS, output_intensities=False,
                            output_df_index=False, verbose=VERBOSE, gen_id='val')
    x, y = zip(*val_iterator)
    x = np.concatenate(x)
    y = np.concatenate(y)
    validation_data = x, y
    callbacks = [ModelCheckpoint(MODEL_CHECKPOINT, monitor='val_accuracy', save_best_only=True),
                 CSV_LOGGER]
    model.fit_generator(
        train_iterator,
        steps_per_epoch=len(train_df) // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=validation_data,
        validation_steps=len(val_df) // BATCH_SIZE,
        workers=3,
        callbacks=callbacks
    )


if __name__ == '__main__':
    train()


